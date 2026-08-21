from __future__ import annotations

import os
import shutil
import time
import uuid
import json
import math
import re
import sqlite3
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

import cadquery as cq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response, Request
from fastapi.middleware.cors import CORSMiddleware

from step_analyzer import analyze_step, _dominant_longitudinal_axis_and_length

try:
    import psycopg
except Exception:
    psycopg = None

BASE = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("STEP_CACHE_DIR", "/tmp/vakstaal_step_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
TTL_HOURS = int(os.environ.get("STEP_TTL_HOURS", "6"))

STEP_MATERIAL_LENGTH_VERSION = 3  # longest physical extent along real profile axis

app = FastAPI(title="Vakstaal STEP Server", version="1.0.0")

# No cookies/auth are used, so wildcard CORS is safe for this API.
# Later this can be restricted to calculator.vakstaal.nl / vercel.app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)





def _physical_material_length_along_axis(
    solid: cq.Shape,
) -> tuple[float, np.ndarray, str]:
    """
    Materiaallengte = langste echte longitudinale zijrand van de koker.

    Waarom:
    max-min van alle bodypunten langs een as kan bij schuine/verstek-einden
    groter zijn dan iedere werkelijke lengtezijde. Dat gaf o.a. 1157,6 mm
    terwijl de langste echte zijde in SolidWorks ~1108,49 mm is.

    Methode:
    1. bepaal de echte longitudinale profielas;
    2. verzamel alleen rechte edges die vrijwel parallel aan die as lopen;
    3. groepeer collineaire/opeenvolgende segmenten per fysieke zijlijn;
    4. neem de langste totale span van zo'n echte longitudinale zijlijn.

    Daardoor:
    - recht/recht: alle lange zijden gelijk -> één daarvan is genoeg;
    - schuin/recht of schuin/schuin: langste echte zijde wint;
    - onderbroken zijde door uitsparing: segmenten op dezelfde lijn worden
      samengenomen tot de totale longitudinale span;
    - wereldoriëntatie van het STEP-bestand maakt niet uit.
    """
    axis, analyzer_length, method = _dominant_longitudinal_axis_and_length(solid)
    axis = np.array(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("Lengteas van STEP-body kon niet worden bepaald.")
    axis = axis / norm

    # Bouw twee dwarse richtingen om fysieke zijlijnen te groeperen.
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(ref, axis))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(axis, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)

    segments = []
    for edge in solid.Edges():
        try:
            if edge.geomType() != "LINE":
                continue
            verts = edge.Vertices()
            if len(verts) < 2:
                continue

            p = np.array(verts[0].toTuple(), dtype=float)
            q = np.array(verts[-1].toTuple(), dtype=float)
            vec = q - p
            length = float(np.linalg.norm(vec))
            if length <= 1e-6:
                continue

            unit = vec / length
            parallel = abs(float(np.dot(unit, axis)))

            # Alleen echte lengte-randen. ~3.6 graden tolerantie.
            if parallel < 0.998:
                continue

            pa = float(np.dot(p, axis))
            pb = float(np.dot(q, axis))
            lo, hi = (pa, pb) if pa <= pb else (pb, pa)

            mid = (p + q) / 2.0
            tu = float(np.dot(mid, u))
            tv = float(np.dot(mid, v))

            segments.append({
                "lo": lo,
                "hi": hi,
                "u": tu,
                "v": tv,
                "len": hi - lo,
            })
        except Exception:
            continue

    if segments:
        # Groepeer segmenten die op dezelfde fysieke longitudinale zijlijn liggen.
        # Tolerantie ruim genoeg voor STEP numerical noise, klein genoeg om
        # verschillende zijden/hoeken niet samen te voegen.
        line_tol = 0.60
        groups = []

        for seg in segments:
            target = None
            for g in groups:
                if abs(seg["u"] - g["u"]) <= line_tol and abs(seg["v"] - g["v"]) <= line_tol:
                    target = g
                    break
            if target is None:
                target = {"u": seg["u"], "v": seg["v"], "intervals": []}
                groups.append(target)
            target["intervals"].append((seg["lo"], seg["hi"]))

        best_span = 0.0

        for g in groups:
            intervals = sorted(g["intervals"], key=lambda x: x[0])
            if not intervals:
                continue

            # We willen de fysieke zijde-span, ook als een uitsparing de edge
            # topologisch in meerdere stukken heeft verdeeld.
            side_lo = min(a for a, _ in intervals)
            side_hi = max(b for _, b in intervals)
            span = float(side_hi - side_lo)
            best_span = max(best_span, span)

        if best_span > 0 and math.isfinite(best_span):
            return best_span, axis, "longest-real-longitudinal-side-v3"

    # Veilige fallback voor exotische geometrie zonder herkenbare rechte lengte-edge.
    fallback = max(0.0, float(analyzer_length or 0.0))
    return fallback, axis, f"{method}-fallback"


def _apply_physical_material_lengths(step_path: Path, result: dict) -> dict:
    """
    Maak de langste fysieke profiel-lengte de enige STEP-materiaallengtebron.

    Past details[i].length_mm aan en bewaart daarnaast diagnosevelden. Daardoor
    gebruiken frontend, materiaalprijs, totalen en nesting vanzelf exact dezelfde
    lengte zonder losse correcties op verschillende plekken.
    """
    result = dict(result or {})
    details = list(result.get("details") or [])

    imported = cq.importers.importStep(str(step_path))
    solids = imported.solids().vals()

    length_audit = []

    for idx, solid in enumerate(solids):
        if idx >= len(details):
            break

        detail = dict(details[idx] or {})
        old_length = float(detail.get("length_mm") or 0.0)

        try:
            physical_length, axis, method = _physical_material_length_along_axis(solid)
        except Exception:
            physical_length = old_length
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
            method = "existing-length-fallback"

        # Geen vroege afronding: intern volledige precisie behouden.
        # De browser toont maximaal 2 decimalen.
        detail["length_mm"] = float(physical_length)
        detail["material_length_mm"] = float(physical_length)
        detail["material_length_method"] = method
        detail["material_length_version"] = STEP_MATERIAL_LENGTH_VERSION
        detail["profile_axis"] = [
            float(axis[0]), float(axis[1]), float(axis[2])
        ]

        details[idx] = detail
        length_audit.append({
            "solid_index": idx + 1,
            "previous_length_mm": old_length,
            "material_length_mm": float(physical_length),
            "difference_mm": float(physical_length - old_length),
            "method": method,
        })

    result["details"] = details
    result["material_length_version"] = STEP_MATERIAL_LENGTH_VERSION
    result["material_length_definition"] = (
        "longest real longitudinal side edge of the profile body"
    )
    result["material_length_audit"] = length_audit
    return result


def _analysis_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "analysis.json"


def _assembly_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "assembly_mesh_physical_cut_v62_material_length.json"


def _load_or_analyze(job_id: str, step_path: Path) -> dict:
    cache = _analysis_cache_path(job_id)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if int(cached.get("material_length_version") or 0) == STEP_MATERIAL_LENGTH_VERSION:
                return cached
        except Exception:
            pass

    result = analyze_step(step_path)
    result = _apply_physical_material_lengths(step_path, result)

    try:
        cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return result


def _trim_interval_from_long_edges(
    solid: cq.Shape,
    axis: np.ndarray,
    target_length: float,
    raw_length: float,
) -> tuple[float, float] | None:
    """
    Recover WHERE the analyzer's shorter intrinsic profile length exists.

    For an end-trimmed member the final STEP body typically contains repeated
    longitudinal edges of the true stock length. Their absolute projections
    define the actual interval that must remain visible. Anything outside this
    interval is trim/extend geometry and is removed from the 3D mesh.
    """
    if raw_length - target_length < 1.5:
        return None

    intervals: list[tuple[float, float]] = []

    for edge in solid.Edges():
        if edge.geomType() != "LINE":
            continue

        verts = edge.Vertices()
        if len(verts) < 2:
            continue

        p = np.array(verts[0].toTuple(), dtype=float)
        q = np.array(verts[-1].toTuple(), dtype=float)
        vec = q - p
        edge_len = float(np.linalg.norm(vec))
        if edge_len <= 1e-6:
            continue

        unit = vec / edge_len
        if abs(float(np.dot(unit, axis))) < 0.998:
            continue

        # Match the exact intrinsic length cluster, preserving .9 mm etc.
        if abs(edge_len - target_length) > 0.30:
            continue

        pa = float(np.dot(p, axis))
        pb = float(np.dot(q, axis))
        intervals.append((min(pa, pb), max(pa, pb)))

    # Hollow rectangular sections usually provide at least 4 matching long
    # edges; requiring this avoids clipping on a coincidental single edge.
    if len(intervals) < 4:
        return None

    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    start = float(np.median(starts))
    end = float(np.median(ends))

    if end <= start:
        return None
    if abs((end - start) - target_length) > 0.8:
        return None

    return start, end


def _clip_solid_to_net_length(
    solid: cq.Shape,
    detail: dict | None,
) -> tuple[cq.Shape, bool, float, float]:
    """
    Feature-preserving net-length clipping.

    The ORIGINAL STEP solid is kept, including holes, slots, tabs, notches and
    cut-outs. Only geometry outside the analyzer's true net stock interval is
    removed. No clean replacement profile is created here.
    """
    axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
    axis = np.array(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    raw_length = float(raw_length)
    target_length = float((detail or {}).get("length_mm") or raw_length)

    # Nothing to trim -> preserve exact original body.
    if raw_length - target_length < 0.75:
        return solid, False, raw_length, target_length

    interval = None

    # If a future/current analyzer provides explicit end trim values, use them.
    trim_start = float((detail or {}).get("trim_start_mm") or 0.0)
    trim_end = float((detail or {}).get("trim_end_mm") or 0.0)

    if trim_start > 0.0 or trim_end > 0.0:
        pmin, pmax = _shape_projection_interval(solid, axis)
        start = pmin + trim_start
        end = pmax - trim_end
        if end > start and abs((end - start) - target_length) <= 1.0:
            interval = (start, end)

    # Intrinsic trim detection: locate the repeated long-edge family that has
    # exactly the net length. This preserves asymmetric trims at the right end.
    if interval is None:
        interval = _trim_interval_from_long_edges(
            solid, axis, target_length, raw_length
        )

    # Fallback: recover the most-supported pair of longitudinal edge planes.
    # This still uses actual geometry, never a fixed trim amount.
    if interval is None:
        edge_intervals: list[tuple[float, float]] = []

        for edge in solid.Edges():
            if edge.geomType() != "LINE":
                continue

            verts = edge.Vertices()
            if len(verts) < 2:
                continue

            p = np.array(verts[0].toTuple(), dtype=float)
            q = np.array(verts[-1].toTuple(), dtype=float)
            vec = q - p
            edge_len = float(np.linalg.norm(vec))
            if edge_len <= 1e-6:
                continue

            unit = vec / edge_len
            if abs(float(np.dot(unit, axis))) < 0.998:
                continue

            # Ignore short hole/tab/slot edges.
            if edge_len < target_length * 0.78:
                continue

            pa = float(np.dot(p, axis))
            pb = float(np.dot(q, axis))
            edge_intervals.append((min(pa, pb), max(pa, pb)))

        # Candidate starts/ends are actual endpoints of near-full-length edges.
        starts = [a for a, _ in edge_intervals]
        ends = [b for _, b in edge_intervals]

        best = None
        for s in starts:
            e = s + target_length

            # Number of longitudinal edges supporting this start and end plane.
            start_support = sum(1 for a in starts if abs(a - s) <= 0.35)
            end_support = sum(1 for b in ends if abs(b - e) <= 0.35)

            # Also reward complete edge intervals matching this exact segment.
            full_support = sum(
                1 for a, b in edge_intervals
                if abs(a - s) <= 0.35 and abs(b - e) <= 0.35
            )

            score = full_support * 4 + min(start_support, end_support) * 2
            if score >= 8 and (best is None or score > best[0]):
                best = (score, s, e)

        if best is not None:
            interval = (best[1], best[2])

    if interval is None:
        # Safer to show original geometry than accidentally remove genuine
        # slots/tabs when the trim location cannot be proven.
        return solid, False, raw_length, target_length

    start, end = interval
    if end <= start:
        return solid, False, raw_length, target_length

    # Build a large clipping prism aligned to the profile axis.
    midpoint = (start + end) / 2.0

    center = np.array(solid.Center().toTuple(), dtype=float)
    origin = center + axis * (midpoint - float(np.dot(center, axis)))

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(ref, axis))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    xdir = np.cross(ref, axis)
    xdir = xdir / max(float(np.linalg.norm(xdir)), 1e-12)

    bb = solid.BoundingBox()
    cross_size = max(
        float(bb.xlen), float(bb.ylen), float(bb.zlen)
    ) * 3.0 + 100.0

    clip_box = (
        cq.Workplane(
            cq.Plane(
                origin=tuple(origin),
                xDir=tuple(xdir),
                normal=tuple(axis),
            )
        )
        .box(
            cross_size,
            cross_size,
            float(end - start),
            centered=(True, True, True),
        )
        .val()
    )

    # Boolean intersection removes only the material outside the two net end
    # planes. Internal holes, slots, tabs and cut-outs remain in the result.
    clipped = solid.intersect(clip_box)

    if clipped.isNull() or clipped.Volume() <= 1e-7:
        return solid, False, raw_length, target_length

    return clipped, True, raw_length, target_length



def _shape_projection_interval(shape: cq.Shape, axis: np.ndarray) -> tuple[float, float]:
    vals: list[float] = []
    for vertex in shape.Vertices():
        vals.append(float(np.dot(np.array(vertex.toTuple(), dtype=float), axis)))
    if not vals:
        center = np.array(shape.Center().toTuple(), dtype=float)
        p = float(np.dot(center, axis))
        return p, p
    return min(vals), max(vals)


def _transverse_profile_direction(
    solid: cq.Shape,
    axis: np.ndarray,
    outer_width: float,
    outer_height: float,
) -> tuple[np.ndarray, bool]:
    """
    Find a real transverse profile direction while rejecting notch/slot edges.

    Rounded RHS/SHS profiles do not contain straight cross-section edges equal
    to the full outside size (40x20 R1.75 has 36.5/16.5 mm flats). The previous
    exact-size match therefore often fell back to an arbitrary axis, which is
    why some 40x20 corner radii were wrongly shown as machining lines.
    """
    candidates: list[tuple[float, float, np.ndarray, bool]] = []
    max_dim = max(float(outer_width), float(outer_height), 1.0)
    min_dim = max(min(float(outer_width), float(outer_height)), 1.0)

    for edge in solid.Edges():
        if edge.geomType() != "LINE":
            continue

        verts = edge.Vertices()
        if len(verts) < 2:
            continue

        p = np.array(verts[0].toTuple(), dtype=float)
        q = np.array(verts[-1].toTuple(), dtype=float)
        vec = q - p
        edge_len = float(np.linalg.norm(vec))
        if edge_len <= 1e-6:
            continue

        unit = vec / edge_len
        parallel = abs(float(np.dot(unit, axis)))
        if parallel > 0.08:
            continue

        dw = abs(edge_len - outer_width)
        dh = abs(edge_len - outer_height)
        matches_width = dw <= dh
        err = min(dw, dh)

        # First preference: long flat side of the original profile. Requiring
        # at least 55% of the largest outside dimension rejects most slot/notch
        # edges while accepting radius-shortened flats (36.5 on a 40 mm side,
        # 93 on a 100 mm side, etc.).
        if edge_len >= max_dim * 0.55 and edge_len <= max_dim * 1.04:
            candidates.append((0.0, -edge_len, unit, matches_width))
            continue

        # Second preference: a convincing smaller flat side. This is only used
        # when no long-side candidate exists.
        relaxed_tol = max(0.6, min_dim * 0.22)
        if err <= relaxed_tol:
            candidates.append((1.0, err, unit, matches_width))

    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1]))
        _, _, unit, matches_width = candidates[0]
        return unit, matches_width

    # Robust arbitrary perpendicular fallback (round profiles / unusual STEP).
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(ref, axis))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    xdir = np.cross(ref, axis)
    xdir = xdir / max(float(np.linalg.norm(xdir)), 1e-12)
    return xdir, True


def _clean_profile_proxy(
    solid: cq.Shape,
    detail: dict | None,
) -> tuple[cq.Shape, bool]:
    """
    Build a clean idealized outer profile at EXACTLY the analyzer's net length.

    This intentionally removes all end trims, mitres, notches, tabs, slots and
    other cut geometry from the 3D viewer. The calculation still comes from the
    real STEP solid; only the visualization is simplified.

    Recognized square/rectangular tubes become clean outer boxes. Since hollow
    tube and solid box have the same visible outside silhouette, this is both
    much faster and visually much cleaner.
    """
    if not detail or not detail.get("recognized"):
        return solid, False

    profile_type = str(detail.get("type") or "").lower()
    outer_width = float(detail.get("outer_width_mm") or 0.0)
    outer_height = float(detail.get("outer_height_mm") or 0.0)
    net_length = float(detail.get("length_mm") or 0.0)

    if net_length <= 0.0:
        return solid, False

    axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
    axis = np.array(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    # Determine exact longitudinal placement from the same repeated long-edge
    # family used for the net length. This preserves asymmetric end trims.
    interval = _trim_interval_from_long_edges(
        solid, axis, net_length, float(raw_length)
    )

    if interval is None:
        pmin, pmax = _shape_projection_interval(solid, axis)
        if abs((pmax - pmin) - net_length) <= 1.0:
            interval = (pmin, pmax)
        else:
            # Last-resort placement: center net stock inside the original body.
            mid = (pmin + pmax) / 2.0
            interval = (mid - net_length / 2.0, mid + net_length / 2.0)

    start, end = interval
    midpoint = (start + end) / 2.0

    original_center = np.array(solid.Center().toTuple(), dtype=float)
    origin = original_center + axis * (
        midpoint - float(np.dot(original_center, axis))
    )

    # Square / rectangular standard profiles.
    if (
        outer_width > 0.1
        and outer_height > 0.1
        and (
            "vierkant" in profile_type
            or "rechthoek" in profile_type
            or "koker" in profile_type
        )
    ):
        xdir, x_matches_width = _transverse_profile_direction(
            solid, axis, outer_width, outer_height
        )

        xdim = outer_width if x_matches_width else outer_height
        ydim = outer_height if x_matches_width else outer_width

        plane = cq.Plane(
            origin=tuple(origin),
            xDir=tuple(xdir),
            normal=tuple(axis),
        )

        proxy = (
            cq.Workplane(plane)
            .box(
                float(xdim),
                float(ydim),
                float(net_length),
                centered=(True, True, True),
            )
            .val()
        )
        return proxy, True

    # Round standard profiles: use clean outside cylinder.
    if "rond" in profile_type and outer_width > 0.1:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(ref, axis))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        xdir = np.cross(ref, axis)
        xdir = xdir / max(float(np.linalg.norm(xdir)), 1e-12)

        plane = cq.Plane(
            origin=tuple(origin),
            xDir=tuple(xdir),
            normal=tuple(axis),
        )

        proxy = (
            cq.Workplane(plane)
            .circle(float(outer_width) / 2.0)
            .extrude(float(net_length) / 2.0, both=True)
            .val()
        )
        return proxy, True

    return solid, False



def _edge_points(edge: cq.Shape) -> list[list[float]]:
    """Sample a topological edge for WebGL feature highlighting."""
    pts: list[list[float]] = []

    try:
        # CadQuery edge discretization gives enough points for circles/arcs.
        raw = edge.discretize(18)
        for p in raw:
            try:
                pts.append([float(p.x), float(p.y), float(p.z)])
            except Exception:
                pts.append([float(p.X), float(p.Y), float(p.Z)])
    except Exception:
        # Some CadQuery/OCC builds do not expose discretize() on every edge
        # type (notably BSPLINE). positionAt() does, so use it before falling
        # back to only the vertices. This is important for oblique cuts through
        # a rounded tube corner, which are often exported as BSPLINE curves.
        try:
            for t in np.linspace(0.0, 1.0, 18):
                p = edge.positionAt(float(t))
                pts.append([float(p.x), float(p.y), float(p.z)])
        except Exception:
            try:
                verts = edge.Vertices()
                for v in verts:
                    x, y, z = v.toTuple()
                    pts.append([float(x), float(y), float(z)])
            except Exception:
                return []

    # Remove consecutive duplicates.
    clean: list[list[float]] = []
    for p in pts:
        if not clean:
            clean.append(p)
            continue
        q = clean[-1]
        if math.dist(p, q) > 1e-5:
            clean.append(p)
    return clean



def _profile_basis_for_features(
    solid: cq.Shape,
    detail: dict | None,
) -> tuple:
    axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
    zdir = np.array(axis, dtype=float)
    zdir /= max(float(np.linalg.norm(zdir)), 1e-12)

    ow = float((detail or {}).get("outer_width_mm") or 0.0)
    oh = float((detail or {}).get("outer_height_mm") or 0.0)
    profile_type = str((detail or {}).get("type") or "").lower()

    xdir, x_matches_width = _transverse_profile_direction(
        solid, zdir, max(ow, 1.0), max(oh, 1.0)
    )
    xdir = np.array(xdir, dtype=float)
    xdir /= max(float(np.linalg.norm(xdir)), 1e-12)
    ydir = np.cross(zdir, xdir)
    ydir /= max(float(np.linalg.norm(ydir)), 1e-12)

    if not x_matches_width:
        xdir, ydir = ydir, xdir

    # v399: use the geometric bounding centre in the real profile basis rather
    # than the mass centre. Notches/holes can shift Center(), which previously
    # made outside/inside classification unstable.
    projections = []
    for v in solid.Vertices():
        p = np.array(v.toTuple(), dtype=float)
        projections.append((
            float(np.dot(p, xdir)),
            float(np.dot(p, ydir)),
            float(np.dot(p, zdir)),
        ))

    if projections:
        xs=[p[0] for p in projections]
        ys=[p[1] for p in projections]
        zs=[p[2] for p in projections]
        xmid=(min(xs)+max(xs))/2.0
        ymid=(min(ys)+max(ys))/2.0
        zmid=(min(zs)+max(zs))/2.0
        c=xdir*xmid + ydir*ymid + zdir*zmid
        half_len=(max(zs)-min(zs))/2.0
    else:
        c=np.array(solid.Center().toTuple(), dtype=float)
        half_len=float(raw_length)/2.0

    # Learn all longitudinal stock seam/tangent positions. They are useful for
    # removing the true profile seams, but curved END arcs are no longer
    # discarded merely because they touch these anchors.
    seam_anchors: list[tuple[float, float]] = []
    anchor_min_len = max(20.0, float(raw_length) * 0.30)

    for edge in solid.Edges():
        if str(edge.geomType() or "").upper() != "LINE":
            continue
        verts=edge.Vertices()
        if len(verts)<2:
            continue
        p=np.array(verts[0].toTuple(),dtype=float)
        q=np.array(verts[-1].toTuple(),dtype=float)
        vec=q-p
        edge_len=float(np.linalg.norm(vec))
        if edge_len<anchor_min_len:
            continue
        unit=vec/max(edge_len,1e-12)
        if abs(float(np.dot(unit,zdir)))<0.9985:
            continue
        mid=((p+q)/2.0)-c
        seam_anchors.append((
            float(np.dot(mid,xdir)),
            float(np.dot(mid,ydir)),
        ))

    anchor_tol=max(0.30,min(max(ow,1.0),max(oh,1.0))*0.010)
    unique_anchors=[]
    for ax,ay in seam_anchors:
        if not any(math.hypot(ax-bx,ay-by)<=anchor_tol for bx,by in unique_anchors):
            unique_anchors.append((ax,ay))

    is_round=("rond" in profile_type) and not any(
        k in profile_type for k in ("vierkant","rechthoek","koker")
    )

    def shell_score_xy(x: float,y: float)->float:
        if is_round:
            radius=max(ow,oh,1.0)/2.0
            return math.hypot(x,y)/radius
        return max(
            abs(x)/max(ow/2.0,1e-6),
            abs(y)/max(oh/2.0,1e-6),
        )

    # Long stock seams exist on both the outside and inside skin. Their
    # normalized radial levels give us the wall-thickness split without ever
    # needing the nominal wall thickness from the file name.
    scores=sorted(
        {round(shell_score_xy(ax,ay),5) for ax,ay in unique_anchors},
        reverse=True
    )
    outer_shell_threshold=0.965
    if scores:
        hi=scores[0]
        lower=[v for v in scores[1:] if hi-v>=0.012]
        if lower:
            outer_shell_threshold=(hi+lower[0])/2.0
        else:
            outer_shell_threshold=hi*0.970
        outer_shell_threshold=max(0.70,min(1.03,outer_shell_threshold))

    return (
        c,xdir,ydir,zdir,ow,oh,half_len,
        unique_anchors,anchor_tol,
        outer_shell_threshold,is_round,float(raw_length)
    )


def _basis_coords(points: list[list[float]], basis: tuple) -> list[tuple[float,float,float]]:
    c,xdir,ydir,zdir,*_ = basis
    result=[]
    for p0 in points:
        p=np.array(p0,dtype=float)-c
        result.append((
            float(np.dot(p,xdir)),
            float(np.dot(p,ydir)),
            float(np.dot(p,zdir)),
        ))
    return result


def _shell_score(x: float,y: float,basis: tuple)->float:
    _,_,_,_,ow,oh,_,_,_,_,is_round,_=basis
    if is_round:
        return math.hypot(x,y)/max(max(ow,oh,1.0)/2.0,1e-6)
    return max(
        abs(x)/max(ow/2.0,1e-6),
        abs(y)/max(oh/2.0,1e-6),
    )


def _near_profile_anchor(x: float,y: float,basis: tuple,factor: float=1.0)->bool:
    *_,seam_anchors,anchor_tol,_,_,_=basis
    tol=anchor_tol*factor
    return any(math.hypot(x-ax,y-ay)<=tol for ax,ay in seam_anchors)


def _is_standard_profile_edge(edge: cq.Shape,basis: tuple)->bool:
    """
    True only for a real longitudinal stock/profile seam.

    v399 deliberately does NOT classify a transverse CIRCLE/BSPLINE as stock
    merely because its endpoints touch radius anchors. Such an arc at an end,
    notch, slot or opening is a real laser path and must remain visible.
    """
    pts=_edge_points(edge)
    if len(pts)<2:
        return False
    coords=_basis_coords(pts,basis)
    _,_,_,_,_,_,_,_,_,_,_,raw_length=basis

    a=np.array(coords[0],dtype=float)
    b=np.array(coords[-1],dtype=float)
    vec=b-a
    chord=float(np.linalg.norm(vec))
    if chord<=1e-7:
        return False

    axis_ratio=abs(float(vec[2]))/chord
    gt=str(edge.geomType() or "").upper()

    if gt=="LINE":
        if axis_ratio>=0.9985:
            mx=float((a[0]+b[0])/2.0)
            my=float((a[1]+b[1])/2.0)
            if _near_profile_anchor(mx,my,basis,1.35):
                return True
        return False

    # Curved longitudinal seam/split edge: only suppress when it truly runs a
    # substantial distance along the tube and stays on one learned stock seam.
    if gt in {"CIRCLE","ELLIPSE","BSPLINE","BEZIER"}:
        if axis_ratio>=0.985 and chord>=max(15.0,raw_length*0.20):
            x0,y0,_=coords[0]
            x1,y1,_=coords[-1]
            if (
                _near_profile_anchor(x0,y0,basis,1.5)
                and _near_profile_anchor(x1,y1,basis,1.5)
            ):
                return True
        return False

    return False


def _edge_is_on_outer_skin(edge: cq.Shape,basis: tuple)->bool:
    """
    A physical tube-laser contour is drawn on the OUTSIDE skin.

    Outer contour edge: both endpoints are on the learned outside shell.
    Inner BREP loop: both endpoints sit below the wall split -> reject.
    Wall-thickness connector: one outer + one inner endpoint -> reject.
    """
    pts=_edge_points(edge)
    if len(pts)<2:
        return False
    coords=_basis_coords(pts,basis)
    threshold=float(basis[9])

    endpoint_scores=[
        _shell_score(coords[0][0],coords[0][1],basis),
        _shell_score(coords[-1][0],coords[-1][1],basis),
    ]

    # A little numerical allowance for spline discretization / STEP tolerance.
    if min(endpoint_scores)<threshold-0.008:
        return False

    # Protect large rounded outside corners: endpoints are on tangency lines,
    # but the middle of the arc naturally lies slightly further inward.
    sample_scores=[
        _shell_score(x,y,basis)
        for x,y,_ in coords
    ]
    return float(np.percentile(sample_scores,20))>=threshold-0.055


def _point_key(p:list[float],tol:float=0.12)->tuple[int,int,int]:
    t=max(tol,1e-6)
    return (
        int(round(float(p[0])/t)),
        int(round(float(p[1])/t)),
        int(round(float(p[2])/t)),
    )


def _connected_edge_components(items:list[dict],tol:float=0.12)->list[list[dict]]:
    if not items:
        return []
    parent=list(range(len(items)))

    def find(i:int)->int:
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i

    def union(a:int,b:int):
        a,b=find(a),find(b)
        if a!=b:
            parent[b]=a

    owners={}
    for i,item in enumerate(items):
        pts=item["pts"]
        for p in (pts[0],pts[-1]):
            key=_point_key(p,tol)
            if key in owners:
                union(i,owners[key])
            else:
                owners[key]=i

    groups={}
    for i,item in enumerate(items):
        groups.setdefault(find(i),[]).append(item)
    return list(groups.values())


def _physical_cut_polylines(
    solid:cq.Shape,
    detail:dict|None,
)->tuple[list[list[list[float]]],list[list[list[float]]],int]:
    """
    Return (base end contour edges, extra machining edges, base contour count).

    v401:
    A physical tube end is ONE laser contour, but STEP can split that contour
    into several line/arc components. Earlier code kept only the longest
    component at each end. That made angled/rounded ends slightly too short
    and moved valid end pieces into "extra machining".

    This version:
    - keeps only the physical OUTER skin
    - finds one terminal family at each longitudinal end
    - grows that family with nearby disconnected line/arc components
    - reports the real number of terminal contours separately (normally 2/body)
    """
    if not detail or not detail.get("recognized"):
        return [],[],0

    profile_type=str(detail.get("type") or "").lower()
    if not any(k in profile_type for k in ("vierkant","rechthoek","koker","rond")):
        return [],[],0

    try:
        basis=_profile_basis_for_features(solid,detail)
    except Exception:
        return [],[],0

    candidates=[]
    try:
        for edge in solid.Edges():
            if _is_standard_profile_edge(edge,basis):
                continue
            if not _edge_is_on_outer_skin(edge,basis):
                continue
            pts=_edge_points(edge)
            if len(pts)<2:
                continue
            coords=_basis_coords(pts,basis)
            zvals=[p[2] for p in coords]
            candidates.append({
                "pts":pts,
                "zmean":sum(zvals)/len(zvals),
                "zmin":min(zvals),
                "zmax":max(zvals),
                "length":sum(
                    math.dist(pts[i],pts[i+1])
                    for i in range(len(pts)-1)
                ),
            })
    except Exception:
        return [],[],0

    if not candidates:
        return [],[],0

    components=_connected_edge_components(candidates,tol=0.20)
    _,_,_,_,ow,oh,half_len,*_=basis

    summaries=[]
    for comp in components:
        total_len=sum(float(e["length"]) for e in comp)
        weighted_z=sum(
            float(e["zmean"])*max(float(e["length"]),1e-6)
            for e in comp
        )/max(total_len,1e-6)

        all_pts=[]
        for e in comp:
            if e.get("pts"):
                all_pts.append(e["pts"][0])
                all_pts.append(e["pts"][-1])

        summaries.append({
            "edges":comp,
            "length":total_len,
            "zmean":weighted_z,
            "zmin":min(float(e["zmin"]) for e in comp),
            "zmax":max(float(e["zmax"]) for e in comp),
            "endpoints":all_pts,
        })

    transverse=max(ow,oh,1.0)
    raw_length=float(basis[-1])

    # Wide enough for a strong mitre, but only used to locate the seed family.
    end_zone=max(transverse*1.40,raw_length*0.065,4.0)

    plus_candidates=[
        c for c in summaries
        if c["zmean"]>=half_len-end_zone
    ]
    minus_candidates=[
        c for c in summaries
        if c["zmean"]<=-half_len+end_zone
    ]

    plus_seed=max(plus_candidates,key=lambda x:x["length"]) if plus_candidates else None
    minus_seed=max(minus_candidates,key=lambda x:x["length"]) if minus_candidates else None

    # Fallback: use the components furthest apart along the member.
    if plus_seed is None and summaries:
        plus_seed=max(summaries,key=lambda x:x["zmean"])
    if minus_seed is None and summaries:
        minus_seed=min(summaries,key=lambda x:x["zmean"])

    def component_distance(a:dict,b:dict)->float:
        best=float("inf")
        for p in a.get("endpoints",[]):
            for q in b.get("endpoints",[]):
                try:
                    best=min(best,math.dist(p,q))
                except Exception:
                    pass
        return best

    # STEP curve endpoints can miss each other by fractions of a mm. In some
    # exporters a rounded corner is split into a separate component several mm
    # away from the straight segment. Absorb only very-near pieces so a nearby
    # hole/slot is not accidentally converted into an end cut.
    join_tol=max(1.0,min(12.0,transverse*0.11))
    axial_family_tol=max(transverse*1.55,raw_length*0.075,6.0)

    def grow_family(seed:dict|None)->list[dict]:
        if seed is None:
            return []
        family=[seed]
        changed=True
        while changed:
            changed=False
            for comp in summaries:
                if comp in family:
                    continue

                close_to_family=any(
                    component_distance(comp,member)<=join_tol
                    for member in family
                )
                same_end_zone=abs(comp["zmean"]-seed["zmean"])<=axial_family_tol

                if close_to_family and same_end_zone:
                    family.append(comp)
                    changed=True
        return family

    plus_family=grow_family(plus_seed)
    minus_family=grow_family(minus_seed)

    # Never let the same component belong to both end families.
    plus_ids={id(c) for c in plus_family}
    minus_family=[c for c in minus_family if id(c) not in plus_ids]

    # A normal open tube body has two terminal cuts. If one family could not be
    # resolved, do not invent a count; the audit UI will expose it immediately.
    base_contour_count=(1 if plus_family else 0)+(1 if minus_family else 0)

    base_ids={id(c) for c in plus_family+minus_family}
    base_lines=[]
    feature_lines=[]

    for comp in summaries:
        target=base_lines if id(comp) in base_ids else feature_lines
        for edge in comp["edges"]:
            target.append(edge["pts"])

    return base_lines,feature_lines,base_contour_count

def _feature_polylines(
    solid:cq.Shape,
    detail:dict|None,
)->list[list[list[float]]]:
    _base,features,_count=_physical_cut_polylines(solid,detail)
    return features


def _base_cut_polylines(
    solid:cq.Shape,
    detail:dict|None,
)->list[list[list[float]]]:
    base,_features,_count=_physical_cut_polylines(solid,detail)
    return base


def _mesh_shape(shape: cq.Shape, *, center_vertices: bool = False) -> dict:
    vertices, triangles = shape.tessellate(0.45, 0.12)

    verts = []
    for v in vertices:
        try:
            verts.append([float(v.x), float(v.y), float(v.z)])
        except Exception:
            verts.append([float(v.X), float(v.Y), float(v.Z)])

    tris = []
    for tri in triangles:
        try:
            a, b, c = tri
            tris.append([int(a), int(b), int(c)])
        except Exception:
            continue

    if not verts or not tris:
        raise ValueError("Geen zichtbare 3D-mesh gevonden.")

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]

    if center_vertices:
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0
        verts = [[x - cx, y - cy, z - cz] for x, y, z in verts]

    size = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1e-6,
    )

    return {
        "vertices": verts,
        "triangles": tris,
        "size": float(size),
    }


def cleanup_old_jobs() -> None:
    cutoff = time.time() - TTL_HOURS * 3600
    for folder in CACHE_DIR.iterdir():
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def job_step_path(job_id: str) -> Path:
    folder = CACHE_DIR / job_id
    candidates = list(folder.glob("source.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="STEP-sessie niet gevonden of verlopen.")
    return candidates[0]



# ============================================================================
# OFFERTE DATABASE + STEP/NEST BESTANDEN
# ============================================================================

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
QUOTE_DB_PATH = Path(os.environ.get("QUOTE_DB_PATH", "/tmp/vakstaal_quotes.sqlite3"))
MAX_QUOTE_FILE_MB = int(os.environ.get("MAX_QUOTE_FILE_MB", "100"))

DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN", "").strip()
DROPBOX_REFRESH_TOKEN = os.environ.get("DROPBOX_REFRESH_TOKEN", "").strip()
DROPBOX_APP_KEY = os.environ.get("DROPBOX_APP_KEY", "").strip()
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET", "").strip()
DROPBOX_ROOT = os.environ.get("DROPBOX_ROOT", "/Offertes").strip() or "/Offertes"

_dropbox_runtime_access_token = DROPBOX_ACCESS_TOKEN


def _dropbox_refresh_access_token() -> str:
    global _dropbox_runtime_access_token

    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        raise HTTPException(
            status_code=503,
            detail=(
                "Dropbox access token is verlopen. Stel DROPBOX_REFRESH_TOKEN, "
                "DROPBOX_APP_KEY en DROPBOX_APP_SECRET in op de server."
            )
        )

    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_APP_KEY,
        "client_secret": DROPBOX_APP_SECRET,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            result=json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8",errors="replace")
        raise HTTPException(
            status_code=503,
            detail=f"Dropbox token vernieuwen mislukt ({exc.code}): {detail[:500]}"
        ) from exc

    token=str(result.get("access_token") or "").strip()
    if not token:
        raise HTTPException(status_code=503,detail="Dropbox gaf geen nieuw access token terug.")

    _dropbox_runtime_access_token=token
    return token


def _dropbox_token(force_refresh: bool=False) -> str:
    if force_refresh:
        return _dropbox_refresh_access_token()
    if _dropbox_runtime_access_token:
        return _dropbox_runtime_access_token
    if DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET:
        return _dropbox_refresh_access_token()
    raise HTTPException(status_code=503,detail="Dropbox-token ontbreekt op de server.")


def _dropbox_headers(force_refresh: bool=False) -> dict:
    return {"Authorization": f"Bearer {_dropbox_token(force_refresh)}"}


def _dropbox_token_expired(detail: str) -> bool:
    value=str(detail or "").lower()
    return "expired_access_token" in value or "invalid_access_token" in value


def _dropbox_rpc(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"https://api.dropboxapi.com/2/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            **_dropbox_headers(),
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox API fout ({exc.code}): {detail[:700]}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox verbinding mislukt: {type(exc).__name__}: {exc}"
        ) from exc


def _dropbox_upload_bytes(path: str, data: bytes) -> dict:
    args={"path":path,"mode":"overwrite","autorename":False,"mute":True}

    def attempt(force_refresh=False):
        req=urllib.request.Request(
            "https://content.dropboxapi.com/2/files/upload",
            data=data,
            method="POST",
            headers={
                **_dropbox_headers(force_refresh),
                "Content-Type":"application/octet-stream",
                "Dropbox-API-Arg":json.dumps(args,separators=(",",":")),
            },
        )
        with urllib.request.urlopen(req,timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return attempt(False)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8",errors="replace")
        if exc.code==401 and _dropbox_token_expired(detail):
            try:
                return attempt(True)
            except HTTPException:
                raise
            except urllib.error.HTTPError as retry:
                retry_detail=retry.read().decode("utf-8",errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Dropbox uploadfout na tokenvernieuwing ({retry.code}): {retry_detail[:700]}"
                ) from retry
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox uploadfout ({exc.code}): {detail[:700]}"
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox upload mislukt: {type(exc).__name__}: {exc}"
        ) from exc

def _dropbox_download_bytes(path: str) -> bytes:
    req = urllib.request.Request(
        "https://content.dropboxapi.com/2/files/download",
        data=b"",
        method="POST",
        headers={
            **_dropbox_headers(),
            "Dropbox-API-Arg": json.dumps({"path": path}, separators=(",", ":")),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox downloadfout ({exc.code}): {detail[:700]}"
        ) from exc


def _dropbox_delete_path(path: str) -> None:
    if not path:
        return
    try:
        _dropbox_rpc("files/delete_v2", {"path": path})
    except HTTPException as exc:
        if "not_found" not in str(exc.detail).lower():
            raise


def _safe_dropbox_name(value: str, fallback: str = "Onbekend") -> str:
    value = str(value or "").strip()
    value = re.sub(r'[<>:"/\\|?*]+', "-", value)
    value = re.sub(r"\s+", " ", value).strip(" .-")
    return (value or fallback)[:120]


def _quote_dropbox_folder(quote_number: str, customer_name: str) -> str:
    year_match = re.search(r"(20\d{2})", str(quote_number or ""))
    year = year_match.group(1) if year_match else str(datetime.now().year)
    q = _safe_dropbox_name(quote_number, "Offerte")
    c = _safe_dropbox_name(customer_name, "Klant")
    return f"{DROPBOX_ROOT.rstrip('/')}/{year}/{q} - {c}"


def _file_dropbox_subfolder(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    lower = str(filename or "").lower()
    if suffix in {".zx", ".nest"}:
        return "Nest"
    if suffix in {".step", ".stp"}:
        production_words = ("productie", "production", "solid_", "onderdeel_", "part_")
        return "Productie STEP" if any(w in lower for w in production_words) else "Origineel"
    return "Overige bestanden"


def _quote_identity(conn, quote_id: str) -> tuple[str, str, str]:
    cur = conn.cursor()
    cur.execute(
        _sql(
            "SELECT quote_number, customer_name, payload_json FROM quotes WHERE id=%s",
            "SELECT quote_number, customer_name, payload_json FROM quotes WHERE id=?"
        ),
        (quote_id,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Offerte niet gevonden.")
    if isinstance(row, sqlite3.Row):
        return row["quote_number"], row["customer_name"], row["payload_json"]
    return row[0], row[1], row[2]


def _sync_quote_json_to_dropbox(conn, quote_id: str) -> str:
    quote_number, customer_name, payload_json = _quote_identity(conn, quote_id)
    folder = _quote_dropbox_folder(quote_number, customer_name)
    try:
        payload = json.loads(payload_json)
    except Exception:
        payload = {}
    wrapper = {
        "quote_id": quote_id,
        "quote_number": quote_number,
        "customer_name": customer_name,
        "saved_at": _utcnow(),
        "payload": payload,
    }
    _dropbox_upload_bytes(
        f"{folder}/offerte.json",
        json.dumps(wrapper, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return folder


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _postgres_enabled() -> bool:
    return bool(DATABASE_URL and psycopg is not None)


def _db_connect():
    if _postgres_enabled():
        return psycopg.connect(DATABASE_URL)

    # Alleen fallback/test. Voor productie gebruiken we jouw Render PostgreSQL.
    QUOTE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(QUOTE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _sql(postgres_sql: str, sqlite_sql: str) -> str:
    return postgres_sql if _postgres_enabled() else sqlite_sql


def _row_to_dict(row, cursor=None):
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if isinstance(row, dict):
        return row
    if cursor is not None and cursor.description:
        names = []
        for col in cursor.description:
            names.append(col.name if hasattr(col, "name") else col[0])
        return {names[i]: row[i] for i in range(len(row))}
    return row


def _init_quote_db() -> None:
    with _db_connect() as conn:
        cur = conn.cursor()

        cur.execute(
            _sql(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    id TEXT PRIMARY KEY,
                    quote_number TEXT UNIQUE NOT NULL,
                    customer_name TEXT NOT NULL,
                    contact_person TEXT,
                    customer_email TEXT,
                    customer_phone TEXT,
                    total_ex_vat DOUBLE PRECISION NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    id TEXT PRIMARY KEY,
                    quote_number TEXT UNIQUE NOT NULL,
                    customer_name TEXT NOT NULL,
                    contact_person TEXT,
                    customer_email TEXT,
                    customer_phone TEXT,
                    total_ex_vat REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        )

        cur.execute(
            _sql(
                """
                CREATE TABLE IF NOT EXISTS quote_files (
                    id TEXT PRIMARY KEY,
                    quote_id TEXT NOT NULL REFERENCES quotes(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    file_kind TEXT,
                    file_size BIGINT NOT NULL,
                    data BYTEA NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS quote_files (
                    id TEXT PRIMARY KEY,
                    quote_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    file_kind TEXT,
                    file_size INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quote_id) REFERENCES quotes(id) ON DELETE CASCADE
                )
                """
            )
        )

        # Dropbox metadata toevoegen zonder bestaande offertes te breken.
        if _postgres_enabled():
            cur.execute("ALTER TABLE quote_files ADD COLUMN IF NOT EXISTS dropbox_path TEXT")
        else:
            cur.execute("PRAGMA table_info(quote_files)")
            sqlite_columns = {row[1] for row in cur.fetchall()}
            if "dropbox_path" not in sqlite_columns:
                cur.execute("ALTER TABLE quote_files ADD COLUMN dropbox_path TEXT")

        cur.execute(
            _sql(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    state_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    state_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        )

        conn.commit()


def _file_kind(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".step", ".stp"}:
        return "STEP"
    if suffix in {".zx", ".nest"}:
        return "ZX/Nest"
    return "Bestand"


def _next_quote_number(conn) -> str:
    year = datetime.now().year
    prefix = f"VAK-{year}-"
    cur = conn.cursor()

    cur.execute(
        _sql(
            "SELECT quote_number FROM quotes WHERE quote_number LIKE %s ORDER BY quote_number DESC LIMIT 1",
            "SELECT quote_number FROM quotes WHERE quote_number LIKE ? ORDER BY quote_number DESC LIMIT 1"
        ),
        (prefix + "%",)
    )

    row = cur.fetchone()
    last = 0

    if row:
        value = row[0] if not isinstance(row, sqlite3.Row) else row["quote_number"]
        try:
            last = int(str(value).rsplit("-", 1)[-1])
        except Exception:
            last = 0

    return f"{prefix}{last + 1:04d}"


def _quote_files(conn, quote_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        _sql(
            """
            SELECT id, filename, content_type, file_kind, file_size, dropbox_path, created_at
            FROM quote_files
            WHERE quote_id=%s
            ORDER BY created_at
            """,
            """
            SELECT id, filename, content_type, file_kind, file_size, dropbox_path, created_at
            FROM quote_files
            WHERE quote_id=?
            ORDER BY created_at
            """
        ),
        (quote_id,)
    )
    result = [_row_to_dict(r, cur) for r in cur.fetchall()]
    for item in result:
        item["storage"] = "dropbox" if item.get("dropbox_path") else "database"
    return result


async def _store_quote_files(conn, quote_id: str, files: list[UploadFile]) -> None:
    quote_number, customer_name, _payload_json = _quote_identity(conn, quote_id)
    folder = _quote_dropbox_folder(quote_number, customer_name)

    for upload in files or []:
        filename = _safe_dropbox_name(upload.filename or "bestand", "bestand")
        data = await upload.read()

        if not data:
            continue

        if len(data) > MAX_QUOTE_FILE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} is groter dan {MAX_QUOTE_FILE_MB} MB."
            )

        kind = _file_kind(filename)
        subfolder = _file_dropbox_subfolder(filename)
        dropbox_path = f"{folder}/{subfolder}/{filename}"

        cur = conn.cursor()
        cur.execute(
            _sql(
                """
                SELECT id, dropbox_path FROM quote_files
                WHERE quote_id=%s AND filename=%s AND file_size=%s
                LIMIT 1
                """,
                """
                SELECT id, dropbox_path FROM quote_files
                WHERE quote_id=? AND filename=? AND file_size=?
                LIMIT 1
                """
            ),
            (quote_id, filename, len(data))
        )
        existing = cur.fetchone()
        if existing:
            existing_path = existing["dropbox_path"] if isinstance(existing, sqlite3.Row) else existing[1]
            if existing_path:
                continue

        uploaded = _dropbox_upload_bytes(dropbox_path, data)
        actual_path = uploaded.get("path_display") or uploaded.get("path_lower") or dropbox_path

        if existing:
            existing_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
            cur.execute(
                _sql(
                    "UPDATE quote_files SET dropbox_path=%s, data=%s WHERE id=%s",
                    "UPDATE quote_files SET dropbox_path=?, data=? WHERE id=?"
                ),
                (actual_path, b"", existing_id)
            )
            continue

        file_id = uuid.uuid4().hex
        cur.execute(
            _sql(
                """
                INSERT INTO quote_files
                (id, quote_id, filename, content_type, file_kind, file_size, data, dropbox_path, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                """
                INSERT INTO quote_files
                (id, quote_id, filename, content_type, file_kind, file_size, data, dropbox_path, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """
            ),
            (
                file_id,
                quote_id,
                filename,
                upload.content_type or "application/octet-stream",
                kind,
                len(data),
                b"",
                actual_path,
                _utcnow(),
            )
        )


def _quote_response(conn, quote_id: str) -> dict:
    cur = conn.cursor()
    cur.execute(
        _sql(
            """
            SELECT id, quote_number, customer_name, contact_person, customer_email,
                   customer_phone, total_ex_vat, payload_json, created_at, updated_at
            FROM quotes
            WHERE id=%s
            """,
            """
            SELECT id, quote_number, customer_name, contact_person, customer_email,
                   customer_phone, total_ex_vat, payload_json, created_at, updated_at
            FROM quotes
            WHERE id=?
            """
        ),
        (quote_id,)
    )

    row = _row_to_dict(cur.fetchone(), cur)
    if not row:
        raise HTTPException(status_code=404, detail="Offerte niet gevonden.")

    try:
        row["payload"] = json.loads(row.pop("payload_json"))
    except Exception:
        row["payload"] = {}
        row.pop("payload_json", None)

    row["files"] = _quote_files(conn, quote_id)
    return row


_init_quote_db()



# ---------------------------------------------------------------------------
# Centrale Vakstaal bibliotheek + standaardinstellingen
# ---------------------------------------------------------------------------

APP_STATE_KEY = "vakstaal_global_state"


@app.get("/api/app-state")
def get_app_state():
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _sql(
                "SELECT payload_json, updated_at FROM app_state WHERE state_key=%s",
                "SELECT payload_json, updated_at FROM app_state WHERE state_key=?"
            ),
            (APP_STATE_KEY,)
        )
        row = cur.fetchone()

        if not row:
            return {
                "ok": True,
                "exists": False,
                "state": None,
                "updated_at": None,
            }

        if isinstance(row, sqlite3.Row):
            payload_json = row["payload_json"]
            updated_at = row["updated_at"]
        else:
            payload_json, updated_at = row

        try:
            state = json.loads(payload_json)
        except Exception:
            state = {}

        return {
            "ok": True,
            "exists": True,
            "state": state,
            "updated_at": updated_at,
        }


@app.put("/api/app-state")
def put_app_state(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ongeldige instellingen.")

    # Alleen globale data bewaren; geen actieve offerte/calculatie.
    state = {
        "schema": int(payload.get("schema") or 1),
        "materials": payload.get("materials") if isinstance(payload.get("materials"), list) else [],
        "settings": payload.get("settings") if isinstance(payload.get("settings"), dict) else {},
        "saved_at": _utcnow(),
    }

    now = _utcnow()
    raw = json.dumps(state, ensure_ascii=False)

    with _db_connect() as conn:
        cur = conn.cursor()

        if _postgres_enabled():
            cur.execute(
                """
                INSERT INTO app_state (state_key, payload_json, updated_at)
                VALUES (%s,%s,%s)
                ON CONFLICT (state_key)
                DO UPDATE SET payload_json=EXCLUDED.payload_json, updated_at=EXCLUDED.updated_at
                """,
                (APP_STATE_KEY, raw, now)
            )
        else:
            cur.execute(
                """
                INSERT INTO app_state (state_key, payload_json, updated_at)
                VALUES (?,?,?)
                ON CONFLICT(state_key)
                DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (APP_STATE_KEY, raw, now)
            )

        conn.commit()

    return {
        "ok": True,
        "saved": True,
        "updated_at": now,
        "material_count": len(state["materials"]),
    }



QUOTE_MULTIPART_MAX_PART_SIZE = int(os.getenv("QUOTE_MULTIPART_MAX_PART_SIZE", str(100 * 1024 * 1024)))
QUOTE_MULTIPART_MAX_FILES = int(os.getenv("QUOTE_MULTIPART_MAX_FILES", "200"))
QUOTE_MULTIPART_MAX_FIELDS = int(os.getenv("QUOTE_MULTIPART_MAX_FIELDS", "50"))


async def _parse_large_quote_form(request: Request):
    """
    Parse quote multipart data with a larger per-part limit than Starlette's
    default 1 MB. STEP/PDF files are stored separately in Dropbox and may
    legitimately be several MB.
    """
    try:
        form = await request.form(
            max_files=QUOTE_MULTIPART_MAX_FILES,
            max_fields=QUOTE_MULTIPART_MAX_FIELDS,
            max_part_size=QUOTE_MULTIPART_MAX_PART_SIZE,
        )
    except TypeError:
        # Compatibility fallback for older Starlette versions.
        form = await request.form()
    payload = str(form.get("payload") or "")
    files = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
    return form, payload, files


@app.get("/api/quotes")
def list_quotes():
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, quote_number, customer_name, contact_person, customer_email,
                   customer_phone, total_ex_vat, created_at, updated_at
            FROM quotes
            ORDER BY updated_at DESC
            """
        )

        rows = [_row_to_dict(r, cur) for r in cur.fetchall()]
        for row in rows:
            row["files"] = _quote_files(conn, row["id"])

        return {
            "ok": True,
            "database": "postgresql" if _postgres_enabled() else "sqlite",
            "quotes": rows,
        }


@app.get("/api/quotes/{quote_id}")
def get_quote(quote_id: str):
    with _db_connect() as conn:
        return _quote_response(conn, quote_id)


@app.post("/api/quotes")
async def create_quote(request: Request):
    form, payload, files = await _parse_large_quote_form(request)
    try:
        try:
            data = json.loads(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Ongeldige offertegegevens.") from exc

        customer_name = str(data.get("customer") or "").strip()
        if not customer_name:
            raise HTTPException(status_code=400, detail="Klantnaam ontbreekt.")

        quote_id = uuid.uuid4().hex
        now = _utcnow()

        with _db_connect() as conn:
            quote_number = _next_quote_number(conn)
            cur = conn.cursor()

            cur.execute(
                _sql(
                    """
                    INSERT INTO quotes
                    (id, quote_number, customer_name, contact_person, customer_email,
                     customer_phone, total_ex_vat, payload_json, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    """
                    INSERT INTO quotes
                    (id, quote_number, customer_name, contact_person, customer_email,
                     customer_phone, total_ex_vat, payload_json, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """
                ),
                (
                    quote_id,
                    quote_number,
                    customer_name,
                    str(data.get("contactPerson") or ""),
                    str(data.get("customerEmail") or ""),
                    str(data.get("customerPhone") or ""),
                    float(data.get("total_ex_vat") or 0),
                    json.dumps(data, ensure_ascii=False),
                    now,
                    now,
                )
            )

            await _store_quote_files(conn, quote_id, files)

            dropbox_warning=""
            try:
                _sync_quote_json_to_dropbox(conn, quote_id)
            except HTTPException as exc:
                dropbox_warning=str(exc.detail)

            # De database is leidend: een Dropbox-tokenfout mag een offerte
            # niet meer ongedaan maken of uit de offertelijst laten verdwijnen.
            conn.commit()

            result=_quote_response(conn, quote_id)
            result["dropbox_warning"]=dropbox_warning
            result["dropbox_ok"]=not bool(dropbox_warning)
            return result
    finally:
        try:
            await form.close()
        except Exception:
            pass



@app.put("/api/quotes/{quote_id}")
async def update_quote(quote_id: str, request: Request):
    form, payload, files = await _parse_large_quote_form(request)
    try:
        try:
            data = json.loads(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Ongeldige offertegegevens.") from exc

        customer_name = str(data.get("customer") or "").strip()
        if not customer_name:
            raise HTTPException(status_code=400, detail="Klantnaam ontbreekt.")

        with _db_connect() as conn:
            cur = conn.cursor()

            cur.execute(
                _sql(
                    "SELECT id FROM quotes WHERE id=%s",
                    "SELECT id FROM quotes WHERE id=?"
                ),
                (quote_id,)
            )

            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Offerte niet gevonden.")

            cur.execute(
                _sql(
                    """
                    UPDATE quotes
                    SET customer_name=%s,
                        contact_person=%s,
                        customer_email=%s,
                        customer_phone=%s,
                        total_ex_vat=%s,
                        payload_json=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    """
                    UPDATE quotes
                    SET customer_name=?,
                        contact_person=?,
                        customer_email=?,
                        customer_phone=?,
                        total_ex_vat=?,
                        payload_json=?,
                        updated_at=?
                    WHERE id=?
                    """
                ),
                (
                    customer_name,
                    str(data.get("contactPerson") or ""),
                    str(data.get("customerEmail") or ""),
                    str(data.get("customerPhone") or ""),
                    float(data.get("total_ex_vat") or 0),
                    json.dumps(data, ensure_ascii=False),
                    _utcnow(),
                    quote_id,
                )
            )

            await _store_quote_files(conn, quote_id, files)

            dropbox_warning=""
            try:
                _sync_quote_json_to_dropbox(conn, quote_id)
            except HTTPException as exc:
                dropbox_warning=str(exc.detail)

            # De database is leidend: een Dropbox-tokenfout mag een offerte
            # niet meer ongedaan maken of uit de offertelijst laten verdwijnen.
            conn.commit()

            result=_quote_response(conn, quote_id)
            result["dropbox_warning"]=dropbox_warning
            result["dropbox_ok"]=not bool(dropbox_warning)
            return result
    finally:
        try:
            await form.close()
        except Exception:
            pass



@app.get("/api/quotes/{quote_id}/files/{file_id}")
def download_quote_file(quote_id: str, file_id: str):
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _sql(
                """
                SELECT filename, content_type, data, dropbox_path
                FROM quote_files
                WHERE id=%s AND quote_id=%s
                """,
                """
                SELECT filename, content_type, data, dropbox_path
                FROM quote_files
                WHERE id=? AND quote_id=?
                """
            ),
            (file_id, quote_id)
        )

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bestand niet gevonden.")

        if isinstance(row, sqlite3.Row):
            filename = row["filename"]
            content_type = row["content_type"]
            data = row["data"]
            dropbox_path = row["dropbox_path"]
        else:
            filename, content_type, data, dropbox_path = row

        if dropbox_path:
            data = _dropbox_download_bytes(dropbox_path)

        if data is None:
            raise HTTPException(status_code=404, detail="Bestandsinhoud niet gevonden.")

        safe_name = str(filename).replace('"', "")
        return Response(
            content=bytes(data),
            media_type=content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}"',
                "X-Content-Type-Options": "nosniff",
            },
        )


@app.delete("/api/quotes/{quote_id}")
def delete_quote(quote_id: str):
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _sql(
                "SELECT quote_number, customer_name FROM quotes WHERE id=%s",
                "SELECT quote_number, customer_name FROM quotes WHERE id=?"
            ),
            (quote_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Offerte niet gevonden.")

        if isinstance(row, sqlite3.Row):
            quote_number = row["quote_number"]
            customer_name = row["customer_name"]
        else:
            quote_number, customer_name = row

        dropbox_folder = _quote_dropbox_folder(quote_number, customer_name)

        cur.execute(
            _sql(
                "DELETE FROM quote_files WHERE quote_id=%s",
                "DELETE FROM quote_files WHERE quote_id=?"
            ),
            (quote_id,)
        )
        cur.execute(
            _sql(
                "DELETE FROM quotes WHERE id=%s",
                "DELETE FROM quotes WHERE id=?"
            ),
            (quote_id,)
        )
        conn.commit()

    _dropbox_delete_path(dropbox_folder)
    return {"ok": True, "id": quote_id, "dropbox_deleted": True}



def _normalize_dropbox_browser_path(value: str) -> str:
    """
    Dropbox list_folder gebruikt "" voor de echte root en "/Map/Submap"
    voor onderliggende mappen. Alleen een map-pad is toegestaan.
    """
    value = str(value or "").strip().replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    if not value or value == "/":
        return ""
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/")


def _dropbox_list_folders(path: str = "") -> dict:
    dbx_path = _normalize_dropbox_browser_path(path)
    payload = {
        "path": dbx_path,
        "recursive": False,
        "include_deleted": False,
        "include_has_explicit_shared_members": False,
        "include_mounted_folders": True,
        "limit": 2000,
    }

    result = _dropbox_rpc("files/list_folder", payload)
    entries = list(result.get("entries") or [])

    while result.get("has_more"):
        cursor = str(result.get("cursor") or "")
        if not cursor:
            break
        result = _dropbox_rpc("files/list_folder/continue", {"cursor": cursor})
        entries.extend(result.get("entries") or [])

    folders = []
    for item in entries:
        if str(item.get(".tag") or "") != "folder":
            continue
        folders.append({
            "name": str(item.get("name") or ""),
            "path_display": str(item.get("path_display") or ""),
            "path_lower": str(item.get("path_lower") or ""),
            "id": str(item.get("id") or ""),
        })

    folders.sort(key=lambda x: x["name"].casefold())
    return {
        "ok": True,
        "path": dbx_path,
        "display_path": dbx_path or "/",
        "folders": folders,
    }


@app.get("/api/dropbox/folders")
def dropbox_folders(path: str = ""):
    """Geeft uitsluitend de mappen in de gekozen Dropbox-map terug."""
    return _dropbox_list_folders(path)


@app.post("/api/dropbox/folders/create")
async def dropbox_create_folder(request: Request):
    """Maakt vanuit de app een echte nieuwe Dropbox-map aan."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    parent = _normalize_dropbox_browser_path(body.get("parent") or "")
    raw_name = str(body.get("name") or "").strip()

    if not raw_name:
        raise HTTPException(status_code=400, detail="Vul een mapnaam in.")

    # Dropbox-mapnamen mogen geen padcomponenten bevatten.
    if "/" in raw_name or "\\" in raw_name:
        raise HTTPException(
            status_code=400,
            detail="Gebruik alleen een mapnaam, zonder / of \\."
        )

    name = _safe_dropbox_name(raw_name, "")
    if not name:
        raise HTTPException(status_code=400, detail="Ongeldige mapnaam.")

    target = f"{parent}/{name}" if parent else f"/{name}"

    try:
        created = _dropbox_rpc(
            "files/create_folder_v2",
            {"path": target, "autorename": False}
        )
    except HTTPException as exc:
        detail = str(exc.detail)
        if "conflict" in detail.lower():
            raise HTTPException(
                status_code=409,
                detail=f"De map '{name}' bestaat hier al."
            ) from exc
        raise

    meta = created.get("metadata") or {}
    return {
        "ok": True,
        "folder": {
            "name": str(meta.get("name") or name),
            "path_display": str(meta.get("path_display") or target),
            "path_lower": str(meta.get("path_lower") or target.lower()),
            "id": str(meta.get("id") or ""),
        },
    }



def _dropbox_get_metadata(path: str):
    path = _normalize_dropbox_browser_path(path)
    if not path:
        return {".tag": "folder", "name": "Dropbox", "path_display": ""}
    try:
        return _dropbox_rpc("files/get_metadata", {"path": path})
    except HTTPException as exc:
        if "not_found" in str(exc.detail).lower():
            return None
        raise


def _dropbox_move_path(from_path: str, to_path: str):
    return _dropbox_rpc("files/move_v2", {
        "from_path": _normalize_dropbox_browser_path(from_path),
        "to_path": _normalize_dropbox_browser_path(to_path),
        "autorename": False,
        "allow_shared_folder": False,
        "allow_ownership_transfer": False,
    })


def _dropbox_create_folder_path(path: str):
    path = _normalize_dropbox_browser_path(path)
    if not path:
        return
    if _dropbox_get_metadata(path):
        return
    parent = "/".join(path.split("/")[:-1])
    if parent:
        _dropbox_create_folder_path(parent)
    _dropbox_rpc("files/create_folder_v2", {"path": path, "autorename": False})


def _dropbox_merge_move(source: str, destination: str, moved: list):
    """
    Verplaats een complete opslagboom. Bestaat de doelmap al, dan worden mappen
    samengevoegd. Bestaande doelbestanden worden nooit stil overschreven.
    """
    source = _normalize_dropbox_browser_path(source)
    destination = _normalize_dropbox_browser_path(destination)
    if not source or not destination:
        raise HTTPException(status_code=400, detail="Dropbox-root zelf kan niet worden verplaatst.")
    if source == destination:
        return
    if destination.startswith(source + "/"):
        raise HTTPException(
            status_code=400,
            detail="De nieuwe opslagmap mag niet binnen de huidige opslagmap liggen."
        )

    source_meta = _dropbox_get_metadata(source)
    if not source_meta:
        return

    destination_meta = _dropbox_get_metadata(destination)
    if not destination_meta:
        parent = "/".join(destination.split("/")[:-1])
        if parent:
            _dropbox_create_folder_path(parent)
        _dropbox_move_path(source, destination)
        moved.append({"from": source, "to": destination})
        return

    if str(source_meta.get(".tag") or "") != "folder" or str(destination_meta.get(".tag") or "") != "folder":
        raise HTTPException(
            status_code=409,
            detail=f"Kan '{source}' niet samenvoegen met bestaand doel '{destination}'."
        )

    listing = _dropbox_rpc("files/list_folder", {
        "path": source,
        "recursive": False,
        "include_deleted": False,
        "include_has_explicit_shared_members": False,
        "include_mounted_folders": True,
        "limit": 2000,
    })
    entries = list(listing.get("entries") or [])
    while listing.get("has_more"):
        listing = _dropbox_rpc("files/list_folder/continue", {"cursor": listing["cursor"]})
        entries.extend(listing.get("entries") or [])

    for item in entries:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        child_source = f"{source}/{name}"
        child_destination = f"{destination}/{name}"
        tag = str(item.get(".tag") or "")
        target_meta = _dropbox_get_metadata(child_destination)

        if tag == "folder" and target_meta and str(target_meta.get(".tag") or "") == "folder":
            _dropbox_merge_move(child_source, child_destination, moved)
        elif target_meta:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Verplaatsen gestopt: '{child_destination}' bestaat al. "
                    "Er is niets overschreven. Geef het bestaande bestand eerst een andere naam."
                )
            )
        else:
            _dropbox_move_path(child_source, child_destination)
            moved.append({"from": child_source, "to": child_destination})

    # Lege bronmap verwijderen.
    try:
        _dropbox_rpc("files/delete_v2", {"path": source})
    except Exception:
        pass


@app.post("/api/dropbox/storage/move")
async def dropbox_move_storage(request: Request):
    """Verplaatst bestaande Vakstaal-opslag veilig naar een nieuw gekozen Dropbox-pad."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    source = _normalize_dropbox_browser_path(body.get("source") or "")
    destination = _normalize_dropbox_browser_path(body.get("destination") or "")

    if not source or not destination:
        raise HTTPException(status_code=400, detail="Bron- en doelmap zijn verplicht.")
    if source == destination:
        return {"ok": True, "moved": [], "message": "Opslaglocatie is niet gewijzigd."}

    moved = []
    _dropbox_merge_move(source, destination, moved)
    return {
        "ok": True,
        "source": source,
        "destination": destination,
        "moved_count": len(moved),
        "moved": moved[:200],
    }


@app.get("/api/dropbox/status")
def dropbox_status():
    try:
        result=_dropbox_rpc("check/user",{"query":"vakstaal-dropbox-test"})
        return {
            "ok":True,
            "connected":True,
            "root":DROPBOX_ROOT,
            "refresh_configured":bool(DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET),
            "dropbox_response":result,
        }
    except HTTPException as exc:
        return {
            "ok":False,
            "connected":False,
            "root":DROPBOX_ROOT,
            "refresh_configured":bool(DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET),
            "detail":str(exc.detail),
        }


@app.get("/api/storage/status")
def storage_status():
    try:
        with _db_connect() as conn:
            cur=conn.cursor()
            cur.execute("SELECT COUNT(*) FROM quotes")
            row=cur.fetchone()
            quote_count=int(row[0] if row else 0)
        return {
            "ok":True,
            "database":"postgresql" if _postgres_enabled() else "sqlite",
            "database_url_configured":bool(DATABASE_URL),
            "quote_count":quote_count,
        }
    except Exception as exc:
        return {
            "ok":False,
            "database":"postgresql" if _postgres_enabled() else "sqlite",
            "database_url_configured":bool(DATABASE_URL),
            "detail":f"{type(exc).__name__}: {exc}",
        }


@app.post("/api/dropbox/test-upload")
def dropbox_test_upload():
    stamp=_utcnow()
    data=("Vakstaal Dropbox koppeling werkt.\nServer test uitgevoerd: "+stamp+"\n").encode("utf-8")
    result=_dropbox_upload_bytes("/vakstaal_dropbox_test.txt", data)
    return {"ok":True,"message":"Testbestand succesvol naar Dropbox geupload.","path":result.get("path_display") or result.get("path_lower"),"size":result.get("size")}

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Vakstaal STEP Server",
        "endpoints": [
            "/health",
            "/api/analyze-step",
            "/api/assembly-mesh/{job_id}",
            "/api/solid-mesh/{job_id}/{solid_index}",
            "/api/quotes",
            "/api/app-state",
            "/api/quotes/{quote_id}",
            "/api/dropbox/status",
            "/api/dropbox/folders",
            "/api/dropbox/folders/create",
            "/api/dropbox/storage/move",
            "/api/dropbox/test-upload",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "Vakstaal STEP Server"}


@app.post("/api/analyze-step")
async def analyze_step_endpoint(file: UploadFile = File(...)):
    cleanup_old_jobs()

    filename = file.filename or "upload.step"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".step", ".stp"}:
        raise HTTPException(status_code=400, detail="Kies een STEP-bestand (.step of .stp).")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Het STEP-bestand is leeg.")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"STEP-bestand is groter dan {MAX_UPLOAD_MB} MB."
        )

    job_id = uuid.uuid4().hex
    folder = CACHE_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    step_path = folder / f"source{suffix}"
    step_path.write_bytes(data)

    try:
        result = analyze_step(step_path)
        result = _apply_physical_material_lengths(step_path, result)
        result["filename"] = filename
        result["job_id"] = job_id
        result["expires_hours"] = TTL_HOURS
        try:
            _analysis_cache_path(job_id).write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return result
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"STEP-analyse mislukt: {exc}") from exc


@app.get("/api/assembly-mesh/{job_id}")
def assembly_mesh(job_id: str):
    """
    Return the complete assembly using NET material geometry.

    Any end portions which the STEP analyzer excluded from the material length
    are physically clipped from the returned mesh, so they are fully invisible
    in both the browser viewer and the PDF snapshot.
    """
    cleanup_old_jobs()
    step_path = job_step_path(job_id)
    cache_path = _assembly_cache_path(job_id)

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        analysis = _load_or_analyze(job_id, step_path)
        details = analysis.get("details") or []

        imported = cq.importers.importStep(str(step_path))
        solids = imported.solids().vals()
        if not solids:
            raise HTTPException(status_code=404, detail="Geen onderdelen gevonden.")

        meshes = []
        all_xyz = []
        trimmed_count = 0

        for solid_index, solid in enumerate(solids, start=1):
            detail = details[solid_index - 1] if solid_index - 1 < len(details) else None

            # Viewer geometry is the exact FINAL STEP solid.
            # The STEP file already contains the real trim result. Do not use
            # net material length as a geometric clipping envelope: tabs,
            # slots, notches and shaped ends may legitimately extend beyond
            # the calculated stock length.
            visible_shape = solid
            was_trimmed = False

            axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
            raw_length = float(raw_length)
            net_length = float((detail or {}).get("length_mm") or raw_length)

            mesh = _mesh_shape(visible_shape, center_vertices=False)
            all_xyz.extend(mesh["vertices"])
            base_cut_lines, feature_lines, base_cut_contour_count = _physical_cut_polylines(solid, detail)

            meshes.append({
                "solid_index": solid_index,
                "vertices": mesh["vertices"],
                "triangles": mesh["triangles"],
                "base_cut_lines": base_cut_lines,
                "base_cut_contour_count": int(base_cut_contour_count),
                "feature_lines": feature_lines,
                "feature_classifier_version": 6,
                "physical_cut_classifier_version": 1,
                "profile_axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "has_extra_features": bool(feature_lines),
                "trimmed_visual": bool(was_trimmed),
                "feature_preserving_trim": False,
                "original_final_step_geometry": True,
                "raw_length_mm": raw_length,
                "net_length_mm": net_length,
                "material_length_mm": net_length,
            })

        if not meshes or not all_xyz:
            raise ValueError("Geen zichtbare 3D-mesh gevonden.")

        xs = [p[0] for p in all_xyz]
        ys = [p[1] for p in all_xyz]
        zs = [p[2] for p in all_xyz]

        result = {
            "job_id": job_id,
            "solid_count": len(solids),
            "mesh_count": len(meshes),
            "trimmed_mesh_count": trimmed_count,
            "net_geometry": True,
            "center": [
                (min(xs) + max(xs)) / 2.0,
                (min(ys) + max(ys)) / 2.0,
                (min(zs) + max(zs)) / 2.0,
            ],
            "size": float(max(
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs),
                1e-6,
            )),
            "meshes": meshes,
        }

        try:
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"3D-overzicht kon niet worden gemaakt: {type(exc).__name__}: {exc}"
        ) from exc


@app.get("/api/solid-mesh/{job_id}/{solid_index}")
def solid_mesh(job_id: str, solid_index: int):
    cleanup_old_jobs()
    step_path = job_step_path(job_id)

    try:
        analysis = _load_or_analyze(job_id, step_path)
        details = analysis.get("details") or []

        imported = cq.importers.importStep(str(step_path))
        solids = imported.solids().vals()

        if solid_index < 1 or solid_index > len(solids):
            raise HTTPException(status_code=404, detail="Onderdeel niet gevonden.")

        solid = solids[solid_index - 1]
        detail = details[solid_index - 1] if solid_index - 1 < len(details) else None

        # Show the exact final body from the uploaded STEP file.
        # Net material length is calculation data only, not a viewer clip.
        visible_shape = solid
        was_trimmed = False

        axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
        raw_length = float(raw_length)
        net_length = float((detail or {}).get("length_mm") or raw_length)

        mesh = _mesh_shape(visible_shape, center_vertices=True)

        raw_base_cut_lines, raw_feature_lines, base_cut_contour_count = _physical_cut_polylines(solid, detail)

        # _mesh_shape(center_vertices=True) centers vertices on the original
        # visible bounding box. Apply the same translation to cut lines.
        bb = visible_shape.BoundingBox()
        cx = (float(bb.xmin) + float(bb.xmax)) / 2.0
        cy = (float(bb.ymin) + float(bb.ymax)) / 2.0
        cz = (float(bb.zmin) + float(bb.zmax)) / 2.0

        base_cut_lines = [
            [[p[0]-cx, p[1]-cy, p[2]-cz] for p in line]
            for line in raw_base_cut_lines
        ]
        feature_lines = [
            [[p[0]-cx, p[1]-cy, p[2]-cz] for p in line]
            for line in raw_feature_lines
        ]

        return {
            "job_id": job_id,
            "solid_index": solid_index,
            "vertices": mesh["vertices"],
            "triangles": mesh["triangles"],
            "base_cut_lines": base_cut_lines,
            "base_cut_contour_count": int(base_cut_contour_count),
            "feature_lines": feature_lines,
            "feature_classifier_version": 6,
            "physical_cut_classifier_version": 1,
            "profile_axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            "has_extra_features": bool(feature_lines),
            "size": mesh["size"],
            "net_geometry": True,
            "trimmed_visual": bool(was_trimmed),
            "feature_preserving_trim": False,
            "original_final_step_geometry": True,
            "raw_length_mm": raw_length,
            "net_length_mm": net_length,
                "material_length_mm": net_length,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"3D-weergave kon niet worden gemaakt: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)