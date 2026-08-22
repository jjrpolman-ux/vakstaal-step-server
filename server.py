from __future__ import annotations
import html

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
from fastapi.responses import HTMLResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
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

STEP_MATERIAL_LENGTH_VERSION = 7  # authoritative material length propagated to all fields

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
    Vakstaal materiaallengte v5.

    Definitie:
    de langste echte longitudinale zijde van de HOOFDKOKERWAND.
    Tabs, sleuven, lokale uitsteeksels en andere bewerkingsranden tellen niet
    mee als materiaallengte.

    Dit sluit aan op de praktische maat in SolidWorks:
    - bepaal eerst de echte profielas;
    - vind de grote longitudinale wandvlakken van de koker;
    - verzamel uitsluitend rechte lengteranden van die hoofdwanden;
    - combineer eventueel gesplitste collineaire stukken;
    - gebruik de langste fysieke wandzijde.

    Als twee of meer langste zijden gelijk zijn maakt het niet uit welke
    daarvan de bron is; de numerieke lengte is hetzelfde.
    """
    analyzer_axis, analyzer_length, analyzer_method = _dominant_longitudinal_axis_and_length(solid)
    axis=np.array(analyzer_axis,dtype=float)
    axis_norm=float(np.linalg.norm(axis))
    if axis_norm<=1e-12:
        raise ValueError("Lengteas van STEP-body kon niet worden bepaald.")
    axis=axis/axis_norm

    # Stabiel richtingsteken voor diagnose/groepering.
    for component in axis:
        if abs(float(component))>1e-9:
            if component<0:
                axis=-axis
            break

    ref=np.array([0.0,0.0,1.0],dtype=float)
    if abs(float(np.dot(ref,axis)))>0.92:
        ref=np.array([0.0,1.0,0.0],dtype=float)
    u=np.cross(axis,ref)
    u=u/max(float(np.linalg.norm(u)),1e-12)
    v=np.cross(axis,u)
    v=v/max(float(np.linalg.norm(v)),1e-12)

    longitudinal_faces=[]

    for face in solid.Faces():
        try:
            # Grote kokerwanden zijn in normale RHS/SHS STEP-bestanden vlak.
            if face.geomType() != "PLANE":
                continue

            verts=face.Vertices()
            if len(verts)<3:
                continue

            projections=[
                float(np.dot(np.array(vertex.toTuple(),dtype=float),axis))
                for vertex in verts
            ]
            face_span=max(projections)-min(projections)
            if face_span<=1e-6:
                continue

            # Een echte hoofdwand loopt over vrijwel de volle profiellengte.
            # Gebruik analyzer_length alleen als grove minimumreferentie,
            # nooit als uiteindelijke materiaallengte.
            reference=max(float(analyzer_length or 0.0),1.0)
            if face_span < reference*0.55:
                continue

            area=float(face.Area())
            if not math.isfinite(area) or area<=1e-6:
                continue

            longitudinal_faces.append({
                "face":face,
                "span":float(face_span),
                "area":area,
            })
        except Exception:
            continue

    candidates=[]

    if longitudinal_faces:
        # De vier buiten- en vier binnenwanden van een koker hebben veruit de
        # grootste longitudinale oppervlakken. Lokale tab-/sleufvlakken zijn
        # veel kleiner. Houd daarom alleen echte hoofdwand-oppervlakken over.
        max_area=max(item["area"] for item in longitudinal_faces)
        main_faces=[
            item for item in longitudinal_faces
            if item["area"] >= max_area*0.45
        ]

        for item in main_faces:
            face=item["face"]
            for edge in face.Edges():
                try:
                    if edge.geomType()!="LINE":
                        continue
                    verts=edge.Vertices()
                    if len(verts)<2:
                        continue

                    p=np.array(verts[0].toTuple(),dtype=float)
                    q=np.array(verts[-1].toTuple(),dtype=float)
                    vec=q-p
                    line_len=float(np.linalg.norm(vec))
                    if line_len<=1e-6:
                        continue

                    unit=vec/line_len
                    if abs(float(np.dot(unit,axis)))<0.9985:
                        continue

                    pa=float(np.dot(p,axis))
                    pb=float(np.dot(q,axis))
                    lo,hi=(pa,pb) if pa<=pb else (pb,pa)

                    mid=(p+q)/2.0
                    candidates.append({
                        "lo":lo,
                        "hi":hi,
                        "u":float(np.dot(mid,u)),
                        "v":float(np.dot(mid,v)),
                        "edge_len":line_len,
                    })
                except Exception:
                    continue

    if candidates:
        # Een echte wandzijde kan door een lokale uitsparing topologisch in
        # meerdere collineaire stukken zijn verdeeld. Groepeer die stukken op
        # hun dwarse positie en gebruik de totale fysieke span.
        line_tol=0.50
        groups=[]

        for seg in sorted(candidates,key=lambda x:x["edge_len"],reverse=True):
            group=None
            for existing in groups:
                if abs(seg["u"]-existing["u"])<=line_tol and abs(seg["v"]-existing["v"])<=line_tol:
                    group=existing
                    break
            if group is None:
                group={"u":seg["u"],"v":seg["v"],"intervals":[]}
                groups.append(group)
            group["intervals"].append((seg["lo"],seg["hi"]))

        spans=[]
        for group in groups:
            intervals=group["intervals"]
            if not intervals:
                continue
            span=float(max(b for _,b in intervals)-min(a for a,_ in intervals))
            if math.isfinite(span) and span>0:
                spans.append(span)

        if spans:
            return max(spans),axis,"longest-main-wall-side-v6"

    # Fallback: langste rechte edge parallel aan de gedetecteerde profielas.
    straight_lengths=[]
    for edge in solid.Edges():
        try:
            if edge.geomType()!="LINE":
                continue
            verts=edge.Vertices()
            if len(verts)<2:
                continue
            p=np.array(verts[0].toTuple(),dtype=float)
            q=np.array(verts[-1].toTuple(),dtype=float)
            vec=q-p
            length=float(np.linalg.norm(vec))
            if length<=1e-6:
                continue
            if abs(float(np.dot(vec/length,axis)))>=0.9985:
                straight_lengths.append(length)
        except Exception:
            continue

    if straight_lengths:
        return max(straight_lengths),axis,"longest-axis-edge-fallback-v6"

    return max(0.0,float(analyzer_length or 0.0)),axis,f"{analyzer_method}-fallback-v6"


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
        # Eén bron van waarheid: alle publieke lengtevelden krijgen exact dezelfde
        # gecorrigeerde materiaallengte. Oude frontendcode kan hierdoor niet meer
        # ongemerkt terugvallen op de vroegere analyzer-lengte.
        detail["length_mm"] = float(physical_length)
        detail["length_m"] = float(physical_length) / 1000.0
        detail["material_length_mm"] = float(physical_length)
        detail["material_length_m"] = float(physical_length) / 1000.0
        detail["material_length_method"] = method
        detail["length_method"] = method
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
            "definition": "langste fysieke longitudinale zijde van de hoofdwand",
        })

    result["details"] = details
    result["material_length_version"] = STEP_MATERIAL_LENGTH_VERSION
    result["material_length_definition"] = (
        "longest real longitudinal side of the main profile wall"
    )
    result["material_length_audit"] = length_audit
    return result


def _analysis_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "analysis.json"


def _assembly_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "assembly_mesh_physical_cut_v66_authoritative_length_v7.json"


def _load_or_analyze(job_id: str, step_path: Path) -> dict:
    cache = _analysis_cache_path(job_id)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            detail_versions = [
                int((d or {}).get("material_length_version") or 0)
                for d in (cached.get("details") or [])
            ]
            if (
                int(cached.get("material_length_version") or 0) == STEP_MATERIAL_LENGTH_VERSION
                and detail_versions
                and all(v == STEP_MATERIAL_LENGTH_VERSION for v in detail_versions)
            ):
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
DROPBOX_REDIRECT_URI = os.environ.get("DROPBOX_REDIRECT_URI", "").strip()
DROPBOX_OAUTH_TOKEN_FILE = Path(os.environ.get("DROPBOX_OAUTH_TOKEN_FILE", "/tmp/vakstaal_dropbox_refresh_token.txt"))
DROPBOX_OAUTH_STATE_FILE = Path(os.environ.get("DROPBOX_OAUTH_STATE_FILE", "/tmp/vakstaal_dropbox_oauth_state.txt"))

_dropbox_runtime_access_token = DROPBOX_ACCESS_TOKEN
_dropbox_runtime_refresh_token = DROPBOX_REFRESH_TOKEN
_dropbox_runtime_root_namespace_id = ''
_dropbox_runtime_home_namespace_id = ''
_dropbox_runtime_account_summary = {}
try:
    if not _dropbox_runtime_refresh_token and DROPBOX_OAUTH_TOKEN_FILE.exists():
        _dropbox_runtime_refresh_token = DROPBOX_OAUTH_TOKEN_FILE.read_text(encoding="utf-8").strip()
except Exception:
    pass


def _dropbox_refresh_access_token() -> str:
    global _dropbox_runtime_access_token

    if not (_dropbox_runtime_refresh_token and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        raise HTTPException(
            status_code=503,
            detail=(
                "Dropbox access token is verlopen. Stel DROPBOX_REFRESH_TOKEN, "
                "DROPBOX_APP_KEY en DROPBOX_APP_SECRET in op de server."
            )
        )

    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": _dropbox_runtime_refresh_token,
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
    if _dropbox_runtime_refresh_token and DROPBOX_APP_KEY and DROPBOX_APP_SECRET:
        return _dropbox_refresh_access_token()
    raise HTTPException(status_code=503,detail="Dropbox-token ontbreekt op de server.")



def _dropbox_basic_rpc(endpoint: str, payload: dict, force_refresh: bool=False) -> dict:
    """
    Dropbox RPC zonder Path-Root. Dit is nodig om eerst users/get_current_account
    te kunnen vragen welke root namespace bij het gekoppelde account hoort.
    """
    req=urllib.request.Request(
        f"https://api.dropboxapi.com/2/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization":f"Bearer {_dropbox_token(force_refresh)}",
            "Content-Type":"application/json",
        },
    )
    try:
        with urllib.request.urlopen(req,timeout=25) as response:
            raw=response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8",errors="replace")
        if exc.code==401 and _dropbox_token_expired(detail) and not force_refresh:
            return _dropbox_basic_rpc(endpoint,payload,True)
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox API fout ({exc.code}): {detail[:700]}"
        ) from exc


def _dropbox_account_context(force: bool=False) -> dict:
    """
    Bepaalt de echte Dropbox-root van het gekoppelde account.

    root_namespace_id = hoogste namespace die Dropbox voor dit account opgeeft.
    home_namespace_id = persoonlijke/home namespace.
    Bij persoonlijke accounts zijn deze doorgaans gelijk. Bij teamaccounts
    kunnen ze verschillen.
    """
    global _dropbox_runtime_root_namespace_id
    global _dropbox_runtime_home_namespace_id
    global _dropbox_runtime_account_summary

    if _dropbox_runtime_account_summary and not force:
        return dict(_dropbox_runtime_account_summary)

    account=_dropbox_basic_rpc("users/get_current_account",{})
    root_info=dict(account.get("root_info") or {})
    root_ns=str(root_info.get("root_namespace_id") or "").strip()
    home_ns=str(root_info.get("home_namespace_id") or "").strip()

    # Bij een normaal persoonlijk account kan Dropbox root_info soms anders
    # structureren; root namespace valt dan terug op home namespace.
    if not root_ns:
        root_ns=home_ns

    _dropbox_runtime_root_namespace_id=root_ns
    _dropbox_runtime_home_namespace_id=home_ns

    name_info=dict(account.get("name") or {})
    summary={
        "account_id":str(account.get("account_id") or ""),
        "display_name":str(name_info.get("display_name") or ""),
        "email":str(account.get("email") or ""),
        "email_verified":bool(account.get("email_verified")),
        "root_namespace_id":root_ns,
        "home_namespace_id":home_ns,
        "root_tag":str(root_info.get(".tag") or ""),
        "root_differs_from_home":bool(root_ns and home_ns and root_ns!=home_ns),
    }
    _dropbox_runtime_account_summary=summary
    return dict(summary)


def _dropbox_path_root_header(force: bool=False) -> dict:
    """
    Forceer iedere files-call naar Dropbox' echte root namespace.
    """
    context=_dropbox_account_context(force)
    root_ns=str(context.get("root_namespace_id") or "").strip()
    if not root_ns:
        return {}
    return {
        "Dropbox-API-Path-Root":json.dumps({
            ".tag":"root",
            "root":root_ns,
        },separators=(",",":"))
    }


def _dropbox_headers(force_refresh: bool=False, include_path_root: bool=True) -> dict:
    headers={"Authorization":f"Bearer {_dropbox_token(force_refresh)}"}
    if include_path_root:
        headers.update(_dropbox_path_root_header(force_refresh))
    return headers


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


def _storage_state_config() -> dict:
    try:
        with _db_connect() as conn:
            cur=conn.cursor()
            cur.execute(_sql("SELECT payload_json FROM app_state WHERE state_key=%s","SELECT payload_json FROM app_state WHERE state_key=?"),("vakstaal_global_state",))
            row=cur.fetchone()
            if not row:return {}
            raw=row["payload_json"] if isinstance(row,sqlite3.Row) else row[0]
            state=json.loads(raw or "{}")
            settings_obj=state.get("settings") if isinstance(state,dict) else {}
            cfg=(settings_obj or {}).get("appStorage") if isinstance(settings_obj,dict) else {}
            return cfg if isinstance(cfg,dict) else {}
    except Exception:return {}

def _storage_clean_part(value: str,fallback: str="") -> str:
    value=str(value or "").strip().replace("\\","/")
    value="/".join(p for p in value.split("/") if p not in ("",".",".."))
    return value or fallback

def _storage_pattern(pattern: str,values: dict,fallback: str) -> str:
    result=str(pattern or fallback)
    for key,value in values.items():result=result.replace("{"+key+"}",_safe_dropbox_name(str(value or ""),key))
    return _safe_dropbox_name(result,fallback)

def _quote_storage_config() -> dict:
    c=_storage_state_config()
    return {"root":_storage_clean_part(c.get("dropboxRoot"),DROPBOX_ROOT.strip("/") or "Offertes"),"use_year":c.get("useYearFolder",True) is not False,
      "pattern":str(c.get("quoteFolderPattern") or "{offertenummer} - {klant}"),"step":_storage_clean_part(c.get("stepFolder"),"Productie STEP"),
      "pdf":_storage_clean_part(c.get("pdfFolder"),"PDF"),"source":_storage_clean_part(c.get("sourceFolder"),"Origineel"),
      "nest":_storage_clean_part(c.get("nestFolder"),"Nest"),"other":_storage_clean_part(c.get("otherFolder"),"Overige bestanden")}

def _webshop_storage_config() -> dict:
    c=_storage_state_config();w=c.get("webshopOrders") if isinstance(c.get("webshopOrders"),dict) else {}
    return {"root":_storage_clean_part(w.get("dropboxRoot"),"Webshop orders"),"use_year":w.get("useYearFolder",True) is not False,
      "pattern":str(w.get("orderFolderPattern") or "{ordernummer} - {klant}"),"confirmation":_storage_clean_part(w.get("confirmationFolder"),"Orderbevestiging"),
      "step":_storage_clean_part(w.get("stepFolder"),"Productie STEP")}

def _quote_dropbox_folder(quote_number: str,customer_name: str) -> str:
    cfg=_quote_storage_config();m=re.search(r"(20\d{2})",str(quote_number or ""));year=m.group(1) if m else str(datetime.now().year)
    name=_storage_pattern(cfg["pattern"],{"offertenummer":quote_number,"klant":customer_name,"jaar":year},_safe_dropbox_name(quote_number,"Offerte"))
    parts=[cfg["root"]]+([year] if cfg["use_year"] else [])+[name]
    return "/"+"/".join(_storage_clean_part(p) for p in parts if _storage_clean_part(p))

def _file_dropbox_subfolder(filename: str) -> str:
    cfg=_quote_storage_config();suffix=Path(filename or "").suffix.lower();lower=str(filename or "").lower()
    if suffix in {".zx",".nest"}:return cfg["nest"]
    if suffix==".pdf":return cfg["pdf"]
    if suffix in {".step",".stp"}:
        return cfg["step"] if any(w in lower for w in ("productie","production","solid_","onderdeel_","part_")) else cfg["source"]
    return cfg["other"]

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

        # Dropbox OAuth refresh-token persistent opslaan. Render's lokale /tmp
        # verdwijnt bij een deploy; PostgreSQL blijft bestaan.
        cur.execute(
            _sql(
                """
                CREATE TABLE IF NOT EXISTS oauth_credentials (
                    provider TEXT PRIMARY KEY,
                    refresh_token TEXT NOT NULL,
                    oauth_state TEXT,
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS oauth_credentials (
                    provider TEXT PRIMARY KEY,
                    refresh_token TEXT NOT NULL,
                    oauth_state TEXT,
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


async def _store_quote_files(conn, quote_id: str, files: list[UploadFile]) -> list[dict]:
    quote_number, customer_name, _payload_json = _quote_identity(conn, quote_id)
    folder = _quote_dropbox_folder(quote_number, customer_name)
    stored_files=[]

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
                stored_files.append({
                    "filename": filename,
                    "dropbox_path": existing_path,
                    "existing": True,
                    "size": len(data),
                })
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
            stored_files.append({
                "filename": filename,
                "dropbox_path": actual_path,
                "existing": True,
                "size": len(data),
            })
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

        stored_files.append({
            "filename": filename,
            "dropbox_path": actual_path,
            "existing": False,
            "size": len(data),
        })

    return stored_files


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



def _save_dropbox_oauth_credentials(refresh_token: str | None = None, oauth_state: str | None = None) -> None:
    """Bewaar de Dropbox refresh-token/state persistent in de centrale database."""
    token = str(refresh_token or "").strip()
    with _db_connect() as conn:
        cur = conn.cursor()
        # Bestaande token behouden wanneer alleen oauth_state wordt bijgewerkt.
        cur.execute(
            _sql(
                "SELECT refresh_token FROM oauth_credentials WHERE provider=%s",
                "SELECT refresh_token FROM oauth_credentials WHERE provider=?"
            ),
            ("dropbox",)
        )
        row = cur.fetchone()
        existing = ""
        if row:
            existing = str(row[0] if not isinstance(row, sqlite3.Row) else row["refresh_token"] or "").strip()
        final_token = token or existing
        # Bij de allereerste OAuth-start is er nog geen refresh-token; dan hoeft
        # alleen de state nog niet in deze tabel te worden geschreven.
        if not final_token:
            return
        now = _utcnow()
        cur.execute(
            _sql(
                """
                INSERT INTO oauth_credentials (provider, refresh_token, oauth_state, updated_at)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (provider) DO UPDATE SET
                    refresh_token=EXCLUDED.refresh_token,
                    oauth_state=EXCLUDED.oauth_state,
                    updated_at=EXCLUDED.updated_at
                """,
                """
                INSERT INTO oauth_credentials (provider, refresh_token, oauth_state, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(provider) DO UPDATE SET
                    refresh_token=excluded.refresh_token,
                    oauth_state=excluded.oauth_state,
                    updated_at=excluded.updated_at
                """
            ),
            ("dropbox", final_token, oauth_state, now)
        )
        conn.commit()


def _load_dropbox_oauth_credentials() -> dict:
    """Laad de laatst geautoriseerde Dropbox-koppeling uit PostgreSQL/SQLite."""
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            cur.execute(
                _sql(
                    "SELECT refresh_token, oauth_state, updated_at FROM oauth_credentials WHERE provider=%s",
                    "SELECT refresh_token, oauth_state, updated_at FROM oauth_credentials WHERE provider=?"
                ),
                ("dropbox",)
            )
            row = cur.fetchone()
            if not row:
                return {}
            if isinstance(row, sqlite3.Row):
                return dict(row)
            return {
                "refresh_token": row[0] or "",
                "oauth_state": row[1] or "",
                "updated_at": row[2] or "",
            }
    except Exception:
        # Een databaseprobleem mag de hele STEP-server niet blokkeren.
        return {}


def _restore_dropbox_oauth_from_database() -> bool:
    global _dropbox_runtime_refresh_token, _dropbox_runtime_access_token
    saved = _load_dropbox_oauth_credentials()
    token = str(saved.get("refresh_token") or "").strip()
    if not token:
        # Een eventueel handmatig ingestelde Render refresh-token blijft fallback.
        return bool(_dropbox_runtime_refresh_token)
    _dropbox_runtime_refresh_token = token
    # Access tokens zijn kortlevend en worden bewust niet persistent opgeslagen.
    # Na een deploy wordt automatisch een nieuw access-token uit de refresh-token gehaald.
    _dropbox_runtime_access_token = ""
    return True

_init_quote_db()
_restore_dropbox_oauth_from_database()


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

    # request.form() levert Starlette UploadFile-objecten op.
    # Alleen op fastapi.UploadFile controleren kan alle echte STEP/PDF-uploads
    # stil wegfilteren terwijl offerte.json wel normaal wordt opgeslagen.
    files = [
        item for item in form.getlist("files")
        if isinstance(item, (UploadFile, StarletteUploadFile))
        or (
            getattr(item, "filename", None) is not None
            and callable(getattr(item, "read", None))
        )
    ]
    return form, payload, files



@app.post("/api/generate-production-step")
async def generate_production_step(request: Request):
    """
    Maak één productie-STEP voor een bibliotheekprofiel.
    Iedere opgegeven lengte wordt als een afzonderlijke solid in hetzelfde
    STEP-bestand geplaatst. Deze route is nodig wanneer een offerte uitsluitend
    uit bibliotheekmateriaal bestaat en dus geen geüploade bron-STEP heeft.
    """
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Ongeldige productie-STEP gegevens.") from exc

    size = str(data.get("size") or "").strip()
    outer_w = float(data.get("outerWidth") or 0)
    outer_h = float(data.get("outerHeight") or 0)
    wall = float(data.get("wallThickness") or 0)
    radius = max(0.0, float(data.get("radius") or 0))
    pieces = data.get("pieces") or []

    lengths = []
    for piece in pieces:
        try:
            length = float((piece or {}).get("lengthMm") or 0)
        except Exception:
            length = 0
        if length > 0:
            lengths.append(length)

    if not (outer_w > 0 and outer_h > 0 and wall > 0 and lengths):
        raise HTTPException(
            status_code=400,
            detail=f"Onvoldoende profielgegevens voor productie-STEP: {size or 'onbekend profiel'}."
        )
    if outer_w <= 2 * wall or outer_h <= 2 * wall:
        raise HTTPException(status_code=400, detail="Wanddikte is ongeldig voor deze kokermaat.")

    try:
        inner_w = outer_w - 2 * wall
        inner_h = outer_h - 2 * wall
        inner_r = max(0.0, radius - wall)

        def rounded_rect_wire(w: float, h: float, r: float):
            # CadQuery rect gebruikt een scherp profiel; bij radius > 0 maken we
            # de echte afgeronde rechthoek via 2D fillet.
            wp = cq.Workplane("XY").rect(w, h)
            if r > 1e-6:
                try:
                    wp = wp.vertices().fillet2D(min(r, w / 2 - 1e-6, h / 2 - 1e-6))
                except Exception:
                    pass
            return wp

        solids = []
        x_offset = 0.0
        spacing = max(outer_w, outer_h) + 25.0

        for length in lengths:
            outer = rounded_rect_wire(outer_w, outer_h, radius).extrude(length)
            inner = rounded_rect_wire(inner_w, inner_h, inner_r).extrude(length + 2.0)
            inner = inner.translate((0, 0, -1.0))
            tube = outer.cut(inner)

            # Solids naast elkaar plaatsen zodat meerdere aantallen als losse
            # bodies in hetzelfde STEP-bestand herkenbaar blijven.
            if x_offset:
                tube = tube.translate((x_offset, 0, 0))
            solids.append(tube)
            x_offset += outer_w + spacing

        assembly = cq.Assembly(name="Vakstaal_productie")
        for idx, solid in enumerate(solids, start=1):
            assembly.add(solid, name=f"Koker_{idx}")

        tmp = CACHE_DIR / f"production_{uuid.uuid4().hex}.step"
        cq.exporters.export(assembly, str(tmp), exportType="STEP")
        content = tmp.read_bytes()
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

        if not content:
            raise RuntimeError("Leeg STEP-bestand gegenereerd.")

        return Response(
            content=content,
            media_type="application/step",
            headers={"Content-Disposition": 'attachment; filename="productie.step"'}
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Productie-STEP kon niet worden gemaakt: {exc}"
        ) from exc


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

            stored_files=await _store_quote_files(conn, quote_id, files)

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
            result["received_upload_count"]=len(files)
            result["stored_upload_count"]=len(stored_files)
            result["stored_uploads"]=stored_files
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

            stored_files=await _store_quote_files(conn, quote_id, files)

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
            result["received_upload_count"]=len(files)
            result["stored_upload_count"]=len(stored_files)
            result["stored_uploads"]=stored_files
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

    # Controleer direct bij Dropbox zelf dat de map werkelijk bestaat.
    # Zo kan de browserinterface nooit melden dat een map is gemaakt terwijl
    # alleen de lokale UI is bijgewerkt.
    verified = _dropbox_rpc("files/get_metadata", {"path": target})
    if str(verified.get(".tag") or "") != "folder":
        raise HTTPException(status_code=502, detail="Dropbox heeft de nieuwe map niet als map bevestigd.")
    meta = verified or meta
    return {
        "ok": True,
        "verified": True,
        "folder": {
            "name": str(meta.get("name") or name),
            "path_display": str(meta.get("path_display") or target),
            "path_lower": str(meta.get("path_lower") or target.lower()),
            "id": str(meta.get("id") or ""),
        },
    }




@app.post("/api/dropbox/folders/rename")
async def dropbox_rename_folder(request: Request):
    """Hernoemt één echte Dropbox-map en verifieert het resultaat bij Dropbox."""
    try:
        body=await request.json()
    except Exception:
        body={}

    path=_normalize_dropbox_browser_path(body.get("path") or "")
    raw_name=str(body.get("new_name") or "").strip()

    if not path:
        raise HTTPException(status_code=400,detail="Selecteer eerst een map.")
    if not raw_name:
        raise HTTPException(status_code=400,detail="Vul een nieuwe mapnaam in.")
    if "/" in raw_name or "\\" in raw_name:
        raise HTTPException(status_code=400,detail="Gebruik alleen een mapnaam, zonder / of \\.")

    current=_dropbox_get_metadata(path)
    if not current or str(current.get(".tag") or "")!="folder":
        raise HTTPException(status_code=404,detail="De geselecteerde Dropbox-map bestaat niet meer.")

    old_name=str(current.get("name") or "")
    new_name=_safe_dropbox_name(raw_name,"")
    if not new_name:
        raise HTTPException(status_code=400,detail="Ongeldige mapnaam.")

    if old_name.casefold()==new_name.casefold() and old_name==new_name:
        return {
            "ok":True,
            "verified":True,
            "old_path":path,
            "new_path":path,
            "name":old_name,
            "message":"De mapnaam is niet gewijzigd."
        }

    parent="/".join(path.split("/")[:-1])
    destination=f"{parent}/{new_name}" if parent else f"/{new_name}"

    target=_dropbox_get_metadata(destination)
    if target:
        raise HTTPException(
            status_code=409,
            detail=f"Er bestaat hier al een map of bestand met de naam '{new_name}'."
        )

    moved=_dropbox_move_path(path,destination)

    old_after=_dropbox_get_metadata(path)
    new_after=_dropbox_get_metadata(destination)
    verified=(
        old_after is None
        and new_after is not None
        and str(new_after.get(".tag") or "")=="folder"
    )
    if not verified:
        raise HTTPException(
            status_code=502,
            detail="Dropbox kon het hernoemen niet volledig bevestigen."
        )

    meta=(moved or {}).get("metadata") or new_after or {}
    return {
        "ok":True,
        "verified":True,
        "old_path":path,
        "new_path":str(meta.get("path_display") or destination),
        "name":str(meta.get("name") or new_name),
    }


@app.post("/api/dropbox/folders/delete")
async def dropbox_delete_folder(request: Request):
    """Verwijdert één echte Dropbox-map, inclusief inhoud, na expliciete frontendbevestiging."""
    try:
        body=await request.json()
    except Exception:
        body={}

    path=_normalize_dropbox_browser_path(body.get("path") or "")
    if not path:
        raise HTTPException(status_code=400,detail="De Dropbox-root kan niet worden verwijderd.")

    current=_dropbox_get_metadata(path)
    if not current or str(current.get(".tag") or "")!="folder":
        raise HTTPException(status_code=404,detail="De geselecteerde Dropbox-map bestaat niet meer.")

    # Tel inhoud zodat frontend een bruikbare bevestiging kan tonen/terugkrijgt.
    entries=_dropbox_recursive_entries(path)
    file_count=sum(1 for x in entries if str(x.get(".tag") or "")=="file")
    folder_count=sum(1 for x in entries if str(x.get(".tag") or "")=="folder")
    name=str(current.get("name") or path.split("/")[-1])

    _dropbox_rpc("files/delete_v2",{"path":path})

    after=_dropbox_get_metadata(path)
    if after is not None:
        raise HTTPException(
            status_code=502,
            detail="Dropbox kon niet bevestigen dat de map werkelijk verwijderd is."
        )

    return {
        "ok":True,
        "verified":True,
        "path":path,
        "name":name,
        "deleted_files":file_count,
        "deleted_subfolders":folder_count,
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


def _dropbox_recursive_snapshot(path: str) -> dict:
    path=_normalize_dropbox_browser_path(path);meta=_dropbox_get_metadata(path)
    if not meta:return {"exists":False,"files":0,"folders":0}
    result=_dropbox_rpc("files/list_folder",{"path":path,"recursive":True,"include_deleted":False,"include_mounted_folders":True,"limit":2000})
    entries=list(result.get("entries") or [])
    while result.get("has_more"):
        result=_dropbox_rpc("files/list_folder/continue",{"cursor":result.get("cursor")});entries.extend(result.get("entries") or [])
    return {"exists":True,"files":sum(1 for x in entries if str(x.get(".tag") or "")=="file"),"folders":sum(1 for x in entries if str(x.get(".tag") or "")=="folder")}


def _dropbox_recursive_entries(path: str) -> list:
    """Alle bestaande Dropbox-items onder path, zonder deleted entries."""
    path=_normalize_dropbox_browser_path(path)
    if not path or not _dropbox_get_metadata(path):
        return []
    result=_dropbox_rpc("files/list_folder",{
        "path":path,
        "recursive":True,
        "include_deleted":False,
        "include_mounted_folders":True,
        "limit":2000,
    })
    entries=list(result.get("entries") or [])
    while result.get("has_more"):
        cursor=str(result.get("cursor") or "")
        if not cursor:
            break
        result=_dropbox_rpc("files/list_folder/continue",{"cursor":cursor})
        entries.extend(result.get("entries") or [])
    return entries


def _dropbox_count_named_folders(root: str, folder_name: str) -> int:
    target=str(folder_name or "").strip().casefold()
    if not target:
        return 0
    return sum(
        1 for item in _dropbox_recursive_entries(root)
        if str(item.get(".tag") or "")=="folder"
        and str(item.get("name") or "").strip().casefold()==target
    )


def _dropbox_rename_structural_folders(root: str, renames: list) -> dict:
    """
    Hernoemt bestaande structurele submappen onder één opslagroot.

    Voorbeeld:
      root=/Offertes
      Productie STEP -> STEP productie

    Alle passende mappen worden daadwerkelijk met files/move_v2 verplaatst.
    Bestaat de nieuwe map al, dan worden de twee mappen veilig samengevoegd.
    """
    root=_normalize_dropbox_browser_path(root)
    if not root:
        raise HTTPException(status_code=400,detail="Een opslagroot is verplicht.")
    if not _dropbox_get_metadata(root):
        return {
            "ok":True,
            "verified":True,
            "root":root,
            "renamed_count":0,
            "message":"De opslagroot bestaat nog niet; er zijn geen bestaande submappen om te hernoemen.",
            "results":[],
        }

    clean=[]
    for item in renames or []:
        old=str((item or {}).get("old") or "").strip().strip("/")
        new=str((item or {}).get("new") or "").strip().strip("/")
        label=str((item or {}).get("label") or old or "map").strip()
        if not old or not new or old.casefold()==new.casefold():
            continue
        if "/" in old or "\\" in old or "/" in new or "\\" in new:
            raise HTTPException(
                status_code=400,
                detail=f"Mapnamen voor '{label}' mogen geen / of \\ bevatten."
            )
        clean.append({"old":old,"new":new,"label":label})

    results=[]
    total_moved=0

    for change in clean:
        old_name=change["old"]
        new_name=change["new"]
        before_old=_dropbox_count_named_folders(root,old_name)
        before_new=_dropbox_count_named_folders(root,new_name)

        if before_old==0:
            results.append({
                **change,
                "matched":0,
                "moved":0,
                "verified":True,
                "message":"Geen bestaande mappen met de oude naam gevonden."
            })
            continue

        # Snapshot opnieuw per wijziging. Diepste paden eerst zodat een bovenliggende
        # move geen nog te verwerken child-path ongeldig maakt.
        entries=[
            item for item in _dropbox_recursive_entries(root)
            if str(item.get(".tag") or "")=="folder"
            and str(item.get("name") or "").strip().casefold()==old_name.casefold()
        ]
        entries.sort(
            key=lambda x: str(x.get("path_display") or x.get("path_lower") or "").count("/"),
            reverse=True
        )

        moved_here=0
        for item in entries:
            source=_normalize_dropbox_browser_path(
                str(item.get("path_display") or item.get("path_lower") or "")
            )
            if not source or not _dropbox_get_metadata(source):
                continue
            parent="/".join(source.split("/")[:-1])
            destination=f"{parent}/{new_name}" if parent else f"/{new_name}"

            target=_dropbox_get_metadata(destination)
            if target and str(target.get(".tag") or "")=="folder":
                moved=[]
                _dropbox_merge_move(source,destination,moved)
                moved_here += max(1,len(moved))
            elif target:
                raise HTTPException(
                    status_code=409,
                    detail=f"Kan '{source}' niet hernoemen: '{destination}' bestaat al als bestand."
                )
            else:
                _dropbox_move_path(source,destination)
                moved_here += 1

        after_old=_dropbox_count_named_folders(root,old_name)
        after_new=_dropbox_count_named_folders(root,new_name)
        verified=(after_old==0 and after_new >= before_new + before_old)

        if not verified:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Hernoemen van '{old_name}' naar '{new_name}' kon niet volledig worden bevestigd. "
                    f"Oude naam vóór: {before_old}, erna: {after_old}. "
                    f"Nieuwe naam vóór: {before_new}, erna: {after_new}."
                )
            )

        total_moved += moved_here
        results.append({
            **change,
            "matched":before_old,
            "moved":moved_here,
            "old_remaining":after_old,
            "new_total":after_new,
            "verified":True,
        })

    return {
        "ok":True,
        "verified":True,
        "root":root,
        "renamed_count":total_moved,
        "results":results,
    }


@app.post("/api/dropbox/storage/rename-subfolders")
async def dropbox_rename_storage_subfolders(request: Request):
    try:
        body=await request.json()
    except Exception:
        body={}
    root=_normalize_dropbox_browser_path(body.get("root") or "")
    renames=body.get("renames") or []
    if not isinstance(renames,list):
        raise HTTPException(status_code=400,detail="renames moet een lijst zijn.")
    return _dropbox_rename_structural_folders(root,renames)


@app.post("/api/dropbox/storage/move")
async def dropbox_move_storage(request: Request):
    try:body=await request.json()
    except Exception:body={}
    source=_normalize_dropbox_browser_path(body.get("source") or "");destination=_normalize_dropbox_browser_path(body.get("destination") or "")
    if not source or not destination:raise HTTPException(status_code=400,detail="Bron- en doelmap zijn verplicht.")
    if source==destination:
        snap=_dropbox_recursive_snapshot(destination);return {"ok":True,"verified":True,"moved":[],"before":snap,"after":snap}
    if destination.startswith(source.rstrip("/")+"/"):raise HTTPException(status_code=400,detail="De nieuwe opslagmap mag niet binnen de oude opslagmap liggen.")
    before=_dropbox_recursive_snapshot(source)
    if not before["exists"]:return {"ok":True,"verified":True,"source":source,"destination":destination,"moved_count":0,"before":before,"after":_dropbox_recursive_snapshot(destination),"message":"Geen bestaande bronmap; niets te verplaatsen."}
    moved=[];_dropbox_merge_move(source,destination,moved)
    after_source=_dropbox_recursive_snapshot(source);after=_dropbox_recursive_snapshot(destination)
    verified=(not after_source["exists"] and after["exists"] and after["files"]>=before["files"])
    if not verified:raise HTTPException(status_code=502,detail=f"Dropbox-verplaatsing niet bevestigd. Vooraf {before['files']} bestanden, doel nu {after['files']}, bron bestaat nog: {after_source['exists']}.")
    return {"ok":True,"verified":True,"source":source,"destination":destination,"moved_count":len(moved),"moved":moved[:200],"before":before,"after":after,"source_removed":True}

@app.post("/api/webshop/orders/archive")
async def archive_webshop_order(order_number: str=Form(...),customer_name: str=Form(""),order_confirmation: UploadFile|None=File(None),step_files: list[UploadFile]=File(default=[])):
    cfg=_webshop_storage_config();year=str(datetime.now().year)
    name=_storage_pattern(cfg["pattern"],{"ordernummer":order_number,"klant":customer_name,"jaar":year},_safe_dropbox_name(order_number,"Order"))
    parts=[cfg["root"]]+([year] if cfg["use_year"] else [])+[name];base="/"+"/".join(_storage_clean_part(p) for p in parts if _storage_clean_part(p));stored=[]
    if order_confirmation is not None and order_confirmation.filename:
        data=await order_confirmation.read();fn=_safe_dropbox_name(order_confirmation.filename,"Orderbevestiging.pdf");path=f"{base}/{cfg['confirmation']}/{fn}";meta=_dropbox_upload_bytes(path,data);stored.append(meta.get("path_display") or path)
    for upload in step_files or []:
        if not upload or not upload.filename:continue
        data=await upload.read();fn=_safe_dropbox_name(upload.filename,"productie.step");path=f"{base}/{cfg['step']}/{fn}";meta=_dropbox_upload_bytes(path,data);stored.append(meta.get("path_display") or path)
    return {"ok":True,"order_number":order_number,"folder":base,"stored":stored,"storage_config":cfg}

@app.get("/api/dropbox/oauth/diagnose")
def dropbox_oauth_diagnose():
    """
    Veilige diagnose: toont alleen of Render de variabelen daadwerkelijk aan
    DEZE draaiende server doorgeeft. Geen keys/secrets zelf worden teruggegeven.
    """
    return {
        "ok": True,
        "service": "Vakstaal STEP Server",
        "app_key_found": bool(DROPBOX_APP_KEY),
        "app_key_length": len(DROPBOX_APP_KEY or ""),
        "app_secret_found": bool(DROPBOX_APP_SECRET),
        "app_secret_length": len(DROPBOX_APP_SECRET or ""),
        "redirect_uri_found": bool(DROPBOX_REDIRECT_URI),
        "redirect_uri_length": len(DROPBOX_REDIRECT_URI or ""),
        "redirect_uri_scheme_ok": str(DROPBOX_REDIRECT_URI or "").startswith(("https://","http://")),
        "refresh_token_found": bool(_dropbox_runtime_refresh_token),
        "access_token_found": bool(_dropbox_runtime_access_token),
        "ready_to_authorize": bool(DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REDIRECT_URI),
        "render_service_name": os.environ.get("RENDER_SERVICE_NAME",""),
        "render_external_hostname": os.environ.get("RENDER_EXTERNAL_HOSTNAME",""),
        "persistent_refresh_token_found": bool(_load_dropbox_oauth_credentials().get("refresh_token")),
        "token_storage": "postgresql" if (_postgres_enabled() and _load_dropbox_oauth_credentials().get("refresh_token")) else ("database" if _load_dropbox_oauth_credentials().get("refresh_token") else "environment/local fallback"),
        "config_source": "runtime os.environ + persistent database OAuth",
    }


@app.get("/api/dropbox/oauth/status")
def dropbox_oauth_status():
    app_key = (os.environ.get("DROPBOX_APP_KEY") or DROPBOX_APP_KEY or "").strip()
    app_secret = (os.environ.get("DROPBOX_APP_SECRET") or DROPBOX_APP_SECRET or "").strip()
    redirect_uri = (os.environ.get("DROPBOX_REDIRECT_URI") or DROPBOX_REDIRECT_URI or "").strip()
    return {
        "ok":True,
        "app_key_configured":bool(app_key),
        "app_secret_configured":bool(app_secret),
        "redirect_uri_configured":bool(redirect_uri),
        "refresh_token_configured":bool(_dropbox_runtime_refresh_token),
        "persistent_refresh_token":bool(_load_dropbox_oauth_credentials().get("refresh_token")),
        "ready_to_connect":bool(app_key and app_secret and redirect_uri),
    }


@app.get("/api/dropbox/oauth/start")
def dropbox_oauth_start():
    app_key = (os.environ.get("DROPBOX_APP_KEY") or DROPBOX_APP_KEY or "").strip()
    app_secret = (os.environ.get("DROPBOX_APP_SECRET") or DROPBOX_APP_SECRET or "").strip()
    redirect_uri = (os.environ.get("DROPBOX_REDIRECT_URI") or DROPBOX_REDIRECT_URI or "").strip()
    missing = [
        name for name,value in [
            ("DROPBOX_APP_KEY",app_key),
            ("DROPBOX_APP_SECRET",app_secret),
            ("DROPBOX_REDIRECT_URI",redirect_uri),
        ] if not value
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Deze draaiende server mist: " + ", ".join(missing)
        )
    state=uuid.uuid4().hex
    try:
        DROPBOX_OAUTH_STATE_FILE.write_text(state,encoding="utf-8")
    except Exception:
        pass
    try:
        _save_dropbox_oauth_credentials(oauth_state=state)
    except Exception:
        pass
    params={
        "client_id":app_key,
        "response_type":"code",
        "redirect_uri":redirect_uri,
        "token_access_type":"offline",
        "state":state,
    }
    return {"ok":True,"authorization_url":"https://www.dropbox.com/oauth2/authorize?"+urllib.parse.urlencode(params)}


@app.get("/api/dropbox/oauth/callback")
def dropbox_oauth_callback(code: str="", state: str="", error: str="", error_description: str=""):
    global _dropbox_runtime_access_token, _dropbox_runtime_refresh_token, _dropbox_runtime_root_namespace_id, _dropbox_runtime_home_namespace_id, _dropbox_runtime_account_summary

    app_key=(os.environ.get("DROPBOX_APP_KEY") or DROPBOX_APP_KEY or "").strip()
    app_secret=(os.environ.get("DROPBOX_APP_SECRET") or DROPBOX_APP_SECRET or "").strip()
    redirect_uri=(os.environ.get("DROPBOX_REDIRECT_URI") or DROPBOX_REDIRECT_URI or "").strip()

    def callback_page(title: str, message: str, ok: bool=False, close_after: bool=False):
        safe_title=html.escape(str(title))
        safe_message=html.escape(str(message))
        accent="#20b875" if ok else "#d77b38"
        close_js=(
            """
            try{
              if(window.opener){
                window.opener.postMessage({type:'vakstaal-dropbox-connected'},'*');
              }
            }catch(e){}
            setTimeout(()=>window.close(),1200);
            """
            if close_after else ""
        )
        return HTMLResponse(f"""<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
body{{font-family:Arial,sans-serif;background:#071f2c;color:#eef8fc;padding:36px}}
.card{{max-width:680px;margin:40px auto;padding:26px;border:1px solid #24556d;border-radius:14px;background:#0a3044}}
h2{{margin:0 0 12px;color:{accent}}}
p{{line-height:1.55;color:#c4d9e3;white-space:pre-wrap}}
</style>
</head>
<body><div class="card"><h2>{safe_title}</h2><p>{safe_message}</p></div>
<script>{close_js}</script>
</body></html>""")

    if error:
        return callback_page(
            "Dropbox-autorisatie geannuleerd",
            error_description or error,
            ok=False,
        )

    if not (app_key and app_secret and redirect_uri):
        missing=[
            name for name,value in [
                ("DROPBOX_APP_KEY",app_key),
                ("DROPBOX_APP_SECRET",app_secret),
                ("DROPBOX_REDIRECT_URI",redirect_uri),
            ] if not value
        ]
        return callback_page(
            "Dropbox-configuratie ontbreekt",
            "Deze draaiende server mist: " + ", ".join(missing),
            ok=False,
        )

    expected=""
    try:
        expected=str(_load_dropbox_oauth_credentials().get("oauth_state") or "").strip()
    except Exception:
        expected=""
    if not expected:
        try:
            if DROPBOX_OAUTH_STATE_FILE.exists():
                expected=DROPBOX_OAUTH_STATE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            expected=""

    if expected and state != expected:
        return callback_page(
            "Dropbox-koppeling geweigerd",
            "De OAuth state komt niet overeen. Start de autorisatie opnieuw vanuit de Vakstaal-app.",
            ok=False,
        )

    if not code:
        return callback_page(
            "Dropbox-koppeling mislukt",
            "Dropbox gaf geen autorisatiecode terug.",
            ok=False,
        )

    body=urllib.parse.urlencode({
        "code":code,
        "grant_type":"authorization_code",
        "client_id":app_key,
        "client_secret":app_secret,
        "redirect_uri":redirect_uri,
    }).encode("utf-8")

    req=urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token",
        data=body,
        method="POST",
        headers={"Content-Type":"application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req,timeout=25) as response:
            result=json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode("utf-8",errors="replace")
        return callback_page(
            "Dropbox-token aanvragen mislukt",
            f"Dropbox gaf HTTP {exc.code}.\n\n{detail[:1000]}",
            ok=False,
        )
    except Exception as exc:
        return callback_page(
            "Dropbox-token aanvragen mislukt",
            f"{type(exc).__name__}: {exc}",
            ok=False,
        )

    access=str(result.get("access_token") or "").strip()
    refresh=str(result.get("refresh_token") or "").strip()

    if not access:
        return callback_page(
            "Dropbox-koppeling mislukt",
            "Dropbox gaf geen access token terug.",
            ok=False,
        )

    _dropbox_runtime_access_token=access
    _dropbox_runtime_root_namespace_id=''
    _dropbox_runtime_home_namespace_id=''
    _dropbox_runtime_account_summary={}

    if refresh:
        _dropbox_runtime_refresh_token=refresh
        try:
            _save_dropbox_oauth_credentials(refresh_token=refresh, oauth_state=None)
        except Exception as exc:
            if _postgres_enabled():
                return callback_page(
                    "Dropbox gekoppeld, maar opslag mislukt",
                    "De autorisatie is gelukt, maar de permanente refresh-token kon niet in PostgreSQL worden opgeslagen. "
                    f"Daardoor zou de koppeling na een deploy opnieuw verdwijnen. Fout: {type(exc).__name__}: {exc}",
                    ok=False,
                )
        # Lokale opslag blijft alleen een extra fallback voor lokaal testen.
        try:
            DROPBOX_OAUTH_TOKEN_FILE.write_text(refresh,encoding="utf-8")
        except Exception:
            pass

    try:
        if DROPBOX_OAUTH_STATE_FILE.exists():
            DROPBOX_OAUTH_STATE_FILE.unlink()
    except Exception:
        pass

    return callback_page(
        "Dropbox is gekoppeld",
        "De Dropbox-autorisatie is gelukt. Dit venster sluit automatisch; ga daarna in Vakstaal verder met Toegang controleren.",
        ok=True,
        close_after=True,
    )


@app.post("/api/dropbox/oauth/exchange")
def dropbox_oauth_exchange(payload: dict):
    """
    OAuth relay for the Vercel frontend.
    Dropbox redirects to the public Vercel app; that page sends code/state here.
    This avoids navigating the user's browser to the Render callback URL.
    """
    code = str((payload or {}).get("code") or "")
    state = str((payload or {}).get("state") or "")
    error = str((payload or {}).get("error") or "")
    error_description = str((payload or {}).get("error_description") or "")

    response = dropbox_oauth_callback(
        code=code,
        state=state,
        error=error,
        error_description=error_description,
    )
    body = ""
    try:
        body = bytes(response.body).decode("utf-8", errors="replace")
    except Exception:
        body = str(response)

    ok = "Dropbox is gekoppeld" in body
    if not ok:
        # Strip HTML for a compact frontend error.
        detail = re.sub(r"<[^>]+>", " ", body)
        detail = re.sub(r"\s+", " ", detail).strip()
        raise HTTPException(status_code=400, detail=detail[:1200] or "Dropbox-koppeling mislukt.")
    return {"ok": True, "connected": True}


@app.get("/api/dropbox/account-context")
def dropbox_account_context():
    """
    Veilige controle van het daadwerkelijk gekoppelde Dropbox-account en de
    namespace die Vakstaal als root gebruikt.
    """
    try:
        context=_dropbox_account_context(force=True)
        listing=_dropbox_list_folders("")
        return {
            "ok":True,
            **context,
            "visible_root_folders":[str(x.get("name") or "") for x in (listing.get("folders") or [])],
            "visible_root_folder_count":len(listing.get("folders") or []),
        }
    except HTTPException as exc:
        return {"ok":False,"detail":str(exc.detail)}


@app.get("/api/dropbox/full-access-check")
def dropbox_full_access_check():
    """
    Diagnoseert wat path="" met de HUIDIGE token werkelijk teruggeeft.
    De API kan het access-type zelf niet betrouwbaar als 'Full Dropbox' labelen,
    daarom rapporteren we de zichtbare rootmappen en laten de UI de gebruiker
    bevestigen dat dit overeenkomt met zijn echte Dropbox-root.
    """
    try:
        context=_dropbox_account_context()
        listing=_dropbox_list_folders("")
        folders=listing.get("folders") or []
        return {
            "ok":True,
            "path":"",
            "folder_count":len(folders),
            "folders":[str(item.get("name") or "") for item in folders],
            "storage_default":DROPBOX_ROOT,
            "display_name":context.get("display_name",""),
            "email":context.get("email",""),
            "root_namespace_id":context.get("root_namespace_id",""),
            "home_namespace_id":context.get("home_namespace_id",""),
            "root_differs_from_home":context.get("root_differs_from_home",False),
        }
    except HTTPException as exc:
        return {"ok":False,"detail":str(exc.detail),"folder_count":0,"folders":[]}


@app.get("/api/dropbox/browser-root")
def dropbox_browser_root():
    """
    Browser-root is ALTIJD Dropbox pad "".
    DROPBOX_ROOT is alleen een standaard opslagpad en heeft geen invloed op
    navigatie. Met een Full Dropbox token is dit de echte account-root.
    Met App Folder credentials kan Dropbox zelf niet hoger tonen.
    """
    try:
        listing=_dropbox_list_folders("")
        return {
            "ok":True,
            "path":"",
            "display_name":"Dropbox",
            "navigation_root":True,
            "storage_default":DROPBOX_ROOT,
            "folder_count":len(listing.get("folders") or []),
            "folders":listing.get("folders") or [],
            "requires_full_dropbox_for_account_root":True,
        }
    except HTTPException as exc:
        return {
            "ok":False,
            "path":"",
            "display_name":"Dropbox",
            "detail":str(exc.detail),
        }


@app.get("/api/dropbox/status")
def dropbox_status():
    try:
        result=_dropbox_rpc("check/user",{"query":"vakstaal-dropbox-test"})
        return {
            "ok":True,
            "connected":True,
            "root":DROPBOX_ROOT,
            "refresh_configured":bool(_dropbox_runtime_refresh_token and DROPBOX_APP_KEY and DROPBOX_APP_SECRET),
            "dropbox_response":result,
        }
    except HTTPException as exc:
        return {
            "ok":False,
            "connected":False,
            "root":DROPBOX_ROOT,
            "refresh_configured":bool(_dropbox_runtime_refresh_token and DROPBOX_APP_KEY and DROPBOX_APP_SECRET),
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
            "/api/dropbox/account-context",
            "/api/dropbox/folders",
            "/api/dropbox/folders/create",
            "/api/dropbox/folders/rename",
            "/api/dropbox/folders/delete",
            "/api/dropbox/full-access-check",
            "/api/dropbox/storage/move",
            "/api/dropbox/storage/rename-subfolders",
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
            net_length = float((detail or {}).get("material_length_mm") or (detail or {}).get("length_mm") or raw_length)

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
                "material_length_mm": float((detail or {}).get("material_length_mm") or net_length),
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
        net_length = float((detail or {}).get("material_length_mm") or (detail or {}).get("length_mm") or raw_length)

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
                "material_length_mm": float((detail or {}).get("material_length_mm") or net_length),
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