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
import urllib.parse
import urllib.error
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

import cadquery as cq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
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



def _analysis_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "analysis.json"


def _assembly_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "assembly_mesh_detail_v50.json"


def _load_or_analyze(job_id: str, step_path: Path) -> dict:
    cache = _analysis_cache_path(job_id)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    result = analyze_step(step_path)
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
    Find a real straight cross-section edge so a rectangular profile keeps
    its original rotation around the longitudinal axis.

    Returns (x_direction, x_matches_width).
    """
    best = None

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

        # A cross-section edge must be almost perpendicular to member axis.
        if parallel > 0.08:
            continue

        dw = abs(edge_len - outer_width)
        dh = abs(edge_len - outer_height)
        err = min(dw, dh)

        # Tolerance allows chamfers/STEP numerical noise, but rejects slots.
        tol = max(0.35, min(outer_width, outer_height) * 0.035)
        if err > tol:
            continue

        matches_width = dw <= dh
        if best is None or err < best[0]:
            best = (err, unit, matches_width)

    if best is not None:
        return best[1], best[2]

    # Robust arbitrary perpendicular fallback.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    axis, raw_length, _method = _dominant_longitudinal_axis_and_length(solid)
    zdir = np.array(axis, dtype=float)
    zdir /= max(float(np.linalg.norm(zdir)), 1e-12)

    ow = float((detail or {}).get("outer_width_mm") or 0.0)
    oh = float((detail or {}).get("outer_height_mm") or 0.0)

    # Reuse a real cross-section edge to keep profile rotation correct.
    xdir, x_matches_width = _transverse_profile_direction(
        solid,
        zdir,
        max(ow, 1.0),
        max(oh, 1.0),
    )
    xdir = np.array(xdir, dtype=float)
    xdir /= max(float(np.linalg.norm(xdir)), 1e-12)
    ydir = np.cross(zdir, xdir)
    ydir /= max(float(np.linalg.norm(ydir)), 1e-12)

    if not x_matches_width:
        # Swap transverse axes so x corresponds to outer_width.
        xdir, ydir = ydir, xdir

    c = np.array(solid.Center().toTuple(), dtype=float)

    vals = []
    for v in solid.Vertices():
        p = np.array(v.toTuple(), dtype=float)
        vals.append(float(np.dot(p - c, zdir)))
    half_len = (max(vals) - min(vals)) / 2.0 if vals else float(raw_length) / 2.0

    return c, xdir, ydir, zdir, ow, oh, half_len


def _is_standard_profile_edge(
    edge: cq.Shape,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float],
) -> bool:
    """
    Fast feature classifier using a profile basis computed ONCE per solid.

    True  = ordinary profile geometry.
    False = likely machining feature (hole/slot/notch/tab/lip/cut-out).
    """
    c, xdir, ydir, zdir, ow, oh, half_len = basis

    if ow <= 0.0 or oh <= 0.0:
        return True

    gt = str(edge.geomType() or "").upper()

    # Circular/spline edges are machining features on standard rectangular tubes.
    if gt in {"CIRCLE", "ELLIPSE", "BSPLINE", "BEZIER"}:
        return False

    pts = _edge_points(edge)
    if len(pts) < 2:
        return True

    hpw = ow / 2.0
    hph = oh / 2.0
    tol_xy = max(0.35, min(ow, oh) * 0.035)
    tol_z = max(0.5, half_len * 0.002)

    coords = []
    for p0 in pts:
        p = np.array(p0, dtype=float) - c
        coords.append((
            float(np.dot(p, xdir)),
            float(np.dot(p, ydir)),
            float(np.dot(p, zdir)),
        ))

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    zs = [p[2] for p in coords]

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)

    # Standard longitudinal corner edges.
    if dz > max(dx, dy) * 5.0:
        on_x = all(abs(abs(x) - hpw) <= tol_xy for x in xs)
        on_y = all(abs(abs(y) - hph) <= tol_xy for y in ys)
        if on_x or on_y:
            return True

    # Standard end perimeter.
    at_end = (
        all(abs(abs(z) - half_len) <= tol_z for z in zs)
        or (
            max(zs) - min(zs) <= tol_z
            and abs(abs(sum(zs) / len(zs)) - half_len) <= tol_z
        )
    )

    if at_end:
        on_x = all(abs(abs(x) - hpw) <= tol_xy for x in xs)
        on_y = all(abs(abs(y) - hph) <= tol_xy for y in ys)
        if on_x or on_y:
            return True

    if gt == "LINE":
        on_outer_x = all(abs(abs(x) - hpw) <= tol_xy for x in xs)
        on_outer_y = all(abs(abs(y) - hph) <= tol_xy for y in ys)

        if (on_outer_x or on_outer_y) and (
            at_end or dz > max(dx, dy) * 5.0
        ):
            return True

    return False



def _feature_polylines(
    solid: cq.Shape,
    detail: dict | None,
) -> list[list[list[float]]]:
    """
    Return likely extra-machining edges.

    Performance-critical: the profile coordinate basis is calculated ONCE per
    solid instead of once per edge. This makes large assemblies much faster.
    """
    if not detail or not detail.get("recognized"):
        return []

    profile_type = str(detail.get("type") or "").lower()
    if not any(k in profile_type for k in ("vierkant", "rechthoek", "koker", "rond")):
        return []

    try:
        basis = _profile_basis_for_features(solid, detail)
    except Exception:
        return []

    features: list[list[list[float]]] = []

    try:
        for edge in solid.Edges():
            if _is_standard_profile_edge(edge, basis):
                continue

            pts = _edge_points(edge)
            if len(pts) >= 2:
                features.append(pts)
    except Exception:
        return []

    return features



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
DROPBOX_APP_KEY = os.environ.get("DROPBOX_APP_KEY", "").strip()
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET", "").strip()
DROPBOX_REFRESH_TOKEN = os.environ.get("DROPBOX_REFRESH_TOKEN", "").strip()
DROPBOX_ROOT = os.environ.get("DROPBOX_ROOT", "/Offertes").strip() or "/Offertes"

# Tijdelijke access token alleen in het geheugen bewaren.
_dropbox_cached_access_token = ""
_dropbox_cached_access_token_expires_at = 0.0


def _dropbox_refresh_configured() -> bool:
    return bool(
        DROPBOX_APP_KEY
        and DROPBOX_APP_SECRET
        and DROPBOX_REFRESH_TOKEN
    )


def _refresh_dropbox_access_token() -> str:
    """Vraag met de permanente refresh token een nieuwe tijdelijke access token op."""
    global _dropbox_cached_access_token
    global _dropbox_cached_access_token_expires_at

    if not _dropbox_refresh_configured():
        raise HTTPException(
            status_code=503,
            detail="Dropbox refresh-token instellingen ontbreken op de server."
        )

    payload = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_APP_KEY,
        "client_secret": DROPBOX_APP_SECRET,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox token vernieuwen mislukt ({exc.code}): {detail[:700]}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox token vernieuwen mislukt: {type(exc).__name__}: {exc}"
        ) from exc

    token = str(result.get("access_token") or "").strip()
    if not token:
        raise HTTPException(
            status_code=502,
            detail="Dropbox gaf geen nieuwe access token terug."
        )

    # Dropbox retourneert expires_in. Houd 90 seconden veiligheidsmarge aan.
    expires_in = max(120, int(result.get("expires_in") or 14400))
    _dropbox_cached_access_token = token
    _dropbox_cached_access_token_expires_at = time.time() + expires_in - 90
    return token


def _dropbox_access_token(*, force_refresh: bool = False) -> str:
    # Permanente servermodus heeft altijd voorrang op een handmatig gegenereerde token.
    if _dropbox_refresh_configured():
        if (
            force_refresh
            or not _dropbox_cached_access_token
            or time.time() >= _dropbox_cached_access_token_expires_at
        ):
            return _refresh_dropbox_access_token()
        return _dropbox_cached_access_token

    # Tijdelijke fallback voor migratie/noodgevallen.
    if DROPBOX_ACCESS_TOKEN:
        return DROPBOX_ACCESS_TOKEN

    raise HTTPException(
        status_code=503,
        detail="Dropbox is niet geconfigureerd: refresh token en access token ontbreken."
    )


def _dropbox_headers(*, force_refresh: bool = False) -> dict:
    return {
        "Authorization": f"Bearer {_dropbox_access_token(force_refresh=force_refresh)}"
    }


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
        if exc.code == 401 and _dropbox_refresh_configured():
            retry = urllib.request.Request(
                f"https://api.dropboxapi.com/2/{endpoint}",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    **_dropbox_headers(force_refresh=True),
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(retry, timeout=25) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Dropbox API fout na tokenvernieuwing ({retry_exc.code}): {retry_detail[:700]}"
                ) from retry_exc
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
    args = {
        "path": path,
        "mode": "overwrite",
        "autorename": False,
        "mute": True,
    }
    req = urllib.request.Request(
        "https://content.dropboxapi.com/2/files/upload",
        data=data,
        method="POST",
        headers={
            **_dropbox_headers(),
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps(args, separators=(",", ":")),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401 and _dropbox_refresh_configured():
            retry = urllib.request.Request(
                "https://content.dropboxapi.com/2/files/upload",
                data=data,
                method="POST",
                headers={
                    **_dropbox_headers(force_refresh=True),
                    "Content-Type": "application/octet-stream",
                    "Dropbox-API-Arg": json.dumps(args, separators=(",", ":")),
                },
            )
            try:
                with urllib.request.urlopen(retry, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Dropbox uploadfout na tokenvernieuwing ({retry_exc.code}): {retry_detail[:700]}"
                ) from retry_exc
        raise HTTPException(
            status_code=502,
            detail=f"Dropbox uploadfout ({exc.code}): {detail[:700]}"
        ) from exc
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
        if exc.code == 401 and _dropbox_refresh_configured():
            retry = urllib.request.Request(
                "https://content.dropboxapi.com/2/files/download",
                data=b"",
                method="POST",
                headers={
                    **_dropbox_headers(force_refresh=True),
                    "Dropbox-API-Arg": json.dumps({"path": path}, separators=(",", ":")),
                },
            )
            try:
                with urllib.request.urlopen(retry, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Dropbox downloadfout na tokenvernieuwing ({retry_exc.code}): {retry_detail[:700]}"
                ) from retry_exc
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
    if suffix == ".pdf":
        return "PDF"
    if suffix in {".zx", ".nest"}:
        return "Nest"
    if suffix in {".step", ".stp"}:
        # Door de calculator gegenereerde bibliotheek-STEP's beginnen met het
        # offertenummer en bevatten niet altijd letterlijk 'productie'.
        production_words = ("productie", "production", "solid_", "onderdeel_", "part_")
        is_generated_quote_step = bool(re.match(r"^VAK-20\d{2}-", str(filename or ""), re.I))
        return "Productie STEP" if (is_generated_quote_step or any(w in lower for w in production_words)) else "Origineel"
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
                WHERE quote_id=%s AND filename=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                """
                SELECT id, dropbox_path FROM quote_files
                WHERE quote_id=? AND filename=?
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            (quote_id, filename)
        )
        existing = cur.fetchone()

        # Probeer Dropbox, maar laat een Dropbox-probleem NOOIT het opslaan van
        # de offerte of de productie-STEP blokkeren. Bij een fout bewaren we de
        # echte bytes in de database en kan een volgende save opnieuw syncen.
        dropbox_ok = False
        actual_path = None
        try:
            uploaded = _dropbox_upload_bytes(dropbox_path, data)
            actual_path = uploaded.get("path_display") or uploaded.get("path_lower") or dropbox_path
            dropbox_ok = True
        except HTTPException:
            actual_path = None

        if existing:
            existing_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
            cur.execute(
                _sql(
                    """
                    UPDATE quote_files
                    SET content_type=%s, file_kind=%s, file_size=%s,
                        dropbox_path=%s, data=%s, created_at=%s
                    WHERE id=%s
                    """,
                    """
                    UPDATE quote_files
                    SET content_type=?, file_kind=?, file_size=?,
                        dropbox_path=?, data=?, created_at=?
                    WHERE id=?
                    """
                ),
                (
                    upload.content_type or "application/octet-stream",
                    kind,
                    len(data),
                    actual_path,
                    b"" if dropbox_ok else data,
                    _utcnow(),
                    existing_id,
                )
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
                b"" if dropbox_ok else data,
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
async def create_quote(
    payload: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
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
        conn.commit()

        # Dropbox is secundaire opslag. Een verwijderde/verlopen Dropbox-entry
        # mag de offerte in PostgreSQL niet terugdraaien.
        try:
            _sync_quote_json_to_dropbox(conn, quote_id)
        except HTTPException:
            pass

        return _quote_response(conn, quote_id)


@app.put("/api/quotes/{quote_id}")
async def update_quote(
    quote_id: str,
    payload: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
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
        conn.commit()

        # Dropbox is secundaire opslag. Een verwijderde/verlopen Dropbox-entry
        # mag de offerte in PostgreSQL niet terugdraaien.
        try:
            _sync_quote_json_to_dropbox(conn, quote_id)
        except HTTPException:
            pass

        return _quote_response(conn, quote_id)


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


@app.get("/api/dropbox/status")
def dropbox_status():
    result = _dropbox_rpc("check/user", {"query": "vakstaal-dropbox-test"})
    return {
        "ok": True,
        "configured": True,
        "root": DROPBOX_ROOT,
        "storage_mode": "dropbox_files_database_metadata",
        "auth_mode": "refresh_token" if _dropbox_refresh_configured() else "access_token_fallback",
        "dropbox_response": result,
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
            feature_lines = _feature_polylines(solid, detail)

            meshes.append({
                "solid_index": solid_index,
                "vertices": mesh["vertices"],
                "triangles": mesh["triangles"],
                "feature_lines": feature_lines,
                "has_extra_features": bool(feature_lines),
                "trimmed_visual": bool(was_trimmed),
                "feature_preserving_trim": False,
                "original_final_step_geometry": True,
                "raw_length_mm": raw_length,
                "net_length_mm": net_length,
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

        raw_feature_lines = _feature_polylines(solid, detail)

        # _mesh_shape(center_vertices=True) centers vertices on the original
        # visible bounding box. Apply the same translation to feature lines.
        bb = visible_shape.BoundingBox()
        cx = (float(bb.xmin) + float(bb.xmax)) / 2.0
        cy = (float(bb.ymin) + float(bb.ymax)) / 2.0
        cz = (float(bb.zmin) + float(bb.zmax)) / 2.0

        feature_lines = [
            [[p[0]-cx, p[1]-cy, p[2]-cz] for p in line]
            for line in raw_feature_lines
        ]

        return {
            "job_id": job_id,
            "solid_index": solid_index,
            "vertices": mesh["vertices"],
            "triangles": mesh["triangles"],
            "feature_lines": feature_lines,
            "has_extra_features": bool(feature_lines),
            "size": mesh["size"],
            "net_geometry": True,
            "trimmed_visual": bool(was_trimmed),
            "feature_preserving_trim": False,
            "original_final_step_geometry": True,
            "raw_length_mm": raw_length,
            "net_length_mm": net_length,
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