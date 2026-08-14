from __future__ import annotations

import os
import shutil
import time
import uuid
import json
import math

import numpy as np
from pathlib import Path

import cadquery as cq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from step_analyzer import analyze_step, _dominant_longitudinal_axis_and_length

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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)



def _analysis_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "analysis.json"


def _assembly_cache_path(job_id: str) -> Path:
    return CACHE_DIR / job_id / "assembly_mesh_original_step_v41.json"


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


def _mesh_shape(shape: cq.Shape, *, center_vertices: bool = False) -> dict:
    vertices, triangles = shape.tessellate(0.9, 0.25)

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

            meshes.append({
                "solid_index": solid_index,
                "vertices": mesh["vertices"],
                "triangles": mesh["triangles"],
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

        return {
            "job_id": job_id,
            "solid_index": solid_index,
            "vertices": mesh["vertices"],
            "triangles": mesh["triangles"],
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
