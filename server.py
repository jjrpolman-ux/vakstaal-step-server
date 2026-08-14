from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

import cadquery as cq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from step_analyzer import analyze_step

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
        return result
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"STEP-analyse mislukt: {exc}") from exc


@app.get("/api/solid-mesh/{job_id}/{solid_index}")
def solid_mesh(job_id: str, solid_index: int):
    cleanup_old_jobs()
    step_path = job_step_path(job_id)

    try:
        imported = cq.importers.importStep(str(step_path))
        solids = imported.solids().vals()

        if solid_index < 1 or solid_index > len(solids):
            raise HTTPException(status_code=404, detail="Onderdeel niet gevonden.")

        solid = solids[solid_index - 1]
        vertices, triangles = solid.tessellate(0.6)

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

        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0

        centered = [[x - cx, y - cy, z - cz] for x, y, z in verts]
        size = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
            1e-6,
        )

        return {
            "job_id": job_id,
            "solid_index": solid_index,
            "vertices": centered,
            "triangles": tris,
            "size": float(size),
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
