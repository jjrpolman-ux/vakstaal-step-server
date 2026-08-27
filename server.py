from __future__ import annotations
import base64
import html

import os
import shutil
import time
import uuid
import json
import math
import zlib
import re
import sqlite3
import smtplib
import ssl
from email.message import EmailMessage
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
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

STEP_MATERIAL_LENGTH_VERSION = 8  # full body projection along longitudinal profile axis
STEP_PROFILE_RECOGNITION_VERSION = 8  # robust main-wall / coaxial-cylinder recognition

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
    Vakstaal materiaallengte v8.

    Enige definitie van de materiaallengte:
    de volledige fysieke projectiespan van de complete STEP-body langs de
    gedetecteerde longitudinale profielas.

    Waarom dit leidend is:
    - afgeronde hoeken en wanddikte mogen de eind-tot-eindmaat niet verkorten;
    - gaten/sleuven binnen de body veranderen de uiterste projectie niet;
    - schuine/eindbewerkte contouren blijven onderdeel van de werkelijke blank;
    - dezelfde meetdefinitie werkt voor RHS/SHS, ronde buis en andere profielen.

    De eerdere methode op basis van een rechte rand van een grote hoofdwand is
    bewust verwijderd. Bij 50x15x1,5 kon die rand door de hoek-/eindgeometrie
    exact 1,5 mm korter zijn dan de echte profielmaat (850 -> 848,5 en
    2015 -> 2013,5).
    """
    analyzer_axis, analyzer_length, analyzer_method = _dominant_longitudinal_axis_and_length(solid)

    axis=np.array(analyzer_axis,dtype=float)
    axis_norm=float(np.linalg.norm(axis))
    if axis_norm<=1e-12:
        raise ValueError("Lengteas van STEP-body kon niet worden bepaald.")
    axis=axis/axis_norm

    # Stabiele richting voor reproduceerbare diagnosevelden.
    for component in axis:
        if abs(float(component))>1e-9:
            if component<0:
                axis=-axis
            break

    projections=[]
    for vertex in solid.Vertices():
        try:
            point=np.array(vertex.toTuple(),dtype=float)
            projections.append(float(np.dot(point,axis)))
        except Exception:
            continue

    if len(projections)>=2:
        physical_length=float(max(projections)-min(projections))
        if math.isfinite(physical_length) and physical_length>1e-6:
            return physical_length,axis,"full-body-axis-projection-v8"

    # Alleen bij een werkelijk onbruikbare topologie terugvallen op de analyzer.
    fallback=max(0.0,float(analyzer_length or 0.0))
    return fallback,axis,f"{analyzer_method}-fallback-v8"


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
            "definition": "volledige fysieke projectiespan van de body langs de profielas",
        })

    result["details"] = details
    result["material_length_version"] = STEP_MATERIAL_LENGTH_VERSION
    result["material_length_definition"] = (
        "full physical body projection span along the longitudinal profile axis"
    )
    result["material_length_audit"] = length_audit
    return result



def _stable_unit(vector) -> np.ndarray:
    a=np.array(vector,dtype=float)
    n=float(np.linalg.norm(a))
    if n<=1e-12:
        return np.array([0.0,0.0,1.0],dtype=float)
    a=a/n
    for component in a:
        if abs(float(component))>1e-9:
            if component<0:
                a=-a
            break
    return a

def _cluster_scalar_values(values, tolerance: float) -> list[float]:
    vals=sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return []
    groups=[[vals[0]]]
    for value in vals[1:]:
        if abs(value-groups[-1][-1])<=tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group)/len(group) for group in groups]

def _profile_number_nl(value: float) -> str:
    n=round(float(value),2)
    if abs(n-round(n))<1e-8:
        return str(int(round(n)))
    return ('%.2f'%n).rstrip('0').rstrip('.').replace('.',',')

def _face_longitudinal_span(face: cq.Shape, axis: np.ndarray) -> float:
    values=[]
    for vertex in face.Vertices():
        try:
            values.append(float(np.dot(np.array(vertex.toTuple(),dtype=float),axis)))
        except Exception:
            continue
    return max(values)-min(values) if values else 0.0

def _robust_round_tube_dimensions(solid: cq.Shape, axis: np.ndarray, length_mm: float):
    radii=[]
    min_span=max(1.0,float(length_mm)*0.35)
    for face in solid.Faces():
        try:
            if face.geomType()!='CYLINDER':
                continue
            adaptor=BRepAdaptor_Surface(face.wrapped)
            cylinder=adaptor.Cylinder()
            direction=cylinder.Axis().Direction()
            cyl_axis=_stable_unit([direction.X(),direction.Y(),direction.Z()])
            if abs(float(np.dot(cyl_axis,axis)))<0.995:
                continue
            if _face_longitudinal_span(face,axis)<min_span:
                continue
            radius=float(cylinder.Radius())
            if radius>0.05 and math.isfinite(radius):
                radii.append(radius)
        except Exception:
            continue
    distinct=_cluster_scalar_values(radii,0.04)
    if len(distinct)<2:
        return None
    outer=max(distinct)
    inner_candidates=[r for r in distinct if r<outer-0.05]
    if not inner_candidates:
        return None
    inner=max(inner_candidates)
    wall=outer-inner
    diameter=outer*2.0
    if not (diameter>1.0 and wall>0.10 and wall<outer*0.48):
        return None
    return diameter,wall

def _robust_rectangular_tube_dimensions(solid: cq.Shape, axis: np.ndarray, length_mm: float):
    min_span=max(1.0,float(length_mm)*0.35)
    families=[]
    for face in solid.Faces():
        try:
            if face.geomType()!='PLANE':
                continue
            adaptor=BRepAdaptor_Surface(face.wrapped)
            direction=adaptor.Plane().Axis().Direction()
            normal=_stable_unit([direction.X(),direction.Y(),direction.Z()])
            if abs(float(np.dot(normal,axis)))>0.08:
                continue
            span=_face_longitudinal_span(face,axis)
            if span<min_span:
                continue
            area=float(face.Area())
            if not math.isfinite(area) or area<=1e-5:
                continue
            family=None
            for candidate in families:
                if abs(float(np.dot(normal,candidate['normal'])))>=0.995:
                    family=candidate
                    break
            if family is None:
                family={'normal':normal,'offsets':[]}
                families.append(family)
            center=np.array(face.Center().toTuple(),dtype=float)
            family['offsets'].append(float(np.dot(center,family['normal'])))
        except Exception:
            continue

    dimensions=[]
    for family in families:
        levels=_cluster_scalar_values(family['offsets'],0.20)
        if len(levels)<4:
            continue
        lo,hi=levels[0],levels[-1]
        inner_lo,inner_hi=levels[1],levels[-2]
        outer_size=hi-lo
        wall_a=inner_lo-lo
        wall_b=hi-inner_hi
        wall=(wall_a+wall_b)/2.0
        if not (
            outer_size>1.0 and wall>0.10 and outer_size>2.0*wall+0.20
            and abs(wall_a-wall_b)<=max(0.30,wall*0.18)
        ):
            continue
        dimensions.append((outer_size,wall))

    if len(dimensions)<2:
        return None
    dimensions=sorted(dimensions,key=lambda item:item[0],reverse=True)[:2]
    width,height=dimensions[0][0],dimensions[1][0]
    walls=[dimensions[0][1],dimensions[1][1]]
    if abs(walls[0]-walls[1])>max(0.30,(sum(walls)/2.0)*0.18):
        return None
    return max(width,height),min(width,height),sum(walls)/2.0

def _apply_robust_standard_profile_recognition(step_path: Path, result: dict) -> dict:
    result=dict(result or {})
    details=[dict(d or {}) for d in (result.get('details') or [])]
    imported=cq.importers.importStep(str(step_path))
    solids=imported.solids().vals()
    audit=[]

    for index,solid in enumerate(solids):
        if index>=len(details):
            break
        detail=details[index]
        old_type=str(detail.get('type') or '')
        old_size=str(detail.get('profile_size') or '')
        length_mm=float(detail.get('material_length_mm') or detail.get('length_mm') or 0.0)

        axis_raw=detail.get('profile_axis')
        if isinstance(axis_raw,(list,tuple)) and len(axis_raw)>=3:
            axis=_stable_unit(axis_raw[:3])
        else:
            try:
                detected_axis,_len,_method=_dominant_longitudinal_axis_and_length(solid)
                axis=_stable_unit(detected_axis)
            except Exception:
                axis=np.array([0.0,0.0,1.0],dtype=float)

        corrected=None
        round_dims=_robust_round_tube_dimensions(solid,axis,length_mm)
        if round_dims:
            diameter,wall=round_dims
            corrected={
                'type':'Rond',
                'profile_size':f"Ø{_profile_number_nl(diameter)}x{_profile_number_nl(wall)}",
                'outer_width_mm':float(diameter),
                'outer_height_mm':float(diameter),
                'outer_diameter_mm':float(diameter),
                'thickness_mm':float(wall),
                'wall_thickness_mm':float(wall),
                'recognized':True,
                'standard_profile':True,
                'profile_shape':'round-tube',
                'profile_recognition_method':'coaxial-cylinder-radii-v8',
            }
        else:
            rect_dims=_robust_rectangular_tube_dimensions(solid,axis,length_mm)
            if rect_dims:
                width,height,wall=rect_dims
                square=abs(width-height)<=max(0.20,max(width,height)*0.004)
                if square:
                    mean=(width+height)/2.0
                    width=height=mean
                corrected={
                    'type':'Vierkant' if square else 'Rechthoekig',
                    'profile_size':f"{_profile_number_nl(width)}x{_profile_number_nl(height)}x{_profile_number_nl(wall)}",
                    'outer_width_mm':float(width),
                    'outer_height_mm':float(height),
                    'thickness_mm':float(wall),
                    'wall_thickness_mm':float(wall),
                    'recognized':True,
                    'standard_profile':True,
                    'profile_shape':'square-tube' if square else 'rectangular-tube',
                    'profile_recognition_method':'main-wall-levels-v8',
                }

        if corrected:
            changed=(old_type!=corrected['type'] or old_size!=corrected['profile_size'] or detail.get('recognized') is not True)
            if changed:
                detail['analyzer_type_before_correction']=old_type
                detail['analyzer_profile_size_before_correction']=old_size
                detail['profile_corrected']=True
            detail.update(corrected)
            if changed and detail.get('warning'):
                detail['analyzer_warning_before_correction']=detail.get('warning')
                detail['warning']=''
        else:
            detail['standard_profile']=bool(
                detail.get('recognized') and any(
                    k in str(detail.get('type') or '').lower()
                    for k in ('vierkant','rechthoek','koker','rond','buis')
                )
            )
            detail.setdefault('profile_recognition_method','legacy-analyzer')

        detail['profile_recognition_version']=STEP_PROFILE_RECOGNITION_VERSION
        details[index]=detail
        audit.append({
            'solid_index':index+1,
            'before_type':old_type,
            'before_profile_size':old_size,
            'after_type':str(detail.get('type') or ''),
            'after_profile_size':str(detail.get('profile_size') or ''),
            'recognized':bool(detail.get('recognized')),
            'standard_profile':bool(detail.get('standard_profile')),
            'method':str(detail.get('profile_recognition_method') or ''),
        })

    result['details']=details
    result['profile_recognition_version']=STEP_PROFILE_RECOGNITION_VERSION
    result['profile_recognition_audit']=audit
    result['recognized_count']=sum(1 for d in details if d.get('recognized'))
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
            recognition_versions=[
                int((d or {}).get("profile_recognition_version") or 0)
                for d in (cached.get("details") or [])
            ]
            if (
                int(cached.get("material_length_version") or 0) == STEP_MATERIAL_LENGTH_VERSION
                and int(cached.get("profile_recognition_version") or 0) == STEP_PROFILE_RECOGNITION_VERSION
                and detail_versions and recognition_versions
                and all(v == STEP_MATERIAL_LENGTH_VERSION for v in detail_versions)
                and all(v == STEP_PROFILE_RECOGNITION_VERSION for v in recognition_versions)
            ):
                return cached
        except Exception:
            pass

    result = analyze_step(step_path)
    result = _apply_physical_material_lengths(step_path, result)
    result = _apply_robust_standard_profile_recognition(step_path, result)

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


def _cut_layer_storage_config() -> dict:
    c=_storage_state_config()
    layers=c.get("cutLayers") if isinstance(c.get("cutLayers"),dict) else {}
    raw=str(layers.get("dropboxRoot") or "Snijlayers").strip().replace("\\","/")
    raw=re.sub(r"^/?Dropbox(?:/|$)","/",raw,flags=re.I)
    return {"root":_storage_clean_part(raw,"Snijlayers")}


def _dropbox_storage_route_state(path: str) -> dict:
    normalized=_normalize_dropbox_browser_path(path)
    configured=bool(normalized)
    if not configured:
        return {"configured":False,"ok":False,"path":"","pathDisplay":""}
    meta=_dropbox_get_metadata(normalized)
    ok=bool(meta and str(meta.get(".tag") or "")=="folder")
    return {
        "configured":True,
        "ok":ok,
        "path":normalized,
        "pathDisplay":str((meta or {}).get("path_display") or normalized),
    }

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
                CREATE TABLE IF NOT EXISTS quote_approvals (
                    quote_id TEXT PRIMARY KEY REFERENCES quotes(id) ON DELETE CASCADE,
                    token TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    viewed_at TEXT,
                    accepted_at TEXT,
                    accepted_by TEXT,
                    note TEXT,
                    email_sent_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS quote_approvals (
                    quote_id TEXT PRIMARY KEY,
                    token TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    viewed_at TEXT,
                    accepted_at TEXT,
                    accepted_by TEXT,
                    note TEXT,
                    email_sent_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(quote_id) REFERENCES quotes(id) ON DELETE CASCADE
                )
                """
            )
        )

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



def _approval_public_base() -> str:
    return (
        str(os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
        or str(os.environ.get("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
    )

def _approval_url(token: str) -> str:
    path=f"/approve/{urllib.parse.quote(str(token or ''))}"
    base=_approval_public_base()
    return f"{base}{path}" if base else path

def _approval_for_quote(conn, quote_id: str, create: bool=True) -> dict | None:
    cur=conn.cursor()
    cur.execute(_sql(
        """SELECT quote_id,token,status,viewed_at,accepted_at,accepted_by,note,email_sent_at,created_at,updated_at
           FROM quote_approvals WHERE quote_id=%s""",
        """SELECT quote_id,token,status,viewed_at,accepted_at,accepted_by,note,email_sent_at,created_at,updated_at
           FROM quote_approvals WHERE quote_id=?"""
    ),(quote_id,))
    row=_row_to_dict(cur.fetchone(),cur)
    if row:
        row["url"]=_approval_url(row.get("token") or "")
        return row
    if not create:
        return None
    now=_utcnow()
    token=uuid.uuid4().hex+uuid.uuid4().hex[:12]
    cur.execute(_sql(
        """INSERT INTO quote_approvals (quote_id,token,status,created_at,updated_at)
           VALUES (%s,%s,%s,%s,%s)""",
        """INSERT INTO quote_approvals (quote_id,token,status,created_at,updated_at)
           VALUES (?,?,?,?,?)"""
    ),(quote_id,token,"pending",now,now))
    return {"quote_id":quote_id,"token":token,"status":"pending","viewed_at":None,
            "accepted_at":None,"accepted_by":None,"note":None,"email_sent_at":None,
            "created_at":now,"updated_at":now,"url":_approval_url(token)}

def _approval_by_token(conn, token: str) -> dict:
    cur=conn.cursor()
    cur.execute(_sql(
        """SELECT a.quote_id,a.token,a.status,a.viewed_at,a.accepted_at,a.accepted_by,a.note,a.email_sent_at,
                  a.created_at,a.updated_at,q.quote_number,q.customer_name,q.contact_person,q.customer_email,q.total_ex_vat
           FROM quote_approvals a JOIN quotes q ON q.id=a.quote_id WHERE a.token=%s""",
        """SELECT a.quote_id,a.token,a.status,a.viewed_at,a.accepted_at,a.accepted_by,a.note,a.email_sent_at,
                  a.created_at,a.updated_at,q.quote_number,q.customer_name,q.contact_person,q.customer_email,q.total_ex_vat
           FROM quote_approvals a JOIN quotes q ON q.id=a.quote_id WHERE a.token=?"""
    ),(token,))
    row=_row_to_dict(cur.fetchone(),cur)
    if not row:
        raise HTTPException(status_code=404, detail="Deze akkoordlink is niet geldig.")
    return row

def _approval_email_configured() -> bool:
    return all(str(os.environ.get(k) or "").strip() for k in
               ("SMTP_HOST","SMTP_USER","SMTP_PASSWORD","QUOTE_APPROVAL_NOTIFY_EMAIL"))

def _send_approval_email(data: dict) -> bool:
    if not _approval_email_configured():
        return False
    host=str(os.environ.get("SMTP_HOST") or "").strip()
    port=int(os.environ.get("SMTP_PORT") or "587")
    user=str(os.environ.get("SMTP_USER") or "").strip()
    password=str(os.environ.get("SMTP_PASSWORD") or "").strip()
    recipient=str(os.environ.get("QUOTE_APPROVAL_NOTIFY_EMAIL") or "").strip()
    sender=str(os.environ.get("SMTP_FROM") or user).strip()
    use_ssl=str(os.environ.get("SMTP_SSL") or "").strip().lower() in {"1","true","yes"} or port==465

    msg=EmailMessage()
    msg["Subject"]=f"Offerte {data.get('quote_number') or ''} is akkoord"
    msg["From"]=sender
    msg["To"]=recipient
    msg.set_content(
        "De klant heeft digitaal akkoord gegeven op een Vakstaal-offerte.\n\n"
        f"Offerte: {data.get('quote_number') or '-'}\n"
        f"Klant: {data.get('customer_name') or '-'}\n"
        f"Akkoord door: {data.get('accepted_by') or '-'}\n"
        f"Datum/tijd: {data.get('accepted_at') or '-'}\n"
        f"Bedrag excl. btw: EUR {float(data.get('total_ex_vat') or 0):.2f}\n"
        + (f"Opmerking: {data.get('note')}\n" if data.get("note") else "")
    )
    ctx=ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host,port,timeout=15,context=ctx) as smtp:
            smtp.login(user,password); smtp.send_message(msg)
    else:
        with smtplib.SMTP(host,port,timeout=15) as smtp:
            smtp.ehlo(); smtp.starttls(context=ctx); smtp.ehlo()
            smtp.login(user,password); smtp.send_message(msg)
    return True

def _approval_html(data: dict, accepted: bool=False) -> str:
    qno=html.escape(str(data.get("quote_number") or "Offerte"))
    customer=html.escape(str(data.get("customer_name") or ""))
    total=f"{float(data.get('total_ex_vat') or 0):.2f}".replace(".",",")
    token=html.escape(str(data.get("token") or ""),quote=True)
    done=accepted or str(data.get("status") or "")=="accepted"
    if done:
        content=f"""
        <div class="badge">✓ AKKOORD ONTVANGEN</div><h1>Bedankt voor uw akkoord</h1>
        <p>Uw akkoord op offerte <b>{qno}</b> is geregistreerd.</p>
        <div class="facts"><div><span>Klant</span><b>{customer}</b></div><div><span>Bedrag excl. btw</span><b>€ {total}</b></div>
        <div><span>Akkoord door</span><b>{html.escape(str(data.get('accepted_by') or '-'))}</b></div>
        <div><span>Geregistreerd</span><b>{html.escape(str(data.get('accepted_at') or '-'))}</b></div></div>
        <p class="muted">Vakstaal heeft uw akkoord ontvangen.</p>"""
    else:
        content=f"""
        <div class="eyebrow">DIGITAAL AKKOORD</div><h1>Offerte {qno}</h1>
        <p>Controleer de gegevens en bevestig hieronder uw akkoord.</p>
        <div class="facts"><div><span>Klant</span><b>{customer}</b></div><div><span>Bedrag excl. btw</span><b>€ {total}</b></div></div>
        <form method="post" action="/approve/{token}/accept">
          <label>Naam van degene die akkoord geeft<input name="accepted_by" required maxlength="120"></label>
          <label>Opmerking <small>(optioneel)</small><textarea name="note" maxlength="1000" rows="3"></textarea></label>
          <label class="check"><input type="checkbox" required><span>Ik geef akkoord op deze offerte.</span></label>
          <button type="submit">✓ Akkoord met offerte</button>
        </form>"""
    return f"""<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{qno} · Vakstaal</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#071e29;color:#eaf7fb;font-family:Arial,sans-serif;padding:24px}}
main{{max-width:650px;margin:35px auto;background:#0a2a38;border:1px solid #245167;border-radius:16px;padding:28px;box-shadow:0 24px 70px #0007}}
.eyebrow{{font-size:10px;letter-spacing:.15em;color:#58cfff;font-weight:900}}h1{{margin:8px 0}}p{{color:#b8ccd5;line-height:1.55}}
.facts{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:22px 0}}.facts div{{padding:13px;border:1px solid #1c4659;border-radius:10px;background:#082431}}
.facts span{{display:block;color:#7fa0ae;font-size:10px;margin-bottom:5px}}label{{display:block;margin:13px 0;font-size:12px;font-weight:800}}
input,textarea{{width:100%;margin-top:6px;padding:11px;border-radius:8px;border:1px solid #356176;background:#061c26;color:#fff}}
.check{{display:flex;gap:9px;align-items:center}}.check input{{width:18px;margin:0}}button{{width:100%;padding:14px;border:0;border-radius:9px;background:#1597d0;color:#fff;font-weight:900;font-size:15px}}
.badge{{display:inline-block;padding:7px 10px;border-radius:999px;background:#103d2c;border:1px solid #2ea769;color:#91edb9;font-size:11px;font-weight:900}}
.muted,small{{color:#7896a3;font-size:10px}}@media(max-width:560px){{body{{padding:10px}}main{{margin:10px auto;padding:20px}}.facts{{grid-template-columns:1fr}}}}
</style></head><body><main>{content}</main></body></html>"""

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
    row["approval"] = _approval_for_quote(conn, quote_id, create=True)
    conn.commit()
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




def _lcm_read_ascii_value(block: bytes, field_index: int) -> str:
    """
    FS Material / LCM Cut-block veld:
      00 01 <field-index> 00 00 <len> <ascii>
    Alleen de tekstvelden uitlezen die we voor basisadvies nodig hebben.
    """
    marker = bytes([0x00, 0x01, field_index, 0x00, 0x00])
    pos = block.find(marker)
    if pos < 0:
        return ""
    pos += len(marker)
    if pos >= len(block):
        return ""

    # In de aangeleverde FS-material files is de stringlengte 1 byte.
    length = block[pos]
    pos += 1
    if length <= 0 or pos + length > len(block):
        return ""
    return block[pos:pos + length].decode("utf-8", errors="ignore").strip()


def _lcm_read_named_ascii(data: bytes, name: bytes) -> str:
    """
    Named stringwaarden zoals WorkSpeed:
      <name> 00 00 <len> <ascii>
    """
    pos = data.find(name)
    if pos < 0:
        return ""
    pos += len(name)
    if pos + 3 > len(data):
        return ""

    # Zoek na de key naar 00 00 LEN.
    for i in range(pos, min(len(data) - 3, pos + 12)):
        if data[i] == 0 and data[i + 1] == 0:
            length = data[i + 2]
            start = i + 3
            if 0 < length <= 64 and start + length <= len(data):
                raw = data[start:start + length]
                if all((32 <= c < 127) for c in raw):
                    return raw.decode("ascii", errors="ignore").strip()
    return ""



def _lcm_read_named_number(data: bytes, name: bytes) -> float | None:
    """Lees de laatste echte numerieke named value voor een FSMATERIAL-key."""
    start = 0
    found = None
    while True:
        pos = data.find(name, start)
        if pos < 0:
            break
        cursor = pos + len(name)
        for i in range(cursor, min(len(data) - 3, cursor + 18)):
            if data[i] == 0 and data[i + 1] == 0:
                length = data[i + 2]
                value_start = i + 3
                if 0 < length <= 48 and value_start + length <= len(data):
                    raw = data[value_start:value_start + length]
                    try:
                        text = raw.decode("ascii").strip()
                    except Exception:
                        text = ""
                    if re.fullmatch(r"[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", text or ""):
                        try:
                            found = float(text.replace(",", "."))
                        except Exception:
                            pass
                    break
        start = pos + 1
    return found


def _lcm_advanced_parameters(data: bytes) -> dict:
    """
    Extra echte FSMATERIAL-velden die in de aangeleverde buis-layers voorkomen.
    Alleen numerieke waarden die werkelijk in het bestand staan worden geretourneerd.
    """
    fields = {
        "ptFollowHPlus": (b"PTFollowHPlus", "Buishoek volg-hoogte +", "mm"),
        "ptConsV": (b"PTConsV", "Buishoek snelheid", ""),
        "ptConsA": (b"PTConsA", "Buishoek acceleratie", ""),
        "ptFreq": (b"PTFreq", "Buishoek frequentie", "Hz"),
        "ptPressure": (b"PTPressure", "Buishoek gasdruk", "bar"),
        "ptPwmRatio": (b"PTPwmRatio", "Buishoek PWM-ratio", ""),
        "ptCurrent": (b"PTCurrent", "Buishoek laserstroom", "%"),
        "ptCornerStandard": (b"PTCornerStandard", "Buishoek standaard", ""),
        "gpPressure": (b"GPPressure", "Groef gasdruk", "bar"),
        "gpCurrent": (b"GPCurrent", "Groef laserstroom", "%"),
        "gpPwmRatio": (b"GpPwmRatio", "Groef PWM-ratio", ""),
        "gpFreq": (b"GpFreq", "Groef frequentie", "Hz"),
        "gpMinAngle": (b"GpMinAngle", "Groef minimale hoek", "rad"),
        "gpFocusPos": (b"GpFocusPos", "Groef focuspositie", "mm"),
        "gpConsV": (b"GpConsV", "Groef snelheid", ""),
        "extGap": (b"ExtGap", "Smart-end buitengap", "mm"),
        "inGap": (b"InGap", "Smart-end binnengap", "mm"),
        "inSpdRate": (b"InSpdRate", "Smart-end binnen snelheidratio", ""),
        "inPresRate": (b"InPresRate", "Smart-end binnen drukratio", ""),
        "extSpdRate": (b"ExtSpdRate", "Smart-end buiten snelheidratio", ""),
        "extPresRate": (b"ExtPresRate", "Smart-end buiten drukratio", ""),
        "inHeight": (b"InHeight", "Smart-end binnenhoogte", "mm"),
        "extHeight": (b"ExtHeight", "Smart-end buitenhoogte", "mm"),
    }
    result = {}
    for key, (raw_name, label, unit) in fields.items():
        value = _lcm_read_named_number(data, raw_name)
        if value is not None and math.isfinite(value):
            result[key] = {"value": value, "label": label, "unit": unit}
    return result


def _lcm_read_note(data: bytes) -> str:
    """
    Note is opgeslagen als:
      Note 00 02 00 00 <len> <utf8>
    """
    pos = data.find(b"Note")
    if pos < 0:
        return ""
    tail = data[pos + 4:pos + 256]
    # Zoek eerste bruikbare lengte + leesbare tekst.
    for i in range(0, max(0, len(tail) - 2)):
        length = tail[i]
        if not (5 <= length <= 180):
            continue
        raw = tail[i + 1:i + 1 + length]
        if len(raw) != length:
            continue
        try:
            text = raw.decode("utf-8").strip("\x00 ").strip()
        except Exception:
            continue
        if len(text) >= 5 and sum(ch.isprintable() for ch in text) / max(1, len(text)) > 0.9:
            if any(c.isalpha() for c in text):
                return text
    return ""


def _lcm_read_immediate_byte(data: bytes, name: bytes) -> int | None:
    pos=data.find(name)
    if pos<0: return None
    cursor=pos+len(name)
    # FSMATERIAL named scalars used by PipeCorner are stored directly before 00 01 00 00.
    marker=data.find(b"\x00\x01\x00\x00",cursor,min(len(data),cursor+12))
    if marker<0 or marker<=cursor: return None
    raw=data[marker-1]
    return int(raw)

def _lcm_freq_from_code(code: int | None) -> float | None:
    # Confirmed in the supplied FS layers: code 9 corresponds to 5000 Hz.
    # Unknown codes are intentionally not guessed.
    return 5000.0 if code==9 else None

# v645: oude _lcm_corner_parameters verwijderd; één leidende implementatie staat bij de machine-LCM builder.

# v645: oude _lcm_cut_parameters_from_block verwijderd; één leidende implementatie staat bij de machine-LCM builder.

def _parse_fs_material_lcm(content: bytes, filename: str = "") -> dict:
    if not content.startswith(b"FSMATERIAL"):
        raise ValueError("Dit bestand is geen herkende FSMATERIAL/LCM-layer.")

    # De zlib-stream staat direct na de FSMATERIAL-header/versiebytes.
    zpos = -1
    for i in range(10, min(len(content) - 2, 64)):
        if content[i] == 0x78 and content[i + 1] in (0x01, 0x5E, 0x9C, 0xDA):
            try:
                data = zlib.decompress(content[i:])
                zpos = i
                break
            except Exception:
                continue
    if zpos < 0:
        raise ValueError("De gecomprimeerde layerdata kon niet worden gelezen.")

    note = _lcm_read_note(data)
    source_text = f"{note} {filename}".strip()

    # Cut-blok: index mapping uit de FSMATERIAL velddefinitie:
    # 0x1A Focus, 0x1B GasPressure.
    cut_start = data.find(b"\x00\x00\x00\x03Cut")
    if cut_start < 0:
        cut_start = data.find(b"Cut")
    pierce_start = data.find(b"Pierce1", cut_start + 1)
    cut_block = data[cut_start:pierce_start if pierce_start > cut_start else cut_start + 512]

    speed_raw = _lcm_read_named_ascii(data, b"WorkSpeed")

    def num(text):
        try:
            return float(str(text).replace(",", "."))
        except Exception:
            return None

    # FS-material versies gebruiken niet altijd dezelfde veldindex voor
    # Focus/GasPressure. Lees daarom de ASCII-getallen uit het Cut-blok en
    # herken de betekenis op basis van een realistische waarderange.
    cut_ascii_values = []
    for match in re.finditer(rb"\x00\x01.\x00\x00([\x01-\x20])([\-0-9.,]+)", cut_block):
        try:
            length = match.group(1)[0]
            raw = match.group(2)[:length].decode("ascii", errors="ignore")
            value = num(raw)
            if value is not None:
                cut_ascii_values.append((raw, value))
        except Exception:
            pass

    focus = next((v for raw, v in cut_ascii_values if -20 <= v < 0), None)
    pressure = next((v for raw, v in cut_ascii_values if 0.1 <= v <= 30), None)
    work_speed_mm_s = num(speed_raw)
    speed_m_min = work_speed_mm_s * 0.06 if work_speed_mm_s is not None else None

    # Nozzle uit naam/note, bv "1.5E Nozzle".
    nozzle = ""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*([A-Za-z])\s*Nozzle", source_text, re.I)
    if m:
        nozzle = f"{m.group(1).replace(',', '.')}{m.group(2).upper()}"

    # Dikte: liefst expliciete "...MM Staal", anders begin van bestandsnaam.
    thickness = None
    for pattern in (
        r"(\d+(?:[.,]\d+)?)\s*MM\b",
        r"^(\d+(?:[.,]\d+)?)\s*MM\b",
    ):
        m = re.search(pattern, source_text, re.I)
        if m:
            thickness = num(m.group(1))
            if thickness:
                break

    # Profielmaat uit note, bv 40x40.
    profile = ""
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\b", source_text, re.I)
    if m:
        profile = f"{m.group(1).replace(',', '.')}x{m.group(2).replace(',', '.')}"

    # v638: radius uit de layernaam/note, bijvoorbeeld R0,75 of r1,5.
    radius_mm = None
    radius_matches = list(re.finditer(
        r"(?:^|[\s_\-])R\s*([0-9]+(?:[.,][0-9]+)?)(?=$|[\s_\-.])",
        source_text,
        re.I,
    ))
    if radius_matches:
        radius_mm = num(radius_matches[-1].group(1))

    # Materiaal uit tekst; huidige calculator focust op staal.
    material = "Staal"
    if re.search(r"rvs|inox|stainless", source_text, re.I):
        material = "RVS"
    elif re.search(r"aluminium|aluminum|\balu\b", source_text, re.I):
        material = "Aluminium"

    # De offerte-gasregel blijft leidend voor materiaalkeuze:
    # <=3 mm N2, >3 mm O2. GasType-enum in LCM is machinespecifiek.
    gas = "oxygen" if (thickness is not None and thickness > 3.0) else "nitrogen"

    advanced_parameters = _lcm_advanced_parameters(data)
    cut_parameters = _lcm_cut_parameters_from_block(cut_block)
    corner_parameters = _lcm_corner_parameters(data)

    return {
        "ok": True,
        "filename": filename,
        "note": note,
        "material": material,
        "thicknessMm": thickness,
        "profile": profile,
        "radiusMm": radius_mm,
        "gas": gas,
        "nozzle": nozzle,
        "gasPressureBar": pressure,
        "cutSpeedMMin": speed_m_min,
        "workSpeedMmS": work_speed_mm_s,
        "focusMm": focus,
        "cutParameters": cut_parameters,
        "cornerParameters": corner_parameters,
        "advancedParameters": advanced_parameters,
        "source": "LCM import",
        "contentBase64": base64.b64encode(content).decode("ascii"),
        "originalFilename": filename,
    }



def _lcm_unpack_payload(content: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Splits een FSMATERIAL bestand in:
      - ongewijzigde header/prefix
      - gedecomprimeerde FSMATERIAL payload
      - eventuele trailing bytes

    De gegenereerde layer gebruikt dus exact dezelfde containeropbouw als
    de echte referentielayer.
    """
    if not content.startswith(b"FSMATERIAL"):
        raise ValueError("Referentielayer is geen FSMATERIAL/LCM-bestand.")

    for i in range(10, min(len(content)-2, 64)):
        if content[i] != 0x78 or content[i+1] not in (0x01,0x5E,0x9C,0xDA):
            continue
        try:
            obj=zlib.decompressobj()
            payload=obj.decompress(content[i:]) + obj.flush()
            return content[:i], payload, obj.unused_data or b""
        except Exception:
            continue
    raise ValueError("De gecomprimeerde LCM-payload kon niet worden geopend.")


def _lcm_pack_payload(prefix: bytes, payload: bytes, trailing: bytes=b"") -> bytes:
    return prefix + zlib.compress(payload, level=9) + bytes(trailing or b"")


def _lcm_number_text(value: float, decimals: int=6) -> bytes:
    n=float(value)
    if not math.isfinite(n):
        raise ValueError("LCM-waarde is niet numeriek.")
    text=f"{n:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("-0",""):
        text="0"
    return text.encode("ascii")


def _lcm_replace_named_ascii(data: bytes, name: bytes, raw_value: bytes) -> tuple[bytes,bool]:
    """
    Named string:
      NAME 00 00 LEN ASCII
    Vervangt alleen de waarde; alle overige payloadbytes blijven intact.
    """
    start=0
    found=None
    while True:
        pos=data.find(name,start)
        if pos<0:
            break
        cursor=pos+len(name)
        for i in range(cursor,min(len(data)-3,cursor+18)):
            if data[i]==0 and data[i+1]==0:
                ln=data[i+2]
                v0=i+3
                if 0 < ln <= 64 and v0+ln <= len(data):
                    old=data[v0:v0+ln]
                    if all(32<=c<127 for c in old):
                        found=(i,v0,ln)
                        break
        start=pos+1
    if not found:
        return data,False
    i,v0,ln=found
    if len(raw_value)>255:
        raise ValueError(f"Nieuwe waarde voor {name!r} is te lang.")
    return data[:i+2]+bytes([len(raw_value)])+raw_value+data[v0+ln:],True


def _lcm_replace_named_number(data: bytes, name: bytes, value: float) -> tuple[bytes,bool]:
    return _lcm_replace_named_ascii(data,name,_lcm_number_text(value))


def _lcm_replace_named_scalar_code(data: bytes, name: bytes, code: int) -> tuple[bytes,bool]:
    """
    FSMATERIAL scalar:
      NAME <code> 00 01 00 00
    Gebruikt o.a. 0x11 voor ingeschakeld en 0x05 voor uitgeschakeld.
    """
    pos=data.find(name)
    if pos<0:
        return data,False
    cursor=pos+len(name)
    marker=data.find(b"\x00\x01\x00\x00",cursor,min(len(data),cursor+12))
    if marker<0 or marker<=cursor:
        return data,False
    return data[:marker-1]+bytes([int(code)&0xFF])+data[marker:],True


def _lcm_replace_cut_ascii_field(data: bytes, field_index: int, value: float) -> tuple[bytes,bool]:
    cut_start=data.find(b"\x00\x00\x00\x03Cut")
    if cut_start<0:
        cut_start=data.find(b"Cut")
    if cut_start<0:
        return data,False
    end=data.find(b"Pierce1",cut_start+1)
    if end<0:
        end=min(len(data),cut_start+700)

    marker=bytes([0x00,0x01,field_index,0x00,0x00])
    pos=data.find(marker,cut_start,end)
    if pos<0:
        return data,False
    len_pos=pos+len(marker)
    if len_pos>=len(data):
        return data,False
    ln=data[len_pos]
    v0=len_pos+1
    if not (0 < ln <= 32 and v0+ln<=len(data)):
        return data,False
    raw=_lcm_number_text(value)
    return data[:len_pos]+bytes([len(raw)])+raw+data[v0+ln:],True


def _lcm_read_cut_scalar_code(data: bytes, field_index: int) -> int | None:
    cut_start=data.find(b"\x00\x00\x00\x03Cut")
    if cut_start<0:
        cut_start=data.find(b"Cut")
    if cut_start<0:
        return None
    end=data.find(b"Pierce1",cut_start+1)
    if end<0:
        end=min(len(data),cut_start+700)
    marker=bytes([0x00,0x01,field_index])
    pos=data.find(marker,cut_start,end)
    if pos<0 or pos+3>=len(data):
        return None
    # Directe bytevelden hebben geen 00 00 + stringlengte.
    if data[pos+3]==0 and pos+4<end and data[pos+4]==0:
        return None
    return int(data[pos+3])


def _lcm_replace_cut_scalar_code(data: bytes, field_index: int, code: int) -> tuple[bytes,bool]:
    cut_start=data.find(b"\x00\x00\x00\x03Cut")
    if cut_start<0:
        cut_start=data.find(b"Cut")
    if cut_start<0:
        return data,False
    end=data.find(b"Pierce1",cut_start+1)
    if end<0:
        end=min(len(data),cut_start+700)
    marker=bytes([0x00,0x01,field_index])
    pos=data.find(marker,cut_start,end)
    if pos<0 or pos+3>=len(data):
        return data,False
    if data[pos+3]==0 and pos+4<end and data[pos+4]==0:
        return data,False
    return data[:pos+3]+bytes([int(code)&0xFF])+data[pos+4:],True


def _lcm_replace_note(data: bytes, note: str) -> tuple[bytes,bool]:
    current=_lcm_read_note(data)
    if not current:
        return data,False
    old=current.encode("utf-8")
    new=str(note or "").encode("utf-8")
    if not new or len(new)>180:
        return data,False

    note_pos=data.find(b"Note")
    if note_pos<0:
        return data,False
    old_pos=data.find(old,note_pos,min(len(data),note_pos+320))
    if old_pos<0 or old_pos<=0:
        return data,False
    length_pos=old_pos-1
    if data[length_pos] != len(old):
        return data,False
    return data[:length_pos]+bytes([len(new)])+new+data[old_pos+len(old):],True


def _lcm_enabled_code(enabled: bool) -> int:
    # Bewezen uit de aangeleverde layers:
    # 0x11 = aangevinkt, 0x05 = niet aangevinkt.
    return 0x11 if bool(enabled) else 0x05


def _lcm_frequency_code(hz: float | None) -> int | None:
    if hz is None:
        return None
    value=float(hz)
    # In de echte aangeleverde FSMATERIAL-layers is code 9 = 5000 Hz.
    # Andere mappings worden niet gegokt.
    return 9 if abs(value-5000.0)<0.5 else None


def _lcm_cut_parameters_from_block(cut_block: bytes) -> dict:
    """
    Bewezen Cut-mapping uit de aangeleverde echte FSMATERIAL-bestanden:
      0x13 = PWM-frequency code (9 -> 5000 Hz)
      0x14 = duty-cycle percentage als directe byte
      0x18 = peak/laser current percentage als ASCII-getal
    Lift Height en Cut Height blijven onbekend totdat hun machinecodering
    met echte referentiebestanden bewezen is.
    """
    ascii_values=[]
    for match in re.finditer(rb"\x00\x01.\x00\x00([\x01-\x20])([\-0-9.,]+)",cut_block):
        try:
            ln=match.group(1)[0]
            txt=match.group(2)[:ln].decode("ascii",errors="ignore")
            ascii_values.append(float(txt.replace(",",".")))
        except Exception:
            pass

    peak=None
    marker=b"\x00\x01\x18\x00\x00"
    pos=cut_block.find(marker)
    if pos>=0 and pos+len(marker)<len(cut_block):
        ln=cut_block[pos+len(marker)]
        raw=cut_block[pos+len(marker)+1:pos+len(marker)+1+ln]
        try:
            peak=float(raw.decode("ascii").replace(",","."))
        except Exception:
            peak=None

    duty=None
    freq=None
    for field_index,key in ((0x14,"duty"),(0x13,"freq")):
        marker=bytes([0x00,0x01,field_index])
        pos=cut_block.find(marker)
        if pos>=0 and pos+3<len(cut_block):
            code=cut_block[pos+3]
            if key=="duty":
                duty=float(code) if 0<=code<=100 else None
            else:
                freq=_lcm_freq_from_code(code)

    return {
        "liftHeightMm": None,
        "cutHeightMm": None,
        "peakPowerPct": peak,
        "dutyCyclePct": duty,
        "frequencyHz": freq,
    }


def _lcm_enabled_from_code(code: int | None) -> bool:
    return int(code or 0)==0x11


def _lcm_corner_parameters(data: bytes) -> dict:
    ratio=_lcm_read_named_number(data,b"PTPwmRatio")
    standard=_lcm_read_named_number(data,b"PTCornerStandard")
    return {
        "enabled": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"UsePTAdjust")),
        "followHeightOffsetMm": _lcm_read_named_number(data,b"PTFollowHPlus"),
        "cornerPressureEnabled": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"UsePtPressure")),
        "cornerPressureBar": _lcm_read_immediate_byte(data,b"PTPressure"),
        "peakPowerEnabled": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"UsePtCurrent")),
        # PTCurrent4 is niet veilig genoeg bewezen om numeriek te herschrijven.
        "peakPowerPct": None,
        "dutyCycleEnabled": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"UsePtPwmRatio")),
        "dutyCyclePct": (ratio*100.0 if ratio is not None and abs(ratio)<=1.5 else ratio),
        "frequencyEnabled": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"UsePtFreq")),
        "frequencyHz": _lcm_freq_from_code(_lcm_read_immediate_byte(data,b"PTFreq")),
        "defineCornerDegPerMm": (standard*180.0/math.pi if standard is not None else None),
        "limitBAxisSpeed": _lcm_enabled_from_code(_lcm_read_immediate_byte(data,b"PTConsEn")),
        "bAxisSpeedRpm": _lcm_read_named_number(data,b"PTConsV"),
        "bAxisAccelerationRadS2": _lcm_read_named_number(data,b"PTConsA"),
    }


def _build_machine_lcm(reference_content: bytes, desired: dict, filename: str) -> tuple[bytes,dict,list[str]]:
    """
    Maakt een echte machine-LCM door één echte referentielayer te klonen.

    Alleen bewezen velden worden aangepast. Gas pressure wordt nooit gewijzigd.
    Onbekende binaire instellingen blijven exact uit de referentielayer komen.
    """
    reference_parsed=_parse_fs_material_lcm(reference_content,filename="reference.lcm")
    target_radius=float(desired.get("radiusMm") or 0)
    ref_radius=float(reference_parsed.get("radiusMm") or 0)

    if not (target_radius>0 and ref_radius>0 and abs(target_radius-ref_radius)<0.011):
        raise ValueError(
            f"Radiusblokkering: doel R{target_radius:g} en referentie R{ref_radius:g} zijn niet exact gelijk."
        )

    prefix,payload,trailing=_lcm_unpack_payload(reference_content)
    changed=[]

    def require(ok: bool, field: str):
        if not ok:
            raise ValueError(f"LCM-veld '{field}' kon niet veilig worden aangepast.")
        changed.append(field)

    # CUT — bewezen schrijfbare velden.
    speed=desired.get("cutSpeedMMin")
    if speed is not None:
        payload,ok=_lcm_replace_named_ascii(
            payload,b"WorkSpeed",_lcm_number_text(float(speed)/0.06)
        )
        require(ok,"Cut Speed")

    focus=desired.get("focusMm")
    if focus is not None:
        payload,ok=_lcm_replace_cut_ascii_field(payload,0x1A,float(focus))
        require(ok,"Focus Pos")

    cut=dict(desired.get("cutParameters") or {})

    # Lift Height / Cut Height zijn nog niet veilig gemapt.
    if cut.get("liftHeightMm") is not None:
        raise ValueError(
            "Lift Height is in dit LCM-formaat nog niet veilig genoeg gemapt om automatisch te herschrijven."
        )
    if cut.get("cutHeightMm") is not None:
        raise ValueError(
            "Cut Height is in dit LCM-formaat nog niet veilig genoeg gemapt om automatisch te herschrijven."
        )

    if cut.get("peakPowerPct") is not None:
        payload,ok=_lcm_replace_cut_ascii_field(payload,0x18,float(cut["peakPowerPct"]))
        require(ok,"Cut Peak Power")

    if cut.get("dutyCyclePct") is not None:
        duty=int(round(float(cut["dutyCyclePct"])))
        if not 0<=duty<=100:
            raise ValueError("Cut Duty Cycle moet tussen 0 en 100 liggen.")
        payload,ok=_lcm_replace_cut_scalar_code(payload,0x14,duty)
        require(ok,"Cut Duty Cycle")

    if cut.get("frequencyHz") is not None:
        code=_lcm_frequency_code(cut["frequencyHz"])
        if code is None:
            raise ValueError(
                f"Cut Frequency {cut['frequencyHz']} Hz kan nog niet veilig naar een FSMATERIAL-code worden vertaald."
            )
        payload,ok=_lcm_replace_cut_scalar_code(payload,0x13,code)
        require(ok,"Cut Frequency")

    # CORNER — alleen bewezen named/scalar velden.
    corner=dict(desired.get("cornerParameters") or {})

    if "enabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePTAdjust",_lcm_enabled_code(corner["enabled"])
        )
        require(ok,"Corner enabled")

    if corner.get("followHeightOffsetMm") is not None:
        payload,ok=_lcm_replace_named_number(
            payload,b"PTFollowHPlus",float(corner["followHeightOffsetMm"])
        )
        require(ok,"Corner Follow height offset")

    if "cornerPressureEnabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePtPressure",_lcm_enabled_code(corner["cornerPressureEnabled"])
        )
        require(ok,"Corner pressure toggle")
    # Drukwaarde zelf bewust NIET wijzigen.

    if "peakPowerEnabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePtCurrent",_lcm_enabled_code(corner["peakPowerEnabled"])
        )
        require(ok,"Corner Peak power toggle")

    # Numerieke Corner Peak Power is nog niet bewezen geschreven.
    # De waarde blijft daarom exact uit de echte referentielayer.

    if "dutyCycleEnabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePtPwmRatio",_lcm_enabled_code(corner["dutyCycleEnabled"])
        )
        require(ok,"Corner Duty cycle toggle")

    if corner.get("dutyCyclePct") is not None:
        ratio=float(corner["dutyCyclePct"])/100.0
        payload,ok=_lcm_replace_named_number(payload,b"PTPwmRatio",ratio)
        require(ok,"Corner Duty cycle")

    if "frequencyEnabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePtFreq",_lcm_enabled_code(corner["frequencyEnabled"])
        )
        require(ok,"Corner Frequency toggle")

    if corner.get("frequencyHz") is not None:
        code=_lcm_frequency_code(corner["frequencyHz"])
        if code is None:
            raise ValueError(
                f"Corner Frequency {corner['frequencyHz']} Hz kan nog niet veilig worden gecodeerd."
            )
        payload,ok=_lcm_replace_named_scalar_code(payload,b"PTFreq",code)
        require(ok,"Corner Frequency")

    if corner.get("defineCornerDegPerMm") is not None:
        rad=float(corner["defineCornerDegPerMm"])*math.pi/180.0
        payload,ok=_lcm_replace_named_number(payload,b"PTCornerStandard",rad)
        require(ok,"Define corner")

    if "limitBAxisSpeed" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"PTConsEn",_lcm_enabled_code(corner["limitBAxisSpeed"])
        )
        require(ok,"Limit B-axis speed")

    if corner.get("bAxisSpeedRpm") is not None:
        payload,ok=_lcm_replace_named_number(payload,b"PTConsV",float(corner["bAxisSpeedRpm"]))
        require(ok,"B-axis speed")

    if corner.get("bAxisAccelerationRadS2") is not None:
        payload,ok=_lcm_replace_named_number(payload,b"PTConsA",float(corner["bAxisAccelerationRadS2"]))
        require(ok,"B-axis acceleration")

    # Interne note alleen aanpassen wanneer het bestaande Note-record veilig herkenbaar is.
    note=str(desired.get("note") or filename.rsplit(".",1)[0])
    payload,note_ok=_lcm_replace_note(payload,note)
    if note_ok:
        changed.append("Layer Note")

    result=_lcm_pack_payload(prefix,payload,trailing)

    # Harde round-trip validatie: nieuw bestand moet opnieuw volledig parseerbaar zijn.
    parsed=_parse_fs_material_lcm(result,filename=filename)

    # Controleer alle velden die onze parser bewezen terug kan lezen.
    def close_num(actual,expected,tol=0.02):
        if expected is None:
            return True
        try:
            return actual is not None and abs(float(actual)-float(expected))<=tol
        except Exception:
            return False

    if speed is not None and not close_num(parsed.get("cutSpeedMMin"),speed,0.02):
        raise ValueError("Validatie mislukt voor Cut Speed.")
    if focus is not None and not close_num(parsed.get("focusMm"),focus,0.02):
        raise ValueError("Validatie mislukt voor Focus Pos.")
    if cut.get("peakPowerPct") is not None and not close_num(
        (parsed.get("cutParameters") or {}).get("peakPowerPct"),cut["peakPowerPct"],0.02
    ):
        raise ValueError("Validatie mislukt voor Cut Peak Power.")
    if cut.get("dutyCyclePct") is not None and not close_num(
        (parsed.get("cutParameters") or {}).get("dutyCyclePct"),cut["dutyCyclePct"],0.5
    ):
        raise ValueError("Validatie mislukt voor Cut Duty Cycle.")
    if cut.get("frequencyHz") is not None and not close_num(
        (parsed.get("cutParameters") or {}).get("frequencyHz"),cut["frequencyHz"],1.0
    ):
        raise ValueError("Validatie mislukt voor Cut Frequency.")

    pc=parsed.get("cornerParameters") or {}
    if corner.get("followHeightOffsetMm") is not None and not close_num(
        pc.get("followHeightOffsetMm"),corner["followHeightOffsetMm"],0.02
    ):
        raise ValueError("Validatie mislukt voor Corner Follow height offset.")
    if corner.get("dutyCyclePct") is not None and not close_num(
        pc.get("dutyCyclePct"),corner["dutyCyclePct"],0.2
    ):
        raise ValueError("Validatie mislukt voor Corner Duty Cycle.")
    if corner.get("frequencyHz") is not None and not close_num(
        pc.get("frequencyHz"),corner["frequencyHz"],1.0
    ):
        raise ValueError("Validatie mislukt voor Corner Frequency.")
    if corner.get("bAxisSpeedRpm") is not None and not close_num(
        pc.get("bAxisSpeedRpm"),corner["bAxisSpeedRpm"],0.02
    ):
        raise ValueError("Validatie mislukt voor B-axis speed.")
    if corner.get("bAxisAccelerationRadS2") is not None and not close_num(
        pc.get("bAxisAccelerationRadS2"),corner["bAxisAccelerationRadS2"],0.02
    ):
        raise ValueError("Validatie mislukt voor B-axis acceleration.")

    return result,parsed,changed


@app.post("/api/cut-layer/build-machine")
async def build_machine_cut_layer(request: Request):
    body=await request.json()
    reference_b64=str(body.get("referenceContentBase64") or "")
    filename=_safe_dropbox_name(
        str(body.get("filename") or "Slimme_snijlayer.LCM"),
        "Slimme_snijlayer.LCM"
    )
    if not filename.lower().endswith(".lcm"):
        filename += ".LCM"

    try:
        reference=base64.b64decode(reference_b64,validate=True)
    except Exception:
        raise HTTPException(status_code=400,detail="De echte referentielayer bevat geen geldige LCM-data.")

    desired=dict(body.get("desired") or {})
    try:
        content,parsed,changed=_build_machine_lcm(reference,desired,filename)
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc))

    return {
        "ok":True,
        "machineReady":True,
        "filename":filename,
        "contentBase64":base64.b64encode(content).decode("ascii"),
        "parsed":parsed,
        "changedFields":changed,
        "validation":"round-trip-ok",
        "sizeBytes":len(content),
    }


@app.post("/api/cut-layer/parse")
async def parse_cut_layer(file: UploadFile = File(...)):
    filename = str(file.filename or "layer.lcm")
    if not filename.lower().endswith(".lcm"):
        raise HTTPException(status_code=400, detail="Kies een .lcm snijlayerbestand.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Het layerbestand is leeg.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Het layerbestand is te groot.")

    try:
        return _parse_fs_material_lcm(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer kon niet worden gelezen: {exc}") from exc


@app.post("/api/generate-production-step")
async def generate_production_step(request: Request):
    """
    Maak één productie-STEP voor een bibliotheekprofiel.

    Ondersteund:
    - vierkante/rechthoekige holle kokers, incl. radius
    - ronde holle buizen

    Iedere opgegeven lengte wordt als een afzonderlijke solid in hetzelfde
    STEP-bestand geplaatst. Hierdoor kan een offerte die uitsluitend uit
    bibliotheekmateriaal bestaat toch complete productie-STEP-bestanden krijgen.
    """
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Ongeldige productie-STEP gegevens.") from exc

    size = str(data.get("size") or "").strip()
    profile_kind = str(data.get("profileKind") or "").strip().lower()
    profile_type = str(data.get("profileType") or "").strip().lower()

    outer_w = float(data.get("outerWidth") or 0)
    outer_h = float(data.get("outerHeight") or 0)
    diameter = float(data.get("diameter") or 0)
    wall = float(data.get("wallThickness") or 0)
    radius = max(0.0, float(data.get("radius") or 0))
    pieces = data.get("pieces") or []

    # Compatibiliteit met oudere frontendversies die profileKind niet meesturen.
    is_round = (
        profile_kind == "round-tube"
        or profile_type in {"rond", "buis", "ronde buis"}
        or "ronde buis" in profile_type
    )
    if is_round:
        profile_kind = "round-tube"
        if diameter <= 0:
            diameter = max(outer_w, outer_h)
        outer_w = diameter
        outer_h = diameter
    else:
        profile_kind = "rectangular-tube"

    lengths = []
    for piece in pieces:
        try:
            length = float((piece or {}).get("lengthMm") or 0)
        except Exception:
            length = 0
        if length > 0:
            lengths.append(length)

    if not lengths or not (wall > 0):
        raise HTTPException(
            status_code=400,
            detail=f"Onvoldoende profielgegevens voor productie-STEP: {size or 'onbekend profiel'}."
        )

    if is_round:
        if not (diameter > 0):
            raise HTTPException(
                status_code=400,
                detail=f"Buitendiameter ontbreekt voor ronde buis: {size or 'onbekend profiel'}."
            )
        if diameter <= 2 * wall:
            raise HTTPException(
                status_code=400,
                detail="Wanddikte is ongeldig voor deze ronde buis."
            )
    else:
        if not (outer_w > 0 and outer_h > 0):
            raise HTTPException(
                status_code=400,
                detail=f"Buitenmaat ontbreekt voor productie-STEP: {size or 'onbekend profiel'}."
            )
        if outer_w <= 2 * wall or outer_h <= 2 * wall:
            raise HTTPException(
                status_code=400,
                detail="Wanddikte is ongeldig voor deze kokermaat."
            )

    try:
        def rounded_rect_solid(w: float, h: float, r: float, length: float):
            """
            Maak een geëxtrudeerde afgeronde rechthoek.
            CadQuery 2.8 ondersteunt fillet2D niet op Workplane; gebruik Sketch.fillet.
            """
            rr = max(0.0, min(float(r or 0), w / 2 - 1e-6, h / 2 - 1e-6))
            if rr > 1e-6:
                sketch = cq.Sketch().rect(w, h).vertices().fillet(rr)
                return cq.Workplane("XY").placeSketch(sketch).extrude(length)
            return cq.Workplane("XY").rect(w, h).extrude(length)

        def rectangular_tube_solid(length: float):
            inner_w = outer_w - 2 * wall
            inner_h = outer_h - 2 * wall
            inner_r = max(0.0, radius - wall)

            outer = rounded_rect_solid(outer_w, outer_h, radius, length)
            # Iets langer uitsnijden voorkomt coplanaire eindvlakken / OCC-artefacts.
            inner = rounded_rect_solid(inner_w, inner_h, inner_r, length + 2.0)
            inner = inner.translate((0, 0, -1.0))
            return outer.cut(inner)

        def round_tube_solid(length: float):
            outer_r = diameter / 2.0
            inner_r = outer_r - wall
            if not (outer_r > 0 and inner_r > 0):
                raise ValueError("Ongeldige diameter/wanddikte voor ronde buis.")

            outer = cq.Workplane("XY").circle(outer_r).extrude(length)
            # Ook hier 1 mm aan iedere zijde doorsteken voor robuuste booleans.
            inner = cq.Workplane("XY").circle(inner_r).extrude(length + 2.0)
            inner = inner.translate((0, 0, -1.0))
            return outer.cut(inner)

        exported_shapes = []
        x_offset = 0.0
        transverse_size = diameter if is_round else max(outer_w, outer_h)
        spacing = max(transverse_size, 1.0) + 25.0

        for length in lengths:
            tube = (
                round_tube_solid(length)
                if is_round
                else rectangular_tube_solid(length)
            )

            if x_offset:
                tube = tube.translate((x_offset, 0, 0))

            # Workplane.cut kan intern een Compound teruggeven. Haal de echte
            # OCC-solids eruit en bouw daarna één valide compound voor export.
            value = tube.val()
            solids_here = list(value.Solids()) if hasattr(value, "Solids") else []
            if solids_here:
                exported_shapes.extend(solids_here)
            elif isinstance(value, cq.Shape):
                exported_shapes.append(value)

            x_offset += transverse_size + spacing

        if not exported_shapes:
            raise RuntimeError("Geen geldige solids voor productie-STEP gegenereerd.")

        compound = cq.Compound.makeCompound(exported_shapes)

        tmp = CACHE_DIR / f"production_{uuid.uuid4().hex}.step"
        cq.exporters.export(compound, str(tmp), exportType="STEP")
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


@app.get("/approve/{token}", response_class=HTMLResponse)
def approval_page(token: str):
    with _db_connect() as conn:
        data=_approval_by_token(conn,token)
        if data.get("status")!="accepted" and not data.get("viewed_at"):
            now=_utcnow(); cur=conn.cursor()
            cur.execute(_sql(
                "UPDATE quote_approvals SET viewed_at=%s,updated_at=%s WHERE token=%s",
                "UPDATE quote_approvals SET viewed_at=?,updated_at=? WHERE token=?"
            ),(now,now,token))
            conn.commit(); data["viewed_at"]=now
        return HTMLResponse(_approval_html(data))

@app.post("/approve/{token}/accept", response_class=HTMLResponse)
def approval_accept(token: str, accepted_by: str=Form(...), note: str=Form("")):
    accepted_by=str(accepted_by or "").strip()
    if not accepted_by:
        raise HTTPException(status_code=400,detail="Naam ontbreekt.")
    with _db_connect() as conn:
        data=_approval_by_token(conn,token)
        if data.get("status")!="accepted":
            now=_utcnow(); cur=conn.cursor()
            cur.execute(_sql(
                """UPDATE quote_approvals SET status=%s,accepted_at=%s,accepted_by=%s,note=%s,updated_at=%s WHERE token=%s""",
                """UPDATE quote_approvals SET status=?,accepted_at=?,accepted_by=?,note=?,updated_at=? WHERE token=?"""
            ),("accepted",now,accepted_by,str(note or "").strip(),now,token))
            conn.commit(); data=_approval_by_token(conn,token)
            try:
                if _send_approval_email(data):
                    cur.execute(_sql(
                        "UPDATE quote_approvals SET email_sent_at=%s WHERE token=%s",
                        "UPDATE quote_approvals SET email_sent_at=? WHERE token=?"
                    ),(now,token)); conn.commit(); data["email_sent_at"]=now
            except Exception as exc:
                print(f"Approval email failed: {exc}")
        return HTMLResponse(_approval_html(data,accepted=True))

@app.get("/api/quotes/{quote_id}/approval")
def quote_approval_status(quote_id: str):
    with _db_connect() as conn:
        approval=_approval_for_quote(conn,quote_id,create=True); conn.commit()
        return {"ok":True,"approval":approval,"email_notifications_configured":_approval_email_configured()}

@app.get("/api/quote-approvals/recent")
def recent_quote_approvals(limit: int=20):
    limit=max(1,min(100,int(limit or 20)))
    with _db_connect() as conn:
        cur=conn.cursor()
        cur.execute(f"""SELECT a.quote_id,a.status,a.viewed_at,a.accepted_at,a.accepted_by,a.note,a.email_sent_at,
                              q.quote_number,q.customer_name,q.total_ex_vat
                       FROM quote_approvals a JOIN quotes q ON q.id=a.quote_id
                       WHERE a.status='accepted' ORDER BY a.accepted_at DESC LIMIT {limit}""")
        return {"ok":True,"approvals":[_row_to_dict(r,cur) for r in cur.fetchall()],
                "email_notifications_configured":_approval_email_configured()}

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
            row["approval"] = _approval_for_quote(conn, row["id"], create=False)

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



@app.get("/api/dropbox/storage/diagnose")
def dropbox_storage_diagnose():
    """Eén centrale diagnose voor alle actieve Vakstaal-opslagroutes."""
    try:
        context=_dropbox_account_context(force=True)
        quote_cfg=_quote_storage_config()
        order_cfg=_webshop_storage_config()
        layer_cfg=_cut_layer_storage_config()
        routes={
            "quotes":_dropbox_storage_route_state("/"+quote_cfg["root"].strip("/")),
            "orders":_dropbox_storage_route_state("/"+order_cfg["root"].strip("/")),
            "cutLayers":_dropbox_storage_route_state("/"+layer_cfg["root"].strip("/")),
        }
        ready=sum(1 for item in routes.values() if item.get("ok"))
        return {
            "ok":ready==len(routes),
            "connected":True,
            "readyRoutes":ready,
            "totalRoutes":len(routes),
            "routes":routes,
            "account":{
                "displayName":context.get("display_name",""),
                "email":context.get("email",""),
            },
        }
    except HTTPException as exc:
        return {
            "ok":False,"connected":False,"readyRoutes":0,"totalRoutes":3,
            "routes":{},"detail":str(exc.detail)
        }


@app.post("/api/dropbox/cut-layers/ensure-folder")
async def dropbox_cut_layer_ensure_folder(request: Request):
    """Maak/bevestig de snijlayer-map via dezelfde centrale Dropbox-verbinding."""
    try:
        body=await request.json()
    except Exception:
        body={}
    raw_path=str(body.get("path") or "").strip()
    raw_path=re.sub(r"^/?Dropbox(?:/|$)","/",raw_path,flags=re.I)
    path=_normalize_dropbox_browser_path(raw_path) or "/Snijlayers"
    _dropbox_create_folder_path(path)
    meta=_dropbox_get_metadata(path)
    if not meta or str(meta.get(".tag") or "")!="folder":
        raise HTTPException(status_code=502,detail="De Dropbox-map voor snijlayers kon niet worden aangemaakt of bevestigd.")
    return {
        "ok":True,
        "connected":True,
        "path":_normalize_dropbox_browser_path(meta.get("path_display") or meta.get("path_lower") or path),
        "pathDisplay":str(meta.get("path_display") or path),
        "folder":meta,
    }


@app.post("/api/dropbox/cut-layers/upload")
async def dropbox_cut_layer_upload(path: str=Form(...), file: UploadFile=File(...)):
    root=_normalize_dropbox_browser_path(path)
    if not root:
        raise HTTPException(status_code=400,detail="Kies eerst een Dropbox-map voor snijlayers.")
    filename=_safe_dropbox_name(file.filename or "snijlayer.lcm","snijlayer.lcm")
    if not filename.lower().endswith(".lcm"):
        raise HTTPException(status_code=400,detail="Alleen .LCM-snijlayerbestanden kunnen hier worden opgeslagen.")
    data=await file.read()
    if not data:
        raise HTTPException(status_code=400,detail="Het snijlayerbestand is leeg.")
    if len(data)>10*1024*1024:
        raise HTTPException(status_code=413,detail="Het snijlayerbestand is groter dan 10 MB.")
    _dropbox_create_folder_path(root)
    target=f"{root}/{filename}"
    meta=_dropbox_upload_bytes(target,data)
    verified=_dropbox_get_metadata(meta.get("path_display") or meta.get("path_lower") or target)
    if not verified or str(verified.get(".tag") or "")!="file":
        raise HTTPException(status_code=502,detail="Dropbox kon de opgeslagen snijlayer niet bevestigen.")
    return {"ok":True,"file":verified}


@app.post("/api/dropbox/cut-layers/delete")
async def dropbox_cut_layer_delete(request: Request):
    try: body=await request.json()
    except Exception: body={}
    path=_normalize_dropbox_browser_path(body.get("path") or "")
    if not path:
        raise HTTPException(status_code=400,detail="Geen Dropbox-snijlayerpad opgegeven.")
    if not path.lower().endswith(".lcm"):
        raise HTTPException(status_code=400,detail="Alleen .LCM-snijlayerbestanden kunnen via deze route worden verwijderd.")
    _dropbox_delete_path(path)
    return {"ok":True,"path":path}


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




# ===== v623 persistent e-Boekhouden settings =====
EBOEK_STATE_KEY="eboekhouden_settings"
EBOEK_CREDENTIAL_PROVIDER="eboekhouden_api"

def _eboek_default_settings():
    return {
        "apiBase":str(os.environ.get("EBOEKHOUDEN_API_BASE") or "https://api.e-boekhouden.nl").rstrip("/"),
        "revenueLedgerCode":str(os.environ.get("EBOEKHOUDEN_REVENUE_LEDGER_CODE") or "8055"),
        "debtorLedgerId":str(os.environ.get("EBOEKHOUDEN_DEBTOR_LEDGER_ID") or ""),
        "invoiceTemplateName":str(os.environ.get("EBOEKHOUDEN_INVOICE_TEMPLATE_NAME") or "Vakstaal"),
        "invoiceTemplateId":str(os.environ.get("EBOEKHOUDEN_INVOICE_TEMPLATE_ID") or ""),
        "paymentTermDays":int(os.environ.get("EBOEKHOUDEN_PAYMENT_TERM") or 30),
        "vatCode":str(os.environ.get("EBOEKHOUDEN_VAT_CODE") or "HOOG_VERK_21"),
        "invoiceDescription":"Werkzaamheden volgens offerte {offertenummer}",
        "createMissingRelation":True,
    }

def _eboek_load_settings():
    result=_eboek_default_settings()
    try:
        with _db_connect() as conn:
            cur=conn.cursor()
            cur.execute(_sql(
                "SELECT payload_json FROM app_state WHERE state_key=%s",
                "SELECT payload_json FROM app_state WHERE state_key=?"
            ),(EBOEK_STATE_KEY,))
            row=cur.fetchone()
            if row:
                raw=row[0] if not isinstance(row,sqlite3.Row) else row["payload_json"]
                parsed=json.loads(raw or "{}")
                if isinstance(parsed,dict): result.update(parsed)
    except Exception as exc:
        print(f"e-Boekhouden settings load warning: {exc}")
    return result

def _eboek_save_settings(settings):
    now=_utcnow()
    payload=json.dumps(settings,ensure_ascii=False)
    with _db_connect() as conn:
        cur=conn.cursor()
        if _postgres_enabled():
            cur.execute("""
                INSERT INTO app_state(state_key,payload_json,updated_at)
                VALUES (%s,%s,%s)
                ON CONFLICT(state_key) DO UPDATE SET payload_json=EXCLUDED.payload_json,updated_at=EXCLUDED.updated_at
            """,(EBOEK_STATE_KEY,payload,now))
        else:
            cur.execute("""
                INSERT INTO app_state(state_key,payload_json,updated_at)
                VALUES (?,?,?)
                ON CONFLICT(state_key) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at
            """,(EBOEK_STATE_KEY,payload,now))
        conn.commit()

def _eboek_stored_api_token():
    try:
        with _db_connect() as conn:
            cur=conn.cursor()
            cur.execute(_sql(
                "SELECT refresh_token FROM oauth_credentials WHERE provider=%s",
                "SELECT refresh_token FROM oauth_credentials WHERE provider=?"
            ),(EBOEK_CREDENTIAL_PROVIDER,))
            row=cur.fetchone()
            if row:
                return str(row[0] if not isinstance(row,sqlite3.Row) else row["refresh_token"]).strip()
    except Exception as exc:
        print(f"e-Boekhouden token load warning: {exc}")
    return ""

def _eboek_store_api_token(token):
    token=str(token or "").strip()
    if not token: return
    now=_utcnow()
    with _db_connect() as conn:
        cur=conn.cursor()
        if _postgres_enabled():
            cur.execute("""
                INSERT INTO oauth_credentials(provider,refresh_token,oauth_state,updated_at)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT(provider) DO UPDATE SET refresh_token=EXCLUDED.refresh_token,updated_at=EXCLUDED.updated_at
            """,(EBOEK_CREDENTIAL_PROVIDER,token,"",now))
        else:
            cur.execute("""
                INSERT INTO oauth_credentials(provider,refresh_token,oauth_state,updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(provider) DO UPDATE SET refresh_token=excluded.refresh_token,updated_at=excluded.updated_at
            """,(EBOEK_CREDENTIAL_PROVIDER,token,"",now))
        conn.commit()
    _EBOEK_SESSION["token"]=""
    _EBOEK_SESSION["expires"]=0.0

def _eboek_mask_token(token):
    token=str(token or "")
    if not token:return ""
    if len(token)<=8:return "••••••••"
    return token[:4]+"••••••••"+token[-4:]


# ===== v622 e-Boekhouden REST integration =====
EBOEKHOUDEN_API_BASE = "https://api.e-boekhouden.nl"
_EBOEK_SESSION = {"token": "", "expires": 0.0}

def _eboek_api_token() -> str:
    return _eboek_stored_api_token() or str(os.environ.get("EBOEKHOUDEN_API_TOKEN") or "").strip()

def _eboek_configured() -> bool:
    return bool(_eboek_api_token())

def _eboek_http(method: str, path: str, *, query=None, body=None, auth=True, retry=True):
    url = str(_eboek_load_settings().get("apiBase") or EBOEKHOUDEN_API_BASE).rstrip("/") + path
    if query:
        clean = {k:v for k,v in query.items() if v not in (None, "")}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)

    headers = {"Accept":"application/json"}
    if body is not None:
        headers["Content-Type"]="application/json"
        raw=json.dumps(body,ensure_ascii=False).encode("utf-8")
    else:
        raw=None

    if auth:
        token=_eboek_session_token()
        headers["Authorization"]="Bearer " + token

    req=urllib.request.Request(url,data=raw,headers=headers,method=method.upper())
    try:
        with urllib.request.urlopen(req,timeout=25) as response:
            payload=response.read()
            return json.loads(payload.decode("utf-8")) if payload else {}
    except urllib.error.HTTPError as exc:
        payload=exc.read().decode("utf-8","replace")
        if auth and retry and exc.code==401:
            _EBOEK_SESSION["token"]=""
            _EBOEK_SESSION["expires"]=0
            return _eboek_http(method,path,query=query,body=body,auth=auth,retry=False)
        try:
            detail=json.loads(payload)
        except Exception:
            detail=payload
        raise HTTPException(status_code=502,detail=f"e-Boekhouden API ({exc.code}): {detail}")
    except Exception as exc:
        if isinstance(exc,HTTPException): raise
        raise HTTPException(status_code=502,detail=f"e-Boekhouden is niet bereikbaar: {exc}")

def _eboek_session_token() -> str:
    api_token=_eboek_api_token()
    if not api_token:
        raise HTTPException(
            status_code=503,
            detail="e-Boekhouden is nog niet gekoppeld. Stel EBOEKHOUDEN_API_TOKEN in op de server."
        )
    now=time.time()
    if _EBOEK_SESSION["token"] and _EBOEK_SESSION["expires"]>now+60:
        return _EBOEK_SESSION["token"]

    data=_eboek_http(
        "POST","/v1/session",
        body={"accessToken":api_token,"source":"Vakstaal"},
        auth=False
    )
    token=str(data.get("token") or data.get("sessionToken") or "").strip()
    if not token:
        raise HTTPException(status_code=502,detail="e-Boekhouden gaf geen sessietoken terug.")
    _EBOEK_SESSION["token"]=token
    _EBOEK_SESSION["expires"]=now+3300
    return token

def _eboek_items(data):
    if isinstance(data,list): return data
    if not isinstance(data,dict): return []
    for key in ("items","data","results","relations","invoices","ledgers","templates"):
        value=data.get(key)
        if isinstance(value,list): return value
    return []

def _eboek_relation_public(rel):
    contact=str(
        rel.get("contact")
        or rel.get("contactPerson")
        or rel.get("contact_person")
        or ""
    ).strip()
    fixed_phone=str(rel.get("phoneNumber") or rel.get("phone_number") or "").strip()
    mobile_phone=str(rel.get("mobilePhoneNumber") or rel.get("mobile_phone_number") or "").strip()
    preferred_phone=fixed_phone or mobile_phone

    return {
        "id":rel.get("id"),
        "code":rel.get("code") or "",
        "type":rel.get("type") or "",
        "name":str(rel.get("name") or "").strip(),
        "contact":contact,
        "contactPerson":contact,
        "address":rel.get("address") or "",
        "postalCode":rel.get("postalCode") or rel.get("postal_code") or "",
        "city":rel.get("city") or "",
        "country":rel.get("country") or "",
        "phoneNumber":fixed_phone,
        "mobilePhoneNumber":mobile_phone,
        "preferredPhoneNumber":preferred_phone,
        "emailAddress":rel.get("emailAddress") or rel.get("email_address") or "",
        "emailAddressInvoice":rel.get("emailAddressInvoice") or rel.get("email_address_invoice") or "",
        "termOfPayment":rel.get("termOfPayment") or rel.get("term_of_payment"),
    }

def _eboek_find_relations(name="", email="", limit=20, partial=False):
    query={"limit":max(1,min(50,int(limit or 20))),"offset":0}
    if name:
        query["name[like]" if partial else "name"] = f"%{name}%" if partial else name
    if email:
        query["email[like]" if partial else "email"] = f"%{email}%" if partial else email
    data=_eboek_http("GET","/v1/relation",query=query)
    return _eboek_items(data)

def _eboek_get_relation(relation_id):
    return _eboek_http("GET",f"/v1/relation/{int(relation_id)}")

def _eboek_hydrate_relation_list_items(items, limit=20):
    """
    GET /v1/relation returns RelationListItem objects with only id/type/code.
    For display/autofill we therefore fetch GET /v1/relation/{id} for every
    candidate, which contains the actual name/contact/address/e-mail fields.
    """
    hydrated=[]
    for item in (items or []):
        if len(hydrated)>=max(1,min(50,int(limit or 20))):
            break
        relation_id=item.get("id") if isinstance(item,dict) else None
        if not relation_id:
            continue
        try:
            detail=_eboek_get_relation(relation_id)
            if isinstance(detail,dict):
                hydrated.append(detail)
        except Exception as exc:
            print(f"e-Boekhouden relation detail {relation_id} warning: {exc}")
    return hydrated

def _eboek_find_or_create_relation(payload):
    relation_id=payload.get("eboekhoudenRelationId")
    if relation_id:
        try:
            rel=_eboek_get_relation(relation_id)
            return rel,False
        except Exception:
            pass

    name=str(payload.get("customer") or "").strip()
    email=str(payload.get("customerEmail") or "").strip()
    if email:
        rows=_eboek_find_relations(email=email,limit=10)
        if rows: return rows[0],False
    if name:
        rows=_eboek_find_relations(name=name,limit=10)
        exact=[r for r in rows if str(r.get("name") or "").strip().casefold()==name.casefold()]
        if exact: return exact[0],False

    if not name:
        raise HTTPException(status_code=400,detail="Klantnaam ontbreekt voor e-Boekhouden.")

    if _eboek_load_settings().get("createMissingRelation") is False:
        raise HTTPException(
            status_code=404,
            detail="Klant bestaat nog niet in e-Boekhouden en automatisch aanmaken staat uit."
        )

    relation_body={
        "type":"B",
        "name":name,
        "contact":str(payload.get("contactPerson") or "").strip() or None,
        "address":str(payload.get("customerAddress") or "").strip() or None,
        "phoneNumber":str(payload.get("customerPhone") or "").strip() or None,
        "emailAddress":email or None,
        "emailAddressInvoice":email or None,
        "termOfPayment":int(_eboek_load_settings().get("paymentTermDays") or 30),
    }
    relation_body={k:v for k,v in relation_body.items() if v not in (None,"")}
    created=_eboek_http("POST","/v1/relation",body=relation_body)
    rid=created.get("id")
    if not rid:
        raise HTTPException(status_code=502,detail="e-Boekhouden heeft de klant aangemaakt maar geen relatie-ID teruggegeven.")
    return _eboek_get_relation(rid),True

def _eboek_find_ledger_id(code: str):
    data=_eboek_http("GET","/v1/ledger",query={"limit":2000,"offset":0,"code":code})
    rows=_eboek_items(data)
    exact=[r for r in rows if str(r.get("code") or "")==str(code)]
    return (exact[0].get("id") if exact else (rows[0].get("id") if rows else None))

def _eboek_find_template_id():
    cfg=_eboek_load_settings()
    configured=str(cfg.get("invoiceTemplateId") or "").strip()
    if configured.isdigit(): return int(configured)
    name=str(cfg.get("invoiceTemplateName") or "Vakstaal").strip()
    data=_eboek_http("GET","/v1/invoicetemplate",query={"limit":200,"offset":0,"name":name,"active":"true"})
    rows=_eboek_items(data)
    if not rows:
        data=_eboek_http("GET","/v1/invoicetemplate",query={"limit":200,"offset":0,"active":"true"})
        rows=_eboek_items(data)
    exact=[r for r in rows if str(r.get("name") or "").strip().casefold()==name.casefold()]
    row=(exact[0] if exact else (rows[0] if len(rows)==1 else None))
    return row.get("id") if row else None


@app.get("/api/eboekhouden/settings")
def eboekhouden_get_settings():
    cfg=_eboek_load_settings()
    token=_eboek_api_token()
    return {
        "ok":True,
        "settings":cfg,
        "tokenConfigured":bool(token),
        "tokenMasked":_eboek_mask_token(token),
        "connected":bool(_EBOEK_SESSION.get("token") and _EBOEK_SESSION.get("expires",0)>time.time())
    }

@app.put("/api/eboekhouden/settings")
async def eboekhouden_put_settings(request: Request):
    incoming=await request.json()
    if not isinstance(incoming,dict):
        raise HTTPException(status_code=400,detail="Ongeldige e-Boekhouden instellingen.")

    current=_eboek_load_settings()
    allowed={
        "apiBase","revenueLedgerCode","debtorLedgerId","invoiceTemplateName",
        "invoiceTemplateId","paymentTermDays","vatCode","invoiceDescription",
        "createMissingRelation"
    }
    for key in allowed:
        if key in incoming:
            current[key]=incoming[key]

    current["apiBase"]=str(current.get("apiBase") or "https://api.e-boekhouden.nl").rstrip("/")
    current["revenueLedgerCode"]=str(current.get("revenueLedgerCode") or "8055").strip()
    current["debtorLedgerId"]=str(current.get("debtorLedgerId") or "").strip()
    current["invoiceTemplateName"]=str(current.get("invoiceTemplateName") or "Vakstaal").strip()
    current["invoiceTemplateId"]=str(current.get("invoiceTemplateId") or "").strip()
    current["paymentTermDays"]=max(0,int(current.get("paymentTermDays") or 30))
    current["vatCode"]=str(current.get("vatCode") or "HOOG_VERK_21").strip()
    current["invoiceDescription"]=str(current.get("invoiceDescription") or "Werkzaamheden volgens offerte {offertenummer}").strip()
    current["createMissingRelation"]=bool(current.get("createMissingRelation",True))

    api_token=str(incoming.get("apiToken") or "").strip()
    if api_token:
        _eboek_store_api_token(api_token)

    _eboek_save_settings(current)
    token=_eboek_api_token()
    return {
        "ok":True,
        "settings":current,
        "tokenConfigured":bool(token),
        "tokenMasked":_eboek_mask_token(token),
        "connected":False
    }

@app.get("/api/eboekhouden/diagnose")
def eboekhouden_diagnose():
    cfg=_eboek_load_settings()
    token=_eboek_api_token()
    result={
        "ok":False,
        "tokenConfigured":bool(token),
        "tokenMasked":_eboek_mask_token(token),
        "sessionOk":False,
        "ledgerOk":False,
        "templateOk":False,
        "ledgerCode":str(cfg.get("revenueLedgerCode") or "8055"),
        "ledgerId":None,
        "templateId":None,
        "templateName":str(cfg.get("invoiceTemplateName") or "Vakstaal"),
        "stage":"token",
        "detail":"",
    }
    if not token:
        result["detail"]="Er is nog geen e-Boekhouden API-token op de server opgeslagen."
        return result

    try:
        _eboek_session_token()
        result["sessionOk"]=True
        result["stage"]="ledger"
    except HTTPException as exc:
        result["detail"]=str(exc.detail)
        return result

    try:
        ledger_id=_eboek_find_ledger_id(result["ledgerCode"])
        result["ledgerId"]=ledger_id
        result["ledgerOk"]=bool(ledger_id)
        if not ledger_id:
            result["detail"]=f"Omzetgrootboek {result['ledgerCode']} is niet gevonden in e-Boekhouden."
    except HTTPException as exc:
        result["detail"]=str(exc.detail)
        return result

    result["stage"]="template"
    try:
        template_id=_eboek_find_template_id()
        result["templateId"]=template_id
        result["templateOk"]=bool(template_id)
        if not template_id and not result["detail"]:
            result["detail"]=f"Factuursjabloon {result['templateName']} is niet gevonden of niet uniek."
    except HTTPException as exc:
        if not result["detail"]:
            result["detail"]=str(exc.detail)
        return result

    result["ok"]=result["sessionOk"] and result["ledgerOk"] and result["templateOk"]
    result["stage"]="ready" if result["ok"] else result["stage"]
    return result

@app.get("/api/eboekhouden/status")
def eboekhouden_status():
    if not _eboek_configured():
        return {"ok":False,"configured":False}
    try:
        _eboek_session_token()
        return {"ok":True,"configured":True}
    except HTTPException as exc:
        return {"ok":False,"configured":True,"detail":exc.detail}

@app.get("/api/eboekhouden/relations/search")
def eboekhouden_relation_search(q: str="", limit: int=10):
    query=str(q or "").strip()
    wanted=max(1,min(20,int(limit or 10)))
    if len(query)<1:
        return {"ok":True,"relations":[]}

    # v632: e-Boekhouden gebruikt zonder filteroperator een exacte vergelijking.
    # Voor live typeahead moet dit dus expliciet NAME[LIKE]=%zoektekst% zijn.
    list_items=_eboek_find_relations(
        name=query,
        limit=max(wanted,20),
        partial=True
    )

    # RelationListItem bevat volgens de REST API alleen id/type/code.
    # Haal daarom de echte relatiedetails op vóór we ze aan de browser geven.
    rows=_eboek_hydrate_relation_list_items(list_items,limit=max(wanted,20))

    # Extra e-mailzoeking alleen wanneer de invoer een e-mailadres lijkt.
    if not rows and "@" in query:
        email_items=_eboek_find_relations(
            email=query,
            limit=max(wanted,20),
            partial=True
        )
        rows=_eboek_hydrate_relation_list_items(email_items,limit=max(wanted,20))

    needle=query.casefold()

    # Sorteer echte bedrijfsnamen: begint-met eerst, daarna bevat.
    starts=[]
    contains=[]
    for relation in rows:
        name=str(relation.get("name") or "").strip()
        if not name:
            continue
        lowered=name.casefold()
        if lowered.startswith(needle):
            starts.append(relation)
        elif needle in lowered:
            contains.append(relation)

    ranked=starts+contains

    # Uniek op echte relatie-ID.
    unique=[]
    seen=set()
    for relation in ranked:
        key=str(relation.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(relation)
        if len(unique)>=wanted:
            break

    return {
        "ok":True,
        "relations":[_eboek_relation_public(r) for r in unique]
    }

@app.post("/api/quotes/{quote_id}/eboekhouden-invoice")
def create_eboekhouden_invoice(quote_id: str):
    with _db_connect() as conn:
        quote=_quote_response(conn,quote_id)

    payload=quote.get("payload") or {}
    total=float(payload.get("total_ex_vat") or quote.get("total_ex_vat") or 0)
    if total<=0:
        raise HTTPException(status_code=400,detail="Het offertebedrag is € 0,00; factuur is niet aangemaakt.")

    relation,created_relation=_eboek_find_or_create_relation(payload)
    relation_id=relation.get("id")
    if not relation_id:
        raise HTTPException(status_code=502,detail="Geen geldig e-Boekhouden relatie-ID gevonden.")

    template_id=_eboek_find_template_id()
    if not template_id:
        raise HTTPException(
            status_code=503,
            detail="Geen uniek actief factuursjabloon gevonden. Stel EBOEKHOUDEN_INVOICE_TEMPLATE_ID in."
        )

    cfg=_eboek_load_settings()
    revenue_code=str(cfg.get("revenueLedgerCode") or "8055").strip()
    revenue_id=_eboek_find_ledger_id(revenue_code)
    if not revenue_id:
        raise HTTPException(status_code=503,detail=f"Grootboek {revenue_code} is niet gevonden in e-Boekhouden.")

    debtor_id=None
    configured_debtor=str(cfg.get("debtorLedgerId") or "").strip()
    if configured_debtor.isdigit():
        debtor_id=int(configured_debtor)

    quote_number=str(quote.get("quote_number") or "").strip()
    term=int(relation.get("termOfPayment") or relation.get("term_of_payment") or cfg.get("paymentTermDays") or 30)

    invoice_body={
        "relationId":int(relation_id),
        "date":datetime.now(timezone.utc).date().isoformat(),
        "termOfPayment":term,
        "templateId":int(template_id),
        "reference":f"Offerte {quote_number}" if quote_number else "Vakstaal offerte",
        "items":[{
            "description":str(cfg.get("invoiceDescription") or "Werkzaamheden volgens offerte {offertenummer}").replace("{offertenummer}",quote_number).strip(),
            "pricePerUnit":round(total,2),
            "quantity":1,
            "vatCode":str(cfg.get("vatCode") or "HOOG_VERK_21"),
            "ledgerId":int(revenue_id),
        }],
    }
    if debtor_id:
        invoice_body["mutation"]={
            "description":f"Verkoopfactuur offerte {quote_number}".strip(),
            "ledgerId":debtor_id,
            "checkPaymentReference":False,
        }

    invoice=_eboek_http("POST","/v1/invoice",body=invoice_body)

    # Persist relation id in quote payload so future invoices/searches are exact.
    payload["eboekhoudenRelationId"]=int(relation_id)
    payload["eboekhoudenLastInvoiceId"]=invoice.get("id")
    payload["eboekhoudenLastInvoiceNumber"]=invoice.get("invoiceNumber") or invoice.get("invoice_number") or ""
    with _db_connect() as conn:
        cur=conn.cursor()
        cur.execute(
            _sql("UPDATE quotes SET payload_json=%s,updated_at=%s WHERE id=%s",
                 "UPDATE quotes SET payload_json=?,updated_at=? WHERE id=?"),
            (json.dumps(payload,ensure_ascii=False),_utcnow(),quote_id)
        )
        conn.commit()

    return {
        "ok":True,
        "relation":_eboek_relation_public(relation),
        "relation_created":created_relation,
        "invoice":invoice,
        "ledger_code":revenue_code,
        "template_id":template_id,
    }


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
        result = _apply_robust_standard_profile_recognition(step_path, result)
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



# ===== v581 Dropbox snijlayerbibliotheek =====
def _dropbox_list_files_recursive(path: str = "", suffix: str = ".lcm") -> list[dict]:
    dbx_path = _normalize_dropbox_browser_path(path)
    payload = {
        "path": dbx_path,
        "recursive": True,
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

    wanted = str(suffix or "").lower()
    files = []
    for item in entries:
        if str(item.get(".tag") or "") != "file":
            continue
        name = str(item.get("name") or "")
        if wanted and not name.lower().endswith(wanted):
            continue
        files.append({
            "name": name,
            "path_display": str(item.get("path_display") or ""),
            "path_lower": str(item.get("path_lower") or ""),
            "id": str(item.get("id") or ""),
            "rev": str(item.get("rev") or ""),
            "server_modified": str(item.get("server_modified") or ""),
            "client_modified": str(item.get("client_modified") or ""),
            "size": int(item.get("size") or 0),
            "content_hash": str(item.get("content_hash") or ""),
        })
    files.sort(key=lambda x: (x["path_display"] or x["name"]).casefold())
    return files


@app.post("/api/dropbox/cut-layers/scan")
async def dropbox_cut_layers_scan(request: Request):
    """Scan een ingestelde Dropbox-map recursief op officiële .LCM snijlayers.

    De browser mag bekende revisies meesturen. Ongewijzigde bestanden worden dan
    niet opnieuw gedownload/geparsed; nieuwe of gewijzigde bestanden wel.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    path = _normalize_dropbox_browser_path(body.get("path") or "")
    if not path:
        raise HTTPException(status_code=400, detail="Kies eerst een Dropbox-map voor snijlayers in App-instellingen.")

    known_raw = body.get("known") or {}
    known = {str(k).lower(): str(v or "") for k, v in known_raw.items()} if isinstance(known_raw, dict) else {}

    files = _dropbox_list_files_recursive(path, ".lcm")
    output = []
    parsed_count = 0
    failed_count = 0

    for meta in files:
        key = str(meta.get("path_lower") or meta.get("path_display") or "").lower()
        unchanged = bool(key and known.get(key) and known.get(key) == str(meta.get("rev") or ""))
        row = {**meta, "status": "unchanged" if unchanged else ("changed" if key in known else "new")}
        if not unchanged:
            try:
                content = _dropbox_download_bytes(meta.get("path_display") or meta.get("path_lower") or "")
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("Layerbestand is groter dan 10 MB.")
                parsed = _parse_fs_material_lcm(content, meta.get("name") or "layer.lcm")
                row["parsed"] = parsed
                parsed_count += 1
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                failed_count += 1
        output.append(row)

    return {
        "ok": True,
        "path": path,
        "file_count": len(files),
        "parsed_count": parsed_count,
        "failed_count": failed_count,
        "files": output,
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)