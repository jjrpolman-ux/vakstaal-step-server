from __future__ import annotations
# Invoice review 2026-09-25b; persistent per-message rejection, safe post-send cleanup.
# v8 merge 2026-09-24: supplied invoice routes + v6 distinct STEP end contours; LCM timing schema 2 retained.
import base64
import html

import os
import shutil
import tempfile
import logging
import unicodedata
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
LCM_TIMING_SCHEMA = 2  # v139: named TimeStay only; no controller delays in pierce time

STEP_PROFILE_RECOGNITION_VERSION = 13  # v771: topology-based outer skin + tabs/end contours
STEP_PHYSICAL_CUT_VERSION = 4  # v6: distinct ends on short members; preserve closed loops
# v966: LCM writer ondersteunt Corner Speed (%) naast Define corner/B-as parameters.

app = FastAPI(title="Vakstaal STEP Server", version="1.0.0")

# v1120: quote overwrite now reconciles current STEP/Nest/LCM manifests and removes obsolete files.
# Authentication and outermost CORS are installed after route declarations.





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


def _robust_rectangular_tube_outer_radius(
    solid: cq.Shape,
    axis: np.ndarray,
    length_mm: float,
    width: float,
    height: float,
    wall: float,
) -> float | None:
    """
    Detecteer de echte buitenhoekradius uit cilindrische hoekvlakken die parallel
    lopen aan de lengte-as. Dwars geboorde gaten vallen hierdoor automatisch af.
    Bij een holle koker zien we doorgaans R buiten en R-wand binnen.
    """
    min_span=max(1.0,float(length_mm)*0.35)
    radii=[]
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
            r=float(cylinder.Radius())
            if not math.isfinite(r) or r<=0.05:
                continue
            if r>min(float(width),float(height))/2.0+0.25:
                continue
            radii.append(r)
        except Exception:
            continue

    distinct=_cluster_scalar_values(radii,0.04)
    if not distinct:
        return None

    # Een buitenradius hoort groter te zijn dan de binnenradius. Neem de grootste
    # longitudinale hoekradius die geometrisch binnen het profiel past.
    candidates=[
        r for r in distinct
        if r>=max(0.10,float(wall)*0.55)
        and r<=min(float(width),float(height))/2.0+0.10
    ]
    if not candidates:
        return None
    return float(max(candidates))


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
    wall=sum(walls)/2.0
    width=max(width,height)
    height=min(dimensions[0][0],dimensions[1][0])
    # Herstel breedte/hoogte expliciet na sorteren.
    raw_sizes=sorted([dimensions[0][0],dimensions[1][0]],reverse=True)
    width,height=raw_sizes[0],raw_sizes[1]
    outer_radius=_robust_rectangular_tube_outer_radius(
        solid,axis,length_mm,width,height,wall
    )
    return width,height,wall,outer_radius

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

        # v670 — KOKER EERST.
        # Afgeronde vierkante/rechthoekige kokers hebben longitudinale cilindervlakken
        # voor hun binnen- en buitenhoekradius. Als 'rond' eerst getest wordt kunnen
        # die radii ten onrechte als buiten-/binnenradius van een ronde buis worden
        # geïnterpreteerd (bijv. R5/R3 => foutief Ø10x2).
        #
        # De lange vlakke buiten- en binnenwanden zijn daarom leidend voor kokers.
        rect_dims=_robust_rectangular_tube_dimensions(solid,axis,length_mm)
        if rect_dims:
            width,height,wall,outer_radius=rect_dims
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
                'profile_recognition_method':'main-wall-levels-radius-v10',
                'detected_outer_radius_mm':float(outer_radius) if outer_radius is not None else None,
                'outer_radius_mm':float(max(2.0,outer_radius or 0.0)),
                'effective_outer_radius_mm':float(max(2.0,outer_radius or 0.0)),
                'minimum_outer_radius_mm':2.0,
            }
        else:
            # Alleen wanneer géén geldige kokerwanden zijn gevonden mag de
            # coaxiale-cylinderherkenning een ronde buis opleveren.
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
                    'profile_recognition_method':'coaxial-cylinder-radii-v10',
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
    return CACHE_DIR / job_id / "assembly_mesh_physical_cut_v68_distinct_ends.json"


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
    xdir -= np.dot(xdir, zdir) * zdir
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
    # v6: a short member can have real stock seams shorter than 20 mm.
    # Require 75% of its length when that is shorter than the old 20 mm floor;
    # leave the threshold for ordinary/long members unchanged.
    anchor_min_len = max(
        min(20.0, float(raw_length) * 0.75), float(raw_length) * 0.30, 1e-6
    )

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
        # v770:
        # Een genormaliseerde buitenhuid hoort rond 1.00 te liggen. Bij schuine,
        # gekeepte of samengestelde kokers kunnen de longitudinale seam-ankers
        # door de lokale geometrie iets boven 1.00 uitkomen. De oude bovengrens
        # 1.03 maakte de buitenhuidfilter dan te streng: echte contourdelen met
        # score ~1.00 werden weggegooid, vooral bij midden-/verstekuitsparingen.
        #
        # 0.985 houdt de scheiding met de binnenhuid (typisch ~0.96 bij 2 mm
        # wand op 100 mm koker) intact, maar laat de echte buitencontour wel door.
        outer_shell_threshold=max(0.70,min(0.985,outer_shell_threshold))

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
                # v1216 — korte axiale stukjes vlak bij een fysiek buiseinde zijn
                # niet automatisch een gewone profielnaad. Bij STEP-einden met
                # een lokaal verspringende/afgeronde contour (zoals 25x25x1,5
                # in het Kastframe-bestand) zijn deze korte lijnstukken juist
                # onderdeel van het echte laser-eindpad. De oude herkenning
                # verwijderde ze, waarna de frontend een kunstmatige 1 mm-kloof
                # in een verder geldige eindcontour zag.
                #
                # Lange langsranden blijven gewoon als standaard profielnaad
                # onderdrukt. Alleen een KORT stukje binnen de terminale zone
                # wordt behouden; de buitenhuidfilter beslist daarna nog steeds
                # of het werkelijk tot de fysieke buitencontour behoort.
                half_len=float(basis[6])
                ow=float(basis[4]); oh=float(basis[5])
                terminal_band=max(1.5,min(8.0,max(ow,oh,1.0)*0.18))
                near_terminal=max(abs(float(a[2])),abs(float(b[2]))) >= half_len-terminal_band
                short_terminal_edge=chord <= max(1.25,terminal_band*0.55)
                if near_terminal and short_terminal_edge:
                    return False
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



def _outer_skin_edge_hashes(
    solid: cq.Shape,
    basis: tuple,
) -> set[int]:
    """
    Return edges that border the REAL outside skin of the hollow profile.

    v771:
    Do not decide outside-vs-inside from radius/box distance alone. Rounded
    100x100 corners can legitimately move far inward in XY, while the inside
    wall can sit only ~2 mm behind the outside wall. A single radial threshold
    therefore either loses real outside contour pieces OR includes the inner
    BREP loop.

    Instead use BREP face orientation:
    - an OUTER longitudinal face has its outward normal pointing away from the
      profile axis;
    - an INNER longitudinal face has its outward normal pointing toward the
      hollow cavity / profile axis.

    Every real tube-laser contour edge on the outside skin borders at least one
    such outer face. This also preserves shaped end tabs, mitres and notches.
    """
    c, _xdir, _ydir, zdir, *_ = basis
    result: set[int] = set()

    for face in solid.Faces():
        try:
            center=np.array(face.Center().toTuple(),dtype=float)
            rel=center-c
            axial=float(np.dot(rel,zdir))
            radial=rel-axial*zdir
            radial_norm=float(np.linalg.norm(radial))
            if radial_norm<=1e-7:
                continue

            normal=np.array(face.normalAt().toTuple(),dtype=float)
            normal_norm=float(np.linalg.norm(normal))
            if normal_norm<=1e-9:
                continue
            normal=normal/normal_norm

            # Skin faces run mainly along the profile axis. End/cut-wall faces
            # normally have a much larger axial normal component.
            axial_normal=abs(float(np.dot(normal,zdir)))
            if axial_normal>0.25:
                continue

            radial_unit=radial/radial_norm
            outward_alignment=float(np.dot(normal,radial_unit))

            # Strong positive alignment = outside skin.
            # Inner hollow faces are normally near -1.0.
            if outward_alignment<0.80:
                continue

            for edge in face.Edges():
                try:
                    result.add(int(edge.hashCode()))
                except Exception:
                    pass
        except Exception:
            continue

    return result


def _closed_cut_component_outline(component: dict, tolerance: float = 0.025):
    """Return the actual ordered closed outline, or None; never fill a gap.

    Match the frontend's closure tolerance. A closed component needs no nearby
    edges added to it: those can belong to the opposite end or a drilled hole.
    Coordinates and edge samples remain unmodified in the returned mesh.
    """
    pending = [list(edge.get("pts") or []) for edge in component.get("edges", [])]
    pending = [points for points in pending if len(points) > 1]
    if not pending:
        return None
    loop = pending.pop(0)
    while math.dist(loop[0], loop[-1]) > tolerance:
        best_distance, best_index, reverse = float("inf"), -1, False
        for index, points in enumerate(pending):
            for at_end, point in ((False, points[0]), (True, points[-1])):
                distance = math.dist(loop[-1], point)
                if distance < best_distance:
                    best_distance, best_index, reverse = distance, index, at_end
        if best_index < 0 or best_distance > tolerance:
            return None
        points = pending.pop(best_index)
        if reverse:
            points = list(reversed(points))
        loop.extend(points[1:])
    if pending or len(loop) < 3:
        return None
    return loop


def _closed_cut_outline_wraps_axis(outline: list, basis: tuple) -> bool:
    """Distinguish a complete end from a closed hole in a profile wall.

    An end runs around the stock axis in the transverse plane. A wall hole does
    not. This is a classification of existing CAD edges, not a generated contour.
    Used only to resolve the old overlapping-end-zone ambiguity.
    """
    xy = [(x, y) for x, y, _ in _basis_coords(outline, basis)]
    if len(xy) < 3 or any(math.hypot(x, y) < 1e-8 for x, y in xy):
        return False
    angle = 0.0
    for (ax, ay), (bx, by) in zip(xy, xy[1:] + xy[:1]):
        angle += math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    return abs(angle) > math.pi


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
        # v771: topology is leidend voor buitenhuidherkenning. Alleen wanneer
        # een exotisch STEP-bestand geen bruikbare face-normalen oplevert,
        # valt de parser terug op de oudere geometrische shell-test.
        outer_skin_edges=_outer_skin_edge_hashes(solid,basis)
        use_topology=bool(outer_skin_edges)

        for edge in solid.Edges():
            if _is_standard_profile_edge(edge,basis):
                continue

            if use_topology:
                try:
                    if int(edge.hashCode()) not in outer_skin_edges:
                        continue
                except Exception:
                    continue
            elif not _edge_is_on_outer_skin(edge,basis):
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

    # v6: wide end zones overlap on short members. Previously both max()
    # calls could pick the SAME component. Deduplication then discarded one
    # entire end, leaving a genuine second end in feature_lines.
    # Resolve that ambiguity with two distinct, axially ordered candidates.
    # Closed wall holes cannot stand in for a missing end.
    closed_outlines = {
        id(component): _closed_cut_component_outline(component)
        for component in summaries
    }
    if plus_seed is not None and plus_seed is minus_seed:
        terminal_candidates = [
            component for component in summaries
            if closed_outlines[id(component)] is None
            or _closed_cut_outline_wraps_axis(closed_outlines[id(component)], basis)
        ]
        if terminal_candidates:
            lower = min(component["zmean"] for component in terminal_candidates)
            upper = max(component["zmean"] for component in terminal_candidates)
            if upper - lower > 0.025:
                split = (lower + upper) / 2.0
                minus_seed = max(
                    (component for component in terminal_candidates if component["zmean"] < split),
                    key=lambda component: component["length"],
                )
                plus_seed = max(
                    (component for component in terminal_candidates if component["zmean"] >= split),
                    key=lambda component: component["length"],
                )
            else:
                # Insufficient independent geometry: keep at most the one end
                # actually present. The existing frontend validation still fails.
                only_seed = max(terminal_candidates, key=lambda component: component["length"])
                plus_seed = only_seed if only_seed["zmean"] >= 0 else None
                minus_seed = only_seed if only_seed["zmean"] < 0 else None
        else:
            plus_seed = minus_seed = None

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

    def grow_family(seed:dict|None,other_seed:dict|None)->list[dict]:
        if seed is None:
            return []
        # A complete end must never absorb the other end or a nearby hole.
        if closed_outlines.get(id(seed)) is not None:
            return [seed]
        family=[seed]
        changed=True
        while changed:
            changed=False
            for comp in summaries:
                if any(comp is member for member in family) or comp is other_seed:
                    continue
                if closed_outlines.get(id(comp)) is not None:
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

    plus_family=grow_family(plus_seed,minus_seed)
    minus_family=grow_family(minus_seed,plus_seed)

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


def _simulation_profile_frame(solid: cq.Shape, detail: dict | None) -> dict:
    # Use the same stock-aligned coordinate system for CAD contours and simulation.
    if not detail or not detail.get("recognized"):
        return {}
    try:
        basis = _profile_basis_for_features(solid, detail)
        return {"profile_axis": [float(v) for v in basis[3]],
                "profile_basis_u": [float(v) for v in basis[1]],
                "profile_basis_v": [float(v) for v in basis[2]]}
    except Exception:
        return {}


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
    if DATABASE_URL and psycopg is None:
        raise RuntimeError("DATABASE_URL is ingesteld maar psycopg ontbreekt; geen stille SQLite-terugval.")
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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS quote_number_counters (
                year INTEGER PRIMARY KEY,
                last_number BIGINT NOT NULL CHECK (last_number >= 0)
            )
        """)

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


def _next_quote_number() -> str:
    """Reserve a number in a short transaction, independent of file uploads.

    Reservations survive failed saves: gaps are allowed, reuse is not.
    The atomic upsert serializes competing reservations across workers.
    """
    year = datetime.now().year
    prefix = f"VAK-{year}-"
    conn = _db_connect()
    try:
        with conn:
            cur = conn.cursor()
            if not _postgres_enabled():
                cur.execute("BEGIN IMMEDIATE")
            cur.execute(_sql(
                "SELECT last_number FROM quote_number_counters WHERE year=%s",
                "SELECT last_number FROM quote_number_counters WHERE year=?"
            ), (year,))
            seed = 0
            if cur.fetchone() is None:
                # Seed a new annual counter numerically, including numbers >9999.
                cur.execute(_sql(
                    "SELECT quote_number FROM quotes WHERE quote_number LIKE %s",
                    "SELECT quote_number FROM quotes WHERE quote_number LIKE ?"
                ), (prefix + "%",))
                for row in cur.fetchall():
                    suffix = str(row[0])[len(prefix):]
                    if suffix and suffix.isascii() and suffix.isdecimal():
                        seed = max(seed, int(suffix))
            cur.execute(_sql(
                """INSERT INTO quote_number_counters (year, last_number)
                   VALUES (%s, %s)
                   ON CONFLICT (year) DO UPDATE SET last_number =
                   GREATEST(quote_number_counters.last_number + 1, excluded.last_number)
                   RETURNING last_number""",
                """INSERT INTO quote_number_counters (year, last_number)
                   VALUES (?, ?)
                   ON CONFLICT (year) DO UPDATE SET last_number =
                   MAX(quote_number_counters.last_number + 1, excluded.last_number)
                   RETURNING last_number"""
            ), (year, seed + 1))
            number = cur.fetchone()[0]
        return f"{prefix}{number:04d}"
    finally:
        conn.close()


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


_PRODUCTION_STEP_FILENAME_RE = re.compile(
    r"^\d+x_.+_\d+(?:\.\d+)?mm_(?:(?:n2|o2)_)?(?:met|zonder)_bewerkingen\.(?:step|stp)$",
    re.IGNORECASE,
)
_FILTERED_STEP_FILENAME_RE = re.compile(r"_OFFERTSELECTIE\.(?:step|stp)$", re.IGNORECASE)
_GENERATED_CUT_LAYER_FILENAME_RE = re.compile(r"__Snijlayer_[^/\\]+\.lcm$", re.IGNORECASE)


def _payload_filename_set(data: dict, key: str) -> set[str]:
    result: set[str] = set()
    raw = data.get(key)
    if not isinstance(raw, list):
        return result
    for item in raw:
        if isinstance(item, dict):
            name = item.get("filename") or item.get("name")
        else:
            name = item
        if name:
            result.add(_safe_dropbox_name(str(name), "bestand").lower())
    return result


def _quote_file_manifest(data: dict) -> dict[str, set[str]]:
    """Authoritative manifest for technical quote files.

    New frontends send every technical category explicitly. Empty lists are
    meaningful: they mean that category should be empty after this save.
    """
    production = _payload_filename_set(data, "production_step_files")
    filtered = _payload_filename_set(data, "filtered_step_files")
    sources = _payload_filename_set(data, "source_step_files")
    nests = _payload_filename_set(data, "source_nest_files")
    cut_layers = _payload_filename_set(data, "cut_layer_files")

    # Nieuwe clients sturen expliciete manifests. Een lege lijst betekent:
    # deze offerte gebruikt geen bronbestand van dit type meer. Alleen oudere
    # clients zonder manifest mogen nog terugvallen op step_filename/nest_filename.
    if not isinstance(data.get("source_step_files"), list):
        step_filename = str(data.get("step_filename") or "").strip()
        if step_filename:
            sources.add(_safe_dropbox_name(step_filename, "bestand").lower())

    if not isinstance(data.get("source_nest_files"), list):
        nest_filename = str(data.get("nest_filename") or "").strip()
        if nest_filename:
            nests.add(_safe_dropbox_name(nest_filename, "bestand").lower())

    return {
        "production": production,
        "filtered": filtered,
        "sources": sources,
        "nests": nests,
        "cut_layers": cut_layers,
    }


def _same_dropbox_path(a: str, b: str) -> bool:
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def _dedupe_quote_file_rows(conn, quote_id: str) -> int:
    """Keep only the newest DB row per filename.

    Older app versions could leave duplicate PDF/STEP rows behind. Dropbox itself
    usually contains only one file when the path is identical; only a genuinely
    different obsolete path is deleted physically.
    """
    cur = conn.cursor()
    cur.execute(
        _sql(
            "SELECT id, filename, dropbox_path, created_at FROM quote_files WHERE quote_id=%s ORDER BY created_at DESC, id DESC",
            "SELECT id, filename, dropbox_path, created_at FROM quote_files WHERE quote_id=? ORDER BY created_at DESC, id DESC",
        ),
        (quote_id,),
    )
    rows = cur.fetchall()
    kept: dict[str, str] = {}
    removed = 0
    for row in rows:
        if isinstance(row, sqlite3.Row):
            file_id, filename, dropbox_path = row["id"], row["filename"], row["dropbox_path"]
        else:
            file_id, filename, dropbox_path = row[0], row[1], row[2]
        key = str(filename or "").lower()
        if key not in kept:
            kept[key] = str(dropbox_path or "")
            continue

        keep_path = kept[key]
        if dropbox_path and not _same_dropbox_path(dropbox_path, keep_path):
            _dropbox_delete_path(str(dropbox_path))
        cur.execute(
            _sql("DELETE FROM quote_files WHERE id=%s", "DELETE FROM quote_files WHERE id=?"),
            (file_id,),
        )
        removed += 1
    return removed


def _reconcile_quote_managed_files(conn, quote_id: str, data: dict) -> list[str]:
    """Delete obsolete technical files after overwriting a quote.

    This is deliberately manifest-driven. Quantity changes, removed products,
    replaced STEP imports and changed cut-layers may all create a new filename.
    Anything no longer present in the CURRENT manifest is removed from Dropbox
    and quote_files.

    PDFs and unrelated user attachments are not touched here.
    """
    manifest = _quote_file_manifest(data)

    reconcile_production = isinstance(data.get("production_step_files"), list)
    reconcile_filtered = isinstance(data.get("filtered_step_files"), list)
    reconcile_sources = isinstance(data.get("source_step_files"), list)
    reconcile_nests = isinstance(data.get("source_nest_files"), list)
    reconcile_cut_layers = isinstance(data.get("cut_layer_files"), list)

    if not any((
        reconcile_production,
        reconcile_filtered,
        reconcile_sources,
        reconcile_nests,
        reconcile_cut_layers,
    )):
        return []

    production = manifest["production"]
    filtered = manifest["filtered"]
    source_names = manifest["sources"]
    nest_names = manifest["nests"]
    cut_layers = manifest["cut_layers"]

    removed_names: list[str] = []
    cur = conn.cursor()
    cur.execute(
        _sql(
            "SELECT id, filename, dropbox_path FROM quote_files WHERE quote_id=%s",
            "SELECT id, filename, dropbox_path FROM quote_files WHERE quote_id=?",
        ),
        (quote_id,),
    )
    rows = cur.fetchall()

    for row in rows:
        if isinstance(row, sqlite3.Row):
            file_id, filename, dropbox_path = row["id"], row["filename"], row["dropbox_path"]
        else:
            file_id, filename, dropbox_path = row[0], row[1], row[2]

        safe_name = _safe_dropbox_name(str(filename or ""), "bestand")
        key = safe_name.lower()
        suffix = Path(safe_name).suffix.lower()

        obsolete = False

        if reconcile_production and _PRODUCTION_STEP_FILENAME_RE.match(safe_name):
            obsolete = key not in production

        elif reconcile_filtered and _FILTERED_STEP_FILENAME_RE.search(safe_name):
            obsolete = key not in filtered

        elif reconcile_cut_layers and _GENERATED_CUT_LAYER_FILENAME_RE.search(safe_name):
            obsolete = key not in cut_layers

        elif (
            reconcile_sources
            and suffix in {".step", ".stp"}
            and not _PRODUCTION_STEP_FILENAME_RE.match(safe_name)
            and not _FILTERED_STEP_FILENAME_RE.search(safe_name)
        ):
            obsolete = key not in source_names

        elif reconcile_nests and suffix in {".zx", ".nest"}:
            obsolete = key not in nest_names

        if not obsolete:
            continue

        # Delete Dropbox first. If that fails, keep DB metadata so the next save
        # can retry instead of silently leaving an orphan.
        if dropbox_path:
            _dropbox_delete_path(str(dropbox_path))

        cur.execute(
            _sql("DELETE FROM quote_files WHERE id=%s", "DELETE FROM quote_files WHERE id=?"),
            (file_id,),
        )
        removed_names.append(safe_name)

    return removed_names


# Compatibility alias for older call sites / diagnostics.
def _reconcile_quote_generated_steps(conn, quote_id: str, data: dict) -> list[str]:
    return _reconcile_quote_managed_files(conn, quote_id, data)


async def _store_quote_files(
    conn,
    quote_id: str,
    files: list[UploadFile],
    managed_step_names: set[str] | None = None,
) -> list[dict]:
    quote_number, customer_name, _payload_json = _quote_identity(conn, quote_id)
    folder = _quote_dropbox_folder(quote_number, customer_name)
    step_folder = _quote_storage_config()["step"]
    managed_step_names = {str(name or "").lower() for name in (managed_step_names or set())}
    stored_files=[]

    for upload in files or []:
        filename = _safe_dropbox_name(upload.filename or "bestand", "bestand")
        data = await upload.read(MAX_QUOTE_FILE_MB * 1024 * 1024 + 1)

        if not data:
            continue

        if len(data) > MAX_QUOTE_FILE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} is groter dan {MAX_QUOTE_FILE_MB} MB."
            )

        kind = _file_kind(filename)
        subfolder = step_folder if filename.lower() in managed_step_names else _file_dropbox_subfolder(filename)
        dropbox_path = f"{folder}/{subfolder}/{filename}"

        cur = conn.cursor()
        cur.execute(
            _sql(
                """
                SELECT id, dropbox_path FROM quote_files
                WHERE quote_id=%s AND filename=%s
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                """
                SELECT id, dropbox_path FROM quote_files
                WHERE quote_id=? AND filename=?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """
            ),
            (quote_id, filename)
        )
        existing = cur.fetchone()
        old_path = ""
        if existing:
            old_path = str(existing["dropbox_path"] if isinstance(existing, sqlite3.Row) else existing[1] or "")

        # Naam en bestandsgrootte bewijzen niet dat de inhoud gelijk is.
        # Iedere aangeleverde versie wordt daarom echt opnieuw opgeslagen.
        uploaded = _dropbox_upload_bytes(dropbox_path, data)
        actual_path = uploaded.get("path_display") or uploaded.get("path_lower") or dropbox_path

        # Productie-STEP stond in oudere versies soms in 'Origineel'. Bij een
        # nieuwe save verhuist het actuele bestand naar Productie STEP en wordt
        # de oude fysieke kopie meteen opgeruimd.
        if old_path and not _same_dropbox_path(old_path, actual_path):
            _dropbox_delete_path(old_path)

        if existing:
            existing_id = existing["id"] if isinstance(existing, sqlite3.Row) else existing[0]
            cur.execute(
                _sql(
                    "UPDATE quote_files SET dropbox_path=%s, data=%s, file_size=%s, content_type=%s, file_kind=%s WHERE id=%s",
                    "UPDATE quote_files SET dropbox_path=?, data=?, file_size=?, content_type=?, file_kind=? WHERE id=?"
                ),
                (actual_path, b"", len(data), upload.content_type or "application/octet-stream", kind, existing_id)
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

def _approval_email_settings() -> dict:
    """Mailinstellingen voor direct akkoordbericht.

    Gebruik dezelfde SMTP-configuratie als de overige Vakstaal-mailfuncties.
    Een apart QUOTE_APPROVAL_NOTIFY_EMAIL blijft mogelijk; als dat niet is
    ingesteld gaat de melding automatisch naar het ingestelde afzender-/loginadres.
    """
    host=str(os.environ.get("SMTP_HOST") or "").strip()
    user=str(os.environ.get("SMTP_USER") or "").strip()
    password=str(os.environ.get("SMTP_PASSWORD") or "").strip()
    if host and user and password:
        port=int(os.environ.get("SMTP_PORT") or "587")
        sender=str(os.environ.get("SMTP_FROM") or user).strip()
        return {
            "host":host,
            "port":port,
            "user":user,
            "password":password,
            "sender":sender or user,
            "recipient":str(os.environ.get("QUOTE_APPROVAL_NOTIFY_EMAIL") or sender or user).strip(),
            "ssl":str(os.environ.get("SMTP_SSL") or "").strip().lower() in {"1","true","yes"} or port==465,
            "source":"SMTP_*",
        }

    host=str(os.environ.get("VAKSTAAL_SMTP_HOST") or "").strip()
    user=str(os.environ.get("VAKSTAAL_SMTP_USER") or "").strip()
    password=str(os.environ.get("VAKSTAAL_SMTP_PASSWORD") or "").strip()
    port=int(os.environ.get("VAKSTAAL_SMTP_PORT") or "465")
    sender=str(os.environ.get("VAKSTAAL_SMTP_FROM") or user).strip()
    return {
        "host":host,
        "port":port,
        "user":user,
        "password":password,
        "sender":sender or user,
        "recipient":str(os.environ.get("QUOTE_APPROVAL_NOTIFY_EMAIL") or sender or user).strip(),
        "ssl":str(os.environ.get("VAKSTAAL_SMTP_SSL") or "true").strip().lower() in {"1","true","yes"} or port==465,
        "source":"VAKSTAAL_SMTP_*",
    }

def _approval_email_configured() -> bool:
    cfg=_approval_email_settings()
    return bool(cfg.get("host") and cfg.get("user") and cfg.get("password") and cfg.get("recipient"))

def _send_approval_email(data: dict) -> bool:
    cfg=_approval_email_settings()
    if not (cfg.get("host") and cfg.get("user") and cfg.get("password") and cfg.get("recipient")):
        return False
    host=str(cfg.get("host") or "").strip()
    port=int(cfg.get("port") or 587)
    user=str(cfg.get("user") or "").strip()
    password=str(cfg.get("password") or "").strip()
    recipient=str(cfg.get("recipient") or "").strip()
    sender=str(cfg.get("sender") or user).strip()
    use_ssl=bool(cfg.get("ssl") or port==465)

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
            "state": _lcm_refresh_state_timing(state),
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
    state = _lcm_refresh_state_timing(state)
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
    pos=data.find(b"Note")
    if pos<0: return ""
    cursor=pos+4
    if data[cursor:cursor+4]==b"\x00\x02\x00\x00" and cursor+4<len(data):
        ln=data[cursor+4]
        raw=data[cursor+5:cursor+5+ln]
        try: return raw.decode("utf-8").strip("\x00 ").strip()
        except Exception: pass
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


_LCM_LASER_FIELD_NAMES = {
    "LaserMode","PwmFreq","GasPressure","BeamSize","Focus","LaserCurrent",
    "Height","GasType","PwmRatio","RatioStart","FreqEnd","PierceMode",
    "CycleTime","RatioEnd","TimeFollow","TimeFocus","TimeBlow","TimeStay",
    "FreqStart","FocusEnd","Flags","SmoothType","NormalE","NormalS","Enable",
    "Freq","KnotCount","Current","Ratio","Bound","Bottom","Left","Top","Right"
}

def _lcm_value_table_and_laser_field_map(data: bytes) -> tuple[dict[int,str],dict[int,str]]:
    end=data.find(b"\x00\x07version")
    if end<0:
        end=min(len(data),900)
    parts=data[:end].split(b"\x00")
    table={}
    field_map={}
    for idx,part in enumerate(parts):
        if not part:
            continue
        try:
            ln=int(part[0])
            raw=part[1:1+ln]
            if len(raw)!=ln:
                continue
            text=raw.decode("ascii")
        except Exception:
            continue
        table[idx]=text
        # FSMATERIAL gebruikt de tabelindex zélf als field-id.
        # Niet opnieuw nummeren na filtering: numerieke value-table items
        # tussen veldnamen zouden anders alle volgende velden verschuiven.
        if text in _LCM_LASER_FIELD_NAMES:
            field_map[int(idx)]=text
    return table,field_map

def _lcm_decode_ref(table: dict[int,str],code: int):
    raw=table.get(int(code))
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",","."))
    except Exception:
        return raw

def _lcm_block_payload(data: bytes,name: bytes,next_name: bytes|None=None) -> bytes:
    marker=b"\x00\x00\x00"+bytes([len(name)])+name
    start=data.find(marker)
    if start<0:
        start=data.find(name)
    if start<0:
        return b""
    end=data.find(next_name,start+len(name)) if next_name else -1
    if end<0:
        end=min(len(data),start+700)
    return data[start:end]

def _lcm_decode_laser_block(data: bytes,name: bytes,next_name: bytes|None=None) -> dict:
    table,field_map=_lcm_value_table_and_laser_field_map(data)
    block=_lcm_block_payload(data,name,next_name)
    result={}
    i=0
    while i<=len(block)-4:
        if block[i:i+2]!=b"\x00\x01":
            i+=1; continue
        field_id=block[i+2]
        field_name=field_map.get(field_id)
        if not field_name:
            i+=3; continue
        if i+6<=len(block) and block[i+3:i+5]==b"\x00\x00":
            ln=block[i+5]
            raw=block[i+6:i+6+ln]
            try: value=float(raw.decode("ascii").replace(",","."))
            except Exception:
                try: value=raw.decode("ascii")
                except Exception: value=None
            result[field_name]=value
            i+=6+ln
        else:
            result[field_name]=_lcm_decode_ref(table,block[i+3])
            i+=4
    return result


def _lcm_decode_cut_field_by_id(data: bytes, field_id: int):
    """
    Fallback voor FSMATERIAL-versies waarin de dynamische veldnaam-tabel afwijkt.
    Bekende Cut veld-ID's uit de aangeleverde buis-LCM's:
      0x13 PwmFreq
      0x15 PwmRatio
      0x18 LaserCurrent
      0x1A Focus
      0x1B GasPressure
      0x1C Height
      0x1D GasType
    """
    table,_=_lcm_value_table_and_laser_field_map(data)
    block=_lcm_block_payload(data,b"Cut",b"Pierce1")
    if not block:
        return None

    marker=bytes([0x00,0x01,int(field_id)&0xff])
    pos=0
    while True:
        pos=block.find(marker,pos)
        if pos<0:
            return None
        if pos+4>len(block):
            return None

        if pos+6<=len(block) and block[pos+3:pos+5]==b"\x00\x00":
            ln=block[pos+5]
            raw=block[pos+6:pos+6+ln]
            try:
                return float(raw.decode("ascii").replace(",","."))
            except Exception:
                try:
                    return raw.decode("ascii")
                except Exception:
                    pass

        code=block[pos+3]
        value=_lcm_decode_ref(table,code)
        if value is not None:
            return value
        pos+=1


def _lcm_named_ref_value(data: bytes,name: bytes):
    table,_=_lcm_value_table_and_laser_field_map(data)
    pos=data.find(name)
    if pos<0: return None
    cursor=pos+len(name)
    marker=data.find(b"\x00\x01\x00\x00",cursor,min(len(data),cursor+16))
    if marker<0 or marker<=cursor: return None
    return _lcm_decode_ref(table,data[marker-1])

def _lcm_named_flag(data: bytes,name: bytes) -> bool:
    pos=data.find(name)
    if pos<0: return False
    cursor=pos+len(name)
    marker=data.find(b"\x00\x01\x00\x00",cursor,min(len(data),cursor+16))
    if marker<0 or marker<=cursor: return False
    code=int(data[marker-1])
    # FSMATERIAL stores enabled flags as 0x11 in some layers and 0x10 in
    # others. The 100x100x3 N2 machine layer uses 0x10 for its active corner
    # technique and B-axis limiter.
    if code in (0x10,0x11): return True
    if code==0x05: return False
    decoded=_lcm_named_ref_value(data,name)
    try: return bool(int(float(decoded)))
    except Exception: return False

def _lcm_gas_name(value) -> str:
    try: n=int(round(float(value)))
    except Exception: return ""
    return {2:"O2",3:"N2"}.get(n,str(n))

def _lcm_named_number_or_ref(data: bytes, name: bytes) -> float | None:
    """
    Lees een numerieke FSMATERIAL named value die óf inline ASCII staat,
    óf als 1-byte referentie naar de value-table is opgeslagen.
    Voorbeeld uit de echte 100x100x2-layer:
      PTConsA 0x22 -> value-table[0x22] == "20".
    """
    direct=_lcm_read_named_number(data,name)
    if direct is not None:
        return direct
    ref=_lcm_named_ref_value(data,name)
    try:
        value=float(ref)
        return value if math.isfinite(value) else None
    except Exception:
        return None



def _lcm_cut_parameters_complete(data: bytes,cut_block: dict) -> dict:
    ratio=cut_block.get("PwmRatio")
    if ratio is None:
        ratio=_lcm_decode_cut_field_by_id(data,0x15)

    cut_height=cut_block.get("Height")
    if cut_height is None:
        cut_height=_lcm_decode_cut_field_by_id(data,0x1C)

    peak_power=cut_block.get("LaserCurrent")
    if peak_power is None:
        peak_power=_lcm_decode_cut_field_by_id(data,0x18)

    frequency=cut_block.get("PwmFreq")
    if frequency is None:
        frequency=_lcm_decode_cut_field_by_id(data,0x13)

    return {
        "liftHeightMm":_lcm_named_number_or_ref(data,b"LiftHeight"),
        "cutHeightMm":cut_height,
        "peakPowerPct":peak_power,
        "dutyCyclePct":(float(ratio)*100.0 if isinstance(ratio,(int,float)) and abs(float(ratio))<=1.5 else ratio),
        "frequencyHz":frequency,
        "beamSizeMm":cut_block.get("BeamSize"),
        # v139: ieder bestand gebruikt zijn eigen veldnaam-/referentietabel.
        # Het zichtbare onderste Pierce time-veld is TimeStay, ook in Cut.
        # CycleTime/FreqEnd/Focus/Blow/Follow zijn GEEN vervangende piercetijden.
        # Bewaar overige velden alleen voor inspectie, niet als extra procestijd.
        "cycleTimeMs":cut_block.get("CycleTime"),
        "pierceTimeMs":cut_block.get("TimeStay"),
        "pierceTimeField":"TimeStay",
        "timingSchema":LCM_TIMING_SCHEMA,
        "timeFocusMs":cut_block.get("TimeFocus"),
        "timeBlowMs":cut_block.get("TimeBlow"),
        "timeFollowMs":cut_block.get("TimeFollow"),
        "timeStayMs":cut_block.get("TimeStay"),
        "laserOffDelayMs":_lcm_named_ref_value(data,b"DelayBeforeLaserOff"),
        "useLowPassFilter":_lcm_named_flag(data,b"UseLowPassFilter"),
        "lowPassFrequencyHz":_lcm_read_named_number(data,b"LowPassFrequency"),
        "slowLead":{"enabled":_lcm_named_flag(data,b"SlowLeadEn")},
        "slowStop":{"enabled":_lcm_named_flag(data,b"SlowEndEn")},
        "dynamicPower":True,
        "dynamicFrequency":True,
    }

def _lcm_pierce_parameters_complete(data: bytes) -> dict:
    step_count=_lcm_named_ref_value(data,b"PierceStepCount")
    try: ptype=int(round(float(step_count or 0)))
    except Exception: ptype=0
    stages={}
    for number,name,next_name in ((1,b"Pierce1",b"Pierce2"),(2,b"Pierce2",b"Pierce3"),(3,b"Pierce3",b"PipeCorner")):
        block=_lcm_decode_laser_block(data,name,next_name)
        ratio=block.get("PwmRatio")
        stages[f"stage{number}"]={
            "mode":"Segment Pie",
            "timeMs":block.get("TimeFollow"),
            "heightMm":block.get("Height"),
            "gas":_lcm_gas_name(block.get("GasType")),
            "pressureBar":block.get("GasPressure"),
            "peakPowerPct":block.get("LaserCurrent"),
            "dutyCyclePct":(float(ratio)*100.0 if isinstance(ratio,(int,float)) and abs(float(ratio))<=1.5 else ratio),
            "frequencyHz":block.get("PwmFreq"),
            "beamSize":block.get("BeamSize"),
            "focusMm":block.get("Focus"),
            "endFocusMm":block.get("FocusEnd"),
            # Alleen het onderste instelbare Pierce time-veld; geen FreqEnd (Hz).
            "pierceTimeMs":block.get("TimeStay"),
            "pierceTimeField":"TimeStay",
            "timingSchema":LCM_TIMING_SCHEMA,
            "timeStayMs":block.get("TimeStay"),
            # Alleen zichtbaar als bronveld. Niet meetellen in tijd/gascalculatie.
            "laserOffGasOnMs":block.get("TimeBlow"),
        }
    return {"type":ptype,**stages}

def _lcm_corner_parameters_complete(data: bytes) -> dict:
    ratio=_lcm_read_named_number(data,b"PTPwmRatio")
    standard=_lcm_read_named_number(data,b"PTCornerStandard")
    use_pressure=_lcm_named_flag(data,b"UsePtPressure")
    use_peak=_lcm_named_flag(data,b"UsePtCurrent")
    use_duty=_lcm_named_flag(data,b"UsePtPwmRatio")
    use_freq=_lcm_named_flag(data,b"UsePtFreq")
    return {
        "enabled":_lcm_named_flag(data,b"UsePTAdjust"),
        "followHeightOffsetMm":_lcm_read_named_number(data,b"PTFollowHPlus"),
        "cornerSpeed":_lcm_named_number_or_ref(data,b"PTCornerSpeed"),
        "useCornerPressure":use_pressure,"cornerPressureEnabled":use_pressure,
        "cornerPressureBar":_lcm_named_ref_value(data,b"PTPressure"),
        "usePeakPower":use_peak,"peakPowerEnabled":use_peak,
        "peakPowerPct":_lcm_read_named_number(data,b"PTCurrent"),
        "useDutyCycle":use_duty,"dutyCycleEnabled":use_duty,
        "dutyCyclePct":(ratio*100.0 if ratio is not None and abs(ratio)<=1.5 else ratio),
        "useFrequency":use_freq,"frequencyEnabled":use_freq,
        "frequencyHz":_lcm_named_ref_value(data,b"PTFreq"),
        "defineCornerDegPerMm":(standard*180.0/math.pi if standard is not None else None),
        "limitBAxisSpeed":_lcm_named_flag(data,b"PTConsEn"),
        "bAxisSpeedRpm":_lcm_named_number_or_ref(data,b"PTConsV"),
        "bAxisAcceleration":_lcm_named_number_or_ref(data,b"PTConsA"),
        "bAxisAccelerationRadS2":_lcm_named_number_or_ref(data,b"PTConsA"),
    }

def _lcm_other_parameters_complete(data: bytes) -> dict:
    root=_lcm_read_named_number(data,b"NanoMicroRootRatio")
    return {
        "vibrationSuppress":_lcm_named_flag(data,b"ShakeResLv"),
        "adjustMicroJoint":_lcm_named_flag(data,b"AdjustNanoMicroLen"),
        "adjustMicroJointMm":_lcm_read_named_number(data,b"NanoMicroLength"),
        "rootRatioEnabled":_lcm_named_flag(data,b"AdjustNanoMicroRatio"),
        "rootRatioPct":(root*100.0 if root is not None and abs(root)<=1.5 else root),
        "microJointSpeedEnabled":_lcm_named_flag(data,b"AdjustNanoMicroSpeed"),
        "microJointSpeedMMin":_lcm_read_named_number(data,b"NanoMicroSpeed"),
        "fixedHeightCutting":_lcm_named_flag(data,b"UseCutHeight"),
        "smartFollowOutTubeLead":not _lcm_named_flag(data,b"NewEcoLeadDisabled"),
        "smartFollowLengthMm":_lcm_named_ref_value(data,b"EcoLeadTraceDetectLen"),
    }


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
    if not str(speed_raw or "").strip():
        speed_named=_lcm_read_named_number(data,b"WorkSpeed")
        speed_raw="" if speed_named is None else str(speed_named)

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
    speed_m_min = (
        _lcm_clean_machine_number(work_speed_mm_s * 0.06)
        if work_speed_mm_s is not None and work_speed_mm_s>0 else None
    )

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
    # Accept the wall thickness immediately after the second dimension:
    # "100x100x3" has no word boundary between "100" and the next "x".
    m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)(?=\s*[x×]\s*\d|\b)", source_text, re.I)
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

    # v860: de echte Cut Gas uit de LCM is de bron van waarheid.
    # De dikteregel is alleen fallback voor oude/afwijkende LCM's zonder leesbare GasType.
    cut_values=_lcm_decode_laser_block(data,b"Cut",b"Pierce1")
    cut_gas_machine=_lcm_gas_name(
        cut_values.get("GasType")
        if cut_values.get("GasType") is not None
        else _lcm_decode_cut_field_by_id(data,0x1D)
    )
    gas_code=str(cut_gas_machine or "").strip().upper()
    if gas_code == "N2":
        gas = "nitrogen"
    elif gas_code == "O2":
        gas = "oxygen"
    else:
        gas = "oxygen" if (thickness is not None and thickness > 3.0) else "nitrogen"

    advanced_parameters = _lcm_advanced_parameters(data)
    cut_parameters=_lcm_cut_parameters_complete(data,cut_values)
    pierce_parameters=_lcm_pierce_parameters_complete(data)
    corner_parameters=_lcm_corner_parameters_complete(data)
    other_parameters=_lcm_other_parameters_complete(data)
    focus=cut_values.get("Focus") if cut_values.get("Focus") is not None else focus
    if focus is None:
        focus=_lcm_decode_cut_field_by_id(data,0x1A)

    pressure=cut_values.get("GasPressure") if cut_values.get("GasPressure") is not None else pressure
    if pressure is None:
        pressure=_lcm_decode_cut_field_by_id(data,0x1B)

    return {
        "ok":True,"lcmTimingSchema":LCM_TIMING_SCHEMA,"filename":filename,"note":note,"material":material,
        "thicknessMm":thickness,"profile":profile,"radiusMm":radius_mm,
        "gas":gas,"nozzle":nozzle,"gasPressureBar":pressure,
        "cutSpeedMMin":speed_m_min,"workSpeedMmS":work_speed_mm_s,
        "focusMm":focus,"cutGasMachine":cut_gas_machine,
        "cutParameters":cut_parameters,"pierceParameters":pierce_parameters,
        "cornerParameters":corner_parameters,"otherParameters":other_parameters,
        "advancedParameters":advanced_parameters,"source":"LCM import",
        "contentBase64":base64.b64encode(content).decode("ascii"),
        "originalFilename":filename,
    }



def _lcm_refresh_state_timing(state: dict) -> dict:
    """Refresh old parsed timing from each layer's OWN embedded LCM.

    A new Dropbox revision is not required to correct an earlier parser bug.
    Only parsed timing metadata changes; the source binary, profile, price,
    gas settings and all quote snapshots are left untouched.
    """
    settings_obj=state.get("settings") if isinstance(state,dict) else None
    layers=settings_obj.get("cutLayerPresets") if isinstance(settings_obj,dict) else None
    if not isinstance(layers,list):
        return state
    parsed_cache={}
    for layer in layers:
        if not isinstance(layer,dict) or not layer.get("contentBase64"):
            continue
        cut=layer.get("cutParameters") or {}
        pp=layer.get("pierceParameters") or {}
        if (cut.get("timingSchema")==LCM_TIMING_SCHEMA and
            all((pp.get(f"stage{i}") or {}).get("timingSchema")==LCM_TIMING_SCHEMA for i in (1,2,3))):
            continue
        raw=str(layer["contentBase64"])
        try:
            if len(raw)>14*1024*1024:
                raise ValueError("LCM te groot voor herinlezen.")
            if raw not in parsed_cache:
                content=base64.b64decode(raw,validate=True)
                parsed_cache[raw]=_parse_fs_material_lcm(content,str(layer.get("originalFilename") or "layer.lcm"))
            parsed=parsed_cache[raw]
            target=layer.setdefault("cutParameters",{})
            for key in ("pierceTimeMs","pierceTimeField","timingSchema","timeStayMs",
                        "timeFocusMs","timeBlowMs","timeFollowMs","cycleTimeMs","laserOffDelayMs"):
                target[key]=parsed["cutParameters"].get(key)
            target_pp=layer.setdefault("pierceParameters",{})
            target_pp["type"]=parsed["pierceParameters"]["type"]
            for i in (1,2,3):
                stage=target_pp.setdefault(f"stage{i}",{})
                source=parsed["pierceParameters"][f"stage{i}"]
                for key in ("pierceTimeMs","pierceTimeField","timingSchema","timeStayMs","timeMs","laserOffGasOnMs"):
                    stage[key]=source.get(key)
            layer["lcmTimingSchema"]=LCM_TIMING_SCHEMA
            layer.pop("lcmTimingReadError",None)
        except Exception as exc:
            # Never pretend that an old FreqEnd/CycleTime mapping is verified.
            layer["lcmTimingReadError"]=f"LCM-piercetijd niet herlezen: {type(exc).__name__}"
    return state


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



def _lcm_clean_machine_number(value, decimals: int = 6):
    """
    Remove harmless floating-point conversion noise from machine values.
    Example: 233.33333 mm/s * 0.06 may decode as 13.9999998 m/min -> 14.0.
    This does not coarsely round genuine values; it only snaps values that are
    already within 1e-6 of a clean decimal representation.
    """
    if value is None:
        return None
    n=float(value)
    if not math.isfinite(n):
        return n

    for d in (0,1,2,3,4,5,decimals):
        factor=10 ** d
        rounded=round(n * factor) / factor
        if abs(n-rounded) <= 1e-6:
            return rounded
    return round(n,decimals)


def _lcm_number_text(value: float, decimals: int=6) -> bytes:
    n=_lcm_clean_machine_number(value,decimals)
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
    """
    Write an exact named numeric field without scanning into the next field.

    Supported real FSMATERIAL forms:
      LEN NAME 00 00 LEN ASCII
      LEN NAME REF 00 01 00 00

    Reference-backed values are converted locally to inline ASCII so changing
    PTConsA/PTCurrent/etc. cannot overwrite the following field name.
    """
    raw=_lcm_number_text(value)
    start=0
    candidates=[]
    while True:
        pos=data.find(name,start)
        if pos<0:
            break
        if pos>0 and data[pos-1]==len(name):
            candidates.append(pos)
        start=pos+1

    if not candidates:
        return data,False

    # Prefer the payload record after the table.
    for pos in reversed(candidates):
        cursor=pos+len(name)

        # Inline number: NAME 00 00 LEN ASCII
        if cursor+3<=len(data) and data[cursor:cursor+2]==b"\x00\x00":
            ln=data[cursor+2]
            v0=cursor+3
            if 0 < ln <= 64 and v0+ln<=len(data):
                old=data[v0:v0+ln]
                if all(32<=c<127 for c in old):
                    return data[:cursor+2]+bytes([len(raw)])+raw+data[v0+ln:],True

        # Reference number: NAME REF 00 01 00 00
        if cursor+5<=len(data) and data[cursor+1:cursor+5]==b"\x00\x01\x00\x00":
            replacement=b"\x00\x00"+bytes([len(raw)])+raw
            return data[:cursor]+replacement+data[cursor+5:],True

    return data,False


def _lcm_replace_named_scalar_code(data: bytes, name: bytes, code: int) -> tuple[bytes,bool]:
    """
    FSMATERIAL named scalar:
      LEN NAME <code> 00 01 00 00

    Search exact field-name occurrences. This avoids e.g. PTFreq accidentally
    matching the tail of UsePtFreq.
    """
    start=0
    candidates=[]
    while True:
        pos=data.find(name,start)
        if pos<0:
            break
        exact_prefix=(pos>0 and data[pos-1]==len(name))
        if exact_prefix:
            cursor=pos+len(name)
            marker=data.find(b"\x00\x01\x00\x00",cursor,min(len(data),cursor+12))
            if marker>=0 and marker>cursor:
                candidates.append((pos,marker))
        start=pos+1

    if not candidates:
        return data,False

    # Payload record is normally the last exact occurrence after the field table.
    _,marker=candidates[-1]
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



def _lcm_laser_field_index(data: bytes, field_name: str) -> int | None:
    """Resolve a laser-block field index from the FSMATERIAL field table."""
    _, field_map = _lcm_value_table_and_laser_field_map(data)
    wanted=str(field_name or "")
    for field_index, name in field_map.items():
        if str(name)==wanted:
            return int(field_index)
    return None


def _lcm_reference_code_for_number(data: bytes, value: float, tol: float=1e-9) -> int | None:
    """
    Return the value-table reference code that decodes to `value`.
    Used for enum/reference fields such as Cut GasType.
    """
    table,_=_lcm_value_table_and_laser_field_map(data)
    target=float(value)
    for code,raw in table.items():
        try:
            if abs(float(str(raw).replace(",","."))-target)<=tol:
                return int(code)
        except Exception:
            continue
    return None


def _lcm_replace_cut_number_by_name(data: bytes, field_name: str, value: float) -> tuple[bytes,bool]:
    """
    Write a numeric Cut field by its real FSMATERIAL field name.

    The machine format permits both value-table references and inline ASCII
    numerics inside a laser block. To avoid changing a shared value-table item,
    a reference-backed field is converted locally to the inline numeric form.
    """
    field_index=_lcm_laser_field_index(data,field_name)
    if field_index is None:
        return data,False

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

    raw=_lcm_number_text(float(value))

    # Existing inline number: 00 01 ID 00 00 LEN ASCII
    if pos+6<=len(data) and data[pos+3:pos+5]==b"\x00\x00":
        ln=data[pos+5]
        v0=pos+6
        if 0 < ln <= 64 and v0+ln<=len(data):
            return data[:pos+5]+bytes([len(raw)])+raw+data[v0+ln:],True

    # Existing value-table reference: 00 01 ID REF
    # Replace only this field with inline numeric encoding.
    replacement=marker+b"\x00\x00"+bytes([len(raw)])+raw
    return data[:pos]+replacement+data[pos+4:],True


def _lcm_replace_cut_reference_number_by_name(data: bytes, field_name: str, numeric_value: float) -> tuple[bytes,bool]:
    """
    Write a Cut enum/reference field to an already existing numeric value-table
    entry, preserving FSMATERIAL's reference encoding.
    """
    field_index=_lcm_laser_field_index(data,field_name)
    if field_index is None:
        return data,False
    ref_code=_lcm_reference_code_for_number(data,numeric_value)
    if ref_code is None:
        return data,False
    return _lcm_replace_cut_scalar_code(data,field_index,ref_code)


def _lcm_machine_gas_number(value) -> int:
    text=str(value or "").strip().upper().replace("₂","2")
    if text in ("O2","OXYGEN","ZUURSTOF","2"):
        return 2
    if text in ("N2","NITROGEN","STIKSTOF","3"):
        return 3
    raise ValueError(f"Onbekend Cut Gas '{value}'. Gebruik N2 of O2.")


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



def _lcm_strict_desired(desired: dict) -> dict:
    """
    Whitelist voor machine-LCM wijzigingen.
    Onbekende velden worden NIET stil genegeerd: de build stopt.
    Daardoor kan een frontendwijziging nooit ongemerkt extra LCM-velden herschrijven.
    Gas pressure / corner pressure zijn expliciet geen schrijfbare layerparameters.
    """
    desired=dict(desired or {})
    allowed_top={
        "radiusMm","cutSpeedMMin","cutGasMachine","focusMm",
        "cutParameters","cornerParameters","note"
    }
    unknown_top=sorted(set(desired)-allowed_top)
    if unknown_top:
        raise ValueError("Niet-toegestane LCM-velden: "+", ".join(unknown_top))

    cut=dict(desired.get("cutParameters") or {})
    allowed_cut={"cutHeightMm","peakPowerPct","dutyCyclePct","frequencyHz"}
    unknown_cut=sorted(set(cut)-allowed_cut)
    if unknown_cut:
        raise ValueError("Niet-toegestane Cut-velden: "+", ".join(unknown_cut))

    corner=dict(desired.get("cornerParameters") or {})
    allowed_corner={
        "enabled","followHeightOffsetMm",
        "peakPowerEnabled","peakPowerPct",
        "dutyCycleEnabled","dutyCyclePct",
        "frequencyEnabled","frequencyHz",
        "cornerSpeed","defineCornerDegPerMm","limitBAxisSpeed",
        "bAxisSpeedRpm","bAxisAccelerationRadS2","bAxisAcceleration"
    }
    unknown_corner=sorted(set(corner)-allowed_corner)
    if unknown_corner:
        raise ValueError("Niet-toegestane Corner-velden: "+", ".join(unknown_corner))

    # Gasdruk mag nooit via een layerwriter worden aangepast.
    forbidden_pressure={
        "gasPressureBar","pressureBar","cornerPressureBar","cornerPressureEnabled"
    }
    if forbidden_pressure & set(desired):
        raise ValueError("Gasdruk is geen softwarematige layerparameter.")
    if forbidden_pressure & set(cut):
        raise ValueError("Gasdruk is geen softwarematige Cut-layerparameter.")
    if forbidden_pressure & set(corner):
        raise ValueError("Corner gasdruk is geen softwarematige layerparameter.")

    clean={k:desired.get(k) for k in allowed_top if k in desired}
    clean["cutParameters"]={k:cut.get(k) for k in allowed_cut if k in cut}
    clean["cornerParameters"]={k:corner.get(k) for k in allowed_corner if k in corner}
    return clean


def _lcm_assert_preserved(reference_parsed: dict, parsed: dict, desired: dict) -> None:
    """
    Controleert parser-bekende velden die NIET door deze build gewijzigd mogen worden.
    Het binaire bestand wordt nog steeds vanuit de originele payload gekloond;
    deze check detecteert daarnaast onbedoelde semantische wijzigingen.
    """
    def same(a,b,tol=1e-9):
        if isinstance(a,(int,float)) or isinstance(b,(int,float)):
            try:
                if a is None and b is None:
                    return True
                return a is not None and b is not None and abs(float(a)-float(b))<=tol
            except Exception:
                return a==b
        return a==b

    # Nozzle wordt fysiek niet softwarematig geschreven; hij wordt uit Note/naam afgeleid.
    # Gas pressure, pierce, other en advanced instellingen moeten inhoudelijk gelijk blijven.
    if not same(reference_parsed.get("gasPressureBar"),parsed.get("gasPressureBar"),0.02):
        raise ValueError("Veiligheidscontrole: gasdruk veranderde onverwacht.")

    for group in ("pierceParameters","otherParameters","advancedParameters"):
        before=reference_parsed.get(group) or {}
        after=parsed.get(group) or {}
        if before!=after:
            raise ValueError(f"Veiligheidscontrole: niet-bewerkbare {group} veranderden onverwacht.")

    # Interne Note moet exact overeenkomen met de gevraagde slimme layernaam.
    wanted_note=str(desired.get("note") or "").strip()
    if wanted_note and str(parsed.get("note") or "").strip()!=wanted_note:
        raise ValueError("Validatie mislukt voor Layer Note.")


def _build_machine_lcm(reference_content: bytes, desired: dict, filename: str) -> tuple[bytes,dict,list[str]]:
    """
    Maakt een echte machine-LCM door één echte referentielayer te klonen.

    Alleen bewezen velden worden aangepast. Gas pressure wordt nooit gewijzigd.
    Onbekende binaire instellingen blijven exact uit de referentielayer komen.
    """
    desired=_lcm_strict_desired(desired)
    reference_parsed=_parse_fs_material_lcm(reference_content,filename="reference.lcm")
    target_radius=float(desired.get("radiusMm") or 0)
    ref_radius=float(reference_parsed.get("radiusMm") or 0)

    # v695: voor rechthoekige/vierkante kokers mag een bewezen nabije layer
    # als technische referentie dienen. Radius en kokermaat worden in de app
    # meegenomen in de leerscore; de writer verandert alleen expliciete,
    # round-trip controleerbare machineparameters.
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
        # Field index differs between real FSMATERIAL versions; resolve by name.
        payload,ok=_lcm_replace_cut_number_by_name(payload,"Focus",float(focus))
        require(ok,"Focus Pos")

    cut_gas=desired.get("cutGasMachine")
    if cut_gas is None:
        cut_gas=desired.get("gas")
    if cut_gas is not None:
        gas_number=_lcm_machine_gas_number(cut_gas)
        payload,ok=_lcm_replace_cut_reference_number_by_name(payload,"GasType",gas_number)
        require(ok,"Cut Gas")

    cut=dict(desired.get("cutParameters") or {})

    # Alleen Cut Height hoort bij de vereenvoudigde kokereditor.
    # Lift Height blijft bewust onaangeroerd.
    if cut.get("liftHeightMm") is not None:
        raise ValueError(
            "Lift Height hoort niet bij de vereenvoudigde kokerlayer en wordt niet herschreven."
        )

    if cut.get("cutHeightMm") is not None:
        # v735: probeer eerst de veldtabel-route; sommige echte FSMATERIAL-versies
        # coderen Height echter als een exact named numeric record. Gebruik dan
        # dezelfde veilige named-field writer als fallback. Round-trip validatie
        # hieronder blijft altijd verplicht, dus er wordt niets blind geschreven.
        payload_before=payload
        payload,ok=_lcm_replace_cut_number_by_name(
            payload,"Height",float(cut["cutHeightMm"])
        )
        if not ok:
            payload=payload_before
            payload,ok=_lcm_replace_named_number(
                payload,b"Height",float(cut["cutHeightMm"])
            )
        require(ok,"Cut Height")

    if cut.get("peakPowerPct") is not None:
        peak=float(cut["peakPowerPct"])
        if not 0<=peak<=100:
            raise ValueError("Cut Peak Power moet tussen 0 en 100 liggen.")
        payload,ok=_lcm_replace_cut_number_by_name(payload,"LaserCurrent",peak)
        require(ok,"Cut Peak Power")

    if cut.get("dutyCyclePct") is not None:
        duty=float(cut["dutyCyclePct"])
        if not 0<=duty<=100:
            raise ValueError("Cut Duty Cycle moet tussen 0 en 100 liggen.")
        # FSMATERIAL PwmRatio is stored as 0..1 in the real layers.
        payload,ok=_lcm_replace_cut_number_by_name(payload,"PwmRatio",duty/100.0)
        require(ok,"Cut Duty Cycle")

    if cut.get("frequencyHz") is not None:
        payload,ok=_lcm_replace_cut_reference_number_by_name(
            payload,"PwmFreq",float(cut["frequencyHz"])
        )
        if not ok:
            raise ValueError(
                f"Cut Frequency {cut['frequencyHz']} Hz bestaat niet in de value-table van deze referentielayer."
            )
        require(ok,"Cut Frequency")

    # CORNER — alleen bewezen named/scalar velden.
    corner=dict(desired.get("cornerParameters") or {})

    if "enabled" in corner:
        # Hoofdschakelaar "Enable corner technique".
        # In de echte FSMATERIAL/LCM is deze optie gemapt op UsePTAdjust.
        # False schakelt Corner Technique daadwerkelijk uit; de onderliggende
        # waarden mogen in het bestand blijven staan maar zijn dan niet actief.
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePTAdjust",_lcm_enabled_code(corner["enabled"])
        )
        require(ok,"Corner enabled")

    if corner.get("followHeightOffsetMm") is not None:
        payload,ok=_lcm_replace_named_number(
            payload,b"PTFollowHPlus",float(corner["followHeightOffsetMm"])
        )
        require(ok,"Corner Follow height offset")

    # Gasdruk/corner-pressure bewust volledig onaangeroerd.

    if "peakPowerEnabled" in corner:
        payload,ok=_lcm_replace_named_scalar_code(
            payload,b"UsePtCurrent",_lcm_enabled_code(corner["peakPowerEnabled"])
        )
        require(ok,"Corner Peak power toggle")

    if corner.get("peakPowerPct") is not None:
        peak=float(corner["peakPowerPct"])
        if not 0<=peak<=100:
            raise ValueError("Corner Peak Power moet tussen 0 en 100 liggen.")
        payload,ok=_lcm_replace_named_number(payload,b"PTCurrent",peak)
        require(ok,"Corner Peak Power")

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
        code=_lcm_reference_code_for_number(payload,float(corner["frequencyHz"]))
        if code is None:
            raise ValueError(
                f"Corner Frequency {corner['frequencyHz']} Hz bestaat niet in de value-table van deze referentielayer."
            )
        payload,ok=_lcm_replace_named_scalar_code(payload,b"PTFreq",code)
        require(ok,"Corner Frequency")

    # v966: Corner Speed is in TubePro een percentage van de normale Cut Speed.
    # 40 betekent dus 40% van de rechte snijsnelheid door de radius/hoek.
    if corner.get("cornerSpeed") is not None:
        corner_speed=float(corner["cornerSpeed"])
        if not 0 < corner_speed <= 100:
            raise ValueError("Corner Speed moet groter dan 0 en maximaal 100% zijn.")
        payload,ok=_lcm_replace_named_number(payload,b"PTCornerSpeed",corner_speed)
        require(ok,"Corner Speed")

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

    b_axis_accel=corner.get("bAxisAccelerationRadS2")
    if b_axis_accel is None:
        b_axis_accel=corner.get("bAxisAcceleration")
    if b_axis_accel is not None:
        payload,ok=_lcm_replace_named_number(payload,b"PTConsA",float(b_axis_accel))
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
    if cut_gas is not None:
        parsed_gas=str(parsed.get("cutGasMachine") or "").upper().replace("₂","2")
        wanted_gas="O2" if _lcm_machine_gas_number(cut_gas)==2 else "N2"
        if parsed_gas!=wanted_gas:
            raise ValueError("Validatie mislukt voor Cut Gas.")
    if cut.get("cutHeightMm") is not None and not close_num(
        (parsed.get("cutParameters") or {}).get("cutHeightMm"),cut["cutHeightMm"],0.02
    ):
        raise ValueError("Validatie mislukt voor Cut Height.")
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

    for requested_key,parsed_key,label in (
        ("enabled","enabled","Corner enabled"),
        ("peakPowerEnabled","peakPowerEnabled","Corner Peak power toggle"),
        ("dutyCycleEnabled","dutyCycleEnabled","Corner Duty cycle toggle"),
        ("frequencyEnabled","frequencyEnabled","Corner Frequency toggle"),
        ("limitBAxisSpeed","limitBAxisSpeed","Limit B-axis speed"),
    ):
        if requested_key in corner and bool(pc.get(parsed_key))!=bool(corner.get(requested_key)):
            raise ValueError(f"Validatie mislukt voor {label}.")

    if corner.get("peakPowerPct") is not None and not close_num(
        pc.get("peakPowerPct"),corner["peakPowerPct"],0.2
    ):
        raise ValueError("Validatie mislukt voor Corner Peak Power.")
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
    if corner.get("cornerSpeed") is not None and not close_num(
        pc.get("cornerSpeed"),corner["cornerSpeed"],0.05
    ):
        raise ValueError("Validatie mislukt voor Corner Speed.")
    if corner.get("defineCornerDegPerMm") is not None and not close_num(
        pc.get("defineCornerDegPerMm"),corner["defineCornerDegPerMm"],0.002
    ):
        raise ValueError("Validatie mislukt voor Define corner.")
    if corner.get("bAxisSpeedRpm") is not None and not close_num(
        pc.get("bAxisSpeedRpm"),corner["bAxisSpeedRpm"],0.02
    ):
        raise ValueError("Validatie mislukt voor B-axis speed.")
    if corner.get("bAxisAccelerationRadS2") is not None and not close_num(
        pc.get("bAxisAccelerationRadS2"),corner["bAxisAccelerationRadS2"],0.02
    ):
        raise ValueError("Validatie mislukt voor B-axis acceleration.")

    # Extra harde controle op alle parser-bekende NIET-bewerkbare groepen.
    _lcm_assert_preserved(reference_parsed,parsed,desired)

    return result,parsed,changed




def _machine_spec_output_text(payload: dict) -> str:
    parts=[]
    for item in payload.get("output") or []:
        if not isinstance(item,dict) or item.get("type")!="message":
            continue
        for content in item.get("content") or []:
            if isinstance(content,dict) and content.get("type")=="output_text":
                value=content.get("text")
                if value:
                    parts.append(str(value))
    return "\n".join(parts).strip()


def _machine_spec_collect_urls(value, found=None):
    if found is None:
        found=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if str(k).lower() in {"url","source_url"} and isinstance(v,str) and v.startswith(("http://","https://")):
                found.add(v.strip())
            _machine_spec_collect_urls(v,found)
    elif isinstance(value,list):
        for item in value:
            _machine_spec_collect_urls(item,found)
    return found


def _machine_spec_norm_url(raw: str) -> str:
    try:
        p=urllib.parse.urlsplit(str(raw or "").strip())
        if p.scheme not in {"http","https"} or not p.netloc:
            return ""
        path=p.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit((p.scheme.lower(),p.netloc.lower(),path,p.query,""))
    except Exception:
        return ""


def _machine_spec_extract_json(raw: str) -> dict:
    s=str(raw or "").strip()
    if s.startswith("```"):
        s=re.sub(r"^```(?:json)?\s*","",s,flags=re.I)
        s=re.sub(r"\s*```$","",s)
    start=s.find("{")
    end=s.rfind("}")
    if start<0 or end<=start:
        raise ValueError("AI response bevat geen JSON-object")
    value=json.loads(s[start:end+1])
    if not isinstance(value,dict):
        raise ValueError("AI response is geen JSON-object")
    return value


def _machine_spec_clean_field(item, allowed_urls: set[str], *, low: float, high: float):
    if not isinstance(item,dict):
        return {"value":None,"source_url":"","source_title":"","evidence":"","confidence":0}
    try:
        value=float(str(item.get("value","")).replace(",","."))
    except Exception:
        value=None
    try:
        confidence=float(item.get("confidence") or 0)
    except Exception:
        confidence=0
    source_url=str(item.get("source_url") or "").strip()
    source_title=str(item.get("source_title") or "").strip()[:180]
    evidence=str(item.get("evidence") or "").strip()[:500]
    norm=_machine_spec_norm_url(source_url)
    source_verified=(not allowed_urls) or (norm in allowed_urls)

    # Alleen concrete waarden van het exacte model met een herleidbare bron.
    if not (value is not None and math.isfinite(value) and low<=value<=high):
        value=None
    if confidence<0.65 or not source_verified:
        value=None
    return {
        "value":value,
        "source_url":source_url if source_verified else "",
        "source_title":source_title,
        "evidence":evidence,
        "confidence":max(0,min(1,confidence)),
        "source_verified":bool(source_verified),
    }


_MACHINE_SPEC_SEARCH_CACHE={}
_MACHINE_SPEC_SEARCH_TTL_S=15*60
_MACHINE_SPEC_USER_AGENT=(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36"
)


def _machine_spec_query_identity(query: str) -> tuple[str,str]:
    tokens=re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*",str(query or ""))
    brand=tokens[0] if tokens else ""
    model_candidates=[t for t in tokens if re.search(r"[A-Za-z]",t) and re.search(r"\d",t)]
    model=max(model_candidates,key=lambda x:len(re.sub(r"[^A-Za-z0-9]","",x))) if model_candidates else ""
    if not model and len(tokens)>1:
        model="".join(tokens[1:])
    norm_model=re.sub(r"[^A-Za-z0-9]","",model).upper()
    return brand,norm_model


def _machine_spec_safe_public_url(raw: str) -> str:
    try:
        p=urllib.parse.urlsplit(str(raw or "").strip())
        if p.scheme not in {"http","https"} or not p.hostname:
            return ""
        host=p.hostname.strip().lower()
        if host in {"localhost","localhost.localdomain"} or host.endswith(".local"):
            return ""
        # Letterlijke private/local IP-adressen nooit ophalen.
        try:
            import ipaddress
            ip=ipaddress.ip_address(host.strip("[]"))
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return ""
        except ValueError:
            pass
        return urllib.parse.urlunsplit((p.scheme,p.netloc,p.path,p.query,""))
    except Exception:
        return ""


def _machine_spec_http_get(url: str, *, timeout: int=14, max_bytes: int=12_000_000):
    safe=_machine_spec_safe_public_url(url)
    if not safe:
        raise ValueError("Ongeldige of niet-openbare bron-URL")
    req=urllib.request.Request(
        safe,
        headers={
            "User-Agent":_MACHINE_SPEC_USER_AGENT,
            "Accept":"text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5",
            "Accept-Language":"nl-NL,nl;q=0.8,en-US;q=0.7,en;q=0.6",
            "Cache-Control":"no-cache",
        },
        method="GET",
    )
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        final_url=_machine_spec_safe_public_url(resp.geturl())
        if not final_url:
            raise ValueError("Bron stuurde door naar een niet-openbare URL")
        content_type=str(resp.headers.get("Content-Type") or "").lower()
        length=resp.headers.get("Content-Length")
        if length:
            try:
                parsed_length=int(length)
            except (TypeError,ValueError):
                parsed_length=0
            if parsed_length>max_bytes:
                raise ValueError("Bronbestand is te groot om veilig te controleren")
        data=resp.read(max_bytes+1)
        if len(data)>max_bytes:
            raise ValueError("Bronbestand is te groot om veilig te controleren")
        return final_url,content_type,data


def _machine_spec_strip_html(raw: bytes) -> tuple[str,str]:
    text=raw.decode("utf-8",errors="replace")
    title=""
    m=re.search(r"(?is)<title[^>]*>(.*?)</title>",text)
    if m:
        title=html.unescape(re.sub(r"(?s)<[^>]+>"," ",m.group(1)))
        title=re.sub(r"\s+"," ",title).strip()[:220]
    text=re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>"," ",text)
    text=re.sub(r"(?i)<br\s*/?>|</(?:p|div|li|tr|td|th|h[1-6]|section|article)\s*>","\n",text)
    text=re.sub(r"(?s)<[^>]+>"," ",text)
    text=html.unescape(text).replace("\xa0"," ")
    lines=[]
    for line in text.splitlines():
        line=re.sub(r"[ \t\r\f\v]+"," ",line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)[:350_000],title


def _machine_spec_pdf_text(raw: bytes) -> str:
    try:
        import io
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(raw),strict=False)
        parts=[]
        total=0
        for page in reader.pages[:60]:
            try:
                value=page.extract_text() or ""
            except Exception:
                value=""
            if value:
                parts.append(value)
                total+=len(value)
            if total>=300_000:
                break
        return "\n".join(parts)[:350_000]
    except Exception:
        return ""


def _machine_spec_unwrap_ddg_url(raw: str) -> str:
    href=html.unescape(str(raw or "").strip())
    if href.startswith("//"):
        href="https:"+href
    try:
        p=urllib.parse.urlsplit(href)
        if "duckduckgo.com" in (p.hostname or "").lower():
            qs=urllib.parse.parse_qs(p.query)
            target=(qs.get("uddg") or [""])[0]
            if target:
                href=urllib.parse.unquote(target)
    except Exception:
        pass
    return _machine_spec_safe_public_url(href)


def _machine_spec_ddg_results(search_html: bytes, limit: int=10):
    source=search_html.decode("utf-8",errors="replace")
    results=[]
    # DuckDuckGo HTML gebruikt result__a voor de echte resultaatlink.
    for m in re.finditer(r"(?is)<a\b([^>]*class=[\"'][^\"']*result__a[^\"']*[\"'][^>]*)>(.*?)</a>",source):
        attrs=m.group(1)
        hm=re.search(r"(?is)href\s*=\s*[\"']([^\"']+)[\"']",attrs)
        if not hm:
            continue
        url=_machine_spec_unwrap_ddg_url(hm.group(1))
        if not url:
            continue
        title=html.unescape(re.sub(r"(?s)<[^>]+>"," ",m.group(2)))
        title=re.sub(r"\s+"," ",title).strip()[:220]
        if url not in {x["url"] for x in results}:
            results.append({"url":url,"title":title,"rank":len(results)+1})
        if len(results)>=limit:
            break
    return results


def _machine_spec_bing_results(search_html: bytes, limit: int=10):
    source=search_html.decode("utf-8",errors="replace")
    results=[]
    for m in re.finditer(r"(?is)<li[^>]*class=[\"'][^\"']*b_algo[^\"']*[\"'][^>]*>.*?<h2[^>]*>\s*<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",source):
        url=_machine_spec_safe_public_url(html.unescape(m.group(1)))
        if not url:
            continue
        title=html.unescape(re.sub(r"(?s)<[^>]+>"," ",m.group(2)))
        title=re.sub(r"\s+"," ",title).strip()[:220]
        if url not in {x["url"] for x in results}:
            results.append({"url":url,"title":title,"rank":len(results)+1})
        if len(results)>=limit:
            break
    return results


def _machine_spec_public_search(query: str, limit: int=10):
    encoded=urllib.parse.quote_plus(query)
    errors=[]
    # Eerste keuze: lichte DuckDuckGo HTML-interface, zonder API-sleutel.
    try:
        url=f"https://html.duckduckgo.com/html/?q={encoded}"
        _,_,raw=_machine_spec_http_get(url,timeout=12,max_bytes=2_500_000)
        found=_machine_spec_ddg_results(raw,limit=limit)
        if found:
            return found,"DuckDuckGo"
    except Exception as exc:
        errors.append(f"DuckDuckGo: {exc}")

    # Fallback: gewone Bing-resultatenpagina. Ook hiervoor is geen API-sleutel nodig.
    try:
        url=f"https://www.bing.com/search?q={encoded}&count={max(5,min(20,limit))}"
        _,_,raw=_machine_spec_http_get(url,timeout=12,max_bytes=2_500_000)
        found=_machine_spec_bing_results(raw,limit=limit)
        if found:
            return found,"Bing"
    except Exception as exc:
        errors.append(f"Bing: {exc}")

    raise RuntimeError("; ".join(errors) or "Geen openbare zoekresultaten ontvangen")


def _machine_spec_source_score(source: dict, brand: str, model_key: str) -> float:
    url=str(source.get("url") or "")
    title=str(source.get("title") or "")
    text=str(source.get("text") or "")
    try:
        host=(urllib.parse.urlsplit(url).hostname or "").lower()
    except Exception:
        host=""
    hay=(title+" "+url+" "+text[:30_000]).lower()
    score=0.0
    compact=re.sub(r"[^A-Za-z0-9]","",title+" "+text[:80_000]).upper()
    if model_key and model_key in compact:
        score+=5.0
    brand_l=str(brand or "").lower()
    if brand_l and (brand_l in host or brand_l in title.lower()):
        score+=2.0
    if url.lower().endswith(".pdf") or "application/pdf" in str(source.get("content_type") or ""):
        score+=1.5
    if any(k in hay for k in ("manual","datasheet","data sheet","specification","specifications","technical","parameter")):
        score+=1.2
    if any(k in host for k in ("youtube.","facebook.","instagram.","pinterest.","alibaba.","made-in-china.","ebay.")):
        score-=2.0
    return score


_MACHINE_SPEC_QUANTITY_RE=re.compile(
    r"(?P<value>\d{1,6}(?:[\.,]\d{1,5})?)\s*"
    r"(?P<unit>mm\s*/\s*s(?:\s*(?:\^?2|²))?|m\s*/\s*s(?:\s*(?:\^?2|²))?|"
    r"mm\s*/\s*min|m\s*/\s*min|mm\s*/\s*s|m\s*/\s*s|g\b|mm\b|cm\b)",
    re.I,
)


def _machine_spec_unit_norm(unit: str) -> str:
    u=str(unit or "").lower().replace(" ","").replace("²","2").replace("^","")
    return u


def _machine_spec_convert_quantity(value: float, unit: str, kind: str):
    u=_machine_spec_unit_norm(unit)
    if kind=="y_speed_mmin":
        if u=="m/min": return value
        if u=="mm/min": return value/1000.0
        if u=="mm/s": return value*60.0/1000.0
        if u=="m/s": return value*60.0
    elif kind=="z_speed_mms":
        if u=="mm/s": return value
        if u=="m/s": return value*1000.0
        if u=="m/min": return value*1000.0/60.0
        if u=="mm/min": return value/60.0
    elif kind=="accel_ms2":
        if u=="g": return value*9.80665
        if u=="m/s2": return value
        if u=="mm/s2": return value/1000.0
    elif kind=="stroke_mm":
        if u=="mm": return value
        if u=="cm": return value*10.0
    return None


_MACHINE_SPEC_FIELD_RULES={
    "machineFeedMaxMMin":{
        "kind":"y_speed_mmin","bounds":(1,500),"tolerance":0.04,
        "patterns":[
            r"\b(?:y\s*[- ]?\s*axis|axis\s*y)\b[^|]{0,55}\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid(?:\s*speed)?|feed(?:ing)?\s*speed)\b",
            r"\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid(?:\s*speed)?|feed(?:ing)?\s*speed)\b[^|]{0,55}\b(?:y\s*[- ]?\s*axis|axis\s*y)\b",
            r"\by\b[^|]{0,24}\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid)\b",
        ]
    },
    "machineFeedAccelMS2":{
        "kind":"accel_ms2","bounds":(0.1,300),"tolerance":0.08,
        "patterns":[
            r"\b(?:y\s*[- ]?\s*axis|axis\s*y)\b[^|]{0,55}\b(?:max(?:imum)?\s*)?accel(?:eration)?\b",
            r"\b(?:max(?:imum)?\s*)?accel(?:eration)?\b[^|]{0,55}\b(?:y\s*[- ]?\s*axis|axis\s*y)\b",
            r"\by\b[^|]{0,24}\baccel(?:eration)?\b",
        ]
    },
    "zAxisMaxStrokeMm":{
        "kind":"stroke_mm","bounds":(1,2000),"tolerance":0.03,
        "patterns":[
            r"\b(?:z\s*[- ]?\s*axis|axis\s*z)\b[^|]{0,55}\b(?:travel|stroke|range|movement|moving\s*range)\b",
            r"\b(?:travel|stroke|range|movement|moving\s*range)\b[^|]{0,55}\b(?:z\s*[- ]?\s*axis|axis\s*z)\b",
            r"\bz\b[^|]{0,24}\b(?:travel|stroke|range)\b",
        ]
    },
    "zAxisMaxSpeedMmS":{
        "kind":"z_speed_mms","bounds":(1,5000),"tolerance":0.04,
        "patterns":[
            r"\b(?:z\s*[- ]?\s*axis|axis\s*z)\b[^|]{0,55}\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid(?:\s*speed)?)\b",
            r"\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid(?:\s*speed)?)\b[^|]{0,55}\b(?:z\s*[- ]?\s*axis|axis\s*z)\b",
            r"\bz\b[^|]{0,24}\b(?:max(?:imum)?\s*)?(?:speed|velocity|rapid)\b",
        ]
    },
    "zAxisAccelMS2":{
        "kind":"accel_ms2","bounds":(0.1,500),"tolerance":0.08,
        "patterns":[
            r"\b(?:z\s*[- ]?\s*axis|axis\s*z)\b[^|]{0,55}\b(?:max(?:imum)?\s*)?accel(?:eration)?\b",
            r"\b(?:max(?:imum)?\s*)?accel(?:eration)?\b[^|]{0,55}\b(?:z\s*[- ]?\s*axis|axis\s*z)\b",
            r"\bz\b[^|]{0,24}\baccel(?:eration)?\b",
        ]
    },
}


def _machine_spec_extract_candidates(source: dict, field_key: str):
    rule=_MACHINE_SPEC_FIELD_RULES[field_key]
    text=str(source.get("text") or "")
    lines=[re.sub(r"\s+"," ",x).strip() for x in text.splitlines() if x.strip()]
    # Tabellen verliezen bij HTML/PDF soms celgrenzen; kijk daarom ook één regel voor/na de match.
    windows=[]
    for i,line in enumerate(lines):
        windows.append(line)
        if i+1<len(lines): windows.append(line+" | "+lines[i+1])
        if i>0: windows.append(lines[i-1]+" | "+line)
    seen=set()
    out=[]
    for window in windows:
        low_window=window.lower()
        for pat in rule["patterns"]:
            alias=re.search(pat,low_window,re.I)
            if not alias:
                continue
            quantities=[]
            for qm in _MACHINE_SPEC_QUANTITY_RE.finditer(window):
                try:
                    raw_value=float(qm.group("value").replace(",","."))
                except Exception:
                    continue
                converted=_machine_spec_convert_quantity(raw_value,qm.group("unit"),rule["kind"])
                if converted is None or not math.isfinite(converted):
                    continue
                lo,hi=rule["bounds"]
                if not (lo<=converted<=hi):
                    continue
                distance=abs(((qm.start()+qm.end())/2)-((alias.start()+alias.end())/2))
                quantities.append((distance,converted,raw_value,qm.group("unit"),qm.group(0)))
            if not quantities:
                continue
            # Technische tabellen schrijven vrijwel altijd label -> waarde. Geef
            # daarom een hoeveelheid ná het gevonden veldlabel voorrang; dit voorkomt
            # dat bij samengevoegde tabelregels bijvoorbeeld de Y-acceleratie als
            # Z-acceleratie wordt meegenomen. Alleen als er rechts niets staat, mag
            # de dichtstbijzijnde waarde links van het label gebruikt worden.
            after=[]
            for q in quantities:
                token=str(q[4])
                pos=window.find(token)
                if pos>=max(0,alias.end()-3):
                    after.append(q)
            pool=after or quantities
            pool.sort(key=lambda x:x[0])
            _,value,raw_value,raw_unit,raw_token=pool[0]
            marker=(round(value,6),window[:220])
            if marker in seen:
                continue
            seen.add(marker)
            evidence=window[:420]
            if abs(value-raw_value)>1e-9 or _machine_spec_unit_norm(raw_unit) not in {"m/min","m/s2","mm","mm/s"}:
                evidence+=f" · omgerekend uit {raw_token.strip()}"
            source_score=float(source.get("score") or 0)
            confidence=max(0.65,min(0.97,0.68+0.035*source_score))
            out.append({
                "value":value,
                "source_url":source.get("url") or "",
                "source_title":source.get("title") or source.get("url") or "",
                "evidence":evidence,
                "confidence":confidence,
                "source_verified":True,
                "source_score":source_score,
            })
    return out


def _machine_spec_resolve_candidates(field_key: str, candidates: list[dict]):
    if not candidates:
        return {"value":None,"source_url":"","source_title":"","evidence":"","confidence":0,"source_verified":False},None
    # Zelfde URL/waarde maar één keer meenemen.
    unique=[]
    seen=set()
    for c in sorted(candidates,key=lambda x:(float(x.get("source_score") or 0),float(x.get("confidence") or 0)),reverse=True):
        marker=(_machine_spec_norm_url(c.get("source_url") or ""),round(float(c.get("value") or 0),5))
        if marker in seen: continue
        seen.add(marker); unique.append(c)
    best=unique[0]
    tol=float(_MACHINE_SPEC_FIELD_RULES[field_key]["tolerance"])
    conflicts=[]
    for other in unique[1:]:
        bv=float(best["value"]); ov=float(other["value"])
        rel=abs(bv-ov)/max(abs(bv),abs(ov),1e-9)
        if rel>tol and float(other.get("source_score") or 0)>=float(best.get("source_score") or 0)-1.5:
            conflicts.append(other)
    if conflicts:
        values=[best]+conflicts[:3]
        summary="; ".join(
            f"{round(float(c['value']),4)} ({urllib.parse.urlsplit(c.get('source_url') or '').hostname or 'bron'})"
            for c in values
        )
        return {"value":None,"source_url":"","source_title":"","evidence":"","confidence":0,"source_verified":False},f"Tegenstrijdige bronnen voor {field_key}: {summary}. Waarde daarom niet automatisch ingevuld."
    return {
        "value":round(float(best["value"]),6),
        "source_url":str(best.get("source_url") or ""),
        "source_title":str(best.get("source_title") or "")[:180],
        "evidence":str(best.get("evidence") or "")[:500],
        "confidence":max(0,min(1,float(best.get("confidence") or 0))),
        "source_verified":True,
    },None


def _machine_spec_fetch_sources(query: str):
    brand,model_key=_machine_spec_query_identity(query)
    if not model_key:
        raise HTTPException(status_code=400,detail="Vul naast het merk ook het exacte machinemodel in.")
    searches=[
        f'"{query}" manual specifications Y axis Z axis',
        f'"{query}" Y axis speed acceleration Z axis travel speed acceleration',
        f'"{query}" datasheet pdf',
    ]
    merged=[]
    providers=[]
    seen=set()
    for q in searches:
        try:
            found,provider=_machine_spec_public_search(q,limit=8)
            providers.append(provider)
        except Exception:
            continue
        for item in found:
            url=_machine_spec_norm_url(item.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(item)
        if len(merged)>=14:
            break
    if not merged:
        raise HTTPException(
            status_code=502,
            detail="De openbare webzoekopdracht leverde geen resultaten op. Controleer de internettoegang van de server en probeer opnieuw."
        )

    # Eerst waarschijnlijk relevante/technische resultaten ophalen.
    brand_l=brand.lower()
    def prelim(item):
        hay=(str(item.get("title") or "")+" "+str(item.get("url") or "")).lower()
        s=0
        if brand_l and brand_l in hay: s+=2
        if str(item.get("url") or "").lower().endswith(".pdf"): s+=2
        if any(k in hay for k in ("manual","datasheet","spec","parameter","technical")): s+=1
        return (-s,int(item.get("rank") or 99))
    merged.sort(key=prelim)

    sources=[]
    fetch_errors=[]
    for item in merged[:10]:
        url=item.get("url") or ""
        try:
            final_url,content_type,raw=_machine_spec_http_get(url,timeout=14,max_bytes=12_000_000)
            title=str(item.get("title") or "")
            if "pdf" in content_type or final_url.lower().endswith(".pdf") or raw[:5]==b"%PDF-":
                page_text=_machine_spec_pdf_text(raw)
            else:
                page_text,page_title=_machine_spec_strip_html(raw)
                if page_title: title=page_title
            if not page_text:
                continue
            compact=re.sub(r"[^A-Za-z0-9]","",title+" "+page_text[:120_000]).upper()
            exact=bool(model_key and model_key in compact)
            source={
                "url":final_url,
                "title":title[:220] or final_url,
                "content_type":content_type,
                "text":page_text,
                "exact_model":exact,
            }
            source["score"]=_machine_spec_source_score(source,brand,model_key)
            sources.append(source)
        except Exception as exc:
            fetch_errors.append(f"{urllib.parse.urlsplit(url).hostname or url}: {exc}")

    exact_sources=[s for s in sources if s.get("exact_model")]
    if not exact_sources:
        raise HTTPException(
            status_code=404,
            detail="Er zijn wel zoekresultaten gevonden, maar geen gecontroleerde bron waarin het exacte machinemodel duidelijk voorkomt. Er worden daarom geen waarden ingevuld."
        )
    exact_sources.sort(key=lambda s:float(s.get("score") or 0),reverse=True)
    return brand,model_key,exact_sources,sorted(set(providers)),fetch_errors


@app.post("/api/machine-specs/search")
async def machine_specs_search(request: Request):
    try:
        body=await request.json()
    except Exception:
        body={}
    query=re.sub(r"\s+"," ",str(body.get("query") or "")).strip()
    if len(query)<3:
        raise HTTPException(status_code=400,detail="Vul merk en exact machinemodel in.")
    if len(query)>160:
        raise HTTPException(status_code=400,detail="Machinemodel is te lang.")

    cache_key=query.casefold()
    cached=_MACHINE_SPEC_SEARCH_CACHE.get(cache_key)
    if cached and time.time()-float(cached.get("at") or 0)<_MACHINE_SPEC_SEARCH_TTL_S:
        return cached["result"]

    try:
        brand,model_key,sources,providers,fetch_errors=_machine_spec_fetch_sources(query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502,detail=f"Online zoeken zonder API-sleutel is mislukt: {exc}")

    fields={}
    warnings=[]
    for key in _MACHINE_SPEC_FIELD_RULES:
        candidates=[]
        for source in sources:
            candidates.extend(_machine_spec_extract_candidates(source,key))
        field,conflict=_machine_spec_resolve_candidates(key,candidates)
        fields[key]=field
        if conflict:
            warnings.append(conflict)

    found=sum(1 for f in fields.values() if isinstance(f.get("value"),(int,float)))
    if found<5:
        warnings.append(
            f"{5-found} van de 5 gevraagde machinewaarden zijn leeg gelaten omdat ze niet expliciet of niet eenduidig in de gecontroleerde bronnen voor dit exacte model stonden."
        )
    if fetch_errors and len(sources)<3:
        warnings.append("Niet alle gevonden bronpagina's konden door de server worden geopend; alleen daadwerkelijk gecontroleerde pagina's zijn gebruikt.")

    result={
        "query":query,
        "machine_name":query,
        "exact_model_match":True,
        "fields":fields,
        "warnings":warnings[:8],
        "provider":"Openbare webzoekopdracht (geen API-sleutel)",
        "search_engines":providers,
        "sources_checked":len(sources),
        "sources":[
            {"url":s["url"],"title":s["title"],"score":round(float(s.get("score") or 0),2)}
            for s in sources[:8]
        ],
    }
    _MACHINE_SPEC_SEARCH_CACHE[cache_key]={"at":time.time(),"result":result}
    return result

@app.post("/api/cut-layer/build-machine")
# v695 — simplified koker layer writer: exact editable Cut/Corner set with round-trip validation.
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
        desired=_lcm_strict_desired(desired)
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
    - ronde, vierkante (met hoekradius), ovale en sleufvormige uitsparingen
    - per bewerking door één wand of beide tegenoverliggende wanden
    - overlappende bewerkingen worden CAD-technisch eerst samengevoegd tot één
      gezamenlijke snijvolume, zodat de uiteindelijke STEP één open contour
      krijgt waar vormen elkaar overlappen.

    v934 merged contour:
    De oorspronkelijke parametrische vormen blijven exact (cirkels blijven
    cirkels, sleuven blijven echte sleuven). De server rasteriseert de contour
    dus niet: hij verenigt de echte CadQuery cutters vóórdat materiaal uit de
    koker wordt gesneden. Dit geeft dezelfde samengestelde opening als de
    frontend, maar zonder verlies van geometrische nauwkeurigheid.

    v774 performance:
    Het STEP-bestand bevat bewust slechts ÉÉN representatieve productie-body
    per exacte productvariant. Het verkochte aantal staat in de bestandsnaam
    en wordt door de frontend/offerte bewaard. Zo worden identieke booleans
    bij grote aantallen niet tientallen of honderden keren opnieuw uitgevoerd.
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
    quantity = max(1, int(data.get("quantity") or 1))
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

    normalized_pieces = []
    for piece in pieces:
        piece = piece or {}
        try:
            length = float(piece.get("lengthMm") or 0)
        except Exception:
            length = 0
        if length <= 0:
            continue

        operations = []
        for raw_op in (piece.get("operations") or []):
            raw_op = raw_op or {}
            op_type = str(raw_op.get("type") or "hole").lower()
            if op_type not in {"hole", "square", "oval", "slot"}:
                continue
            try:
                diameter_mm = float(raw_op.get("diameterMm") or 0)
                width_mm = float(raw_op.get("widthMm") or 0)
                operation_length_mm = float(raw_op.get("lengthMm") or 0)
                corner_radius_mm = float(raw_op.get("cornerRadiusMm") or 0)
                position_mm = float(raw_op.get("positionMm") or 0)
                offset_mm = float(raw_op.get("offsetMm") or 0)
            except Exception:
                continue

            face = str(raw_op.get("face") or "top").lower()
            if face not in {"top", "bottom", "left", "right"}:
                face = "top"
            through = "both" if str(raw_op.get("through") or "single").lower() == "both" else "single"

            if op_type == "hole":
                width_mm = diameter_mm
                operation_length_mm = diameter_mm
            elif op_type == "square":
                size_mm = width_mm if width_mm > 0 else operation_length_mm
                width_mm = size_mm
                operation_length_mm = size_mm
                corner_radius_mm = max(0.0, min(corner_radius_mm, size_mm / 2.0))
            elif op_type == "slot":
                operation_length_mm = max(operation_length_mm, width_mm)

            if width_mm > 0 and operation_length_mm > 0 and 0 < position_mm < length:
                operations.append({
                    "type": op_type,
                    "diameterMm": diameter_mm,
                    "widthMm": width_mm,
                    "lengthMm": operation_length_mm,
                    "cornerRadiusMm": corner_radius_mm,
                    "positionMm": position_mm,
                    "offsetMm": offset_mm,
                    "face": face,
                    "through": through,
                })

        normalized_pieces.append({"lengthMm": length, "operations": operations})

    # v774: één body is voldoende voor productie. Oudere frontends kunnen nog
    # meerdere identieke pieces sturen; neem daarom defensief alleen de eerste.
    # Nieuwe frontend v820 stuurt sowieso slechts één representatief piece.
    if normalized_pieces:
        normalized_pieces = [normalized_pieces[0]]

    lengths = [p["lengthMm"] for p in normalized_pieces]

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

        def _operation_cutter_on_plane(
            op_type: str,
            operation_length_mm: float,
            width_mm: float,
            corner_radius_mm: float,
            plane: cq.Plane,
            travel: float,
        ):
            wp = cq.Workplane(plane)

            if op_type == "hole":
                return wp.circle(width_mm / 2.0).extrude(travel)

            if op_type == "square":
                radius = max(
                    0.0,
                    min(
                        float(corner_radius_mm or 0),
                        operation_length_mm / 2.0,
                        width_mm / 2.0,
                    ),
                )
                if radius > 1e-6:
                    sketch = cq.Sketch().rect(
                        operation_length_mm,
                        width_mm,
                    ).vertices().fillet(radius)
                    return wp.placeSketch(sketch).extrude(travel)
                return wp.rect(operation_length_mm, width_mm).extrude(travel)

            if op_type == "oval":
                return wp.ellipse(
                    operation_length_mm / 2.0,
                    width_mm / 2.0,
                ).extrude(travel)

            if op_type == "slot":
                return wp.slot2D(
                    operation_length_mm,
                    width_mm,
                    0,
                ).extrude(travel)

            raise ValueError(f"Onbekende profielbewerking: {op_type}")

        def apply_profile_operations(tube, length: float, operations: list):
            """
            Bouw eerst alle exacte CAD-cutters en verenig die daarna vóór de cut.

            Daardoor worden overlappende gaten/sleuven/vierkante uitsparingen
            geometrisch één opening. De resulterende STEP bevat dus niet langer
            een interne contourlijn waar twee bewerkingen elkaar overlappen.

            Belangrijk: dit is een echte CAD-union van de parametrische vormen,
            geen polygon/raster-benadering uit de browser.
            """
            if not operations:
                return tube

            cross_w = diameter if is_round else outer_w
            cross_h = diameter if is_round else outer_h
            cutters = []

            for op in operations:
                op_type = str(op.get("type") or "hole").lower()
                if op_type not in {"hole", "square", "oval", "slot"}:
                    continue

                op_length = float(op.get("lengthMm") or 0)
                op_width = float(op.get("widthMm") or 0)
                corner_radius = float(op.get("cornerRadiusMm") or 0)
                pos = float(op.get("positionMm") or 0)
                offset = float(op.get("offsetMm") or 0)
                face = str(op.get("face") or "top").lower()
                through = str(op.get("through") or "single").lower()

                if op_type == "hole":
                    d = float(op.get("diameterMm") or op_width or 0)
                    op_length = d
                    op_width = d

                if not (op_length > 0 and op_width > 0 and 0 < pos < length):
                    continue

                extra = 2.0
                if face in {"top", "bottom"}:
                    travel = cross_h + extra * 2.0 if through == "both" else (cross_h / 2.0 + extra if is_round else wall + extra * 2.0)
                    if face == "top":
                        origin = (offset, cross_h / 2.0 + extra, pos)
                        normal = (0, -1, 0)
                    else:
                        origin = (offset, -cross_h / 2.0 - extra, pos)
                        normal = (0, 1, 0)
                    plane = cq.Plane(
                        origin=cq.Vector(*origin),
                        xDir=cq.Vector(0, 0, 1),
                        normal=cq.Vector(*normal),
                    )
                else:
                    travel = cross_w + extra * 2.0 if through == "both" else (cross_w / 2.0 + extra if is_round else wall + extra * 2.0)
                    if face == "right":
                        origin = (cross_w / 2.0 + extra, offset, pos)
                        normal = (-1, 0, 0)
                    else:
                        origin = (-cross_w / 2.0 - extra, offset, pos)
                        normal = (1, 0, 0)
                    plane = cq.Plane(
                        origin=cq.Vector(*origin),
                        xDir=cq.Vector(0, 0, 1),
                        normal=cq.Vector(*normal),
                    )

                cutter = _operation_cutter_on_plane(
                    op_type,
                    op_length,
                    op_width,
                    corner_radius,
                    plane,
                    travel,
                )
                cutters.append(cutter)

            if not cutters:
                return tube

            # Exacte CAD-union. Bij overlappende vormen verdwijnt hierdoor de
            # interne overlaprand; losse vormen blijven als losse solids binnen
            # dezelfde compound/union bestaan en worden nog steeds correct
            # uitgesneden.
            try:
                merged_cutter = cutters[0]
                for cutter in cutters[1:]:
                    merged_cutter = merged_cutter.union(cutter)
                try:
                    merged_cutter = merged_cutter.clean()
                except Exception:
                    pass
                return tube.cut(merged_cutter)
            except Exception:
                # Robuuste fallback voor exotische OCC-booleans: behoud dezelfde
                # eindgeometrie door de originele cutters één voor één te snijden.
                result = tube
                for cutter in cutters:
                    result = result.cut(cutter)
                return result

        # v774: bouw maar één representatieve body.
        piece = normalized_pieces[0]
        length = float(piece["lengthMm"])

        tube = (
            round_tube_solid(length)
            if is_round
            else rectangular_tube_solid(length)
        )

        tube = apply_profile_operations(
            tube,
            length,
            piece.get("operations") or [],
        )

        value = tube.val()
        exported_shapes = list(value.Solids()) if hasattr(value, "Solids") else []
        if not exported_shapes and isinstance(value, cq.Shape):
            exported_shapes = [value]

        if not exported_shapes:
            raise RuntimeError("Geen geldige solids voor productie-STEP gegenereerd.")

        export_shape = (
            exported_shapes[0]
            if len(exported_shapes) == 1
            else cq.Compound.makeCompound(exported_shapes)
        )

        tmp = CACHE_DIR / f"production_{uuid.uuid4().hex}.step"
        cq.exporters.export(export_shape, str(tmp), exportType="STEP")
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
            headers={
                "Content-Disposition": 'attachment; filename="productie.step"',
                "X-Vakstaal-Quantity": str(quantity),
                "X-Vakstaal-Bodies": "1",
                "X-Vakstaal-Operation-Union": "exact-cad-v1",
                "X-Vakstaal-Production-Step": "merged-overlap-contours",
            }
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

def _machine_hours_number(value, default=0.0):
    try:
        number=float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def _quote_production_minutes(payload: dict) -> float:
    """Totale productietijd van één opgeslagen offerte: machinetijd + laadtijd."""
    if not isinstance(payload, dict):
        return 0.0

    # Nieuwe offertes bewaren exact dezelfde KPI-waarde als de frontend.
    direct=_machine_hours_number(payload.get("totalProductionMinutes"), -1)
    if direct >= 0:
        return max(0.0,direct)

    # Bestaande offertes: de definitieve regels bevatten de naar producten
    # verdeelde machine- en laadtijd. De som is dezelfde definitie als de
    # zichtbare 'Totale productietijd' in de offerte.
    lines=payload.get("committedQuoteMaterialLines")
    if not isinstance(lines,list):
        lines=payload.get("calculationMaterialLines")
    if isinstance(lines,list):
        total=0.0
        has_time=False
        for line in lines:
            if not isinstance(line,dict) or line.get("enabled") is False:
                continue
            machine=_machine_hours_number(line.get("machineMinutes"),0)
            loading=_machine_hours_number(line.get("loadMinutes"),0)
            if machine>0 or loading>0:
                has_time=True
            total += max(0.0,machine)+max(0.0,loading)
        if has_time:
            return max(0.0,total)

    # Legacy fallback: vóór regelgebonden laadtijd werd machinetijd als totaal
    # bewaard. Dit is bewust de laatste fallback en wordt nooit bij nieuwere
    # offertes gebruikt.
    machine=_machine_hours_number(
        payload.get("automaticMachineMinutesExact",payload.get("minutes",0)),0
    )
    return max(0.0,machine)


@app.get("/api/quotes")
def list_quotes():
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, quote_number, customer_name, contact_person, customer_email,
                   customer_phone, total_ex_vat, payload_json, created_at, updated_at
            FROM quotes
            ORDER BY updated_at DESC
            """
        )

        rows = [_row_to_dict(r, cur) for r in cur.fetchall()]
        # Alleen de lichte lifecycle-velden uit payload_json zijn nodig om in het
        # offerte-overzicht de actuele status te tonen. De volledige payload gaat
        # bewust niet mee in /api/quotes.
        by_id = {row["id"]: row for row in rows}
        for row in rows:
            raw_payload = row.pop("payload_json", "")
            try:
                payload = json.loads(raw_payload or "{}")
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}

            mail_history = payload.get("quoteMailHistory")
            if not isinstance(mail_history, list):
                mail_history = []
            last_mail = next(
                (item for item in reversed(mail_history) if isinstance(item, dict)),
                None,
            )

            row["_list_lifecycle"] = {
                "last_mail_at": str((last_mail or {}).get("sent_at") or (last_mail or {}).get("sentAt") or ""),
                "last_mail_to": str((last_mail or {}).get("recipient") or (last_mail or {}).get("to") or ""),
                "invoice_id": str(payload.get("eboekhoudenLastInvoiceId") or ""),
                "invoice_number": str(payload.get("eboekhoudenLastInvoiceNumber") or ""),
                "invoiced_at": str(payload.get("eboekhoudenInvoicedAt") or ""),
                "invoice_mailed_at": str(payload.get("eboekhoudenInvoiceMailedAt") or ""),
            }
            row["files"] = []
            row["approval"] = None

        if rows:
            cur.execute("""
                SELECT f.quote_id, f.id, f.filename, f.content_type, f.file_kind,
                       f.file_size, f.dropbox_path, f.created_at
                FROM quote_files f JOIN quotes q ON q.id = f.quote_id
                ORDER BY f.created_at
            """)
            for record in cur.fetchall():
                item = _row_to_dict(record, cur)
                quote = by_id.get(item.pop("quote_id"))
                if quote is not None:
                    item["storage"] = "dropbox" if item.get("dropbox_path") else "database"
                    quote["files"].append(item)

            cur.execute("""
                SELECT a.quote_id, a.token, a.status, a.viewed_at, a.accepted_at,
                       a.accepted_by, a.note, a.email_sent_at, a.created_at, a.updated_at
                FROM quote_approvals a JOIN quotes q ON q.id = a.quote_id
            """)
            for record in cur.fetchall():
                approval = _row_to_dict(record, cur)
                quote = by_id.get(approval["quote_id"])
                if quote is not None:
                    approval["url"] = _approval_url(approval.get("token") or "")
                    quote["approval"] = approval

        # Eén eenduidige actuele status voor de lijst. Prioriteit volgt de echte
        # offerte-lifecycle: gefactureerd > geaccepteerd > verstuurd > opgeslagen.
        # Bekijken blijft detailinformatie bij de e-mail/acceptatielink.
        for row in rows:
            lifecycle = row.pop("_list_lifecycle", {}) or {}
            approval = row.get("approval") or {}

            invoice_id = str(lifecycle.get("invoice_id") or "")
            invoice_number = str(lifecycle.get("invoice_number") or "")
            invoice_mailed_at = str(lifecycle.get("invoice_mailed_at") or "")
            invoiced_at = str(lifecycle.get("invoiced_at") or "")
            last_mail_at = str(lifecycle.get("last_mail_at") or "")
            last_mail_to = str(lifecycle.get("last_mail_to") or "")

            if invoice_id or invoice_number:
                row["quote_status"] = "invoiced"
                row["quote_status_label"] = "Gefactureerd & gemaild" if invoice_mailed_at else "Gefactureerd"
                row["quote_status_detail"] = (
                    f"Factuur {invoice_number}" if invoice_number else "Factuur aangemaakt"
                )
                row["quote_status_at"] = invoice_mailed_at or invoiced_at
            elif str(approval.get("status") or "").lower() == "accepted":
                row["quote_status"] = "accepted"
                row["quote_status_label"] = "Geaccepteerd"
                row["quote_status_detail"] = str(approval.get("accepted_by") or "Klant")
                row["quote_status_at"] = str(approval.get("accepted_at") or "")
            elif last_mail_at:
                row["quote_status"] = "mailed"
                row["quote_status_label"] = "Verstuurd"
                row["quote_status_detail"] = last_mail_to or "Per e-mail verstuurd"
                row["quote_status_at"] = last_mail_at
            else:
                row["quote_status"] = "saved"
                row["quote_status_label"] = "Opgeslagen"
                row["quote_status_detail"] = "Nog niet verstuurd"
                row["quote_status_at"] = str(row.get("updated_at") or row.get("created_at") or "")

        return {
            "ok": True,
            "database": "postgresql" if _postgres_enabled() else "sqlite",
            "quotes": rows,
        }


@app.get("/api/machine-production-hours")
def machine_production_hours():
    """Gefactureerde productie-uren, één keer per gefactureerde offerte."""
    with _db_connect() as conn:
        cur=conn.cursor()
        cur.execute(
            """
            SELECT id, quote_number, customer_name, total_ex_vat,
                   payload_json, created_at, updated_at
            FROM quotes
            ORDER BY updated_at DESC
            """
        )
        rows=[_row_to_dict(r,cur) for r in cur.fetchall()]

    result=[]
    total_minutes=0.0
    for row in rows:
        try:
            payload=json.loads(row.get("payload_json") or "{}")
            if not isinstance(payload,dict):
                payload={}
        except Exception:
            payload={}

        invoice_id=str(payload.get("eboekhoudenLastInvoiceId") or "").strip()
        invoice_number=str(payload.get("eboekhoudenLastInvoiceNumber") or "").strip()
        invoiced_at=str(payload.get("eboekhoudenInvoicedAt") or "").strip()
        if not (invoice_id or invoice_number or invoiced_at):
            continue

        minutes=_quote_production_minutes(payload)
        total_minutes += minutes
        result.append({
            "id":row.get("id"),
            "quote_number":str(row.get("quote_number") or ""),
            "customer_name":str(row.get("customer_name") or ""),
            "invoice_id":invoice_id,
            "invoice_number":invoice_number,
            "invoiced_at":invoiced_at or str(row.get("updated_at") or ""),
            "created_at":str(row.get("created_at") or ""),
            "updated_at":str(row.get("updated_at") or ""),
            "production_minutes":round(minutes,6),
            "production_hours":round(minutes/60.0,6),
            "total_ex_vat":_machine_hours_number(row.get("total_ex_vat"),0),
        })

    result.sort(key=lambda x: x.get("invoiced_at") or "",reverse=True)
    return {
        "ok":True,
        "source":"invoiced_quotes",
        "total_minutes":round(total_minutes,6),
        "total_hours":round(total_minutes/60.0,6),
        "quote_count":len(result),
        "quotes":result,
    }


@app.get("/api/quotes/{quote_id}")
def get_quote(quote_id: str):
    with _db_connect() as conn:
        return _quote_response(conn, quote_id)


def _validated_quote_payload(payload: str) -> dict:
    """Shared input validation before quote creation or update touches storage."""
    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    try:
        data = json.loads(payload, parse_constant=reject_constant)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail="Ongeldige offertegegevens.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Offertegegevens moeten een JSON-object zijn.")
    total = data.get("total_ex_vat")
    try:
        if isinstance(total, bool) or (total is not None and not isinstance(total, (str, int, float))):
            raise ValueError("Unsupported price type")
        number = float(total or 0)
        if not math.isfinite(number):
            raise ValueError("Non-finite total")
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail="Het offertetotaal moet een geldig, eindig getal zijn.") from exc
    return data


@app.post("/api/quotes")
async def create_quote(request: Request):
    form, payload, files = await _parse_large_quote_form(request)
    try:
        data = _validated_quote_payload(payload)

        customer_name = str(data.get("customer") or "").strip()
        if not customer_name:
            raise HTTPException(status_code=400, detail="Klantnaam ontbreekt.")

        quote_id = uuid.uuid4().hex
        now = _utcnow()

        quote_number = _next_quote_number()
        with _db_connect() as conn:
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

            file_manifest = _quote_file_manifest(data)
            managed_step_names = file_manifest['production'] | file_manifest['filtered']
            stored_files=await _store_quote_files(
                conn,
                quote_id,
                files,
                managed_step_names=managed_step_names,
            )
            deduplicated_files=_dedupe_quote_file_rows(conn, quote_id)
            removed_obsolete_files=_reconcile_quote_managed_files(conn, quote_id, data)

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
            result["deduplicated_file_rows"]=deduplicated_files
            result["removed_obsolete_files"]=removed_obsolete_files
            result["removed_obsolete_generated_files"]=removed_obsolete_files
            return result
    finally:
        try:
            await form.close()
        except Exception:
            pass



def _locked_quote_row(conn, quote_id):
    if not _postgres_enabled():
        conn.execute('BEGIN IMMEDIATE')
    cur = conn.cursor()
    cur.execute(_sql('SELECT updated_at, payload_json FROM quotes WHERE id=%s FOR UPDATE',
                     'SELECT updated_at, payload_json FROM quotes WHERE id=?'), (quote_id,))
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail='Offerte niet gevonden.')
    if hasattr(row, 'keys'):
        return str(row['updated_at']), row['payload_json']
    return str(row[0]), row[1]


def _require_quote_revision(conn, quote_id, expected):
    revision, _payload = _locked_quote_row(conn, quote_id)
    if not isinstance(expected, str) or not expected or expected != revision:
        raise HTTPException(status_code=409, detail=(
            'Deze offerte is gewijzigd of dit concept heeft geen bekende serverversie. '
            'Je wijzigingen blijven lokaal bewaard. Bewaar wat je wilt overnemen en '
            'open daarna de actuele offerte uit de bibliotheek. Er is niets overschreven.'))
    return json.loads(_payload or '{}')


def _merge_quote_event(quote_id, original_revision, fields, history_key, event):
    """Mail/invoice completion merges metadata into the latest quote, never old calculations."""
    with _db_connect() as conn:
        revision, raw = _locked_quote_row(conn, quote_id)
        payload = json.loads(raw or '{}')
        history_raw = payload.get(history_key)
        history = [item for item in (history_raw if isinstance(history_raw, list) else []) if isinstance(item, dict)][-19:]
        history.append(event)
        payload.update(fields)
        payload[history_key] = history
        now = _utcnow()
        conn.cursor().execute(
            _sql('UPDATE quotes SET payload_json=%s,updated_at=%s WHERE id=%s',
                 'UPDATE quotes SET payload_json=?,updated_at=? WHERE id=?'),
            (json.dumps(payload, ensure_ascii=False), now, quote_id))
        conn.commit()
    # If someone changed calculations during the external action, do not grant
    # the old browser permission to overwrite those new calculations.
    return now if revision == str(original_revision) else None


@app.put("/api/quotes/{quote_id}")
async def update_quote(quote_id: str, request: Request):
    form, payload, files = await _parse_large_quote_form(request)
    try:
        data = _validated_quote_payload(payload)

        customer_name = str(data.get("customer") or "").strip()
        if not customer_name:
            raise HTTPException(status_code=400, detail="Klantnaam ontbreekt.")

        with _db_connect() as conn:
            cur = conn.cursor()

            stored_payload = _require_quote_revision(conn, quote_id, data.pop("expected_updated_at", None))
            # Server events are authoritative, including history absent from an old UI.
            for key, value in stored_payload.items():
                if (key == 'quoteMailHistory' or key.startswith('quoteLastMailed')
                        or key.startswith('eboekhoudenInvoice') or key.startswith('eboekhoudenLastInvoice')
                        or key == 'eboekhoudenInvoicedAt'):
                    data[key] = value

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

            file_manifest = _quote_file_manifest(data)
            managed_step_names = file_manifest['production'] | file_manifest['filtered']
            stored_files=await _store_quote_files(
                conn,
                quote_id,
                files,
                managed_step_names=managed_step_names,
            )
            deduplicated_files=_dedupe_quote_file_rows(conn, quote_id)
            removed_obsolete_files=_reconcile_quote_managed_files(conn, quote_id, data)

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
            result["deduplicated_file_rows"]=deduplicated_files
            result["removed_obsolete_files"]=removed_obsolete_files
            result["removed_obsolete_generated_files"]=removed_obsolete_files
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

    if not expected or not state or not __import__('hmac').compare_digest(state, expected):
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

def _eboek_payload_address_parts(payload: dict) -> dict:
    street=str(payload.get("customerStreet") or "").strip()
    house=str(payload.get("customerHouseNumber") or "").strip()
    postal=str(payload.get("customerPostalCode") or "").strip()
    city=str(payload.get("customerCity") or "").strip()

    legacy=str(payload.get("customerAddress") or "").strip()
    if legacy and (not street or not postal or not city):
        chunks=[part.strip() for part in legacy.split(",") if part.strip()]
        first=chunks[0] if chunks else ""
        if first and not street:
            match=re.match(r"^(.*?\D)\s+(\d+[A-Za-z0-9\-/ ]*)$",first)
            if match:
                street=match.group(1).strip()
                if not house:
                    house=match.group(2).strip()
            else:
                street=first
        if len(chunks)>1 and not postal:
            postal=chunks[1]
        if len(chunks)>2 and not city:
            city=", ".join(chunks[2:])

    address_line=" ".join(part for part in (street,house) if part).strip()
    return {
        "street":street,
        "houseNumber":house,
        "address":address_line,
        "postalCode":postal,
        "city":city,
    }


def _eboek_find_or_create_relation(payload, force_create=False):
    relation_id=payload.get("eboekhoudenRelationId")
    if relation_id:
        try:
            rel=_eboek_get_relation(relation_id)
            return rel,False
        except Exception:
            pass

    name=str(payload.get("customer") or "").strip()
    email=str(payload.get("customerEmail") or "").strip()

    # RelationListItem bevat alleen id/type/code. Hydrateer eerst voordat we
    # namen/e-mail vergelijken, anders kan dezelfde klant dubbel aangemaakt worden.
    if email:
        items=_eboek_find_relations(email=email,limit=10)
        rows=_eboek_hydrate_relation_list_items(items,limit=10)
        exact_email=[
            r for r in rows
            if email.casefold() in {
                str(r.get("emailAddress") or r.get("email_address") or "").strip().casefold(),
                str(r.get("emailAddressInvoice") or r.get("email_address_invoice") or "").strip().casefold(),
            }
        ]
        if exact_email:
            return exact_email[0],False
        if rows:
            return rows[0],False

    if name:
        items=_eboek_find_relations(name=name,limit=10)
        rows=_eboek_hydrate_relation_list_items(items,limit=10)
        exact=[r for r in rows if str(r.get("name") or "").strip().casefold()==name.casefold()]
        if exact:
            return exact[0],False

    if not name:
        raise HTTPException(status_code=400,detail="Klantnaam ontbreekt voor e-Boekhouden.")

    if not force_create and _eboek_load_settings().get("createMissingRelation") is False:
        raise HTTPException(
            status_code=404,
            detail="Klant bestaat nog niet in e-Boekhouden en automatisch aanmaken staat uit."
        )

    cfg=_eboek_load_settings()
    address=_eboek_payload_address_parts(payload)
    customer_type=str(payload.get("customerType") or "").strip().lower()

    relation_body={
        "type":"P" if customer_type=="private" else "B",
        "name":name,
        "contact":str(payload.get("contactPerson") or "").strip() or None,
        "address":address.get("address") or None,
        "postalCode":address.get("postalCode") or None,
        "city":address.get("city") or None,
        "phoneNumber":str(payload.get("customerPhone") or "").strip() or None,
        "emailAddress":email or None,
        "emailAddressInvoice":email or None,
        "termOfPayment":int(cfg.get("paymentTermDays") or 30),
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

@app.get("/api/eboekhouden/relations/{relation_id}")
def eboekhouden_relation_detail(relation_id: int):
    rel=_eboek_get_relation(relation_id)
    if not isinstance(rel,dict) or not rel.get("id"):
        raise HTTPException(status_code=404,detail="e-Boekhouden relatie is niet gevonden.")
    return {
        "ok":True,
        "relation":_eboek_relation_public(rel)
    }

# ============================================================================
# v775 — klantofferte per e-mail verzenden
# Gebruikt dezelfde SMTP-server als de bestaande akkoord-notificaties.
# De ontvanger wordt ALTIJD uit de opgeslagen offerte gelezen; de browser kan
# dus niet een willekeurig extern ontvangeradres aan dit endpoint meegeven.
# ============================================================================

def _quote_mail_smtp_settings() -> dict:
    """Gebruik eerst de gewone SMTP_* instellingen en val daarna terug op
    dezelfde VAKSTAAL_SMTP_* instellingen die de bestaande login/reset-mail gebruikt.
    Daardoor hoeft er geen tweede mailserverconfiguratie naast de bestaande te staan.
    """
    generic={
        "host":str(os.environ.get("SMTP_HOST") or "").strip(),
        "port":int(os.environ.get("SMTP_PORT") or "587"),
        "user":str(os.environ.get("SMTP_USER") or "").strip(),
        "password":str(os.environ.get("SMTP_PASSWORD") or "").strip(),
        "from":str(os.environ.get("SMTP_FROM") or "").strip(),
        "ssl":str(os.environ.get("SMTP_SSL") or "").strip().lower() in {"1","true","yes"},
        "source":"SMTP_*",
    }
    if generic["host"] and generic["user"] and generic["password"]:
        generic["ssl"]=bool(generic["ssl"] or generic["port"]==465)
        return generic

    vakstaal={
        "host":str(os.environ.get("VAKSTAAL_SMTP_HOST") or "").strip(),
        "port":int(os.environ.get("VAKSTAAL_SMTP_PORT") or "465"),
        "user":str(os.environ.get("VAKSTAAL_SMTP_USER") or "").strip(),
        "password":str(os.environ.get("VAKSTAAL_SMTP_PASSWORD") or "").strip(),
        "from":str(os.environ.get("VAKSTAAL_SMTP_FROM") or "").strip(),
        "ssl":True,
        "source":"VAKSTAAL_SMTP_*",
    }
    vakstaal["ssl"]=bool(vakstaal["port"]==465 or str(os.environ.get("VAKSTAAL_SMTP_SSL") or "true").strip().lower() in {"1","true","yes"})
    return vakstaal


def _quote_mail_smtp_configured() -> bool:
    cfg=_quote_mail_smtp_settings()
    return bool(cfg.get("host") and cfg.get("user") and cfg.get("password"))


def _quote_mail_clean_header(value: str, fallback: str = "") -> str:
    return str(value or fallback).replace("\r", " ").replace("\n", " ").strip()


def _quote_mail_valid_email(value: str) -> bool:
    value = _quote_mail_clean_header(value)
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))


def _send_quote_mail_smtp(
    *,
    recipient: str,
    sender_email: str,
    sender_name: str,
    quote_number: str,
    customer_name: str,
    pdf_bytes: bytes,
    pdf_filename: str,
    mail_subject: str = "",
    mail_body: str = "",
    signature_logo_bytes: bytes = b"",
    signature_logo_type: str = "image/png",
    signature_logo_filename: str = "vakstaal-handtekening.png",
) -> None:
    if not _quote_mail_smtp_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "De mailserver is nog niet ingesteld. "
                "De offerte-mail gebruikt automatisch de bestaande SMTP_* of VAKSTAAL_SMTP_* instellingen van de Vakstaal-server."
            ),
        )

    recipient = _quote_mail_clean_header(recipient)
    sender_email = _quote_mail_clean_header(sender_email)
    sender_name = _quote_mail_clean_header(sender_name, "Vakstaal")
    quote_number = _quote_mail_clean_header(quote_number, "Offerte")
    customer_name = _quote_mail_clean_header(customer_name, "klant")
    pdf_filename = _quote_mail_clean_header(pdf_filename, f"{quote_number}.pdf")

    if not _quote_mail_valid_email(recipient):
        raise HTTPException(status_code=400, detail="Bij deze offerte staat geen geldig klant-e-mailadres.")
    if not _quote_mail_valid_email(sender_email):
        raise HTTPException(status_code=400, detail="Het ingestelde afzender e-mailadres is niet geldig.")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="De klantofferte-PDF ontbreekt.")

    smtp_cfg=_quote_mail_smtp_settings()
    host=str(smtp_cfg.get("host") or "").strip()
    port=int(smtp_cfg.get("port") or 587)
    user=str(smtp_cfg.get("user") or "").strip()
    password=str(smtp_cfg.get("password") or "").strip()
    configured_from=_quote_mail_clean_header(smtp_cfg.get("from") or user)
    use_ssl=bool(smtp_cfg.get("ssl") or port==465)

    subject = _quote_mail_clean_header(
        mail_subject,
        f"Offerte {quote_number} van {sender_name}"
    )
    body = str(mail_body or "").strip() or (
        f"Beste {customer_name},\n\n"
        f"In de bijlage ontvangt u onze offerte {quote_number}.\n\n"
        "In de PDF kunt u de offerte bekijken en, wanneer beschikbaar, digitaal accepteren.\n\n"
        "Met vriendelijke groet,\n"
        f"{sender_name}\n"
        f"{sender_email}"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["Reply-To"] = sender_email
    msg["To"] = recipient
    msg.set_content(body)

    # HTML-versie gebruikt exact dezelfde tekst, veilig ge-escaped.
    html_body = "<div style=\"font-family:Arial,sans-serif;font-size:14px;line-height:1.55;color:#1d2b33\">" + \
        html.escape(body).replace("\n","<br>") + "</div>"

    if signature_logo_bytes:
        html_body += (
            "<div style=\"margin-top:22px\">"
            "<img src=\"cid:vakstaal-signature-logo\" "
            "style=\"max-width:220px;max-height:90px;display:block\" "
            "alt=\"Vakstaal\">"
            "</div>"
        )

    msg.add_alternative(html_body, subtype="html")

    if signature_logo_bytes:
        html_part = msg.get_payload()[-1]
        logo_type = str(signature_logo_type or "image/png").lower()
        if "/" in logo_type:
            maintype, subtype = logo_type.split("/",1)
        else:
            maintype, subtype = "image", "png"
        if maintype != "image":
            maintype, subtype = "image", "png"
        html_part.add_related(
            signature_logo_bytes,
            maintype=maintype,
            subtype=subtype,
            cid="<vakstaal-signature-logo>",
            filename=_quote_mail_clean_header(signature_logo_filename, "vakstaal-handtekening.png"),
        )

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=pdf_filename,
    )

    ctx = ssl.create_default_context()
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=25, context=ctx) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg, from_addr=configured_from or user, to_addrs=[recipient])
        else:
            with smtplib.SMTP(host, port, timeout=25) as smtp:
                smtp.ehlo()
                smtp.starttls(context=ctx)
                smtp.ehlo()
                smtp.login(user, password)
                smtp.send_message(msg, from_addr=configured_from or user, to_addrs=[recipient])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"De mailserver kon de offerte niet verzenden: {exc}",
        ) from exc


@app.get("/api/mail/status")
def quote_mail_status():
    cfg=_quote_mail_smtp_settings()
    return {
        "ok": True,
        "configured": _quote_mail_smtp_configured(),
        "smtp_user": str(cfg.get("user") or "").strip(),
        "default_from": str(cfg.get("from") or "").strip(),
        "config_source": str(cfg.get("source") or ""),
    }


@app.post("/api/quotes/{quote_id}/send-email")
async def send_quote_email(
    quote_id: str,
    pdf: UploadFile = File(...),
    sender_email: str = Form(...),
    sender_name: str = Form("Vakstaal"),
    mail_subject: str = Form(""),
    mail_body: str = Form(""),
    signature_logo: UploadFile = File(None),
):
    with _db_connect() as conn:
        quote = _quote_response(conn, quote_id)

    payload = quote.get("payload") or {}
    recipient = str(
        quote.get("customer_email")
        or payload.get("customerEmail")
        or ""
    ).strip()
    customer_name = str(
        quote.get("customer_name")
        or payload.get("customer")
        or "klant"
    ).strip()
    quote_number = str(quote.get("quote_number") or "Offerte").strip()

    content = await pdf.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="De offerte-PDF is groter dan 20 MB.")

    signature_logo_bytes=b""
    signature_logo_type="image/png"
    signature_logo_filename="vakstaal-handtekening.png"
    if signature_logo is not None:
        signature_logo_bytes=await signature_logo.read()
        if len(signature_logo_bytes)>1024*1024:
            raise HTTPException(status_code=413,detail="Het handtekeninglogo is groter dan 1 MB.")
        signature_logo_type=str(signature_logo.content_type or "image/png")
        signature_logo_filename=str(signature_logo.filename or "vakstaal-handtekening.png")

    _send_quote_mail_smtp(
        recipient=recipient,
        sender_email=sender_email,
        sender_name=sender_name,
        quote_number=quote_number,
        customer_name=customer_name,
        pdf_bytes=content,
        pdf_filename=pdf.filename or f"{quote_number}.pdf",
        mail_subject=mail_subject,
        mail_body=mail_body,
        signature_logo_bytes=signature_logo_bytes,
        signature_logo_type=signature_logo_type,
        signature_logo_filename=signature_logo_filename,
    )

    # v874 — succesvolle offerte-mail duurzaam registreren bij de offerte.
    # Alleen NA succesvolle SMTP-verzending opslaan, zodat de status nooit
    # onterecht "verstuurd" kan worden.
    sent_at=_utcnow()
    clean_sender=_quote_mail_clean_header(sender_email)
    filename=pdf.filename or f"{quote_number}.pdf"
    history=payload.get("quoteMailHistory")
    if not isinstance(history,list):
        history=[]
    history=[
        item for item in history
        if isinstance(item,dict)
    ][-49:]
    history.append({
        "sent_at":sent_at,
        "recipient":recipient,
        "sender_email":clean_sender,
        "filename":filename,
    })
    payload["quoteMailHistory"]=history
    payload["quoteLastMailedAt"]=sent_at
    payload["quoteLastMailedTo"]=recipient
    payload["quoteLastMailedFrom"]=clean_sender

    event_revision = _merge_quote_event(
        quote_id, quote.get('updated_at'),
        {key: payload[key] for key in ('quoteLastMailedAt','quoteLastMailedTo','quoteLastMailedFrom')},
        'quoteMailHistory', history[-1])

    return {
        "ok": True,
        "quote_id": quote_id,
        "quote_number": quote_number,
        "recipient": recipient,
        "sender_email": clean_sender,
        "filename": filename,
        "sent_at": sent_at,
        "updated_at": event_revision,
        "mail_count": len(history),
    }

@app.post("/api/quotes/{quote_id}/eboekhouden-invoice")
async def create_eboekhouden_invoice(quote_id: str, request: Request):
    try:
        options=await request.json()
        if not isinstance(options,dict):
            options={}
    except Exception:
        options={}

    send_email=bool(options.get("send_email"))
    requested_recipient=str(options.get("recipient_email") or "").strip()
    sender_email=str(options.get("sender_email") or "").strip()
    sender_name=str(options.get("sender_name") or "Vakstaal").strip() or "Vakstaal"
    mail_subject=str(options.get("mail_subject") or "").strip()
    mail_body_html=str(options.get("mail_body_html") or "").strip()

    with _db_connect() as conn:
        quote=_quote_response(conn,quote_id)

    payload=quote.get("payload") or {}
    total=float(payload.get("total_ex_vat") or quote.get("total_ex_vat") or 0)
    if total<=0:
        raise HTTPException(status_code=400,detail="Het offertebedrag is € 0,00; factuur is niet aangemaakt.")

    relation,created_relation=_eboek_find_or_create_relation(payload,force_create=True)
    relation_id=relation.get("id")
    if not relation_id:
        raise HTTPException(status_code=502,detail="Geen geldig e-Boekhouden relatie-ID gevonden.")

    invoice_recipient=""
    if send_email:
        invoice_recipient=(
            requested_recipient
            or str(payload.get("customerEmail") or "").strip()
        )
        if not _quote_mail_valid_email(invoice_recipient):
            raise HTTPException(
                status_code=400,
                detail="Voor factuurmail ontbreekt een geldig klant-e-mailadres."
            )
        if not _quote_mail_valid_email(sender_email):
            raise HTTPException(
                status_code=400,
                detail="Voor factuurmail ontbreekt een geldig afzender e-mailadres."
            )
        if not mail_subject:
            raise HTTPException(status_code=400,detail="Onderwerp voor de factuurmail ontbreekt.")
        if not mail_body_html:
            raise HTTPException(status_code=400,detail="Tekst voor de factuurmail ontbreekt.")

        # e-Boekhouden mailt de factuur aan het factuur-e-mailadres van de relatie.
        # Zorg dat dit exact gelijk loopt met het e-mailadres uit de open Vakstaal-offerte.
        current_invoice_email=str(
            relation.get("emailAddressInvoice")
            or relation.get("email_address_invoice")
            or ""
        ).strip()
        if current_invoice_email.casefold()!=invoice_recipient.casefold():
            relation_name=str(relation.get("name") or payload.get("customer") or "").strip()
            if not relation_name:
                raise HTTPException(
                    status_code=400,
                    detail="Klantnaam ontbreekt; factuur-e-mailadres kon niet in e-Boekhouden worden bijgewerkt."
                )
            _eboek_http(
                "PATCH",
                f"/v1/relation/{int(relation_id)}",
                body={
                    "name":relation_name,
                    "emailAddressInvoice":invoice_recipient,
                }
            )
            relation=_eboek_get_relation(relation_id)

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

    if send_email:
        # De moderne e-Boekhouden REST API mailt de officiële factuur direct
        # wanneer het `email` object bij POST /v1/invoice wordt meegestuurd.
        # Hierdoor beheert e-Boekhouden zelf de factuur-PDF én verzendstatus.
        invoice_body["email"]={
            "fromEmail":sender_email,
            "fromName":sender_name,
            "subject":mail_subject,
            "body":mail_body_html,
            "attachUbl":False,
        }

    invoice=_eboek_http("POST","/v1/invoice",body=invoice_body)

    # GET /v1/invoice/{id} bevat o.a. de officiële deelbare PDF-url.
    invoice_id=invoice.get("id") if isinstance(invoice,dict) else None
    invoice_detail={}
    if invoice_id:
        try:
            invoice_detail=_eboek_http("GET",f"/v1/invoice/{int(invoice_id)}")
        except Exception as exc:
            print(f"e-Boekhouden invoice detail {invoice_id} warning: {exc}")
    if isinstance(invoice_detail,dict) and invoice_detail:
        merged=dict(invoice)
        merged.update(invoice_detail)
        invoice=merged

    # Persist relation id + lifecycle-status in quote payload so future
    # invoices/searches én het klantstatusvenster exact blijven.
    invoiced_at=_utcnow()
    payload["eboekhoudenRelationId"]=int(relation_id)
    payload["eboekhoudenLastInvoiceId"]=invoice.get("id")
    payload["eboekhoudenLastInvoiceNumber"]=invoice.get("invoiceNumber") or invoice.get("invoice_number") or ""
    payload["eboekhoudenInvoicedAt"]=invoiced_at

    pdf_url=str(
        invoice.get("urlPdfFile")
        or invoice.get("url_pdf_file")
        or ""
    ).strip()
    if pdf_url:
        payload["eboekhoudenInvoicePdfUrl"]=pdf_url

    if send_email:
        payload["eboekhoudenInvoiceMailedAt"]=invoiced_at
        payload["eboekhoudenInvoiceMailedTo"]=invoice_recipient
        payload["eboekhoudenInvoiceMailedFrom"]=sender_email
        payload["eboekhoudenInvoiceMailStatus"]="sent"
        invoice_mail_history=payload.get("eboekhoudenInvoiceMailHistory")
        if not isinstance(invoice_mail_history,list):
            invoice_mail_history=[]
        invoice_mail_history=[
            item for item in invoice_mail_history
            if isinstance(item,dict)
        ][-19:]
        invoice_mail_history.append({
            "invoice_id":invoice.get("id"),
            "invoice_number":invoice.get("invoiceNumber") or invoice.get("invoice_number") or "",
            "sent_at":invoiced_at,
            "recipient":invoice_recipient,
            "sender_email":sender_email,
            "pdf_url":pdf_url,
            "via":"e-Boekhouden",
            "status":"sent",
        })
        payload["eboekhoudenInvoiceMailHistory"]=invoice_mail_history

    invoice_history=payload.get("eboekhoudenInvoiceHistory")
    if not isinstance(invoice_history,list):
        invoice_history=[]
    invoice_history=[
        item for item in invoice_history
        if isinstance(item,dict)
    ][-19:]
    invoice_history.append({
        "id":invoice.get("id"),
        "number":invoice.get("invoiceNumber") or invoice.get("invoice_number") or "",
        "invoiced_at":invoiced_at,
        "emailed":bool(send_email),
        "emailed_to":invoice_recipient if send_email else "",
        "emailed_from":sender_email if send_email else "",
        "pdf_url":pdf_url,
    })
    payload["eboekhoudenInvoiceHistory"]=invoice_history
    event_revision = _merge_quote_event(
        quote_id, quote.get('updated_at'),
        {key: value for key, value in payload.items()
         if key.startswith('eboekhouden') and key != 'eboekhoudenInvoiceHistory'},
        'eboekhoudenInvoiceHistory', invoice_history[-1])

    return {
        "ok":True,
        "quote_id":quote_id,
        "updated_at":event_revision,
        "relation":_eboek_relation_public(relation),
        "relation_created":created_relation,
        "invoice":invoice,
        "invoiced_at":invoiced_at,
        "emailed":bool(send_email),
        "emailed_to":invoice_recipient if send_email else "",
        "emailed_from":sender_email if send_email else "",
        "invoice_pdf_url":pdf_url,
        "email_status":"sent" if send_email else "not_sent",
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
            "/api/filter-step/{job_id}",
            "/api/filter-step-file",
            "/api/step-source/{job_id}",
            "/api/quotes/{quote_id}/filtered-step",
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


# v9: exact STEP selection export. No re-analysis, reconstruction or quote writes.
# The ordering is the same importStep(...).solids().vals() used by the analyzer.
_V9_STEP_LOG = logging.getLogger("vakstaal.step_selection")


def _v9_step_name_key(value: str) -> str:
    return unicodedata.normalize("NFC", str(value or "").replace("\\", "/").rsplit("/", 1)[-1]).casefold()


def _v9_selection_indices(payload: dict) -> list[int]:
    raw = payload.get("solid_indices") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        raise HTTPException(422, "Geen onderdelen geselecteerd voor het STEP-selectiebestand.")
    if len(raw) > 100000 or any(type(i) is not int or i < 1 for i in raw):
        raise HTTPException(422, "STEP-onderdeelnummers moeten positieve gehele getallen zijn.")
    return sorted(set(raw))


def _v9_cached_step(job_id: str) -> Path:
    # job IDs from /api/analyze-step are UUID hex strings, never filesystem paths.
    if not re.fullmatch(r"[0-9a-fA-F]{32}", job_id or ""):
        raise HTTPException(404, "STEP-sessie niet gevonden of verlopen.")
    return job_step_path(job_id)


def _v9_step_bytes_response(data: bytes, filename: str, count: int | None = None) -> Response:
    name = str(filename or "selectie.step").replace("\\", "/").rsplit("/", 1)[-1]
    name = name.replace("\r", "").replace("\n", "")
    ascii_name = name.encode("ascii", "replace").decode("ascii").replace('"', "_")
    headers = {
        "Content-Disposition": f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{urllib.parse.quote(name, safe="")}',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }
    if count is not None:
        headers["X-STEP-Selected-Count"] = str(count)
    return Response(content=data, media_type="application/step", headers=headers)


def _v9_export_step_selection(data: bytes, indices: list[int], filename: str) -> Response:
    if not data:
        raise HTTPException(422, "Het originele STEP-bestand is leeg.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"STEP-bestand is groter dan {MAX_UPLOAD_MB} MB.")
    try:
        with tempfile.TemporaryDirectory(prefix="vakstaal_step_selection_") as work:
            source = Path(work) / "source.step"
            target = Path(work) / "selection.step"
            source.write_bytes(data)
            solids = cq.importers.importStep(str(source)).solids().vals()
            if not solids:
                raise HTTPException(422, "Het originele STEP-bestand bevat geen solids.")
            invalid = [i for i in indices if i > len(solids)]
            if invalid:
                raise HTTPException(422, "Geselecteerde onderdelen bestaan niet in dit oorspronkelijke STEP-bestand: "
                                    + ", ".join(map(str, invalid[:12]))
                                    + f". Het bestand bevat {len(solids)} onderdelen; controleer de bronversie.")
            chosen = [solids[i - 1] for i in indices]
            shape = chosen[0] if len(chosen) == 1 else cq.Compound.makeCompound(chosen)
            cq.exporters.export(shape, str(target), exportType="STEP")
            result = target.read_bytes()
        if not result:
            raise RuntimeError("Empty STEP export")
        _V9_STEP_LOG.info("step_selection.complete source_count=%d selected_count=%d", len(solids), len(chosen))
        stem = Path(str(filename).replace("\\", "/").rsplit("/", 1)[-1]).stem or "STEP"
        return _v9_step_bytes_response(result, stem + "_OFFERTSELECTIE.step", len(chosen))
    except HTTPException:
        raise
    except Exception as exc:
        _V9_STEP_LOG.exception("step_selection.failed")
        raise HTTPException(422, "Het STEP-selectiebestand kon niet worden gemaakt. Controleer het originele STEP-bestand; er is niets aan de offerte gewijzigd.") from exc


@app.get("/api/step-source/{job_id}")
def v9_download_step_source(job_id: str):
    path = _v9_cached_step(job_id)
    return _v9_step_bytes_response(path.read_bytes(), "bron.step")


@app.post("/api/filter-step/{job_id}")
def v9_filter_step_session(job_id: str, payload: dict):
    indices = _v9_selection_indices(payload)
    path = _v9_cached_step(job_id)
    return _v9_export_step_selection(path.read_bytes(), indices, "STEP.step")


@app.post("/api/filter-step-file")
def v9_filter_step_upload(file: UploadFile = File(...), solid_indices: str = Form(...)):
    # sync route: CAD work runs in FastAPI's worker thread, not its async event loop.
    name = file.filename or "bron.step"
    if Path(name).suffix.lower() not in {".step", ".stp"}:
        raise HTTPException(422, "Kies het oorspronkelijke STEP-bestand (.step of .stp).")
    try:
        raw = json.loads(solid_indices)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Ongeldige STEP-selectie.") from exc
    indices = _v9_selection_indices({"solid_indices": raw})
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    return _v9_export_step_selection(data, indices, name)


@app.post("/api/quotes/{quote_id}/filtered-step")
def v9_filter_saved_quote_step(quote_id: str, payload: dict):
    indices = _v9_selection_indices(payload)
    name = str(payload.get("source_filename") or "").strip()
    if Path(name).suffix.lower() not in {".step", ".stp"}:
        raise HTTPException(422, "De naam van het originele STEP-bronbestand ontbreekt.")
    # Search only this quote, and only an exact original filename. A production
    # or previously filtered file is never substituted by a similar stem.
    with _db_connect() as conn:
        key = _v9_step_name_key(_safe_dropbox_name(name, "bestand"))
        candidates = [f for f in _quote_files(conn, quote_id) if _v9_step_name_key(f.get("filename")) == key]
    if not candidates:
        raise HTTPException(404, "Het originele STEP-bronbestand is niet aan deze offerte gekoppeld.")
    if len(candidates) != 1:
        raise HTTPException(409, "Het originele STEP-bronbestand is niet eenduidig. Er wordt geen willekeurig bestand gekozen.")
    # Reuse the existing scoped database/Dropbox read path. Nothing is saved here.
    original = download_quote_file(quote_id, candidates[0]["id"])
    return _v9_export_step_selection(bytes(original.body), indices, name)


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
                "feature_classifier_version": 8,
                "physical_cut_classifier_version": STEP_PHYSICAL_CUT_VERSION,
                "profile_axis": [float(v) for v in axis],
                **_simulation_profile_frame(solid, detail),
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
            "feature_classifier_version": 8,
            "physical_cut_classifier_version": STEP_PHYSICAL_CUT_VERSION,
            "profile_axis": [float(v) for v in axis],
            **_simulation_profile_frame(solid, detail),
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

# Filename recognition only: no SolidWorks geometry is decoded here.
def _profile_source_dimensions(filename: str, path: str = "") -> dict:
    """Recognize explicit profile names and folder hints, never infer geometry."""
    import re
    import math
    from pathlib import PurePosixPath
    if not filename.lower().endswith('.sldlfp'):
        raise ValueError('Geen SLDLFP-profielbestand.')
    stem = filename[:-7].strip()
    shape_patterns = {
        'Vierkant': r'vierkante?(?:\s+kokers?)?|square',
        'Rechthoek': r'rechthoek(?:ige?)?(?:\s+kokers?)?|reachthoekig(?:\s+kokers?)?|rectangular',
        'Rond': r'ronde?(?:\s+buizen|\s+buis)?|round(?:\s+tube)?',
        'Strip': r'strip(?:pen)?(?:\s+\d+(?:[.,]\d+)?\s*mm)?|plat(?:te)?\s*staal|flat(?:\s+bar)?',
    }
    def shape_hint(text):
        return {shape for shape,pattern in shape_patterns.items()
                if re.fullmatch(pattern, text.strip(), re.IGNORECASE)}

    # Consume only known prefixes; arbitrary text such as "copy" remains a warning.
    prefix = re.match(r'^(ronde?\s+buis|ronde?\s+buizen|ronde?|round(?:\s+tube)?|strip(?:pen)?|plat(?:te)?\s*staal|flat(?:\s+bar)?|vierkante?(?:\s+koker)?|square|rechthoek(?:ige?)?(?:\s+koker)?|rectangular|koker)\s+', stem, re.IGNORECASE)
    name_hints = shape_hint(prefix.group(1)) if prefix else set()
    dimensions = stem[prefix.end():].strip() if prefix else stem
    if dimensions.startswith(('Ø','ø','⌀')):
        name_hints.add('Rond')
    match = re.fullmatch(r'(?:[Øø⌀]\s*)?(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)(?:\s*[xX×]\s*(\d+(?:[.,]\d+)?))?(?:\s*mm)?(?:\s*[rR]\s*(\d+(?:[.,]\d+)?)(?:\s*mm)?)?', dimensions)
    if not match:
        raise ValueError('Naam niet herkend; gebruik bijvoorbeeld Ronde buis 101,6x3,6, strip 30x5 of 15x15x1 R1,5.')
    a,b,c,r = [float(v.replace(',', '.')) if v is not None else None for v in match.groups()]
    if any(not math.isfinite(v) or v <= 0 or v > 100000 for v in (a,b,c,r) if v is not None):
        raise ValueError('Ongeldige profielmaat.')
    folder_hints = set()
    for folder in PurePosixPath(path.replace('\\','/')).parts[:-1]:
        folder_hints.update(shape_hint(folder))
    hints = folder_hints | name_hints
    if len(hints)>1:
        raise ValueError('Profielvorm in bestandsnaam en/of mappen spreekt elkaar tegen; controleer het bestand.')
    hint = next(iter(hints), None)
    if hint == 'Strip':
        if c is not None or r is not None:
            raise ValueError('Een strip verwacht breedte x dikte, zonder derde maat of radius.')
        shape,width,height,thickness = 'Strip',max(a,b),min(a,b),min(a,b)
    elif c is None:
        if hint != 'Rond':
            raise ValueError('Twee maten zijn niet eenduidig: vermeld Ronde buis of Strip in de naam of map.')
        shape,width,height,thickness = 'Rond',a,a,b
        if r is not None:
            raise ValueError('Radiusaanduiding bij een ronde buis vereist controle.')
    else:
        shape = 'Vierkant' if a == b else 'Rechthoek'
        width,height,thickness = max(a,b),min(a,b),c
    if hint and shape != hint:
        raise ValueError('De maten passen niet bij de opgegeven profielvorm.')
    if shape != 'Strip' and thickness*2 >= min(width,height):
        raise ValueError('Wanddikte is te groot voor een hol profiel.')
    if r is not None and r > min(width,height)/2:
        raise ValueError('Radius is groter dan de halve profielmaat.')
    number = lambda v: format(v, '.12g')
    size = 'x'.join(number(v) for v in ((width,thickness) if shape in ('Rond','Strip') else (width,height,thickness)))
    return {'shape':shape,'size':size,'width_mm':width,'height_mm':height,
            'thickness_mm':thickness,'radius_mm':r,'is_solid':shape=='Strip',
            'dimension_key':shape.casefold()+':'+size,
            'source':'filename','geometry_verified':False}


@app.post('/api/dropbox/library-sources/scan')
async def dropbox_library_sources_scan(request: Request):
    """Read-only source preview. Does not mutate the material library or prices."""
    from starlette.concurrency import run_in_threadpool
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='Ongeldig verzoek.')
    if not isinstance(body, dict) or body.get('kind') not in ('profiles','prices'):
        raise HTTPException(status_code=400, detail='Kies profielen of prijslijsten.')
    if not isinstance(body.get('path'), str):
        raise HTTPException(status_code=400, detail='Kies eerst een Dropbox-map.')
    path = _normalize_dropbox_browser_path(body['path'])
    if not path:
        raise HTTPException(status_code=400, detail='Kies een submap; de volledige Dropbox wordt niet gescand.')
    kind = body['kind']
    files = await run_in_threadpool(_dropbox_list_files_recursive, path, '.sldlfp' if kind == 'profiles' else '')
    if kind == 'prices':
        files = [f for f in files if f['name'].lower().endswith(('.csv','.xlsx','.xls','.pdf'))]
    rows = []
    for meta in files[:1000]:
        row = {**meta, 'status':'found'}
        if kind == 'profiles':
            try:
                row['profile'] = _profile_source_dimensions(meta['name'], meta['path_display'] or meta['path_lower'])
                row['status'] = 'recognized'
            except ValueError as exc:
                row.update(status='review', error=str(exc))
        rows.append(row)
    return {'ok':True, 'kind':kind, 'files':rows, 'file_count':len(files),
            'truncated':len(files)>1000, 'applied':False}


# TransIP inkoopfacturen: secrets blijven uitsluitend op de server.
import imaplib as _inv_imaplib
import email as _inv_email
import email.utils as _inv_email_utils
import hashlib as _inv_hashlib
import hmac as _inv_hmac
import io as _inv_io
import subprocess as _inv_subprocess
import unicodedata as _inv_unicode
from datetime import timedelta as _inv_timedelta
from email import policy as _inv_policy
from email.utils import parsedate_to_datetime as _inv_parsedate
try:
    from pypdf import PdfReader as _InvPdfReader
except ImportError:
    _InvPdfReader = None

_INV_ADDRESSES = {
    "administratie@vakstaal.nl": "VAKSTAAL_IMAP_ADMINISTRATIE_PASSWORD",
    "info@vakstaal.nl": "VAKSTAAL_IMAP_INFO_PASSWORD",
}
_INV_MAX_PDF = 5_000_000
_INV_FORWARD_FROM = "info@vakstaal.nl"


def _inv_settings():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_account_settings ("
                     "id INTEGER PRIMARY KEY, destination TEXT NOT NULL, rules_json TEXT NOT NULL, "
                     "auto INTEGER NOT NULL, move_to_trash INTEGER NOT NULL, access_enabled INTEGER NOT NULL, "
                     "auto_enabled_at TEXT NOT NULL)")
        row = conn.execute("SELECT destination,rules_json,auto,move_to_trash,access_enabled,auto_enabled_at "
                           "FROM invoice_account_settings WHERE id=1").fetchone()
    if not row:
        return {"destination": "", "rules": [], "auto": False, "move_to_trash": True,
                "access_enabled": False, "auto_enabled_at": "", "configured": False}
    return {"destination": row[0], "rules": json.loads(row[1]), "auto": bool(row[2]),
            "move_to_trash": True, "access_enabled": bool(row[4]),
            "auto_enabled_at": row[5], "configured": True}


def _inv_supplier_directory():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_supplier_names "
                     "(sender TEXT PRIMARY KEY, company TEXT NOT NULL, updated_at TEXT NOT NULL)")
        return {row[0]: row[1] for row in conn.execute(
            "SELECT sender,company FROM invoice_supplier_names").fetchall()}


def _inv_supplier_family(filename):
    return "family:paynl" if re.match(r"^PAYNL-\d{6,47}\b", str(filename or ""), re.I) else ""


def _inv_known_supplier(suggestion, sender, directory, filename=""):
    address = _inv_email_utils.parseaddr(str(sender or ""))[1].strip().lower()
    if address.rsplit("@", 1)[-1] in {"vakstaal.nl", "hotmail.com", "hotmail.nl", "outlook.com",
                                      "outlook.nl", "live.nl", "live.com", "gmail.com", "icloud.com"}:
        address = ""
    company = directory.get(_inv_supplier_family(filename)) or directory.get(address)
    if not company:
        return dict(suggestion)
    result = {**suggestion, "company": company}
    result["complete"] = bool(result.get("year") and result.get("number") and company
                              and result.get("number_basis") != "bestandsnaam")
    result["filename"] = (f"{result['year']} {company} {result['number']}.pdf"
                          if result["complete"] else "")
    result["missing"] = [key for key in ("year", "company", "number") if not result.get(key)]
    result["supplier_learned"] = True
    return result


def _inv_smtp_settings():
    """Authenticeer factuurmails als het TransIP-postvak dat e-Boekhouden toelaat."""
    password = os.environ.get(_INV_ADDRESSES[_INV_FORWARD_FROM], "")
    if not password:
        raise HTTPException(503, "Het TransIP-wachtwoord van info@vakstaal.nl ontbreekt op de server.")
    return {"host": "smtp.transip.email", "port": 465, "user": _INV_FORWARD_FROM,
            "password": password, "from": _INV_FORWARD_FROM, "ssl": True}


def _inv_authorize(request: Request):
    secret = os.environ.get("VAKSTAAL_INVOICE_API_KEY", "")
    supplied = request.headers.get("X-Invoice-Key", "")
    if not secret:
        raise HTTPException(503, "Factuurkoppeling is niet ingesteld.")
    if supplied:
        if _inv_hmac.compare_digest(supplied, secret):
            return
        raise HTTPException(403, "Factuurtoegangscode is ongeldig.")
    if not _inv_settings()["access_enabled"]:
        raise HTTPException(403, "Vul eerst de factuurtoegangscode in.")


@app.get("/api/invoice-imap/preferences")
def invoice_imap_preferences():
    # Deze route blijft binnen de bestaande verplichte beheerderslogin.
    return _inv_settings()


@app.post("/api/invoice-imap/preferences")
def invoice_imap_preferences_save(payload: dict, request: Request):
    _inv_authorize(request)
    destination = str(payload.get("destination") or "").strip()
    if destination and not re.fullmatch(r"[^\s@]+@e-boekhouden\.nl", destination, re.I):
        raise HTTPException(422, "Controleer het e-Boekhouden-adres.")
    rules = payload.get("rules")
    if not isinstance(rules, list) or len(rules) > 100 or any(not isinstance(rule, str)
            or len(rule) > 254 for rule in rules):
        raise HTTPException(422, "Controleer de leveranciersregels.")
    rules = [rule.strip().lower() for rule in rules if rule.strip()]
    for rule in rules:
        if not re.fullmatch(r"(?:[a-z0-9._%+\-]+@)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)+", rule):
            raise HTTPException(422, "Controleer de leveranciersregels.")
    auto = payload.get("auto") is True
    if auto and (not destination or not rules):
        raise HTTPException(422, "Voor automatisch versturen zijn bestemming en leveranciers nodig.")
    remember = payload.get("remember_access") is True
    auto_enabled_at = str(payload.get("auto_enabled_at") or "")[:40] if auto else ""
    if auto:
        try:
            parsed = datetime.fromisoformat(auto_enabled_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
                raise ValueError("Ongeldige starttijd")
        except ValueError as exc:
            raise HTTPException(422, "Controleer de starttijd voor automatisch versturen.") from exc
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_account_settings ("
                     "id INTEGER PRIMARY KEY, destination TEXT NOT NULL, rules_json TEXT NOT NULL, "
                     "auto INTEGER NOT NULL, move_to_trash INTEGER NOT NULL, access_enabled INTEGER NOT NULL, "
                     "auto_enabled_at TEXT NOT NULL)")
        conn.execute(f"INSERT INTO invoice_account_settings "
                     f"(id,destination,rules_json,auto,move_to_trash,access_enabled,auto_enabled_at) "
                     f"VALUES (1,{marker},{marker},{marker},{marker},{marker},{marker}) "
                     "ON CONFLICT (id) DO UPDATE SET destination=excluded.destination, "
                     "rules_json=excluded.rules_json,auto=excluded.auto, "
                     "move_to_trash=excluded.move_to_trash,access_enabled=excluded.access_enabled, "
                     "auto_enabled_at=excluded.auto_enabled_at",
                     (destination, json.dumps(rules), int(auto), 1,
                      int(remember), auto_enabled_at))
    return {"saved": True, "access_enabled": remember}


@app.post("/api/invoice-imap/forget")
def invoice_imap_forget(request: Request):
    _inv_authorize(request)
    with _db_connect() as conn:
        conn.execute("UPDATE invoice_account_settings SET access_enabled=0 WHERE id=1")
    return {"access_enabled": False}


@app.post("/api/invoice-imap/supplier-name")
def invoice_imap_supplier_name(payload: dict, request: Request):
    _inv_authorize(request)
    sender = _inv_email_utils.parseaddr(str(payload.get("sender") or ""))[1].strip().lower()
    company = _inv_name_part(payload.get("company"), 55).lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", sender) or not 2 <= len(company) <= 55:
        raise HTTPException(422, "Controleer afzender en bedrijfsnaam.")
    _inv_learn_supplier(sender, company, payload.get("filename", ""))
    return {"saved": True, "company": company}


def _inv_learn_supplier(sender, company, filename=""):
    sender = _inv_email_utils.parseaddr(str(sender or ""))[1].lower()
    domain = sender.rsplit("@", 1)[-1]
    keys = []
    family = _inv_supplier_family(filename)
    if family:
        keys.append(family)
    # Een doorgestuurde mail kan facturen van verschillende leveranciers bevatten.
    if "@" in sender and domain not in {"vakstaal.nl", "hotmail.com", "hotmail.nl", "outlook.com",
                                        "outlook.nl", "live.nl", "live.com", "gmail.com", "icloud.com"}:
        keys.append(sender)
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_supplier_names "
                     "(sender TEXT PRIMARY KEY, company TEXT NOT NULL, updated_at TEXT NOT NULL)")
        for key in keys:
            conn.execute(f"INSERT INTO invoice_supplier_names(sender,company,updated_at) "
                         f"VALUES ({marker},{marker},{marker}) ON CONFLICT (sender) DO UPDATE SET "
                         "company=excluded.company,updated_at=excluded.updated_at",
                         (key, company, datetime.now(timezone.utc).isoformat()))


def _inv_mailbox(address: str, *, writable: bool = False):
    env = _INV_ADDRESSES.get(address)
    if not env:
        raise HTTPException(400, "Ongeldig postvak.")
    password = os.environ.get(env, "")
    if not password:
        raise HTTPException(503, f"Postvak {address} is nog niet ingesteld op de server.")
    try:
        imap = _inv_imaplib.IMAP4_SSL("imap.transip.email", 993, timeout=25)
        imap.login(address, password)
        result, _ = imap.select("INBOX", readonly=not writable)
        if result != "OK":
            raise ValueError("Postvak IN is niet beschikbaar")
        return imap
    except Exception as exc:
        raise HTTPException(502, f"Verbinding met {address} mislukt; controleer de serverinstellingen.") from exc


def _inv_attachments(message):
    for part in message.walk():
        name = part.get_filename() or ""
        if not name.lower().endswith(".pdf"):
            continue
        data = part.get_payload(decode=True)
        if data and len(data) <= _INV_MAX_PDF and data.startswith(b"%PDF-"):
            yield name, data


def _inv_invoice_attachments(message):
    """Return only the PDF attachment(s) that represent the payable invoice.

    WEX/EssoCardOnline mails contain two PDFs: an ``invoice_ES...`` electronic
    invoice document and a ``PAY_...`` PDF. For Vakstaal the PAY PDF is the
    document that must be read, shown and forwarded. When a PAY attachment is
    present in a WEX/Esso mail it is therefore authoritative; the companion
    invoice_ES PDF is intentionally ignored by the invoice workflow.

    The rule is deliberately scoped to WEX/Esso messages and falls back to all
    valid PDFs if WEX changes its filename format, so unrelated suppliers are
    unaffected and a renamed invoice is never silently lost.
    """
    pdfs = list(_inv_attachments(message))
    if not pdfs:
        return []
    sender = _inv_email_utils.parseaddr(str(message.get("From", "")))[1].lower()
    subject = str(message.get("Subject", ""))
    wex_esso = ("wexeurope" in sender or
                bool(re.search(r"\besso\s*card(?:online)?\b", subject, re.I)))
    if not wex_esso:
        return pdfs
    preferred = [(name, pdf) for name, pdf in pdfs
                 if re.match(r"^PAY_[^/\\]*\.pdf$", Path(str(name)).name, re.I)]
    return preferred or pdfs


def _inv_pdf_text(pdf: bytes) -> tuple[str, str]:
    """Lees iedere pagina; gebruik OCR voor pagina's met weinig tekst."""
    pages = []
    source = "onleesbaar"
    try:
        reader = _InvPdfReader(_inv_io.BytesIO(pdf), strict=False)
        if reader.is_encrypted:
            return "", "versleuteld"
        for page in reader.pages:
            text = ""
            try:
                content = page.get_contents()
                if not content or len(content.get_data()) <= 8_000_000:
                    plain = page.extract_text() or ""
                    try:
                        layout = page.extract_text(extraction_mode="layout") or ""
                    except Exception:
                        layout = ""
                    text = plain
                    if layout.strip() and layout.strip() != plain.strip():
                        text += "\n" + layout
            except Exception:
                pass
            pages.append(text)
    except Exception:
        pass
    try:
        import fitz
        with fitz.open(stream=pdf, filetype="pdf") as document:
            if document.needs_pass:
                return "", "versleuteld"
            for index, page in enumerate(document):
                while len(pages) <= index:
                    pages.append("")
                candidate = page.get_text("text", sort=True) or ""
                if len(candidate.strip()) > len(pages[index].strip()):
                    pages[index] = candidate
                if len(pages[index].strip()) >= 300 or not page.get_images(full=False):
                    continue
                if page.rect.width * page.rect.height > 2_000_000:
                    continue
                image = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).tobytes("png")
                for language in ("nld+eng", "eng"):
                    try:
                        result = _inv_subprocess.run(["tesseract", "stdin", "stdout", "-l", language],
                            input=image, capture_output=True, timeout=20, check=True)
                        recognized = result.stdout.decode("utf-8", "replace")
                        if len(recognized.strip()) > len(pages[index].strip()):
                            pages[index] = recognized
                            source = "OCR-scan"
                        break
                    except (_inv_subprocess.CalledProcessError, _inv_subprocess.TimeoutExpired, FileNotFoundError):
                        continue
    except Exception:
        pass
    extracted = "\n".join(pages)
    if extracted.strip() and source != "OCR-scan":
        source = "PDF-tekst"
    return extracted, source


def _inv_classify(text: str):
    lines = [re.sub(r"\s+", " ", line).strip().lower() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "unreadable", "Geen leesbare tekst in PDF; controleer deze handmatig."
    header = " ".join(lines[:16])[:1000]
    normalized = " ".join(lines)
    other = re.compile(r"\b(?:proforma|orderbevestiging|opdrachtbevestiging|quotation|offerte|pakbon|delivery note|tekening|leveringsvoorwaarden)\b")
    invoice = re.compile(r"\b(?:factuur|invoice|creditnota|credit note)\b")
    # Koptekst is leidend: 'factuuradres' of 'factuurvoorwaarden' op een offerte
    # mag nooit voldoende zijn om de offerte als factuur aan te merken.
    header_title = bool(re.search(r"\b(?:factuur|invoice|creditnota|credit note)\b",header))
    # Alleen een documenttitel maakt dit een offerte/order. Een verwijzing
    # zoals 'uw offerte 123' op een echte factuur is geen uitsluitingsgrond.
    other_title = next((i for i, line in enumerate(lines[:30]) if re.fullmatch(
        r"(?:offerte|quotation|orderbevestiging|opdrachtbevestiging|pakbon|delivery note|"
        r"tekening|leveringsvoorwaarden|pro\s*forma(?:\s+factuur)?)"
        r"(?:\s*[:#-]?\s*[a-z]*[-/]?\d[\w/-]*)?", line)), None)
    invoice_title = next((i for i, line in enumerate(lines[:30]) if re.fullmatch(
        r"(?:factuur|invoice|creditnota|credit note)(?:\s*[:#-]?\s*[a-z0-9/-]*\d[a-z0-9/-]*)?",
        line)), None)
    negative = other_title is not None and (invoice_title is None or other_title < invoice_title)
    if negative:
        return "other", "Documenttitel geeft een offerte, order of ander document aan."
    if not header_title and not invoice.search(normalized[:1600]):
        return "other", "Geen factuurkop in de PDF gevonden."
    title = header_title or any(re.fullmatch(r"(?:factuur|invoice|creditnota|credit note)(?:\s*[:#-]?\s*[a-z0-9/-]*\d[a-z0-9/-]*)?", line) for line in lines)
    number = bool(re.search(r"\b(?:factuur(?:nummer|nr\.?|\s*nummer|\s*nr\.?)?|invoice\s*(?:no\.?|number|#)|creditnota\s*(?:nr\.?|nummer)?)\s*[:#-]?\s*(?=[a-z0-9/-]*\d)[a-z0-9][a-z0-9/-]{3,}\b", normalized))
    total = bool(re.search(r"\b(?:totaal(?:\s*(?:te\s*betalen|incl(?:usief)?\.?\s*btw|bedrag))?|amount\s*due|total\s*(?:amount|due)?|te\s*betalen|grand\s*total)\s*[:€\s]{0,20}(?:eur\s*)?\d[\d.,]*", normalized))
    issuer = bool(re.search(r"\b(?:btw(?:-?nummer)?|vat\s*(?:id|number|no)|iban|kvk)\b", normalized))
    extracted_fields = _inv_name_suggestion(text, "")
    number = number or bool(extracted_fields["number"])
    total = total or _inv_total_amount(text) is not None
    dated_invoice = bool(extracted_fields["year"] and re.search(
        r"\b(?:factuurdatum|invoice\s*date|datum\s*factuur)\b", normalized))
    if title and number and issuer and (total or dated_invoice):
        return "invoice", "Factuurkop, nummer, totaal en afzendergegevens gevonden."
    if title or number:
        return "review", "Mogelijke factuur; controleer nummer, totaal en leverancier in de PDF."
    return "other", "Geen duidelijke factuurgegevens in de PDF."


def _inv_kind(text: str, filename: str, source: str = "PDF-tekst"):
    if source == "versleuteld":
        return "unreadable", "PDF is versleuteld en kan niet worden uitgelezen."
    # Een expliciete documentnaam gaat voor toevallige verwijzingen naar
    # factuurnummers, IBAN's of betaalvoorwaarden elders in een offerte.
    if re.search(r"\b(offerte|quotation|proforma|orderbevestiging|opdrachtbevestiging|pakbon|tekening|voorwaarden)\b",
                 re.sub(r"[_-]+", " ", filename), re.I):
        return "other", "Bestandsnaam duidt op een ander document."
    kind, reason = _inv_classify(text)
    if source == "OCR-scan" and kind == "invoice":
        return "invoice", "Factuurgegevens herkend met OCR; controleer de voorgestelde naam vóór verzending."
    return kind, reason


def _inv_total_amount(text: str):
    """Geef alleen een ondubbelzinnig factuurtotaal uit de PDF terug."""
    labels = (
        (3, re.compile(r"\b(?:totaal\s+te\s+betalen|te\s+betalen|amount\s+due|balance\s+due)\b", re.I)),
        (2, re.compile(r"\b(?:totaal\s*(?:incl\.?\s*btw|inclusief\s*btw|bedrag)|factuurtotaal|factuurbedrag|"
                       r"verschuldigd\s+bedrag|invoice\s+total|total\s+amount|grand\s+total|total\s+incl\.?\s*vat)\b", re.I)),
        (1, re.compile(r"\b(?:totaal|total)\b", re.I)),
    )
    money = re.compile(r"(?<![\w])(?:€\s*|EUR\s*)?([+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)[,.]\d{2})(?!\d)", re.I)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    candidates = []
    for index, line in enumerate(lines):
        if re.search(r"\b(?:subtotaal|subtotal|excl(?:usief)?\.?\s*(?:btw|vat)|btw\s*(?:bedrag|amount)|vat\s*amount)\b", line, re.I):
            continue
        for rank, pattern in labels:
            label = pattern.search(line)
            if not label:
                continue
            tail = line[label.end():]
            values = money.findall(tail)
            if not values and index + 1 < len(lines):
                following = lines[index + 1]
                if re.fullmatch(r"(?:€\s*|EUR\s*)?[+-]?[\d.,\s]+", following, re.I):
                    values = money.findall(following)
            if len(values) == 1:
                raw = values[0]
                separator = max(raw.rfind(","), raw.rfind("."))
                integer = re.sub(r"[^\d]", "", raw[:separator])
                cents = raw[separator+1:]
                if integer and len(cents) == 2:
                    sign = "-" if raw.startswith("-") else ""
                    candidates.append((rank, f"{sign}{int(integer)},{cents}"))
            break
    if not candidates:
        return None
    highest = max(rank for rank, _ in candidates)
    distinct = {value for rank, value in candidates if rank == highest}
    return next(iter(distinct)) if len(distinct) == 1 else None


def _inv_name_part(value: str, maximum: int = 64) -> str:
    plain = _inv_unicode.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    plain = re.sub(r"[/\\\\]+", "-", plain)
    plain = re.sub(r"[^A-Za-z0-9._ -]+", " ", plain)
    return re.sub(r"\s+", " ", plain).strip(" ._-")[:maximum].strip(" ._-")


def _inv_name_suggestion(text: str, sender: str):
    """Gebruik alleen aantoonbare PDF-gegevens; maildomeinen zijn geen leverancier."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    opening = "\n".join(lines)
    year = ""
    date = re.search(r"\b(?:factuurdatum|invoice\s*date|date\s*of\s*issue|datum\s*factuur)\b"
                     r"[^\n\d]{0,28}(?:\d{1,2}[./-]\d{1,2}[./-](20\d{2})|"
                     r"(20\d{2})[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+(20\d{2}))",
                     opening, re.I)
    if date:
        year = next((part for part in date.groups() if part), "")
    if not year:
        for position, line in enumerate(lines):
            if not re.search(r"\b(?:factuurdatum|invoice\s*date|date\s*of\s*issue|datum)\b", line, re.I):
                continue
            nearby = line + " " + (lines[position+1] if position+1 < len(lines) else "")
            found = re.search(r"(?:\d{1,2}[./-]\d{1,2}[./-]|\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+)(20\d{2})"
                              r"|\b(20\d{2})[./-]\d{1,2}[./-]\d{1,2}", nearby)
            if found:
                year = next(part for part in found.groups() if part)
                break
    number = ""
    number_label = re.compile(r"\b(?:factuur\s*(?:nummer|nr\.?|no\.?)|factuurnr\.?|"
                              r"invoice\s*(?:number|no\.?|#)|creditnota\s*(?:nummer|nr\.?)|factuur(?=\s*[:#.-]?\s*\d))"
                              r"\s*[:#.-]?\s*([A-Za-z0-9][A-Za-z0-9._/-]{1,45})", re.I)
    number_only = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/ -]{0,47}")
    for position, line in enumerate(lines):
        match = number_label.search(line)
        if match and re.search(r"\d", match.group(1)):
            number = _inv_name_part(match.group(1), 48)
            break
        if re.search(r"\b(?:factuurnummer|factuurnr\.?|invoice\s*(?:no|number))\b", line, re.I):
            # PDF's with a label column often put its value on the next line.
            for candidate in lines[position+1:position+4]:
                if re.search(r"\b(?:datum|date|bedrag|amount|klant|customer|totaal|total)\b", candidate, re.I):
                    break
                value = number_only.fullmatch(candidate.strip(" :#"))
                if value and re.search(r"\d", value.group()):
                    number = _inv_name_part(value.group(), 48)
                    break
            if number:
                break
    header_company = ""
    for line in lines:
        company = re.match(r"^(.{2,65}?)\s+(?:B\.?V\.?|GmbH|Ltda?|Inc\.?|N\.?V\.?)\b", line, re.I)
        if company and not re.search(r"\b(?:vakstaal|factuur|invoice|klant|aan:|ship\s+to)\b", company.group(1), re.I):
            header_company = _inv_name_part(company.group(1), 55).lower()
            break
    if not header_company:
        for position, line in enumerate(lines):
            labelled = re.match(r"^(?:leverancier|afzender|supplier|seller|from)\s*:?\s*(.{2,65})$", line, re.I)
            candidate = labelled.group(1) if labelled else (lines[position+1] if re.fullmatch(
                r"(?:leverancier|afzender|supplier|seller|from)\s*:?", line, re.I)
                and position+1 < len(lines) else "")
            if candidate and not re.search(r"\b(?:vakstaal|factuur|invoice|betaling|klant)\b", candidate, re.I):
                header_company = _inv_name_part(candidate, 55).lower()
                break
    company = _inv_name_part(header_company, 55).lower()
    fields = {"year": year, "company": company, "number": number}
    return {**fields, "complete": bool(all(fields.values())),
            "filename": f"{year} {company} {number}.pdf" if all(fields.values()) else "",
            "basis": "PDF-tekst", "missing": [key for key, value in fields.items() if not value]}


def _inv_name_with_filename(text: str, sender: str, filename: str):
    """Een bestandsnaam kan helpen bij handmatige controle, maar is geen PDF-bewijs."""
    suggestion = _inv_name_suggestion(text, sender)
    if _inv_supplier_family(filename) and not suggestion["company"]:
        suggestion["company"] = "pay"
        suggestion["company_basis"] = "bestandsnaam"
        suggestion["complete"] = False
    if suggestion["number"]:
        suggestion["missing"] = [key for key in ("year", "company", "number") if not suggestion[key]]
        return suggestion
    base = re.sub(r"\.pdf$", "", str(filename or ""), flags=re.I).strip()
    match = re.search(r"\b(?:factuur(?:nummer|nr)?|invoice)\s*[-_ #.:]*"
                      r"([A-Za-z0-9][A-Za-z0-9._/-]{2,47})\b", base, re.I)
    if not match and _inv_supplier_family(filename):
        # Pay vermeldt het PAYNL-kenmerk vooraan in de bijlagenaam. Alleen
        # als voorstel tonen; dit mag nooit automatische verzending starten.
        match = re.match(r"(PAYNL-\d{6,47})\b", base, re.I)
    if match and re.search(r"\d", match.group(1)):
        suggestion["number"] = _inv_name_part(match.group(1), 48)
        suggestion["number_basis"] = "bestandsnaam"
        suggestion["complete"] = False
        suggestion["filename"] = ""
        suggestion["missing"] = [key for key in ("year", "company", "number") if not suggestion[key]]
    return suggestion


def _inv_final_filename(text: str, sender: str, supplied=None):
    suggested = _inv_name_suggestion(text, sender)
    if supplied is not None and not isinstance(supplied, dict):
        raise HTTPException(400, "Ongeldige factuurnaam.")
    fields = {key: str((supplied or {}).get(key) or suggested[key]).strip()
              for key in ("year", "company", "number")}
    if not re.fullmatch(r"20\d{2}", fields["year"]):
        raise HTTPException(422, "Factuurjaartal ontbreekt; vul dit uit de PDF in.")
    if not 2 <= len(fields["company"]) <= 70 or re.search(r"[\x00-\x1f\x7f]", fields["company"]):
        raise HTTPException(422, "Bedrijfsnaam ontbreekt of is ongeldig; controleer de PDF.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/ -]{0,47}", fields["number"]) or not re.search(r"\d", fields["number"]):
        raise HTTPException(422, "Factuurnummer ontbreekt of is ongeldig; controleer de PDF.")
    company = _inv_name_part(fields["company"], 55).lower()
    number = _inv_name_part(fields["number"], 48)
    if len(company) < 2 or len(number) < 1:
        raise HTTPException(422, "Vul bedrijfsnaam en factuurnummer in.")
    return {"year": fields["year"], "company": fields["company"], "number": fields["number"],
            "filename": f"{fields['year']} {company} {number}.pdf"}


def _inv_review_entries():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_pdf_review ("
                     "reference TEXT PRIMARY KEY, details TEXT NOT NULL, approved INTEGER NOT NULL, "
                     "updated_at TEXT NOT NULL)")
        return {row[0]: {"invoice_name": json.loads(row[1]), "approved": bool(row[2])}
                for row in conn.execute("SELECT reference,details,approved FROM invoice_pdf_review").fetchall()}


def _inv_draft_details(value):
    if not isinstance(value, dict):
        raise HTTPException(422, "Ongeldige factuurgegevens.")
    fields = {}
    for key, maximum in (("year", 4), ("company", 70), ("number", 48)):
        raw = value.get(key, "")
        if not isinstance(raw, str) or len(raw) > maximum:
            raise HTTPException(422, "Controleer de ingevulde factuurgegevens.")
        fields[key] = raw.strip()
    # Een concept mag onvolledig zijn. Pas bij bevestigen zijn alle velden verplicht.
    try:
        final = _inv_final_filename("", "", fields)
        return {**final, "complete": True, "basis": "Handmatig opgeslagen"}
    except HTTPException:
        return {**fields, "complete": False, "filename": "", "basis": "Concept"}


@app.post("/api/invoice-imap/review")
def invoice_imap_review(payload: dict, request: Request):
    _inv_authorize(request)
    digest = str(payload.get("digest", ""))
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise HTTPException(422, "Ongeldige PDF-verwijzing.")
    message_key = payload.get("message_key")
    if message_key is not None and (not isinstance(message_key, str) or not re.fullmatch(r"[a-f0-9]{64}", message_key)):
        raise HTTPException(422, "Ongeldige mailidentiteit.")
    if message_key in _inv_rejected_messages():
        raise HTTPException(409, "Deze mail is niet geaccepteerd. Het concept is niet gewijzigd.")
    details = _inv_draft_details(payload.get("invoice_name"))
    confirm = payload.get("confirm") is True
    sender, filename = str(payload.get("sender", "")), str(payload.get("filename", ""))
    if confirm:
        if not details["complete"]:
            raise HTTPException(422, "Vul jaartal, bedrijfsnaam en factuurnummer in voordat je bevestigt.")
        original, filename, pdf = (_inv_resend_document if payload.get("previously_sent") is True else _inv_document)(
            str(payload.get("mailbox", "")), str(payload.get("uid", "")), digest)
        _inv_assert_mail_active(str(payload.get("mailbox", "")), original, payload)
        _, source = _inv_pdf_text(pdf)
        if source == "versleuteld":
            raise HTTPException(409, "Deze PDF is versleuteld; gebruik een leesbare PDF.")
        sender = str(original.get("From", ""))
    _inv_review_entries()  # Zorg ook bij de eerste opslag dat de tabel bestaat.
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        conn.execute(f"INSERT INTO invoice_pdf_review(reference,details,approved,updated_at) "
                     f"VALUES ({marker},{marker},{marker},{marker}) ON CONFLICT (reference) DO UPDATE SET "
                     "details=excluded.details,approved=excluded.approved,updated_at=excluded.updated_at",
                     (digest, json.dumps(details, ensure_ascii=False), int(confirm),
                      datetime.now(timezone.utc).isoformat()))
    company = _inv_name_part(details["company"], 55).lower()
    if confirm and len(company) >= 2:
        _inv_learn_supplier(sender, company, filename)
    return {"saved": True, "approved": confirm, "invoice_name": details}


def _inv_approved_name(digest, supplied=None):
    review = _inv_review_entries().get(digest)
    if not review or not review["approved"]:
        raise HTTPException(409, "Bevestig deze PDF eerst met ‘Factuur accepteren’.")
    details = review["invoice_name"]
    if supplied is not None:
        given = _inv_final_filename("", "", supplied)
        if any(given[key] != details[key] for key in ("year", "company", "number")):
            raise HTTPException(409, "Factuurgegevens zijn gewijzigd. Bevestig de factuur opnieuw.")
    return details


def _inv_analysis_key(digest: str, filename: str) -> str:
    return digest + ":" + _inv_hashlib.sha256(filename.casefold().encode("utf-8")).hexdigest()[:16]


def _inv_analysis_cache():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_pdf_analysis "
                     "(reference TEXT PRIMARY KEY, classification TEXT NOT NULL, "
                     "reason TEXT NOT NULL, text_source TEXT NOT NULL)")
        return {row[0]: (row[1], row[2], row[3]) for row in conn.execute(
            "SELECT reference, classification, reason, text_source FROM invoice_pdf_analysis").fetchall()}


def _inv_save_analysis(rows):
    if not rows:
        return
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        for entry in rows:
            conn.execute(f"INSERT INTO invoice_pdf_analysis "
                         f"(reference, classification, reason, text_source) "
                         f"VALUES ({marker}, {marker}, {marker}, {marker}) "
                         "ON CONFLICT (reference) DO NOTHING", entry)


def _inv_name_cache():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_pdf_names "
                     "(reference TEXT PRIMARY KEY, details TEXT NOT NULL)")
        return {row[0]: json.loads(row[1]) for row in conn.execute(
            "SELECT reference, details FROM invoice_pdf_names").fetchall()}


def _inv_save_names(rows):
    if not rows:
        return
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        for key, details in rows:
            conn.execute(f"INSERT INTO invoice_pdf_names(reference,details) VALUES ({marker},{marker}) "
                         "ON CONFLICT (reference) DO UPDATE SET details=excluded.details",
                         (key, json.dumps(details, ensure_ascii=False)))


def _inv_amount_cache():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_pdf_amounts "
                     "(reference TEXT PRIMARY KEY, amount TEXT NOT NULL)")
        return {row[0]: json.loads(row[1]) for row in conn.execute(
            "SELECT reference, amount FROM invoice_pdf_amounts").fetchall()}


def _inv_save_amounts(rows):
    if not rows:
        return
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        for digest, amount in rows:
            conn.execute(f"INSERT INTO invoice_pdf_amounts(reference,amount) VALUES ({marker},{marker}) "
                         "ON CONFLICT (reference) DO UPDATE SET amount=excluded.amount",
                         (digest, json.dumps(amount)))


def _inv_confirmed_digests():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_mail_confirmed "
                     "(reference TEXT PRIMARY KEY, sent_at TEXT NOT NULL)")
        return {row[0] for row in conn.execute("SELECT reference FROM invoice_mail_confirmed").fetchall()}


def _inv_dismissed_digests():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_mail_dismissed "
                     "(reference TEXT PRIMARY KEY, dismissed_at TEXT NOT NULL)")
        return {row[0] for row in conn.execute("SELECT reference FROM invoice_mail_dismissed").fetchall()}


def _inv_message_key(address: str, message) -> str:
    """Stable mail identity, scoped to the mailbox; never a sender/PDF-wide exclusion.

    Ignore IMAP UIDs, which can change after moving a message. Include content
    evidence to distinguish a supplier reusing a Message-ID. Without Message-ID,
    use the complete message (including transport headers) to avoid collapsing
    separate deliveries that happen to have identical attachments.
    """
    message_id = str(message.get("Message-ID", "")).strip()
    if message_id:
        evidence = [message_id] + [str(message.get(k, "")) for k in ("From", "Date", "Subject")]
        for part in message.walk():
            if part.is_multipart():
                continue
            data = part.get_payload(decode=True)
            if data is None:
                data = str(part.get_payload()).encode("utf-8", "replace")
            evidence.append(part.get_content_type() + ":" + _inv_hashlib.sha256(data).hexdigest())
        identity = json.dumps(evidence, ensure_ascii=False).encode("utf-8")
    else:
        identity = message.as_bytes(policy=_inv_policy.default)
    return _inv_hashlib.sha256(address.strip().lower().encode("utf-8") + b"\n" + identity).hexdigest()


def _inv_rejected_messages():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_mail_rejections "
                     "(reference TEXT PRIMARY KEY, mailbox TEXT NOT NULL, rejected_at TEXT NOT NULL)")
        return {row[0] for row in conn.execute(
            "SELECT reference FROM invoice_mail_rejections WHERE rejected_at <> ''").fetchall()}


def _inv_delivery_digests():
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_mail_delivery "
                     "(reference TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
        return {row[0] for row in conn.execute("SELECT reference FROM invoice_mail_delivery").fetchall()
                if re.fullmatch(r"[a-f0-9]{64}", row[0])}


def _inv_validate_message_key(payload, address, message):
    actual = _inv_message_key(address, message)
    expected = payload.get("message_key")
    if expected is not None:
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise HTTPException(422, "Ongeldige mailidentiteit.")
        if not _inv_hmac.compare_digest(expected, actual):
            raise HTTPException(409, "Deze mail is gewijzigd of verplaatst. Scan het postvak opnieuw.")
    return actual


def _inv_assert_mail_active(address, message, payload=None):
    key = _inv_validate_message_key(payload or {}, address, message)
    if key in _inv_rejected_messages():
        raise HTTPException(409, "Deze specifieke mail is niet geaccepteerd en blijft in je postvak staan.")
    return key


def _inv_lock_mail_decision(conn, key, address):
    """Serialize rejection and send reservations, also across server processes.

    A SQLite INSERT obtains its write lock; PostgreSQL locks this individual
    source-mail row until the transaction commits. No SMTP/IMAP writes here.
    """
    marker = "%s" if _postgres_enabled() else "?"
    conn.execute(f"INSERT INTO invoice_mail_rejections(reference,mailbox,rejected_at) "
                 f"VALUES ({marker},{marker},'') ON CONFLICT (reference) DO NOTHING", (key, address))
    row = conn.execute(f"SELECT rejected_at FROM invoice_mail_rejections WHERE reference={marker}" +
                       (" FOR UPDATE" if _postgres_enabled() else ""), (key,)).fetchone()
    return bool(row and row[0])


def _inv_trash_folder(imap):
    result, folders = imap.list()
    if result != "OK":
        return None
    candidates = []
    for raw in folders or []:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        match = re.match(r'^\(([^)]*)\)\s+(?:"[^"]*"|\S+)\s+(?:"([^"]+)"|(\S+))$', line)
        if not match:
            continue
        flags, quoted, bare = match.groups()
        name = quoted or bare
        if "\\Noselect" in flags:
            continue
        if "\\Trash" in flags:
            return name
        if name.casefold().split("/")[-1].split(".")[-1] in {
                "trash", "prullenbak", "deleted items", "deleted messages", "bin"}:
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def _inv_has_unprocessed_pdf_parts(message, pdfs):
    """Never remove hidden/oversized/broken PDF attachments just because scanning skipped them."""
    parts = [part for part in message.walk() if not part.is_multipart() and (
        (part.get_filename() or "").lower().endswith(".pdf") or
        part.get_content_type().lower() == "application/pdf")]
    return len(parts) != len(pdfs)


def _inv_quoted_folder(name):
    # IMAP mailbox names can contain spaces, quotes, or backslashes.
    return '"' + str(name).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _inv_move_completed_mail(address: str, uid: str, expected_digest: str = ""):
    # A rejected source mail always stays put. All PDFs of other mails must be submitted.
    imap = None
    try:
        imap = _inv_mailbox(address, writable=True)
        message = _inv_fetch(imap, uid)
        if _inv_message_key(address, message) in _inv_rejected_messages():
            return False, "Deze mail is niet geaccepteerd en blijft in Postvak IN staan."
        all_pdfs = list(_inv_attachments(message))
        if not all_pdfs or _inv_has_unprocessed_pdf_parts(message, all_pdfs):
            return False, "Deze bronmail bevat een te grote, onleesbare of niet herkende PDF en blijft in Postvak IN."
        pdfs = _inv_invoice_attachments(message)
        if not pdfs:
            return False, "In deze bronmail is geen bruikbare factuur-PDF gevonden; de mail blijft in Postvak IN."
        actual = {_inv_hashlib.sha256(pdf).hexdigest() for _, pdf in pdfs}
        if expected_digest and expected_digest not in actual:
            return False, "De bronmail komt niet meer overeen met de factuur; de mail is niet verplaatst."
        confirmed = _inv_confirmed_digests()
        # At least one successfully submitted PDF is required; never delete a merely approved draft.
        if not actual or not actual.issubset(confirmed):
            return False, "Deze bronmail heeft nog niet afgehandelde PDF-bijlagen en blijft in Postvak IN."
        trash = _inv_trash_folder(imap)
        if not trash:
            return False, "De prullenbakmap van dit postvak is niet herkenbaar; de mail blijft in Postvak IN."
        try:
            result, _ = imap.uid("MOVE", uid, _inv_quoted_folder(trash))
        except _inv_imaplib.IMAP4.error:
            result = "NO"
        if result != "OK":
            return False, "Veilig verplaatsen naar de prullenbak is niet gelukt; controleer Postvak IN."
        return True, "Bronmail naar de prullenbak verplaatst."
    except Exception:
        return False, "Factuur doorgestuurd, maar verplaatsen naar de prullenbak is mislukt; controleer Postvak IN."
    finally:
        try:
            if imap is not None: imap.logout()
        except Exception: pass


@app.post("/api/invoice-imap/cleanup")
def invoice_imap_cleanup(payload: dict, request: Request):
    """Retry moving submitted source mails, without ever submitting a PDF a second time."""
    _inv_authorize(request)
    documents = payload.get("documents")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 100:
        raise HTTPException(422, "Selecteer 1 tot 100 bronmails om af te handelen.")
    confirmed = _inv_confirmed_digests()
    groups = {}
    # Validate the entire request before making mailbox changes.
    for doc in documents:
        if not isinstance(doc, dict):
            raise HTTPException(422, "Ongeldige bronmail.")
        address, uid, digest = (str(doc.get(key, "")) for key in ("mailbox", "uid", "digest"))
        if address not in _INV_ADDRESSES or not re.fullmatch(r"[0-9]{1,20}", uid) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise HTTPException(422, "Ongeldige bronmail of PDF-verwijzing.")
        if digest not in confirmed:
            raise HTTPException(409, "Deze PDF is niet succesvol doorgestuurd; de bronmail blijft staan.")
        groups.setdefault((address, uid), digest)
    results = []
    for (address, uid), digest in groups.items():
        moved, message = _inv_move_completed_mail(address, uid, digest)
        results.append({"mailbox": address, "uid": uid, "moved_to_trash": moved, "move_message": message})
    return {"results": results, "moved": sum(item["moved_to_trash"] for item in results)}


def _inv_fetch(imap, uid: str):
    if not uid.isascii() or not uid.isdigit() or len(uid)>20:
        raise HTTPException(400, "Ongeldige mailreferentie.")
    # PEEK voorkomt dat een scan nieuwe mails in Outlook/TransIP als gelezen markeert.
    result, data = imap.uid("FETCH", uid, "(BODY.PEEK[])")
    if result != "OK" or not data or not any(isinstance(item, tuple) for item in data):
        raise HTTPException(404, "Mail is niet meer beschikbaar.")
    raw = next(item[1] for item in data if isinstance(item, tuple))
    if len(raw) > 12_000_000:
        raise HTTPException(413, "Mail is te groot.")
    return _inv_email.message_from_bytes(raw, policy=_inv_policy.default)


def _inv_document(address: str, uid: str, digest: str):
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise HTTPException(400, "Ongeldige bijlagereferentie.")
    imap = _inv_mailbox(address)
    try:
        msg = _inv_fetch(imap, uid)
        for name, pdf in _inv_invoice_attachments(msg):
            if _inv_hmac.compare_digest(_inv_hashlib.sha256(pdf).hexdigest(), digest):
                return msg, name, pdf
        raise HTTPException(404, "PDF is verwijderd of gewijzigd.")
    finally:
        try: imap.logout()
        except Exception: pass


def _inv_resend_document(address: str, uid: str, digest: str):
    """Een eerder verplaatste mail is voor opnieuw verzenden ook in Prullenbak te vinden."""
    try:
        return _inv_document(address, uid, digest)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
    imap = _inv_mailbox(address)
    try:
        trash = _inv_trash_folder(imap)
        if not trash or imap.select(_inv_quoted_folder(trash), readonly=True)[0] != "OK":
            raise HTTPException(404, "De originele mail is niet meer te vinden in de prullenbak.")
        since = (datetime.now(timezone.utc)-_inv_timedelta(days=90)).strftime("%d-%b-%Y")
        result, found = imap.uid("SEARCH", None, "SINCE", since)
        if result == "OK":
            for trash_uid in reversed((found[0] or b"").split()[-250:]):
                try:
                    message = _inv_fetch(imap, trash_uid.decode("ascii"))
                    for name, pdf in _inv_invoice_attachments(message):
                        if _inv_hmac.compare_digest(_inv_hashlib.sha256(pdf).hexdigest(), digest):
                            return message, name, pdf
                except (HTTPException, UnicodeDecodeError):
                    continue
        raise HTTPException(404, "De originele PDF staat niet meer in Postvak IN of Prullenbak.")
    finally:
        try: imap.logout()
        except Exception: pass


@app.get("/api/invoice-imap/status")
def invoice_imap_status(request: Request):
    _inv_authorize(request)
    return {"accounts": [{"address": address, "configured": bool(os.environ.get(env))}
                          for address, env in _INV_ADDRESSES.items()], "pdf_reader": _InvPdfReader is not None,
            "forwarding_from": _INV_FORWARD_FROM,
            "message_rejection_ready": True,
            "forwarding_ready": bool(os.environ.get(_INV_ADDRESSES[_INV_FORWARD_FROM]))}


@app.get("/api/invoice-imap/scan")
def invoice_imap_scan(request: Request, mailbox: str = ""):
    _inv_authorize(request)
    if _InvPdfReader is None:
        raise HTTPException(503, "PDF-herkenning ontbreekt op de server: installeer pypdf.")
    if mailbox and mailbox not in _INV_ADDRESSES:
        raise HTTPException(400, "Onbekend postvak.")
    since = (datetime.now(timezone.utc)-_inv_timedelta(days=30)).strftime("%d-%b-%Y")
    docs, errors = [], []
    rejected_messages = _inv_rejected_messages()
    analysis, fresh_analysis = _inv_analysis_cache(), []
    names, fresh_names = _inv_name_cache(), []
    amounts, fresh_amounts = _inv_amount_cache(), []
    suppliers = _inv_supplier_directory()
    reviews = _inv_review_entries()
    for address in ((mailbox,) if mailbox else _INV_ADDRESSES):
        try:
            imap = _inv_mailbox(address)
            try:
                result, found = imap.uid("SEARCH", None, "SINCE", since)
                if result != "OK": raise ValueError("Zoeken mislukt")
                for uidbytes in reversed((found[0] or b"").split()[-120:]):
                    uid = uidbytes.decode("ascii")
                    try: msg = _inv_fetch(imap, uid)
                    except HTTPException: continue
                    message_key = _inv_message_key(address, msg)
                    if message_key in rejected_messages:
                        continue
                    sender = _inv_email_utils.parseaddr(str(msg.get("From", "")))[1].lower()
                    received = str(msg.get("Date", ""))
                    try: received = _inv_parsedate(received).astimezone(timezone.utc).isoformat()
                    except Exception: received = ""
                    for filename, pdf in _inv_invoice_attachments(msg):
                        digest = _inv_hashlib.sha256(pdf).hexdigest()
                        # Toon ook een tweede mailkopie uit het andere postvak.
                        # De verzendregistratie blijft per PDF-digest uniek.
                        analysis_key = _inv_analysis_key(digest, filename) + ":class-v3"
                        text = None
                        if analysis_key not in analysis:
                            text, source = _inv_pdf_text(pdf)
                            kind, reason = _inv_kind(text, filename, source)
                            analysis[analysis_key] = (kind, reason, source)
                            fresh_analysis.append((analysis_key, kind, reason, source))
                        kind, reason, source = analysis[analysis_key]
                        name = None
                        if kind in {"invoice", "review"}:
                            name_key = _inv_analysis_key(digest, filename) + ":n4"
                            name = names.get(name_key)
                            if not isinstance(name, dict) or name.get("extractor_version") != 4:
                                if text is None:
                                    text, source = _inv_pdf_text(pdf)
                                name = _inv_name_with_filename(text, sender, filename)
                                name["basis"] = source
                                name["extractor_version"] = 4
                                if source != "PDF-tekst":
                                    name["complete"] = False
                                names[name_key] = name
                                fresh_names.append((name_key, name))
                        amount = None
                        if kind in {"invoice", "review"}:
                            amount_key = digest + ":amount-v2"
                            if amount_key not in amounts:
                                if text is None:
                                    text, _ = _inv_pdf_text(pdf)
                                amounts[amount_key] = _inv_total_amount(text)
                                fresh_amounts.append((amount_key, amounts[amount_key]))
                            amount = amounts[amount_key]
                        visible_name = _inv_known_supplier(name, sender, suppliers, filename) if name else None
                        if visible_name and (source != "PDF-tekst" or kind != "invoice"):
                            visible_name["complete"] = False
                        review = reviews.get(digest, {})
                        docs.append({"mailbox": address, "uid": uid, "digest": digest, "message_key": message_key,
                                     "sender": sender, "received": received,
                                     "subject": str(msg.get("Subject", ""))[:200],
                                     "filename": filename[:200], "size": len(pdf),
                                     "classification": "invoice" if review.get("approved") else kind, "reason": reason,
                                     "text_source": source, "invoice_name": visible_name,
                                     "invoice_draft": review.get("invoice_name"), "approved": bool(review.get("approved")),
                                     "amount": amount})
            finally:
                try: imap.logout()
                except Exception: pass
        except Exception:
            errors.append(f"{address}: postvak niet ingesteld of niet bereikbaar")
    _inv_save_analysis(fresh_analysis)
    _inv_save_names(fresh_names)
    _inv_save_amounts(fresh_amounts)
    confirmed = _inv_confirmed_digests()
    pending = _inv_delivery_digests() - confirmed
    for doc in docs:
        doc["sent"] = doc["digest"] in confirmed
        doc["delivery_uncertain"] = doc["digest"] in pending
    return {"documents": docs, "errors": errors,
            "scanned_mailboxes": [mailbox] if mailbox else list(_INV_ADDRESSES)}


@app.post("/api/invoice-imap/dismiss")
def invoice_imap_dismiss(payload: dict, request: Request):
    """Exclude a specific source mail. Never move, flag, or delete that mail."""
    _inv_authorize(request)
    documents = payload.get("documents")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 100:
        raise HTTPException(400, "Selecteer 1 tot 100 documenten.")
    groups = {}
    # Read-only verification of every requested source, before recording anything.
    for item in documents:
        if not isinstance(item, dict):
            raise HTTPException(400, "Ongeldige documentselectie.")
        address, uid, digest = (str(item.get(field, "")) for field in ("mailbox", "uid", "digest"))
        message, _, _ = _inv_document(address, uid, digest)
        key = _inv_validate_message_key(item, address, message)
        group = groups.setdefault(key, {"mailbox": address, "uid": uid, "message_key": key,
                                       "requested": set(), "actual": set()})
        group["requested"].add(digest)
        group["actual"].update(_inv_hashlib.sha256(pdf).hexdigest() for _, pdf in _inv_attachments(message))
    _inv_rejected_messages()
    _inv_confirmed_digests()
    _inv_delivery_digests()
    with _db_connect() as conn:
        marker = "%s" if _postgres_enabled() else "?"
        # Fixed ordering avoids deadlocks when two clients select multiple mails.
        for key in sorted(groups):
            _inv_lock_mail_decision(conn, key, groups[key]["mailbox"])
        confirmed = {row[0] for row in conn.execute("SELECT reference FROM invoice_mail_confirmed").fetchall()}
        delivering = {row[0] for row in conn.execute("SELECT reference FROM invoice_mail_delivery").fetchall()}
        for group in groups.values():
            if group["requested"] & confirmed:
                raise HTTPException(409, "Dit document is al doorgestuurd en blijft onder Doorgestuurd staan.")
            if group["actual"] & (delivering - confirmed):
                raise HTTPException(409, "Een verzending uit deze mail loopt nog of is onzeker. Controleer eerst de ontvangst.")
        now = datetime.now(timezone.utc).isoformat()
        for key in groups:
            conn.execute(f"UPDATE invoice_mail_rejections SET rejected_at={marker} "
                         f"WHERE reference={marker}", (now, key))
    return {"dismissed": len(groups), "mailbox_unchanged": True, "scope": "message",
            "messages": [{key: group[key] for key in ("mailbox", "uid", "message_key")}
                         for group in groups.values()]}


@app.post("/api/invoice-imap/trash")
def invoice_imap_trash(payload: dict, request: Request):
    _inv_authorize(request)
    address, uid = (str(payload.get(field, "")) for field in ("mailbox", "uid"))
    requested = payload.get("digests")
    if not isinstance(requested, list) or not 1 <= len(requested) <= 30 or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in requested):
        raise HTTPException(400, "Ongeldige selectie van PDF-bijlagen.")
    imap = _inv_mailbox(address, writable=True)
    try:
        message = _inv_fetch(imap, uid)
        all_valid_pdfs = list(_inv_attachments(message))
        if _inv_has_unprocessed_pdf_parts(message, all_valid_pdfs):
            raise HTTPException(409, "Deze mail heeft ook een te grote of onleesbare PDF; verplaats hem handmatig.")
        originals = _inv_invoice_attachments(message)
        actual = {_inv_hashlib.sha256(pdf).hexdigest() for _, pdf in originals}
        if not actual or actual != set(requested):
            raise HTTPException(409, "Deze mail bevat ook andere PDF’s. Selecteer alle PDF’s uit deze mail of verplaats de mail handmatig.")
        confirmed = _inv_confirmed_digests()
        accepted = {key for key, entry in _inv_review_entries().items() if entry["approved"]}
        for filename, pdf in originals:
            digest = _inv_hashlib.sha256(pdf).hexdigest()
            text, source = _inv_pdf_text(pdf)
            kind, _ = _inv_kind(text, filename, source)
            if kind != "invoice" and digest not in confirmed and digest not in accepted:
                raise HTTPException(409, "Deze mail bevat een PDF die niet als factuur is bevestigd; verplaats de mail handmatig.")
        trash = _inv_trash_folder(imap)
        if not trash:
            raise HTTPException(503, "Prullenbakmap niet gevonden; de mail staat nog in Postvak IN.")
        try:
            result, _ = imap.uid("MOVE", uid, _inv_quoted_folder(trash))
        except _inv_imaplib.IMAP4.error as exc:
            raise HTTPException(502, "Verplaatsen naar de prullenbak is mislukt.") from exc
        if result != "OK":
            raise HTTPException(502, "Verplaatsen naar de prullenbak is mislukt; de mail staat nog in Postvak IN.")
        return {"moved": True, "mailbox": address, "uid": uid}
    finally:
        try: imap.logout()
        except Exception: pass


@app.post("/api/invoice-imap/pdf")
def invoice_imap_pdf(payload: dict, request: Request):
    _inv_authorize(request)
    _, name, pdf = (_inv_resend_document if payload.get("previously_sent") is True else _inv_document)(
        str(payload.get("mailbox", "")), str(payload.get("uid", "")), str(payload.get("digest", "")))
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "inline; filename=invoice.pdf", "Cache-Control": "no-store"})


@app.post("/api/invoice-imap/name")
def invoice_imap_name(payload: dict, request: Request):
    _inv_authorize(request)
    review = _inv_review_entries().get(str(payload.get("digest", "")))
    if review:
        return review["invoice_name"]
    original, filename, pdf = _inv_resend_document(str(payload.get("mailbox", "")),
        str(payload.get("uid", "")), str(payload.get("digest", ""))) if payload.get("previously_sent") is True else _inv_document(
        str(payload.get("mailbox", "")), str(payload.get("uid", "")), str(payload.get("digest", "")))
    text, source = _inv_pdf_text(pdf)
    suggestion = _inv_known_supplier(_inv_name_with_filename(text, str(original.get("From", "")), filename),
                                     str(original.get("From", "")), _inv_supplier_directory(), filename)
    suggestion["basis"] = source
    if source != "PDF-tekst":
        suggestion["complete"] = False  # OCR en onleesbare PDF's altijd zelf controleren.
    return suggestion


@app.post("/api/invoice-imap/send")
def invoice_imap_send(payload: dict, request: Request):
    _inv_authorize(request)
    destination = str(payload.get("destination", "")).strip()
    if not re.fullmatch(r"[^@\s]+@e-boekhouden\.nl", destination, re.I):
        raise HTTPException(400, "Ongeldig e-Boekhouden-adres.")
    repeat_id = str(payload.get("resend_id", ""))
    repeat = bool(repeat_id)
    if repeat and not re.fullmatch(r"[a-f0-9-]{36}", repeat_id):
        raise HTTPException(400, "Ongeldige verzendbevestiging.")
    digest = str(payload.get("digest", ""))
    final_name = _inv_approved_name(digest, payload.get("invoice_name"))
    if repeat and digest not in _inv_confirmed_digests():
        raise HTTPException(409, "Deze PDF is nog niet eerder doorgestuurd.")
    original, filename, pdf = (_inv_resend_document if repeat else _inv_document)(
        str(payload.get("mailbox", "")), str(payload.get("uid", "")), digest)
    source_address = str(payload.get("mailbox", ""))
    message_key = _inv_assert_mail_active(source_address, original, payload)
    text, source = _inv_pdf_text(pdf)
    kind, _ = _inv_kind(text, filename, source)
    if source == "versleuteld":
        raise HTTPException(409, "Deze PDF is versleuteld; gebruik een leesbare PDF.")
    cfg = _inv_smtp_settings()
    # Reserveer vóór het verzenden: bij een timeout kan SMTP al afgeleverd hebben.
    reference = digest + (":repeat:" + repeat_id if repeat else "")
    _inv_delivery_digests()
    with _db_connect() as conn:
        if _inv_lock_mail_decision(conn, message_key, source_address):
            raise HTTPException(409, "Deze mail is niet geaccepteerd en wordt niet doorgestuurd.")
        marker = "%s" if _postgres_enabled() else "?"
        result = conn.execute(f"INSERT INTO invoice_mail_delivery (reference, created_at) VALUES ({marker}, {marker}) ON CONFLICT (reference) DO NOTHING",
                              (reference, datetime.now(timezone.utc).isoformat()))
        if result.rowcount == 0:
            raise HTTPException(409, "Deze PDF is al verzonden of de ontvangst moet nog worden gecontroleerd.")
    mail = EmailMessage()
    mail["From"] = cfg.get("from") or cfg["user"]
    mail["To"] = destination
    mail["Subject"] = "Inkoopfactuur: " + str(original.get("Subject", ""))[:120].replace("\n", " ").replace("\r", " ")
    mail.set_content("Afzender: " + _inv_email_utils.parseaddr(str(original.get("From", "")))[1] + "\nBronpostvak: " + str(payload.get("mailbox", "")))
    mail.add_attachment(pdf, maintype="application", subtype="pdf", filename=final_name["filename"])
    try:
        context = ssl.create_default_context()
        if cfg.get("ssl"):
            with smtplib.SMTP_SSL(cfg["host"], int(cfg["port"]), timeout=25, context=context) as smtp:
                smtp.login(cfg["user"], cfg["password"])
                refused = smtp.send_message(mail)
        else:
            with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=25) as smtp:
                smtp.starttls(context=context)
                smtp.login(cfg["user"], cfg["password"])
                refused = smtp.send_message(mail)
        if refused:
            raise ValueError("De ontvangende mailserver heeft de ontvanger geweigerd.")
    except Exception as exc:
        raise HTTPException(502, "SMTP-verzending mislukt; controleer eerst of de mail toch is aangekomen.") from exc
    with _db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS invoice_mail_confirmed "
                     "(reference TEXT PRIMARY KEY, sent_at TEXT NOT NULL)")
        marker = "%s" if _postgres_enabled() else "?"
        conn.execute(f"INSERT INTO invoice_mail_confirmed (reference, sent_at) VALUES ({marker}, {marker}) "
                     "ON CONFLICT (reference) DO NOTHING", (reference, datetime.now(timezone.utc).isoformat()))
    moved, move_message = (False, "")
    if not repeat:
        moved, move_message = _inv_move_completed_mail(str(payload.get("mailbox", "")), str(payload.get("uid", "")), digest)
    return {"sent": True, "resent": repeat, "submitted_to_smtp": True,
            "forwarding_from": cfg["from"], "filename": final_name["filename"],
            "moved_to_trash": moved, "move_message": move_message}


from vakstaal_auth import install_auth

install_auth(app, _db_connect, _postgres_enabled)
# Outermost: authentication errors also receive the restricted CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("VAKSTAAL_APP_ORIGIN", "https://vakstaal-calculator.vercel.app").rstrip("/")],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Invoice-Key"],
    expose_headers=["Content-Disposition", "Retry-After", "X-STEP-Selected-Count"],
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
