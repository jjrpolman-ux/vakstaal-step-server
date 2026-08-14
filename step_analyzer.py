from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np


def _cluster(values: list[float], tol: float = 0.25) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for value in sorted(v for v in values if v > 1e-6):
        if not groups or abs(value - groups[-1]["mean"]) > tol:
            groups.append({"values": [value], "mean": value})
        else:
            groups[-1]["values"].append(value)
            groups[-1]["mean"] = sum(groups[-1]["values"]) / len(groups[-1]["values"])
    return groups


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = sorted(set((round(float(x), 8), round(float(y), 8)) for x, y in points))
    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _minimum_rectangle_dimensions(points: np.ndarray) -> tuple[float, float]:
    hull = _convex_hull_2d(points)
    if len(hull) < 2:
        return 0.0, 0.0

    best = None
    for i in range(len(hull)):
        delta = hull[(i + 1) % len(hull)] - hull[i]
        angle = math.atan2(delta[1], delta[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rotation = np.array([[c, -s], [s, c]])
        rotated = hull @ rotation.T
        extents = rotated.max(axis=0) - rotated.min(axis=0)
        area = extents[0] * extents[1]
        if best is None or area < best[0]:
            best = (area, float(extents[0]), float(extents[1]))

    return tuple(sorted(best[1:], reverse=True))  # type: ignore[return-value]


def _principal_dimensions(solid: cq.Shape) -> tuple[float, float, float]:
    vertices = solid.Vertices()
    points = np.array([v.toTuple() for v in vertices], dtype=float)

    if len(points) < 4:
        bb = solid.BoundingBox()
        dims = sorted([bb.xlen, bb.ylen, bb.zlen], reverse=True)
        return float(dims[0]), float(dims[1]), float(dims[2])

    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    length_axis = eigenvectors[:, 0]
    length = float(np.ptp(centered @ length_axis))

    e2, e3 = eigenvectors[:, 1], eigenvectors[:, 2]
    projected = np.column_stack((centered @ e2, centered @ e3))
    width, height = _minimum_rectangle_dimensions(projected)
    return length, width, height



def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    vector = vector / norm
    for component in vector:
        if abs(component) > 1e-8:
            if component < 0:
                vector = -vector
            break
    return vector


def _snap_profile_dimension(value: float) -> float:
    """
    Snap STEP geometry to common nominal steel profile sizes.

    Tabs, slots and weldment details can move the dominant detected face by a
    few millimetres.  The tolerance therefore grows slightly with profile size:
    around a 100 mm profile we allow roughly +/-3.5 mm.
    """
    common = [10, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80,
              90, 100, 120, 125, 140, 150, 160, 180, 200, 220, 250, 300]
    nearest = min(common, key=lambda x: abs(x - value))
    tolerance = max(1.0, min(4.0, float(nearest) * 0.035))
    return float(nearest) if abs(nearest - value) <= tolerance else value


def _snap_wall_thickness(value: float | None) -> float | None:
    if value is None:
        return None
    common = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    nearest = min(common, key=lambda x: abs(x - value))
    return float(nearest) if abs(nearest - value) <= 0.30 else value


def _robust_rectangular_profile(solid: cq.Shape, length_axis: np.ndarray) -> dict[str, Any] | None:
    """
    Recover the original RHS/SHS cross-section from dominant long planar faces.

    This deliberately ignores small faces created by tabs, slots and local cut-outs.
    For a 100x100x1.5 tube, the dominant side planes are typically at +/-50
    and +/-48.5 mm. A protruding tab may add a face at e.g. +60 or +70 mm,
    but its area is tiny compared with the long tube faces and is rejected.
    """
    length_axis = np.asarray(length_axis, dtype=float)
    length_axis /= max(np.linalg.norm(length_axis), 1e-12)

    face_data: list[dict[str, Any]] = []
    for face in solid.Faces():
        if face.geomType() != "PLANE":
            continue
        try:
            normal = np.array(face.normalAt().toTuple(), dtype=float)
        except Exception:
            continue
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-9:
            continue
        normal /= n_norm

        # End faces point along the tube axis; side faces are nearly perpendicular.
        if abs(float(np.dot(normal, length_axis))) > 0.22:
            continue

        transverse = normal - float(np.dot(normal, length_axis)) * length_axis
        t_norm = np.linalg.norm(transverse)
        if t_norm < 1e-8:
            continue
        direction = _canonical_direction(transverse / t_norm)
        center = np.array(face.Center().toTuple(), dtype=float)
        area = float(face.Area())
        if area <= 1e-4:
            continue
        face_data.append({"area": area, "dir": direction, "center": center, "face": face})

    if len(face_data) < 4:
        return None

    # Cluster side-face normals into the two cross-section directions, weighted by area.
    direction_groups: list[dict[str, Any]] = []
    for item in sorted(face_data, key=lambda x: x["area"], reverse=True):
        placed = False
        for group in direction_groups:
            dot = float(np.dot(item["dir"], group["dir"]))
            if abs(dot) >= 0.985:
                group["items"].append(item)
                group["area"] += item["area"]
                placed = True
                break
        if not placed:
            direction_groups.append({"dir": item["dir"], "items": [item], "area": item["area"]})

    direction_groups.sort(key=lambda g: g["area"], reverse=True)
    axes: list[dict[str, Any]] = []
    for group in direction_groups:
        if not axes:
            axes.append(group)
        elif abs(float(np.dot(group["dir"], axes[0]["dir"]))) <= 0.20:
            axes.append(group)
            break
    if len(axes) < 2:
        return None

    dimensions: list[float] = []
    thickness_samples: list[float] = []
    axis_quality: list[float] = []

    for axis_group in axes:
        direction = axis_group["dir"]
        offsets: list[tuple[float, float]] = []
        for item in face_data:
            if abs(float(np.dot(item["dir"], direction))) >= 0.985:
                offsets.append((float(np.dot(item["center"], direction)), item["area"]))
        if len(offsets) < 4:
            return None

        offsets.sort(key=lambda x: x[0])
        plane_groups: list[dict[str, Any]] = []
        for offset, area in offsets:
            if not plane_groups or abs(offset - plane_groups[-1]["mean"]) > 0.35:
                plane_groups.append({"values": [offset], "mean": offset, "area": area})
            else:
                group = plane_groups[-1]
                group["values"].append(offset)
                group["area"] += area
                group["mean"] = sum(group["values"]) / len(group["values"])

        max_area = max(g["area"] for g in plane_groups)
        # Long tube faces dominate local tab/slot faces. 12% leaves room for
        # substantial cut-outs while still rejecting protrusion faces.
        major = [g for g in plane_groups if g["area"] >= max_area * 0.12]
        if len(major) < 4:
            return None

        major.sort(key=lambda g: g["mean"])
        outer_low, outer_high = major[0], major[-1]
        dimension = outer_high["mean"] - outer_low["mean"]
        if dimension <= 1.0:
            return None
        dimensions.append(dimension)

        mid = (outer_low["mean"] + outer_high["mean"]) / 2.0
        negative = [g for g in major if g["mean"] < mid]
        positive = [g for g in major if g["mean"] > mid]
        if len(negative) >= 2:
            t = negative[1]["mean"] - negative[0]["mean"]
            if 0.3 <= t <= dimension * 0.20:
                thickness_samples.append(t)
        if len(positive) >= 2:
            t = positive[-1]["mean"] - positive[-2]["mean"]
            if 0.3 <= t <= dimension * 0.20:
                thickness_samples.append(t)

        outer_area = outer_low["area"] + outer_high["area"]
        axis_quality.append(min(1.0, outer_area / max(1e-9, 2 * max_area)))

    if len(dimensions) != 2:
        return None

    width, height = sorted(dimensions, reverse=True)
    width = _snap_profile_dimension(width)
    height = _snap_profile_dimension(height)

    face_thickness = None
    if thickness_samples:
        face_thickness = float(np.median(thickness_samples))
        face_thickness = _snap_wall_thickness(face_thickness)

    # Square classification after nominal dimension snapping. This deliberately
    # turns e.g. 103.4 x 100 and 100 x 98.3 back into nominal 100 x 100.
    profile_type = "Vierkant" if abs(width - height) <= max(1.2, min(width, height) * 0.045) else "Rechthoekig"
    if profile_type == "Vierkant":
        square = _snap_profile_dimension((width + height) / 2.0)
        width = height = square

    # A second thickness candidate comes from volume/length using the NOMINAL
    # profile size. Across several bodies this is very useful for tabs/slots:
    # local faces can suggest 2.0 mm while the common material volume still
    # clusters around the actual 1.5 mm wall.
    # Use the already-known longitudinal size when available; otherwise derive
    # it directly from the solid here. This avoids an undefined local `length`.
    profile_length = _solid_length_mm(solid)
    volume_thickness = _estimated_rectangular_thickness(solid, width, height, profile_length)
    volume_thickness = _snap_wall_thickness(volume_thickness)

    thickness = face_thickness if face_thickness is not None else volume_thickness

    return {
        "type": profile_type,
        "outer_width_mm": width,
        "outer_height_mm": height,
        "thickness_mm": thickness,
        "face_thickness_mm": face_thickness,
        "volume_thickness_mm": volume_thickness,
        "thickness_source": "dominant-faces" if face_thickness is not None else ("volume-estimate" if volume_thickness is not None else "unknown"),
        "profile_source": "dominant-faces",
        "confidence": round(float(np.mean(axis_quality)), 3) if axis_quality else 0.0,
    }

def _exact_rectangular_thickness(solid: cq.Shape, width: float, height: float, length: float) -> float | None:
    line_lengths = [edge.Length() for edge in solid.Edges() if edge.geomType() == "LINE"]
    short_lines = [value for value in line_lengths if value < length * 0.6]
    groups = _cluster(short_lines, 0.25)
    means = [group["mean"] for group in groups if len(group["values"]) >= 2 and group["mean"] > 1]

    best = None
    for inner_w in means:
        if inner_w >= width - 0.5:
            continue
        t_w = (width - inner_w) / 2
        if not 0.5 <= t_w <= 15:
            continue
        for inner_h in means:
            if inner_h >= height - 0.5:
                continue
            t_h = (height - inner_h) / 2
            if not 0.5 <= t_h <= 15:
                continue
            if abs(t_w - t_h) <= 0.35:
                score = abs(t_w - t_h)
                if best is None or score < best[0]:
                    best = (score, (t_w + t_h) / 2)
    return best[1] if best else None


def _solid_length_mm(solid) -> float:
    """Return the longest bounding-box dimension in mm as profile length."""
    try:
        bb = solid.bounding_box()
        dims = [float(bb.xlen), float(bb.ylen), float(bb.zlen)]
        return max(dims)
    except Exception:
        try:
            bb = solid.BoundingBox()
            dims = [float(bb.XLength), float(bb.YLength), float(bb.ZLength)]
            return max(dims)
        except Exception:
            return 0.0


def _estimated_rectangular_thickness(solid: cq.Shape, width: float, height: float, length: float) -> float | None:
    area = solid.Volume() / max(length, 1e-6)
    discriminant = (width + height) ** 2 - 4 * area
    if discriminant < 0:
        return None
    thickness = ((width + height) - math.sqrt(discriminant)) / 4
    if 0.3 <= thickness <= min(width, height) / 2:
        return thickness
    return None


def _round_profile(solid: cq.Shape, length: float, width: float, height: float) -> dict[str, Any] | None:
    radii: list[float] = []
    for edge in solid.Edges():
        if edge.geomType() != "CIRCLE":
            continue
        try:
            radius = float(edge.radius())
        except Exception:
            radius = edge.Length() / (2 * math.pi)
        radii.append(radius)

    groups = _cluster(radii, 0.20)
    groups = sorted(
        [group for group in groups if len(group["values"]) >= 2],
        key=lambda group: group["mean"],
        reverse=True,
    )
    if not groups:
        return None

    outer_radius = groups[0]["mean"]
    if outer_radius * 2 < max(width, height) * 0.70:
        return None

    inner_radius = groups[1]["mean"] if len(groups) > 1 else None
    thickness = outer_radius - inner_radius if inner_radius else None
    return {
        "type": "Rond",
        "diameter_mm": outer_radius * 2,
        "thickness_mm": thickness,
        "length_mm": length,
        "thickness_source": "edges" if thickness is not None else "unknown",
    }


def _round_nearest(value: float, step: float = 0.1) -> float:
    return round(value / step) * step


def _fmt_number(value: float | None) -> str:
    if value is None:
        return ""
    rounded = _round_nearest(value, 0.1)
    if abs(rounded - round(rounded)) < 0.05:
        return str(int(round(rounded)))
    return f"{rounded:.1f}".replace(".", ",")


def analyze_solid(solid: cq.Shape, index: int) -> dict[str, Any]:
    # PCA is used only to find the long axis. Cross-section dimensions are then
    # recovered from dominant planar faces so tabs/slots do not inflate the size.
    vertices = solid.Vertices()
    points = np.array([v.toTuple() for v in vertices], dtype=float)
    if len(points) >= 4:
        centered = points - points.mean(axis=0)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        length_axis = eigenvectors[:, np.argmax(eigenvalues)]
        length_axis = length_axis / max(np.linalg.norm(length_axis), 1e-12)
        length = float(np.ptp(centered @ length_axis))
    else:
        bb = solid.BoundingBox()
        dims = [bb.xlen, bb.ylen, bb.zlen]
        axis_index = int(np.argmax(dims))
        length = float(dims[axis_index])
        length_axis = np.eye(3)[axis_index]

    # Keep old OBB dimensions only as a fallback and for round-profile heuristics.
    _old_length, fallback_width, fallback_height = _principal_dimensions(solid)
    if fallback_height > fallback_width:
        fallback_width, fallback_height = fallback_height, fallback_width

    result: dict[str, Any] = {
        "solid_index": index,
        "length_mm": length,
        "length_m": length / 1000.0,
        "recognized": False,
        "warning": None,
    }

    if length < max(fallback_width, fallback_height) * 1.35:
        result.update({
            "type": "Onbekend",
            "profile_size": "Vrije vorm",
            "outer_width_mm": fallback_width,
            "outer_height_mm": fallback_height,
            "warning": "Onderdeel is niet lang genoeg om betrouwbaar als profiel te herkennen.",
        })
        return result

    round_info = _round_profile(solid, length, fallback_width, fallback_height)
    if round_info:
        result.update(round_info)
        result["recognized"] = True
        d = round_info["diameter_mm"]
        t = _snap_wall_thickness(round_info["thickness_mm"])
        result["thickness_mm"] = t
        result["profile_size"] = f"Ø{_fmt_number(d)}" + (f"x{_fmt_number(t)}" if t else "")
        if t is None:
            result["warning"] = "Diameter herkend, wanddikte niet betrouwbaar bepaald."
        return result

    robust = _robust_rectangular_profile(solid, length_axis)
    if robust:
        width = float(robust["outer_width_mm"])
        height = float(robust["outer_height_mm"])
        thickness = robust.get("thickness_mm")
        profile_type = robust["type"]
        result.update(robust)
        result["recognized"] = True
        if profile_type == "Vierkant":
            base = f"{_fmt_number(width)}x{_fmt_number(height)}"
        else:
            base = f"{_fmt_number(width)}x{_fmt_number(height)}"
        result["profile_size"] = base + (f"x{_fmt_number(thickness)}" if thickness else "")
        if thickness is None:
            result["warning"] = "Kokermaat uit dominante langsvlakken herkend; wanddikte kon niet betrouwbaar worden bepaald."
        return result

    # Fallback for unusual geometry where dominant side planes cannot be isolated.
    width, height = fallback_width, fallback_height
    profile_type = "Vierkant" if abs(width - height) < 0.6 else "Rechthoekig"
    thickness = _exact_rectangular_thickness(solid, width, height, length)
    source = "edges"
    if thickness is None:
        thickness = _estimated_rectangular_thickness(solid, width, height, length)
        source = "volume-estimate" if thickness is not None else "unknown"
    thickness = _snap_wall_thickness(thickness)

    result.update({
        "type": profile_type,
        "outer_width_mm": width,
        "outer_height_mm": height,
        "thickness_mm": thickness,
        "thickness_source": source,
        "recognized": True,
        "profile_source": "bounding-fallback",
    })
    base = (f"{_fmt_number((width + height) / 2)}x{_fmt_number((width + height) / 2)}"
            if profile_type == "Vierkant" else f"{_fmt_number(width)}x{_fmt_number(height)}")
    result["profile_size"] = base + (f"x{_fmt_number(thickness)}" if thickness else "")
    result["warning"] = "Kon de oorspronkelijke kokerdoorsnede niet uit dominante vlakken halen; controleer deze maat handmatig."
    return result


def _harmonize_rectangular_profiles(details: list[dict[str, Any]]) -> None:
    """
    Use consensus across equal nominal profiles.

    A frame usually contains several bodies made from the same stock.  Tabs and
    slots differ per body, so their local wall-face measurements may vary, while
    the volume-derived wall thickness tends to cluster around the original stock.
    We use the median volume candidate per nominal profile and snap it to a
    standard wall thickness. This fixes cases such as a 100x100x1.5 frame being
    reported as several 98-103 mm / 2 mm variants.
    """
    clusters: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        if not item.get("recognized") or item.get("type") == "Rond":
            continue
        w = item.get("outer_width_mm")
        h = item.get("outer_height_mm")
        if w is None or h is None:
            continue
        key = (str(item.get("type")), round(float(w), 1), round(float(h), 1))
        clusters[key].append(item)

    for (_ptype, nominal_w, nominal_h), items in clusters.items():
        candidates = [
            float(item["volume_thickness_mm"])
            for item in items
            if item.get("volume_thickness_mm") is not None
            and 0.5 <= float(item["volume_thickness_mm"]) <= 12.0
        ]

        consensus = None
        if len(candidates) >= 2:
            consensus = _snap_wall_thickness(float(np.median(candidates)))

        # Only use consensus when a body does not already have a reliable
        # dominant-face wall measurement. Do not overwrite a correct 2.0 mm wall.
        if consensus is not None:
            for item in items:
                if item.get("thickness_mm") is None or item.get("thickness_source") == "volume-estimate":
                    item["thickness_mm"] = consensus
                    item["thickness_source"] = "profile-consensus"

        # Rebuild the visible profile string from the harmonized nominal size.
        for item in items:
            w = float(item.get("outer_width_mm", nominal_w))
            h = float(item.get("outer_height_mm", nominal_h))
            t = item.get("thickness_mm")
            if abs(w - h) <= max(1.2, min(w, h) * 0.045):
                sq = _snap_profile_dimension((w + h) / 2.0)
                item["type"] = "Vierkant"
                item["outer_width_mm"] = sq
                item["outer_height_mm"] = sq
                base = f"{_fmt_number(sq)}x{_fmt_number(sq)}"
            else:
                base = f"{_fmt_number(w)}x{_fmt_number(h)}"
            item["profile_size"] = base + (f"x{_fmt_number(t)}" if t is not None else "")

            if consensus is not None and item.get("thickness_source") == "profile-consensus":
                item["warning"] = None


def _group_key(item: dict[str, Any]) -> tuple:
    if not item.get("recognized"):
        return ("Onbekend", item.get("solid_index"))
    return (item.get("type"), item.get("profile_size"))


def analyze_step(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    imported = cq.importers.importStep(str(path))
    solids = imported.solids().vals()
    if not solids:
        raise ValueError("Geen solids gevonden in het STEP-bestand.")

    details = [analyze_solid(solid, i + 1) for i, solid in enumerate(solids)]
    _harmonize_rectangular_profiles(details)

    grouped: dict[tuple, dict[str, Any]] = {}
    for item in details:
        key = _group_key(item)
        if key not in grouped:
            grouped[key] = {
                "type": item.get("type", "Onbekend"),
                "profile_size": item.get("profile_size", "Onbekend"),
                "count": 0,
                "total_length_m": 0.0,
                "lengths_m": [],
                "recognized": bool(item.get("recognized")),
                "warnings": [],
                "thickness_source": item.get("thickness_source"),
            }
        row = grouped[key]
        row["count"] += 1
        row["total_length_m"] += float(item.get("length_m", 0.0))
        row["lengths_m"].append(float(item.get("length_m", 0.0)))
        warning = item.get("warning")
        if warning and warning not in row["warnings"]:
            row["warnings"].append(warning)

    groups = list(grouped.values())
    for group in groups:
        group["total_length_m"] = round(group["total_length_m"], 4)
        group["lengths_m"] = [round(x, 4) for x in group["lengths_m"]]

    type_order = {"Vierkant": 0, "Rechthoekig": 1, "Rond": 2, "Hoeklijn": 3, "Onbekend": 99}
    groups.sort(key=lambda g: (type_order.get(g["type"], 50), g["profile_size"]))

    recognized = sum(1 for item in details if item.get("recognized"))
    return {
        "analyzer_version": "4.0-length-fix",
        "filename": path.name,
        "solid_count": len(solids),
        "recognized_count": recognized,
        "unrecognized_count": len(solids) - recognized,
        "groups": groups,
        "details": details,
    }
