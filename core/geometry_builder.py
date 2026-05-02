from __future__ import annotations

import math
from typing import Any

import numpy as np


PROPERTY_PRESETS = {
    "conductive clay": {
        "resistivity": 20.0,
        "conductivity": 0.05,
        "vp": 1200.0,
        "velocity": 1200.0,
        "epsilon_r": 25.0,
        "permittivity": 25.0,
        "susceptibility": 1e-6,
    },
    "resistive boulder": {
        "resistivity": 2000.0,
        "conductivity": 0.0005,
        "vp": 3500.0,
        "velocity": 3500.0,
        "epsilon_r": 5.0,
        "permittivity": 5.0,
        "susceptibility": 1e-5,
    },
    "air-filled void": {
        "resistivity": 1e5,
        "conductivity": 1e-5,
        "vp": 340.0,
        "velocity": 340.0,
        "epsilon_r": 1.0,
        "permittivity": 1.0,
        "susceptibility": 0.0,
    },
    "water-filled void": {
        "resistivity": 30.0,
        "conductivity": 0.033333,
        "vp": 1500.0,
        "velocity": 1500.0,
        "epsilon_r": 80.0,
        "permittivity": 80.0,
        "susceptibility": 0.0,
    },
    "saturated sand": {
        "resistivity": 80.0,
        "conductivity": 0.0125,
        "vp": 1700.0,
        "velocity": 1700.0,
        "epsilon_r": 25.0,
        "permittivity": 25.0,
        "susceptibility": 1e-6,
    },
    "bedrock high velocity": {
        "resistivity": 1000.0,
        "conductivity": 0.001,
        "vp": 3500.0,
        "velocity": 3500.0,
        "epsilon_r": 6.0,
        "permittivity": 6.0,
        "susceptibility": 1e-5,
    },
    "custom": {
        "resistivity": 100.0,
        "conductivity": 0.01,
        "vp": 1500.0,
        "velocity": 1500.0,
        "epsilon_r": 10.0,
        "permittivity": 10.0,
        "susceptibility": 1e-6,
    },
}


SIZE_PRESETS = {
    "small": {
        "radius": 1.0,
        "width": 3.0,
        "height": 1.0,
    },
    "medium": {
        "radius": 2.0,
        "width": 8.0,
        "height": 2.0,
    },
    "large": {
        "radius": 4.0,
        "width": 15.0,
        "height": 4.0,
    },
    "custom": {
        "radius": 2.0,
        "width": 8.0,
        "height": 2.0,
    },
}


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def layer_boundary_depths(layers: list[dict]) -> list[float]:
    boundaries = []
    cumulative = 0.0

    for layer in layers[:-1]:
        cumulative += _f(layer.get("thickness", 1.0), 1.0)
        boundaries.append(cumulative)

    return boundaries


def layer_mid_depths(layers: list[dict], model_depth: float) -> list[float]:
    mids = []
    top = 0.0

    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            bottom = model_depth
        else:
            bottom = top + _f(layer.get("thickness", model_depth), model_depth)

        bottom = min(bottom, model_depth)
        mids.append(0.5 * (top + bottom))
        top = bottom

    return mids


def depth_from_position(position: str, layers: list[dict], model_depth: float) -> float:
    position = str(position).strip().lower()
    mids = layer_mid_depths(layers, model_depth)
    boundaries = layer_boundary_depths(layers)

    if position.startswith("in layer"):
        try:
            idx = int(position.split()[-1]) - 1
            idx = max(0, min(idx, len(mids) - 1))
            return mids[idx]
        except Exception:
            return mids[0] if mids else 0.25 * model_depth

    if position.startswith("between layer"):
        try:
            parts = position.replace("between layer", "").replace("and", "").split()
            idx = int(parts[0]) - 1
            idx = max(0, min(idx, len(boundaries) - 1))
            return boundaries[idx]
        except Exception:
            return 0.25 * model_depth

    mapping = {
        "very shallow": 0.10 * model_depth,
        "shallow": 0.20 * model_depth,
        "middle depth": 0.45 * model_depth,
        "deep": 0.70 * model_depth,
        "very deep": 0.85 * model_depth,
    }

    return mapping.get(position, 0.30 * model_depth)


def x_from_position(horizontal: str, model_length: float, custom_x: float) -> float:
    horizontal = str(horizontal).strip().lower()

    mapping = {
        "far left": -0.40 * model_length,
        "left": -0.25 * model_length,
        "centre": 0.0,
        "center": 0.0,
        "right": 0.25 * model_length,
        "far right": 0.40 * model_length,
        "custom": custom_x,
    }

    return mapping.get(horizontal, 0.0)


def rotated_rectangle_points(cx: float, cz: float, width: float, height: float, angle_deg: float) -> list[list[float]]:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    local = np.array(
        [
            [-0.5 * width, -0.5 * height],
            [0.5 * width, -0.5 * height],
            [0.5 * width, 0.5 * height],
            [-0.5 * width, 0.5 * height],
        ],
        dtype=float,
    )

    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = local @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cz

    return pts.tolist()


def rotated_ellipse_points(
    cx: float,
    cz: float,
    width: float,
    height: float,
    angle_deg: float,
    n: int = 72,
) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 0.5 * width * np.cos(theta)
    z = 0.5 * height * np.sin(theta)

    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)

    xr = c * x - s * z + cx
    zr = s * x + c * z + cz

    return np.column_stack([xr, zr]).tolist()


def buried_channel_points(cx: float, cz: float, width: float, height: float) -> list[list[float]]:
    half = 0.5 * width
    shoulder = 0.25 * width

    return [
        [cx - half, cz - 0.25 * height],
        [cx - shoulder, cz - 0.75 * height],
        [cx, cz - height],
        [cx + shoulder, cz - 0.75 * height],
        [cx + half, cz - 0.25 * height],
        [cx + half, cz + 0.35 * height],
        [cx - half, cz + 0.35 * height],
    ]


def interface_bump_points(cx: float, cz: float, width: float, height: float) -> list[list[float]]:
    half = 0.5 * width

    return [
        [cx - half, cz],
        [cx - 0.25 * half, cz - height],
        [cx + 0.25 * half, cz - height],
        [cx + half, cz],
        [cx + half, cz + 0.4 * height],
        [cx - half, cz + 0.4 * height],
    ]


def tilted_layer_points(model_length: float, centre_depth: float, thickness: float, angle_deg: float) -> list[list[float]]:
    xmin = -0.5 * model_length
    xmax = 0.5 * model_length

    theta = math.radians(angle_deg)
    dz_left = -0.5 * model_length * math.tan(theta)
    dz_right = 0.5 * model_length * math.tan(theta)

    z1_left = -(centre_depth + dz_left - 0.5 * thickness)
    z1_right = -(centre_depth + dz_right - 0.5 * thickness)
    z2_right = -(centre_depth + dz_right + 0.5 * thickness)
    z2_left = -(centre_depth + dz_left + 0.5 * thickness)

    return [
        [xmin, z1_left],
        [xmax, z1_right],
        [xmax, z2_right],
        [xmin, z2_left],
    ]


def property_preset(name: str) -> dict:
    key = str(name).strip().lower()
    return dict(PROPERTY_PRESETS.get(key, PROPERTY_PRESETS["custom"]))


def build_anomaly_from_target(target: dict, layers: list[dict], domain: dict) -> dict:
    model_length = _f(domain.get("length", 80.0), 80.0)
    model_depth = _f(domain.get("depth", 20.0), 20.0)

    target_type = str(target.get("target_type", "circle / sphere-like body")).strip().lower()
    position_mode = str(target.get("position_mode", "in layer 1"))
    horizontal = str(target.get("horizontal_position", "centre"))
    size_name = str(target.get("size", "medium")).strip().lower()
    preset_name = str(target.get("property_preset", "conductive clay")).strip().lower()

    size = dict(SIZE_PRESETS.get(size_name, SIZE_PRESETS["medium"]))

    radius = _f(target.get("radius", size["radius"]), size["radius"])
    width = _f(target.get("width", size["width"]), size["width"])
    height = _f(target.get("height", size["height"]), size["height"])
    angle = _f(target.get("angle", 0.0), 0.0)

    custom_x = _f(target.get("custom_x", 0.0), 0.0)
    custom_depth = _f(target.get("custom_depth", depth_from_position(position_mode, layers, model_depth)), 0.25 * model_depth)

    if str(position_mode).strip().lower() == "custom depth":
        depth = custom_depth
    else:
        depth = depth_from_position(position_mode, layers, model_depth)

    cx = x_from_position(horizontal, model_length, custom_x)
    cz = -abs(depth)

    props = property_preset(preset_name)

    if preset_name == "custom":
        props["resistivity"] = _f(target.get("resistivity", props["resistivity"]), props["resistivity"])
        props["conductivity"] = _f(target.get("conductivity", props["conductivity"]), props["conductivity"])
        props["vp"] = _f(target.get("vp", props["vp"]), props["vp"])
        props["velocity"] = props["vp"]
        props["epsilon_r"] = _f(target.get("epsilon_r", props["epsilon_r"]), props["epsilon_r"])
        props["permittivity"] = props["epsilon_r"]
        props["susceptibility"] = _f(target.get("susceptibility", props["susceptibility"]), props["susceptibility"])

    if "circle" in target_type or "sphere" in target_type:
        anomaly = {
            "type": "circle",
            "center": [cx, cz],
            "radius": radius,
        }

    elif "ellipse" in target_type or "lens" in target_type:
        anomaly = {
            "type": "polygon",
            "points": rotated_ellipse_points(cx, cz, width, height, angle),
        }

    elif "tilted rectangle" in target_type or "dyke" in target_type:
        anomaly = {
            "type": "polygon",
            "points": rotated_rectangle_points(cx, cz, width, height, angle),
        }

    elif "rectangle" in target_type or "block" in target_type:
        anomaly = {
            "type": "polygon",
            "points": rotated_rectangle_points(cx, cz, width, height, 0.0),
        }

    elif "channel" in target_type:
        anomaly = {
            "type": "polygon",
            "points": buried_channel_points(cx, cz, width, height),
        }

    elif "interface bump" in target_type:
        anomaly = {
            "type": "polygon",
            "points": interface_bump_points(cx, cz, width, height),
        }

    elif "tilted layer" in target_type:
        anomaly = {
            "type": "polygon",
            "points": tilted_layer_points(model_length, abs(depth), height, angle),
        }

    else:
        anomaly = {
            "type": "circle",
            "center": [cx, cz],
            "radius": radius,
        }

    anomaly.update(props)
    anomaly["target_description"] = dict(target)

    return anomaly


def build_anomalies_from_targets(targets: list[dict], layers: list[dict], domain: dict) -> list[dict]:
    return [build_anomaly_from_target(t, layers, domain) for t in targets]


def anomaly_to_target_spec(anomaly: dict, index: int = 0) -> dict:
    typ = str(anomaly.get("type", "circle")).lower()

    if typ == "circle":
        center = anomaly.get("center", [0.0, -4.0])
        radius = _f(anomaly.get("radius", 2.0), 2.0)

        if radius <= 1.2:
            size = "small"
        elif radius <= 3.0:
            size = "medium"
        else:
            size = "large"

        return {
            "target_type": "Circle / sphere-like body",
            "position_mode": "Custom depth",
            "horizontal_position": "Custom",
            "size": size,
            "property_preset": "custom",
            "custom_x": _f(center[0], 0.0),
            "custom_depth": abs(_f(center[1], -4.0)),
            "radius": radius,
            "width": 2.0 * radius,
            "height": 2.0 * radius,
            "angle": 0.0,
            "resistivity": _f(anomaly.get("resistivity", 100.0), 100.0),
            "conductivity": _f(anomaly.get("conductivity", 0.01), 0.01),
            "vp": _f(anomaly.get("vp", anomaly.get("velocity", 1500.0)), 1500.0),
            "epsilon_r": _f(anomaly.get("epsilon_r", anomaly.get("permittivity", 10.0)), 10.0),
            "susceptibility": _f(anomaly.get("susceptibility", 1e-6), 1e-6),
        }

    points = np.asarray(anomaly.get("points", [[0.0, -4.0], [1.0, -4.0], [1.0, -5.0], [0.0, -5.0]]), dtype=float)
    cx = float(np.mean(points[:, 0]))
    cz = float(np.mean(points[:, 1]))
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))

    return {
        "target_type": "Rectangle / block",
        "position_mode": "Custom depth",
        "horizontal_position": "Custom",
        "size": "custom",
        "property_preset": "custom",
        "custom_x": cx,
        "custom_depth": abs(cz),
        "radius": min(width, height) / 2.0,
        "width": width,
        "height": height,
        "angle": 0.0,
        "resistivity": _f(anomaly.get("resistivity", 100.0), 100.0),
        "conductivity": _f(anomaly.get("conductivity", 0.01), 0.01),
        "vp": _f(anomaly.get("vp", anomaly.get("velocity", 1500.0)), 1500.0),
        "epsilon_r": _f(anomaly.get("epsilon_r", anomaly.get("permittivity", 10.0)), 10.0),
        "susceptibility": _f(anomaly.get("susceptibility", 1e-6), 1e-6),
    }


def raster_model_preview(
    layers: list[dict],
    anomalies: list[dict],
    domain: dict,
    property_name: str = "resistivity",
    nx: int = 280,
    nz: int = 140,
):
    model_length = _f(domain.get("length", 80.0), 80.0)
    model_depth = _f(domain.get("depth", 20.0), 20.0)

    xmin = -0.5 * model_length
    xmax = 0.5 * model_length

    x = np.linspace(xmin, xmax, nx)

    # Row 0 = bottom, last row = surface.
    # This matches imshow(..., origin="lower").
    z = np.linspace(-model_depth, 0.0, nz)

    if not layers:
        layers = [{"thickness": model_depth, property_name: 100.0}]

    cumulative_depths = []
    cumulative = 0.0

    for layer in layers:
        cumulative += _f(layer.get("thickness", model_depth), model_depth)
        cumulative_depths.append(cumulative)

    model = np.zeros((nz, nx), dtype=float)

    for iz, zz in enumerate(z):
        depth_below_surface = abs(zz)

        layer_idx = len(layers) - 1

        for i, bottom_depth in enumerate(cumulative_depths):
            if depth_below_surface <= bottom_depth:
                layer_idx = i
                break

        layer = layers[layer_idx]

        if property_name == "vp":
            value = layer.get("vp", layer.get("velocity", 1500.0))
        elif property_name == "epsilon_r":
            value = layer.get("epsilon_r", layer.get("permittivity", 10.0))
        else:
            value = layer.get(property_name, layer.get("resistivity", 100.0))

        model[iz, :] = _f(value, 100.0)

    xx, zz = np.meshgrid(x, z)

    for anomaly in anomalies:
        if property_name == "vp":
            value = anomaly.get("vp", anomaly.get("velocity", 1500.0))
        elif property_name == "epsilon_r":
            value = anomaly.get("epsilon_r", anomaly.get("permittivity", 10.0))
        else:
            value = anomaly.get(property_name, anomaly.get("resistivity", 50.0))

        value = _f(value, 50.0)

        typ = str(anomaly.get("type", "circle")).lower()

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -4.0])
            radius = _f(anomaly.get("radius", 2.0), 2.0)

            mask = (xx - _f(cx, 0.0)) ** 2 + (zz - _f(cz, -4.0)) ** 2 <= radius**2
            model[mask] = value

        elif typ == "polygon":
            from matplotlib.path import Path as MplPath

            points = np.asarray(anomaly.get("points", []), dtype=float)

            if len(points) >= 3:
                pts = np.column_stack([xx.ravel(), zz.ravel()])
                mask = MplPath(points).contains_points(pts).reshape(model.shape)
                model[mask] = value

    extent = [xmin, xmax, -model_depth, 0.0]

    return model, extent

