
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from solvers.FDEM1DFWD_RC import FDEM1DFWD_RC
from solvers.FDEM1DSENS_RC import FDEM1DSENS_RC


MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _progress(scenario: dict, text: str):
    cb = scenario.get("_progress_callback")
    if cb is not None:
        cb(str(text))


def _as_float(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _layer_conductivity(layer):
    if "conductivity" in layer:
        return max(_as_float(layer["conductivity"], 0.01), 0.0)

    if "resistivity" in layer:
        rho = _as_float(layer["resistivity"], 100.0)
        if rho <= 0:
            raise ValueError("Layer resistivity must be > 0.")
        return 1.0 / rho

    return 0.01


def _layer_permittivity(layer):
    if "absolute_permittivity" in layer:
        return max(_as_float(layer["absolute_permittivity"], EPS0), EPS0)

    if "permittivity" in layer:
        eps_r = _as_float(layer["permittivity"], 10.0)
        return max(eps_r, 1.0) * EPS0

    if "epsilon_r" in layer:
        eps_r = _as_float(layer["epsilon_r"], 10.0)
        return max(eps_r, 1.0) * EPS0

    return 10.0 * EPS0


def _layer_susceptibility(layer):
    if "susceptibility" in layer:
        return _as_float(layer["susceptibility"], 1e-6)

    if "sus" in layer:
        return _as_float(layer["sus"], 1e-6)

    return 1e-6


def _layer_property(layer):
    return {
        "con": _layer_conductivity(layer),
        "perm": _layer_permittivity(layer),
        "sus": _layer_susceptibility(layer),
    }


def _default_layers():
    return [
        {
            "thickness": 1.0,
            "conductivity": 0.01,
            "permittivity": 10.0,
            "susceptibility": 1e-6,
        },
        {
            "thickness": 999.0,
            "conductivity": 0.02,
            "permittivity": 12.0,
            "susceptibility": 1e-6,
        },
    ]


def _layers(scenario: dict):
    layers = scenario.get("layers", [])
    return layers if layers else _default_layers()


def _build_fdem_model_from_layers(layers):
    con = []
    perm = []
    sus = []
    thick = []

    if len(layers) == 1:
        layer = layers[0]
        props = _layer_property(layer)
        con = [props["con"]]
        perm = [props["perm"]]
        sus = [props["sus"]]
        thick = []
    else:
        for i, layer in enumerate(layers):
            props = _layer_property(layer)
            con.append(props["con"])
            perm.append(props["perm"])
            sus.append(props["sus"])

            if i < len(layers) - 1:
                thick.append(max(_as_float(layer.get("thickness", 1.0), 1.0), 1e-6))

    return {
        "con": np.asarray(con, dtype=float),
        "perm": np.asarray(perm, dtype=float),
        "sus": np.asarray(sus, dtype=float),
        "thick": np.asarray(thick, dtype=float),
    }


def _base_layer_index_at_depth(depth, layers):
    cumulative = 0.0

    for i, layer in enumerate(layers[:-1]):
        cumulative += max(_as_float(layer.get("thickness", 1.0), 1.0), 0.0)
        if depth < cumulative:
            return i

    return len(layers) - 1


def _estimate_anomaly_equivalent_layer(scenario: dict):
    anomalies = scenario.get("anomalies", [])

    if not anomalies:
        return None

    # Use first anomaly for 1D equivalent-layer screening.
    a = anomalies[0]
    typ = str(a.get("type", "unknown")).strip().lower()

    depths = []
    xvals = []

    if typ == "circle":
        cx, cz = a.get("center", [0.0, -5.0])
        r = max(_as_float(a.get("radius", 1.0), 1.0), 0.1)
        depths = [abs(_as_float(cz, -5.0)) - r, abs(_as_float(cz, -5.0)) + r]
        xvals = [_as_float(cx, 0.0)]

    elif typ in {"ellipse", "elliptical"}:
        cx, cz = a.get("center", [0.0, -5.0])
        h = max(_as_float(a.get("height", a.get("radius_z", 1.0) * 2.0), 2.0), 0.1)
        centre = abs(_as_float(cz, -5.0))
        depths = [centre - 0.5 * h, centre + 0.5 * h]
        xvals = [_as_float(cx, 0.0)]

    elif typ in {"polygon", "rectangle", "block"}:
        pts = a.get("points", [])
        if len(pts) >= 3:
            xvals = [_as_float(p[0], 0.0) for p in pts]
            depths = [abs(_as_float(p[1], 0.0)) for p in pts]

    if not depths:
        return None

    z0 = max(0.0, float(np.nanmin(depths)))
    z1 = max(z0 + 0.1, float(np.nanmax(depths)))
    zc = 0.5 * (z0 + z1)
    thickness = max(z1 - z0, 0.1)
    xc = float(np.nanmean(xvals)) if xvals else 0.0

    # Target properties: use anomaly values if present, else inherited defaults.
    con = _layer_conductivity(a) if ("conductivity" in a or "resistivity" in a) else None
    perm = _layer_permittivity(a) if ("permittivity" in a or "epsilon_r" in a or "absolute_permittivity" in a) else None
    sus = _layer_susceptibility(a) if ("susceptibility" in a or "sus" in a) else None

    return {
        "x_center": xc,
        "z_top": z0,
        "z_bottom": z1,
        "z_center": zc,
        "thickness": thickness,
        "conductivity": con,
        "permittivity": perm,
        "susceptibility": sus,
        "source_type": typ,
    }


def _build_equivalent_target_model(scenario: dict, target: dict):
    layers = _layers(scenario)

    if target is None:
        return _build_fdem_model_from_layers(layers), None

    finite_bounds = [0.0]
    cumulative = 0.0

    for layer in layers[:-1]:
        cumulative += max(_as_float(layer.get("thickness", 1.0), 1.0), 0.0)
        finite_bounds.append(cumulative)

    finite_bounds.extend([target["z_top"], target["z_bottom"]])
    finite_bounds = sorted(set(round(float(b), 8) for b in finite_bounds if float(b) >= 0.0))

    # Remove duplicate/too-close boundaries.
    clean = []
    for b in finite_bounds:
        if not clean or abs(b - clean[-1]) > 1e-6:
            clean.append(b)
    finite_bounds = clean

    eq_layers = []

    for i in range(len(finite_bounds) - 1):
        top = finite_bounds[i]
        bottom = finite_bounds[i + 1]

        if bottom <= top:
            continue

        mid = 0.5 * (top + bottom)
        idx = _base_layer_index_at_depth(mid, layers)
        base = dict(layers[idx])

        if target["z_top"] <= mid < target["z_bottom"]:
            if target["conductivity"] is not None:
                base["conductivity"] = target["conductivity"]
                if "resistivity" in base:
                    base.pop("resistivity", None)
            if target["permittivity"] is not None:
                base["absolute_permittivity"] = target["permittivity"]
            if target["susceptibility"] is not None:
                base["susceptibility"] = target["susceptibility"]

        base["thickness"] = bottom - top
        eq_layers.append(base)

    # Final half-space.
    last_top = finite_bounds[-1] if finite_bounds else 0.0
    idx = _base_layer_index_at_depth(last_top + 1.0, layers)
    halfspace = dict(layers[idx])

    if target["z_top"] <= last_top < target["z_bottom"]:
        if target["conductivity"] is not None:
            halfspace["conductivity"] = target["conductivity"]
            halfspace.pop("resistivity", None)
        if target["permittivity"] is not None:
            halfspace["absolute_permittivity"] = target["permittivity"]
        if target["susceptibility"] is not None:
            halfspace["susceptibility"] = target["susceptibility"]

    halfspace["thickness"] = 999.0
    eq_layers.append(halfspace)

    return _build_fdem_model_from_layers(eq_layers), eq_layers


def _build_sensor(scenario, frequency):
    em = scenario.get("survey", {}).get("em", {})

    coil_spacing = _as_float(em.get("coil_spacing", em.get("spacing", 1.66)), 1.66)
    height = _as_float(em.get("height", 0.0), 0.0)

    return {
        "x": coil_spacing,
        "y": 0.0,
        "z": 0.0,
        "height": height,
        "freq": float(frequency),
        "mom": _as_float(em.get("moment", 1.0), 1.0),
        "ori": str(em.get("orientation", "ZZ")),
    }


def _run_response(scenario: dict, model: dict, frequencies: np.ndarray):
    ip_values = []
    qp_values = []

    for freq in frequencies:
        sensor = _build_sensor(scenario, freq)
        ip, qp = FDEM1DFWD_RC(sensor, {k: np.asarray(v, dtype=float).copy() for k, v in model.items()})
        ip_values.append(float(np.asarray(ip).squeeze()))
        qp_values.append(float(np.asarray(qp).squeeze()))

    return np.asarray(ip_values, dtype=float), np.asarray(qp_values, dtype=float)


def _run_sensitivity(scenario: dict, model: dict, frequencies: np.ndarray, parameter: str):
    parameter = str(parameter).strip().lower()
    if parameter not in {"con", "sus", "perm"}:
        parameter = "con"

    sens_all = []

    for freq in frequencies:
        sensor = _build_sensor(scenario, freq)
        mcopy = {k: np.asarray(v, dtype=float).copy() for k, v in model.items()}

        try:
            sens_ip, sens_qp, error = FDEM1DSENS_RC(sensor, mcopy, parameter)
            sens = np.sqrt(np.asarray(sens_ip, dtype=float) ** 2 + np.asarray(sens_qp, dtype=float) ** 2)
            sens_all.append(sens)
        except Exception:
            continue

    if not sens_all:
        return None

    max_len = max(len(s) for s in sens_all)
    arr = np.zeros((len(sens_all), max_len), dtype=float)

    for i, s in enumerate(sens_all):
        arr[i, :len(s)] = s

    combined = np.nanmean(np.abs(arr), axis=0)

    if np.nanmax(combined) > 0:
        combined = combined / np.nanmax(combined)

    layer_index = np.arange(1, len(combined) + 1, dtype=float)

    return np.column_stack([layer_index, combined])


def _model_text(model: dict, title: str):
    lines = [title, ""]
    con = model["con"]
    perm = model["perm"]
    sus = model["sus"]
    thick = model["thick"]

    top = 0.0

    for i in range(len(con)):
        if i < len(thick):
            bottom = top + thick[i]
            depth_str = f"{top:.3g}–{bottom:.3g} m"
            top = bottom
        else:
            depth_str = f">{top:.3g} m"

        eps_r = perm[i] / EPS0

        lines.append(
            f"Layer {i + 1}: depth={depth_str}, "
            f"sigma={con[i]:.6g} S/m, "
            f"rho={(1.0 / con[i]) if con[i] > 0 else np.inf:.6g} ohm m, "
            f"epsilon_r={eps_r:.6g}, "
            f"sus={sus[i]:.6g}"
        )

    return "\n".join(lines)


def run_em(scenario):
    _progress(scenario, "Preparing EM/FDEM 1D layered model...")

    em = scenario.get("survey", {}).get("em", {})

    frequencies = em.get("frequencies", None)
    if frequencies is None:
        frequencies = [em.get("freq", 1000.0)]

    frequencies = np.asarray([float(f) for f in frequencies], dtype=float)
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0)]

    if frequencies.size == 0:
        raise ValueError("EM frequencies must contain at least one positive value.")

    run_target = bool(em.get("run_target_equivalent", True))
    show_sensitivity = bool(em.get("show_sensitivity", True))
    sensitivity_parameter = str(em.get("sensitivity_parameter", "con")).strip().lower()
    topography_mode = str(em.get("topography_mode", "horizontal_interfaces")).strip().lower()

    base_model = _build_fdem_model_from_layers(_layers(scenario))

    _progress(
        scenario,
        (
            "Running EM/FDEM baseline response...\n"
            f"  frequencies={frequencies.tolist()} Hz\n"
            f"  orientation={em.get('orientation', 'ZZ')}\n"
            f"  coil_spacing={em.get('coil_spacing', em.get('spacing', 1.66))} m\n"
            f"  height={em.get('height', 0.0)} m"
        ),
    )

    ip_base, qp_base = _run_response(scenario, base_model, frequencies)

    target = _estimate_anomaly_equivalent_layer(scenario) if run_target else None
    target_model = None
    eq_layers = None
    ip_target = None
    qp_target = None

    if target is not None:
        _progress(
            scenario,
            (
                "Running EM/FDEM target-equivalent response...\n"
                f"  equivalent target depth={target['z_top']:.3g}–{target['z_bottom']:.3g} m\n"
                f"  equivalent target thickness={target['thickness']:.3g} m"
            ),
        )

        target_model, eq_layers = _build_equivalent_target_model(scenario, target)
        ip_target, qp_target = _run_response(scenario, target_model, frequencies)

    plots = {}

    plots["EM IP frequency response"] = {
        "type": "line",
        "array": np.column_stack([frequencies, ip_base]),
        "title": "EM/FDEM in-phase response vs frequency",
        "xlabel": "frequency [Hz]",
        "ylabel": "IP [ppm]",
        "labels": {1: "background IP"},
    }

    plots["EM QP frequency response"] = {
        "type": "line",
        "array": np.column_stack([frequencies, qp_base]),
        "title": "EM/FDEM quadrature response vs frequency",
        "xlabel": "frequency [Hz]",
        "ylabel": "QP [ppm]",
        "labels": {1: "background QP"},
    }

    if ip_target is not None and qp_target is not None:
        dip = ip_target - ip_base
        dqp = qp_target - qp_base

        plots["EM target-minus-background difference"] = {
            "type": "line",
            "array": np.column_stack([frequencies, dip, dqp]),
            "title": "EM/FDEM equivalent target response: target minus background",
            "xlabel": "frequency [Hz]",
            "ylabel": "difference [ppm]",
            "labels": {1: "ΔIP", 2: "ΔQP"},
        }

        bg_rms = float(np.sqrt(np.nanmean(ip_base**2 + qp_base**2)))
        diff_rms = float(np.sqrt(np.nanmean(dip**2 + dqp**2)))
        ratio = diff_rms / max(bg_rms, 1e-30)

        plots["EM detectability metric"] = {
            "type": "text",
            "title": "EM/FDEM equivalent-target detectability",
            "text": (
                "EM/FDEM equivalent-target detectability metric\n\n"
                f"RMS(background IP/QP) = {bg_rms:.6g} ppm\n"
                f"RMS(target - background) = {diff_rms:.6g} ppm\n"
                f"difference/background ratio = {ratio:.6g}\n\n"
                "Interpretation:\n"
                "< 0.01  : very weak / probably not detectable\n"
                "0.01-0.05: weak, may require excellent repeatability\n"
                "> 0.05  : potentially detectable as bulk 1D response\n\n"
                "Important limitation: this is not a true 2D cavity response. "
                "The anomaly is represented as an equivalent 1D layer."
            ),
        }

    if show_sensitivity:
        sens = _run_sensitivity(scenario, base_model, frequencies, sensitivity_parameter)

        if sens is not None:
            plots[f"EM {sensitivity_parameter} sensitivity by layer"] = {
                "type": "line",
                "array": sens,
                "title": f"Normalised EM/FDEM sensitivity to {sensitivity_parameter} by layer",
                "xlabel": "layer index",
                "ylabel": "normalised sensitivity",
                "labels": {1: f"{sensitivity_parameter} sensitivity"},
            }

    plots["EM background model table"] = {
        "type": "text",
        "title": "EM/FDEM background 1D model",
        "text": _model_text(base_model, "Background 1D layered model"),
    }

    if target_model is not None:
        plots["EM target-equivalent model table"] = {
            "type": "text",
            "title": "EM/FDEM target-equivalent 1D model",
            "text": _model_text(target_model, "Target-equivalent 1D layered model"),
        }

    data_cols = [frequencies, ip_base, qp_base]
    data_names = ["frequency_hz", "ip_background_ppm", "qp_background_ppm"]

    if ip_target is not None and qp_target is not None:
        data_cols.extend([ip_target, qp_target, ip_target - ip_base, qp_target - qp_base])
        data_names.extend([
            "ip_target_ppm",
            "qp_target_ppm",
            "delta_ip_ppm",
            "delta_qp_ppm",
        ])

    data = np.column_stack(data_cols)

    info_lines = [
        "EM/FDEM forward model completed",
        "Implementation: professor 1D FDEM reflection-coefficient solver with digital Hankel filtering.",
        "Template compatibility: scenario layers are used directly; 2D anomaly geometry is converted to an equivalent 1D target layer.",
        "Limitation: this is not a true 2D/3D EM cavity model.",
        f"frequencies_hz={frequencies.tolist()}",
        f"orientation={str(em.get('orientation', 'ZZ'))}",
        f"coil_spacing_m={_as_float(em.get('coil_spacing', em.get('spacing', 1.66)), 1.66):.6g}",
        f"height_m={_as_float(em.get('height', 0.0), 0.0):.6g}",
        f"topography_mode={topography_mode}",
        "topography_note=approximate only; 1D FDEM cannot represent lateral topography explicitly",
        f"run_target_equivalent={run_target}",
        f"show_sensitivity={show_sensitivity}",
        f"sensitivity_parameter={sensitivity_parameter}",
        f"data_columns={data_names}",
        f"background_conductivity_S_m={base_model['con'].tolist()}",
        f"background_thickness_m={base_model['thick'].tolist()}",
        f"background_ip_ppm={ip_base.tolist()}",
        f"background_qp_ppm={qp_base.tolist()}",
    ]

    if target is not None:
        info_lines.extend(
            [
                f"equivalent_target_source_type={target['source_type']}",
                f"equivalent_target_x_center_m={target['x_center']:.6g}",
                f"equivalent_target_z_top_m={target['z_top']:.6g}",
                f"equivalent_target_z_bottom_m={target['z_bottom']:.6g}",
                f"equivalent_target_thickness_m={target['thickness']:.6g}",
            ]
        )

    if ip_target is not None and qp_target is not None:
        info_lines.extend(
            [
                f"target_ip_ppm={ip_target.tolist()}",
                f"target_qp_ppm={qp_target.tolist()}",
                f"delta_ip_ppm={(ip_target - ip_base).tolist()}",
                f"delta_qp_ppm={(qp_target - qp_base).tolist()}",
            ]
        )

    return {
        "model": np.vstack([ip_base, qp_base]),
        "data": data,
        "plots": plots,
        "info": "\n".join(info_lines),
    }
