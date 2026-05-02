from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


C0 = 299_792_458.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GPRInputInfo:
    input_path: Path
    base_name: str
    output_dir: Path
    length: float
    depth: float
    domain_y: float
    air_thickness: float
    dx: float
    dz: float
    frequency_hz: float
    time_window_s: float
    trace_spacing: float
    n_traces: int
    src_x: float
    rx_x: float
    antenna_z: float
    x_positions: np.ndarray
    input_text: str


def _progress(scenario: dict, text: str):
    cb = scenario.get("_progress_callback")
    if cb is not None:
        cb(str(text))


def _to_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            value = value.strip()
            if value.lower().endswith("e6"):
                return float(value[:-2]) * 1e6
        return float(value)
    except Exception:
        return float(default)


def _get_frequency_hz(gpr: dict) -> float:
    if "frequency_mhz" in gpr:
        return _to_float(gpr.get("frequency_mhz"), 100.0) * 1e6

    if "freq_mhz" in gpr:
        return _to_float(gpr.get("freq_mhz"), 100.0) * 1e6

    if "freq" in gpr:
        raw = gpr.get("freq")
        val = _to_float(raw, 100e6)

        if isinstance(raw, str):
            raw_s = raw.strip().lower()
            if raw_s.endswith("mhz"):
                return float(raw_s.replace("mhz", "").strip()) * 1e6
            if raw_s.endswith("ghz"):
                return float(raw_s.replace("ghz", "").strip()) * 1e9

        if val < 1e5:
            return val * 1e6

        return val

    return 100e6


def _domain(scenario: dict):
    domain = scenario.get("domain", {})
    gpr = scenario.get("survey", {}).get("gpr", {})

    length = _to_float(domain.get("length", 12.0), 12.0)
    depth = _to_float(domain.get("depth", 4.5), 4.5)
    dx = _to_float(domain.get("dx", gpr.get("dx", 0.01)), 0.01)
    dz = _to_float(domain.get("dz", dx), dx)

    length = max(length, 2.0)
    depth = max(depth, 1.0)
    dx = max(dx, 0.002)
    dz = max(dz, 0.002)

    return length, depth, dx, dz




def _read_topography_csv(path: str | None):
    """
    Read SwissTopo/map.geo.admin.ch profile CSV or simple x,z CSV.
    Supported column names include:
      Distance / Altitude
      x / z
      distance / elevation
      chainage / height
    Returns raw x, z arrays.
    """
    if not path:
        return None

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"GPR topography CSV not found: {path}")

    last_error = None

    for delimiter in [";", ",", None]:
        try:
            arr = np.genfromtxt(
                path,
                delimiter=delimiter,
                names=True,
                dtype=float,
                encoding=None,
                autostrip=True,
                invalid_raise=False,
            )

            if arr.dtype.names is None:
                continue

            names = list(arr.dtype.names)
            lower = {name.lower().strip(): name for name in names}

            x_candidates = [
                "distance", "distanz", "x", "chainage", "profile_x",
                "dist", "abstand"
            ]
            z_candidates = [
                "altitude", "elevation", "height", "hoehe", "höhe",
                "z", "topography", "topo"
            ]

            x_col = next((lower[c] for c in x_candidates if c in lower), None)
            z_col = next((lower[c] for c in z_candidates if c in lower), None)

            if x_col is None or z_col is None:
                continue

            x = np.asarray(arr[x_col], dtype=float)
            z = np.asarray(arr[z_col], dtype=float)

            x = np.atleast_1d(x)
            z = np.atleast_1d(z)

            mask = np.isfinite(x) & np.isfinite(z)
            x = x[mask]
            z = z[mask]

            if len(x) < 2:
                continue

            order = np.argsort(x)
            x = x[order]
            z = z[order]

            # Remove duplicate x values.
            x_unique, idx = np.unique(x, return_index=True)
            z_unique = z[idx]

            if len(x_unique) < 2:
                continue

            return x_unique, z_unique

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Could not parse GPR topography CSV: {path}. "
        "Expected columns like Distance/Altitude or x/z."
    ) from last_error


def _topography_profile_for_gprmax(scenario: dict, length: float):
    """
    Convert topography to centred GUI coordinates:
      x_local in [-length/2, +length/2]
      elevation normalised so min elevation = 0
    """
    csv_path = scenario.get("files", {}).get("elevation_csv", "")
    csv_topo = _read_topography_csv(csv_path) if csv_path else None

    if csv_topo is not None:
        raw_x, raw_z = csv_topo

        raw_span = float(np.nanmax(raw_x) - np.nanmin(raw_x))
        if raw_span <= 0:
            raise ValueError("Topography CSV distance range is zero.")

        # Map the CSV profile onto the current gprMax local model window.
        x_local = (raw_x - np.nanmin(raw_x)) / raw_span * length - 0.5 * length
        z_local = raw_z - np.nanmin(raw_z)

        return x_local, z_local, f"CSV: {csv_path}"

    topo = scenario.get("topography", [])

    if topo and len(topo) >= 2:
        x = np.asarray([float(pt[0]) for pt in topo], dtype=float)
        z = np.asarray([float(pt[1]) for pt in topo], dtype=float)

        order = np.argsort(x)
        x = x[order]
        z = z[order]

        # If the topography is given as 0..length, centre it.
        if np.nanmin(x) >= -1e-9 and np.nanmax(x) <= length + 1e-9:
            x = x - 0.5 * length

        z = z - np.nanmin(z)
        return x, z, "scenario.topography"

    return (
        np.asarray([-0.5 * length, 0.5 * length], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        "flat default"
    )


def _make_gprmax_surface_function(scenario: dict, length: float, air_clearance: float):
    """
    gprMax z is positive downward from the top of the model.
    Elevation is positive upward, so:
      highest elevation -> shallowest ground surface
      lowest elevation  -> deepest ground surface
    """
    topo_x, topo_elev, topo_source = _topography_profile_for_gprmax(scenario, length)

    topo_elev = topo_elev - float(np.nanmin(topo_elev))
    relief = float(np.nanmax(topo_elev) - np.nanmin(topo_elev))

    def surface_z_down(local_x):
        local_x_arr = np.asarray(local_x, dtype=float)
        elev = np.interp(local_x_arr, topo_x, topo_elev)
        return air_clearance + (relief - elev)

    return surface_z_down, relief, topo_source, topo_x, topo_elev


def _layers(scenario: dict):
    layers = scenario.get("layers", [])

    if not layers:
        layers = [
            {
                "thickness": 2.0,
                "resistivity": 1000.0,
                "conductivity": 0.001,
                "vp": 600.0,
                "velocity": 600.0,
                "epsilon_r": 6.0,
                "permittivity": 6.0,
                "susceptibility": 1e-6,
            },
            {
                "thickness": 999.0,
                "resistivity": 500.0,
                "conductivity": 0.002,
                "vp": 1200.0,
                "velocity": 1200.0,
                "epsilon_r": 9.0,
                "permittivity": 9.0,
                "susceptibility": 1e-6,
            },
        ]

    clean = []

    for layer in layers:
        rho = _to_float(layer.get("resistivity", 1000.0), 1000.0)
        sig = _to_float(layer.get("conductivity", 1.0 / rho if rho > 0 else 0.001), 0.001)
        eps = _to_float(layer.get("epsilon_r", layer.get("permittivity", 9.0)), 9.0)

        clean.append(
            {
                "thickness": _to_float(layer.get("thickness", 999.0), 999.0),
                "resistivity": rho,
                "conductivity": max(sig, 0.0),
                "epsilon_r": max(eps, 1.0),
            }
        )

    return clean


def _anomaly_props(anomaly: dict):
    rho = _to_float(anomaly.get("resistivity", 80.0), 80.0)
    sig = _to_float(anomaly.get("conductivity", 1.0 / rho if rho > 0 else 0.0125), 0.0125)
    eps = _to_float(anomaly.get("epsilon_r", anomaly.get("permittivity", 25.0)), 25.0)

    return {
        "resistivity": rho,
        "conductivity": max(sig, 0.0),
        "epsilon_r": max(eps, 1.0),
    }


def _make_property_models(scenario: dict):
    length, depth, dx, dz = _domain(scenario)
    gpr = scenario.get("survey", {}).get("gpr", {})
    run_actual = bool(gpr.get("run_actual_gprmax", True))

    nx = int(round(length / dx)) + 1
    nz = int(round(depth / dz)) + 1

    if run_actual:
        max_cells = nx * nz

        if max_cells > 650_000:
            raise ValueError(
                "Actual gprMax model is too large for interactive GUI use.\n"
                f"Model length={length:g} m, depth={depth:g} m, dx={dx:g} m, dz={dz:g} m\n"
                f"Grid would be {nx} x {nz} = {max_cells:,} cells before PML.\n\n"
                "Use a smaller actual-FDTD test model first, e.g.:\n"
                "Model length: 12 m\n"
                "Model depth: 4.5 m\n"
                "Grid spacing: 0.01 m\n"
                "Trace spacing: 0.10 m\n"
                "Target radius: 0.75–1.0 m"
            )
    else:
        nx = min(max(nx, 100), 1200)
        nz = min(max(nz, 80), 600)

    x = np.linspace(-0.5 * length, 0.5 * length, nx)
    z = np.linspace(0.0, depth, nz)

    layers = _layers(scenario)

    eps = np.full((nz, nx), layers[-1]["epsilon_r"], dtype=float)
    sigma = np.full((nz, nx), layers[-1]["conductivity"], dtype=float)

    topography_mode = str(gpr.get("topography_mode", "horizontal_interfaces")).strip().lower()
    follow_topography = topography_mode in {
        "parallel_to_topography",
        "layers_follow_topography",
        "follow_topography",
    }

    topo_x, topo_elev, _ = _topography_profile_for_gprmax(scenario, length)
    topo_elev = topo_elev - float(np.nanmin(topo_elev))
    relief = float(np.nanmax(topo_elev) - np.nanmin(topo_elev))
    topo_depth = relief - np.interp(x, topo_x, topo_elev)

    cumulative = np.r_[0.0, np.cumsum([layer["thickness"] for layer in layers])]

    for ix in range(nx):
        # Preview coordinates are depth-positive downward.
        # If enabled, layer thickness is measured below local surface.
        z_eff = z - topo_depth[ix] if follow_topography else z

        for il, layer in enumerate(layers):
            top = cumulative[il]
            bottom = cumulative[il + 1] if il + 1 < len(cumulative) else depth
            mask_z = (z_eff >= top) & (z_eff < bottom)
            eps[mask_z, ix] = layer["epsilon_r"]
            sigma[mask_z, ix] = layer["conductivity"]

    anomalies = scenario.get("anomalies", [])

    X, Z = np.meshgrid(x, z)

    for anomaly in anomalies:
        props = _anomaly_props(anomaly)
        typ = str(anomaly.get("type", "circle")).strip().lower()

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -2.5])
            cx = _to_float(cx, 0.0)
            depth_c = abs(_to_float(cz, -2.5))
            r = max(_to_float(anomaly.get("radius", 0.75), 0.75), dx)

            mask = (X - cx) ** 2 + (Z - depth_c) ** 2 <= r**2

        elif typ in {"ellipse", "elliptical"}:
            cx, cz = anomaly.get("center", [0.0, -2.5])
            cx = _to_float(cx, 0.0)
            depth_c = abs(_to_float(cz, -2.5))
            rx = max(_to_float(anomaly.get("radius_x", anomaly.get("width", 2.0) / 2.0), 1.0), dx)
            rz = max(_to_float(anomaly.get("radius_z", anomaly.get("height", 1.0) / 2.0), 0.5), dz)
            mask = ((X - cx) / rx) ** 2 + ((Z - depth_c) / rz) ** 2 <= 1.0

        elif typ in {"polygon", "rectangle", "block"}:
            points = anomaly.get("points", [])
            if len(points) >= 3:
                px = np.asarray([_to_float(p[0], 0.0) for p in points], dtype=float)
                pz = np.asarray([abs(_to_float(p[1], 0.0)) for p in points], dtype=float)

                xmin = float(np.min(px))
                xmax = float(np.max(px))
                zmin = float(np.min(pz))
                zmax = float(np.max(pz))

                mask = (X >= xmin) & (X <= xmax) & (Z >= zmin) & (Z <= zmax)
            else:
                mask = np.zeros_like(eps, dtype=bool)

        else:
            mask = np.zeros_like(eps, dtype=bool)

        eps[mask] = props["epsilon_r"]
        sigma[mask] = props["conductivity"]

    return x, z, eps, sigma


def _validate_gprmax_sampling(scenario: dict, eps: np.ndarray):
    gpr = scenario.get("survey", {}).get("gpr", {})
    length, depth, dx, dz = _domain(scenario)

    freq_hz = _get_frequency_hz(gpr)
    max_freq_hz = 2.8 * freq_hz
    eps_max = float(np.nanmax(eps))

    wavelength_min = C0 / (max_freq_hz * math.sqrt(max(eps_max, 1.0)))

    cells_x = wavelength_min / dx
    cells_z = wavelength_min / dz

    required = 10.0

    if cells_x < required or cells_z < required:
        recommended = wavelength_min / required
        raise ValueError(
            "gprMax sampling is too coarse for this model.\n"
            f"frequency = {freq_hz / 1e6:.1f} MHz\n"
            f"estimated maximum significant frequency = {max_freq_hz / 1e6:.1f} MHz\n"
            f"maximum epsilon_r = {eps_max:.3g}\n"
            f"current grid dx = {dx:.4g} m, dz = {dz:.4g} m\n"
            f"minimum wavelength estimate = {wavelength_min:.4g} m\n"
            f"cells per wavelength = x:{cells_x:.2f}, z:{cells_z:.2f}\n"
            f"recommended grid spacing <= {recommended:.4g} m\n\n"
            "Fix options:\n"
            "1. Lower antenna frequency.\n"
            "2. Lower high-permittivity target values.\n"
            "3. Reduce grid spacing.\n"
            "4. Use synthetic preview mode instead of actual gprMax FDTD."
        )


def _target_detectability_notes(scenario: dict, eps: np.ndarray, sigma: np.ndarray) -> str:
    gpr = scenario.get("survey", {}).get("gpr", {})
    freq_hz = _get_frequency_hz(gpr)
    _, _, dx, dz = _domain(scenario)

    eps_max = float(np.nanmax(eps))
    sig_max = float(np.nanmax(sigma))
    max_freq_hz = 2.8 * freq_hz
    wavelength_min = C0 / (max_freq_hz * math.sqrt(max(eps_max, 1.0)))
    cells = min(wavelength_min / dx, wavelength_min / dz)

    notes = [
        "GPR feasibility checks",
        f"frequency_mhz={freq_hz / 1e6:.3g}",
        f"max_epsilon_r={eps_max:.6g}",
        f"max_conductivity_S_m={sig_max:.6g}",
        f"minimum_wavelength_estimate_m={wavelength_min:.6g}",
        f"minimum_cells_per_wavelength={cells:.3g}",
    ]

    if cells < 12:
        notes.append("warning=grid is just acceptable; use smaller dx for cleaner gprMax results")

    if sig_max > 0.02:
        notes.append("warning=conductivity is high; GPR attenuation may hide the anomaly")

    if freq_hz < 150e6:
        notes.append("note=low frequency improves penetration but weakens small-target resolution")

    return "\n".join(notes)


def _moving_average_1d(a: np.ndarray, n: int):
    if n <= 1:
        return a.copy()

    n = int(n)
    kernel = np.ones(n, dtype=float) / float(n)

    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=a)


def _dewow(data: np.ndarray, window: int = 81):
    if data.size == 0:
        return data

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    window = min(window, max(3, data.shape[0] // 3 * 2 + 1))

    return data - _moving_average_1d(data, window)


def _background_remove(data: np.ndarray):
    if data.size == 0:
        return data

    return data - np.nanmean(data, axis=1, keepdims=True)


def _agc(data: np.ndarray, window: int = 121, eps: float = 1e-12):
    if data.size == 0:
        return data

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    power = _moving_average_1d(data**2, window)
    rms = np.sqrt(np.maximum(power, eps))

    return data / rms


def _normalise_display(data: np.ndarray, clip_percentile: float = 99.0):
    if data.size == 0:
        return data

    scale = np.nanpercentile(np.abs(data), clip_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(data))

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return np.clip(data / scale, -1.0, 1.0)


def _ricker_wavelet(freq_hz: float, time_window_s: float):
    t = np.linspace(0.0, min(time_window_s, 8.0 / freq_hz), 600)
    t0 = 1.5 / freq_hz
    a = (math.pi * freq_hz * (t - t0)) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return t, w


def _safe_extent_x(x_positions: np.ndarray):
    if x_positions.size == 0:
        return [-1, 1]

    if x_positions.size == 1:
        return [float(x_positions[0]) - 0.5, float(x_positions[0]) + 0.5]

    dx = float(np.median(np.diff(x_positions)))

    return [
        float(x_positions[0] - 0.5 * dx),
        float(x_positions[-1] + 0.5 * dx),
    ]


def _clip_box_to_domain(xmin, xmax, zmin, zmax, length, domain_z, air_thickness, dx, dz):
    eps = max(dx, dz)
    xmin = max(0.0 + eps, float(xmin))
    xmax = min(length - eps, float(xmax))
    zmin = max(air_thickness + eps, float(zmin))
    zmax = min(domain_z - eps, float(zmax))

    return xmin, xmax, zmin, zmax


def _write_gprmax_input(scenario: dict, include_anomalies: bool, base_name: str) -> GPRInputInfo:
    length, depth, dx, dz = _domain(scenario)
    gpr = scenario.get("survey", {}).get("gpr", {})

    output_dir = Path(scenario.get("files", {}).get("output_dir", PROJECT_ROOT / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_hz = _get_frequency_hz(gpr)
    time_window_s = _to_float(gpr.get("time_window_ns", 80.0), 80.0) * 1e-9
    trace_spacing = _to_float(gpr.get("trace_spacing", 0.10), 0.10)

    trace_spacing = max(trace_spacing, dx)

    domain_y = 0.1

    # Topography-aware gprMax geometry.
    # air_clearance is the air gap above the highest point of the surface.
    air_clearance = max(0.5, 10.0 * dz)
    surface_z_down, topo_relief, topo_source, topo_x, topo_elev = _make_gprmax_surface_function(
        scenario=scenario,
        length=length,
        air_clearance=air_clearance,
    )

    air_thickness = air_clearance + topo_relief
    domain_z = depth + air_clearance + topo_relief

    pml_margin = max(12.0 * dx, 0.75)
    rx_sep = _to_float(gpr.get("antenna_separation", 0.50), 0.50)
    rx_sep = max(rx_sep, 5.0 * dx, 0.10)

    src_x = pml_margin
    rx_x = pml_margin + rx_sep

    usable_profile_length = length - rx_x - pml_margin
    if usable_profile_length <= trace_spacing:
        raise ValueError(
            "GPR profile is too short for safe source/receiver stepping.\n"
            f"length={length:g} m, pml_margin={pml_margin:g} m, rx_x={rx_x:g} m\n"
            "Increase model length or reduce antenna separation."
        )

    # gprMax checks that the stepped source/receiver remain inside the domain.
    # Use floor without +1 to avoid "Source(s)/Receiver(s) will be stepped outside".
    n_traces = int(np.floor(usable_profile_length / trace_spacing))
    n_traces = max(n_traces, 2)

    max_traces = int(gpr.get("max_traces", 160))
    if n_traces > max_traces:
        n_traces = max_traces

    x_positions = rx_x + np.arange(n_traces, dtype=float) * trace_spacing - 0.5 * length

    antenna_z = max(air_thickness * 0.5, 2.0 * dz)

    layers = _layers(scenario)
    anomalies = scenario.get("anomalies", []) if include_anomalies else []

    input_path = output_dir / f"{base_name}.in"

    lines = []
    lines.append("#title: generated_forward_modelling_gpr")
    lines.append(f"#domain: {length:.6f} {domain_y:.6f} {domain_z:.6f}")
    lines.append(f"#dx_dy_dz: {dx:.6f} {domain_y:.6f} {dz:.6f}")
    lines.append(f"#time_window: {time_window_s:.12e}")
    lines.append("")
    lines.append("#material: 1 0 1 0 air")

    for i, layer in enumerate(layers, start=1):
        lines.append(
            f"#material: {layer['epsilon_r']:.8g} {layer['conductivity']:.8g} 1 0 layer_{i}"
        )

    for i, anomaly in enumerate(anomalies, start=1):
        props = _anomaly_props(anomaly)
        lines.append(
            f"#material: {props['epsilon_r']:.8g} {props['conductivity']:.8g} 1 0 anomaly_{i}"
        )

    lines.append("")
    lines.append(f"#waveform: ricker 1.0 {frequency_hz:.8e} source_wavelet")
    lines.append(f"#hertzian_dipole: y {src_x:.6f} {0.5 * domain_y:.6f} {antenna_z:.6f} source_wavelet")
    lines.append(f"#rx: {rx_x:.6f} {0.5 * domain_y:.6f} {antenna_z:.6f}")
    lines.append(f"#src_steps: {trace_spacing:.6f} 0 0")
    lines.append(f"#rx_steps: {trace_spacing:.6f} 0 0")
    lines.append("")
    # Topography is represented as staircase columns because gprMax uses a Cartesian grid.
    # Mode:
    #   horizontal_interfaces  -> geological interfaces remain flat in absolute z.
    #   layers_follow_topography / parallel_to_topography -> layer thickness follows local surface.
    topography_mode = str(gpr.get("topography_mode", "horizontal_interfaces")).strip().lower()
    follow_topography = topography_mode in {
        "layers_follow_topography",
        "follow_topography",
        "parallel_to_topography",
    }

    topo_step = max(4.0 * dx, 0.25)
    n_topo_cols = int(np.ceil(length / topo_step))
    n_topo_cols = min(max(n_topo_cols, 2), 240)

    edges = np.linspace(0.0, length, n_topo_cols + 1)

    for j in range(n_topo_cols):
        x0 = float(edges[j])
        x1 = float(edges[j + 1])
        xm_local = 0.5 * (x0 + x1) - 0.5 * length
        surface = float(surface_z_down(xm_local))

        # Air above ground surface.
        lines.append(
            f"#box: {x0:.6f} 0.000000 0.000000 "
            f"{x1:.6f} {domain_y:.6f} {surface:.6f} air"
        )

        current_depth = 0.0

        for i, layer in enumerate(layers, start=1):
            if follow_topography:
                # Layer thickness measured below local ground surface.
                top = surface + current_depth
            else:
                # Interfaces remain horizontally flat in the gprMax absolute z frame.
                # Avoid overwriting air by clipping the layer top below local surface.
                top = max(air_thickness + current_depth, surface)

            bottom = min(domain_z, top + layer["thickness"])

            if bottom > top:
                lines.append(
                    f"#box: {x0:.6f} 0.000000 {top:.6f} "
                    f"{x1:.6f} {domain_y:.6f} {bottom:.6f} layer_{i}"
                )

            current_depth += layer["thickness"]

            if current_depth >= depth or bottom >= domain_z:
                break

    lines.append("")

    for i, anomaly in enumerate(anomalies, start=1):
        typ = str(anomaly.get("type", "circle")).strip().lower()

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -2.5])
            cx = _to_float(cx, 0.0) + 0.5 * length
            zc = float(surface_z_down(cx - 0.5 * length)) + abs(_to_float(cz, -2.5))
            radius = max(_to_float(anomaly.get("radius", 0.75), 0.75), dx)

            eps_clip = max(dx, dz)
            cx = min(max(cx, radius + eps_clip), length - radius - eps_clip)
            zc = min(max(zc, air_thickness + radius + eps_clip), domain_z - radius - eps_clip)

            if 0.0 < cx < length and air_thickness < zc < domain_z:
                lines.append(
                    f"#cylinder: {cx:.6f} 0.000000 {zc:.6f} "
                    f"{cx:.6f} {domain_y:.6f} {zc:.6f} {radius:.6f} anomaly_{i}"
                )

        elif typ in {"ellipse", "elliptical"}:
            cx, cz = anomaly.get("center", [0.0, -2.5])
            cx = _to_float(cx, 0.0) + 0.5 * length
            zc = float(surface_z_down(cx - 0.5 * length)) + abs(_to_float(cz, -2.5))
            width = max(_to_float(anomaly.get("width", 2.0), 2.0), dx)
            height = max(_to_float(anomaly.get("height", 1.0), 1.0), dz)

            xmin = cx - 0.5 * width
            xmax = cx + 0.5 * width
            zmin = zc - 0.5 * height
            zmax = zc + 0.5 * height

            xmin, xmax, zmin, zmax = _clip_box_to_domain(
                xmin, xmax, zmin, zmax, length, domain_z, air_thickness, dx, dz
            )

            if xmax > xmin and zmax > zmin:
                lines.append(
                    f"#box: {xmin:.6f} 0.000000 {zmin:.6f} "
                    f"{xmax:.6f} {domain_y:.6f} {zmax:.6f} anomaly_{i}"
                )

        elif typ in {"polygon", "rectangle", "block"}:
            points = anomaly.get("points", [])

            if len(points) >= 3:
                px = np.asarray([_to_float(p[0], 0.0) + 0.5 * length for p in points], dtype=float)
                local_px = px - 0.5 * length
                pz = np.asarray(
                    [
                        float(surface_z_down(local_px[k])) + abs(_to_float(points[k][1], 0.0))
                        for k in range(len(points))
                    ],
                    dtype=float,
                )

                xmin, xmax, zmin, zmax = _clip_box_to_domain(
                    float(np.min(px)),
                    float(np.max(px)),
                    float(np.min(pz)),
                    float(np.max(pz)),
                    length,
                    domain_z,
                    air_thickness,
                    dx,
                    dz,
                )

                if xmax > xmin and zmax > zmin:
                    lines.append(
                        f"#box: {xmin:.6f} 0.000000 {zmin:.6f} "
                        f"{xmax:.6f} {domain_y:.6f} {zmax:.6f} anomaly_{i}"
                    )

    lines.append("")
    lines.append(
        f"#geometry_view: 0 0 0 {length:.6f} {domain_y:.6f} {domain_z:.6f} "
        f"{dx:.6f} {domain_y:.6f} {dz:.6f} generated_geometry n"
    )
    lines.append("")

    text = "\n".join(lines)
    input_path.write_text(text)

    return GPRInputInfo(
        input_path=input_path,
        base_name=base_name,
        output_dir=output_dir,
        length=length,
        depth=depth,
        domain_y=domain_y,
        air_thickness=air_thickness,
        dx=dx,
        dz=dz,
        frequency_hz=frequency_hz,
        time_window_s=time_window_s,
        trace_spacing=trace_spacing,
        n_traces=n_traces,
        src_x=src_x,
        rx_x=rx_x,
        antenna_z=antenna_z,
        x_positions=x_positions,
        input_text=text,
    )


def _run_command(cmd: list[str], cwd: Path, progress_callback=None):
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    lines = []

    if proc.stdout is not None:
        for line in proc.stdout:
            lines.append(line)

            if progress_callback is not None:
                progress_callback(line.rstrip())

    rc = proc.wait()

    return rc, "".join(lines)


def _remove_previous_outputs(output_dir: Path, base_name: str):
    patterns = [
        f"{base_name}*.out",
        f"{base_name}*.vti",
        f"{base_name}_merged.out",
        "generated_geometry*.vti",
    ]

    for pat in patterns:
        for p in output_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def _read_hdf5_trace_file(path: Path, component: str):
    with h5py.File(path, "r") as f:
        dt = float(f.attrs.get("dt", 1.0))

        rx_path = f"/rxs/rx1/{component}"
        if rx_path not in f:
            available = []
            if "/rxs/rx1" in f:
                available = list(f["/rxs/rx1"].keys())
            raise KeyError(f"Component {component} not found in {path}. Available: {available}")

        arr = np.asarray(f[rx_path], dtype=float)

    return arr, dt


def _read_gprmax_bscan(output_dir: Path, base_name: str, n_traces: int, component: str):
    merged = output_dir / f"{base_name}_merged.out"

    if merged.exists():
        arr, dt = _read_hdf5_trace_file(merged, component)

        if arr.ndim == 1:
            arr = arr[:, None]

        if arr.shape[0] == n_traces and arr.shape[1] != n_traces:
            arr = arr.T

        return arr, dt, merged

    traces = []
    dt = None

    for i in range(1, n_traces + 1):
        path = output_dir / f"{base_name}{i}.out"

        if not path.exists():
            path = output_dir / f"{base_name}{i:03d}.out"

        if not path.exists():
            continue

        arr_i, dt_i = _read_hdf5_trace_file(path, component)

        if arr_i.ndim > 1:
            arr_i = arr_i.ravel()

        traces.append(arr_i)
        dt = dt_i

    if not traces:
        raise FileNotFoundError(f"No gprMax output files found for base {base_name} in {output_dir}")

    min_len = min(len(t) for t in traces)
    traces = [t[:min_len] for t in traces]

    return np.column_stack(traces), float(dt), output_dir / f"{base_name}*.out"


def _run_actual_gprmax(info: GPRInputInfo, component: str, scenario: dict):
    progress_callback = scenario.get("_progress_callback")

    _remove_previous_outputs(info.output_dir, info.base_name)

    # Exact safety correction for gprMax source/rx stepping.
    # gprMax model run 1 uses the initial position.
    # Model run N uses initial + (N - 1) * step.
    trace_spacing = float(info.trace_spacing)
    safety_margin = max(float(info.dx), 1e-6)
    max_start_x = max(float(info.src_x), float(info.rx_x))

    if trace_spacing > 0:
        safe_n_traces = int(
            math.floor(
                (float(info.length) - max_start_x - safety_margin) / trace_spacing
            )
        ) + 1
        safe_n_traces = max(1, safe_n_traces)
    else:
        safe_n_traces = int(info.n_traces)

    original_n_traces = int(info.n_traces)
    actual_n_traces = min(original_n_traces, int(safe_n_traces))
    actual_n_traces = max(1, actual_n_traces)

    # Keep plotting positions consistent with the number actually run.
    info.x_positions = info.x_positions[:actual_n_traces]
    info.n_traces = actual_n_traces

    final_source_x = float(info.src_x) + (actual_n_traces - 1) * trace_spacing
    final_receiver_x = float(info.rx_x) + (actual_n_traces - 1) * trace_spacing

    if progress_callback is not None:
        progress_callback(
            "\n".join(
                [
                    "gprMax trace-count safety check:",
                    f"  domain_length = {info.length:g} m",
                    f"  source_start_x = {info.src_x:g} m",
                    f"  receiver_start_x = {info.rx_x:g} m",
                    f"  trace_spacing = {trace_spacing:g} m",
                    f"  original_n_traces = {original_n_traces}",
                    f"  safe_n_traces = {safe_n_traces}",
                    f"  used_n_traces = {actual_n_traces}",
                    f"  final_source_x = {final_source_x:g} m",
                    f"  final_receiver_x = {final_receiver_x:g} m",
                    f"  right_domain_limit = {info.length:g} m",
                ]
            )
        )

    cmd1 = [
        sys.executable,
        "-m",
        "gprMax",
        info.input_path.name,
        "-n",
        str(actual_n_traces),
    ]

    if progress_callback is not None:
        progress_callback(f"Running gprMax: {' '.join(cmd1)}")

    rc1, out1 = _run_command(cmd1, cwd=info.output_dir, progress_callback=progress_callback)

    if rc1 != 0:
        raise RuntimeError(
            "gprMax execution failed.\n"
            f"Command: {' '.join(cmd1)}\n"
            f"Output:\n{out1}"
        )

    cmd2 = [
        sys.executable,
        "-m",
        "tools.outputfiles_merge",
        info.base_name,
    ]

    if progress_callback is not None:
        progress_callback(f"Merging gprMax outputs: {' '.join(cmd2)}")

    rc2, out2 = _run_command(cmd2, cwd=info.output_dir, progress_callback=progress_callback)

    if rc2 != 0:
        if progress_callback is not None:
            progress_callback("Merge failed; attempting to read individual trace files.")

    bscan, dt, source_path = _read_gprmax_bscan(
        output_dir=info.output_dir,
        base_name=info.base_name,
        n_traces=actual_n_traces,
        component=component,
    )

    return {
        "bscan": bscan,
        "dt": dt,
        "source_path": str(source_path),
        "stdout": out1 + "\n" + out2,
    }

def _synthetic_preview_bscan(scenario: dict, x_positions: np.ndarray, time_ns: np.ndarray):
    length, depth, dx, dz = _domain(scenario)
    layers = _layers(scenario)
    anomalies = scenario.get("anomalies", [])

    bscan = np.zeros((len(time_ns), len(x_positions)), dtype=float)

    freq = _get_frequency_hz(scenario.get("survey", {}).get("gpr", {}))
    period_ns = 1e9 / freq
    pulse_width = max(1.5 * period_ns, 2.0)

    cumulative = 0.0
    eps_above = layers[0]["epsilon_r"]

    for i, layer in enumerate(layers[:-1]):
        cumulative += layer["thickness"]

        if cumulative >= depth:
            break

        v = C0 / math.sqrt(max(eps_above, 1.0))
        twt_ns = 2.0 * cumulative / v * 1e9

        amp = 0.3 + 0.1 * i
        bscan += amp * np.exp(-0.5 * ((time_ns[:, None] - twt_ns) / pulse_width) ** 2)

        eps_above = layer["epsilon_r"]

    for anomaly in anomalies:
        typ = str(anomaly.get("type", "circle")).strip().lower()

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -2.5])
            cx = _to_float(cx, 0.0)
            zc = abs(_to_float(cz, -2.5))
            r = _to_float(anomaly.get("radius", 1.0), 1.0)
        else:
            points = anomaly.get("points", [])
            if len(points) >= 3:
                px = np.asarray([_to_float(p[0], 0.0) for p in points])
                pz = np.asarray([abs(_to_float(p[1], 0.0)) for p in points])
                cx = float(np.mean(px))
                zc = float(np.mean(pz))
                r = max(float(np.ptp(px)) * 0.25, 0.5)
            else:
                continue

        eps_bg = max(float(np.nanmedian([l["epsilon_r"] for l in layers])), 1.0)
        v = C0 / math.sqrt(eps_bg)

        for j, x in enumerate(x_positions):
            dist = math.sqrt((x - cx) ** 2 + zc**2)
            twt_ns = 2.0 * dist / v * 1e9

            amp = 0.35 * max(r, 0.5)
            bscan[:, j] += amp * np.exp(-0.5 * ((time_ns - twt_ns) / pulse_width) ** 2)

    return bscan


def _build_plots(
    scenario: dict,
    x: np.ndarray,
    z: np.ndarray,
    eps: np.ndarray,
    sigma: np.ndarray,
    info: GPRInputInfo,
    actual_full=None,
    actual_background=None,
    component: str = "Ey",
):
    plots = {}

    extent_model = [float(x[0]), float(x[-1]), float(z[-1]), float(z[0])]

    plots["Permittivity model"] = {
        "type": "image",
        "array": eps,
        "extent": extent_model,
        "origin": "upper",
        "title": "GPR relative permittivity model",
        "xlabel": "x [m]",
        "ylabel": "depth [m]",
        "clabel": "relative permittivity [-]",
        "colorbar": True,
    }

    plots["Conductivity model"] = {
        "type": "image",
        "array": sigma,
        "extent": extent_model,
        "origin": "upper",
        "title": "GPR electrical conductivity model",
        "xlabel": "x [m]",
        "ylabel": "depth [m]",
        "clabel": "conductivity [S/m]",
        "colorbar": True,
    }

    if actual_full is not None:
        raw = np.asarray(actual_full["bscan"], dtype=float)
        dt = float(actual_full["dt"])
        time_ns = np.arange(raw.shape[0], dtype=float) * dt * 1e9

        full_dewow = _dewow(raw)
        full_processed = _normalise_display(_agc(_background_remove(full_dewow)))

        x_extent = _safe_extent_x(info.x_positions)
        extent_data = [x_extent[0], x_extent[1], float(time_ns[-1]), float(time_ns[0])]

        plots["Actual gprMax B-scan"] = {
            "type": "image",
            "array": full_processed,
            "extent": extent_data,
            "origin": "upper",
            "title": "Actual gprMax B-scan: dewow + background removal + AGC",
            "xlabel": "x [m]",
            "ylabel": "time [ns]",
            "clabel": f"{component} normalised amplitude",
            "colorbar": True,
            "vmin": -1.0,
            "vmax": 1.0,
        }

        plots["Raw B-scan"] = {
            "type": "image",
            "array": raw,
            "extent": extent_data,
            "origin": "upper",
            "title": "Raw actual gprMax B-scan",
            "xlabel": "x [m]",
            "ylabel": "time [ns]",
            "clabel": f"{component} amplitude",
            "colorbar": True,
        }

        if actual_background is not None:
            bg = np.asarray(actual_background["bscan"], dtype=float)

            nt = min(raw.shape[0], bg.shape[0])
            nx = min(raw.shape[1], bg.shape[1])

            raw2 = raw[:nt, :nx]
            bg2 = bg[:nt, :nx]
            diff = raw2 - bg2

            time_ns_diff = time_ns[:nt]
            x_positions_diff = info.x_positions[:nx]
            x_extent_diff = _safe_extent_x(x_positions_diff)
            extent_diff = [x_extent_diff[0], x_extent_diff[1], float(time_ns_diff[-1]), float(time_ns_diff[0])]

            diff_processed = _normalise_display(_agc(_dewow(diff)))

            plots["Anomaly-only difference B-scan"] = {
                "type": "image",
                "array": diff_processed,
                "extent": extent_diff,
                "origin": "upper",
                "title": "Detectability plot: full model minus background model",
                "xlabel": "x [m]",
                "ylabel": "time [ns]",
                "clabel": f"{component} normalised difference",
                "colorbar": True,
                "vmin": -1.0,
                "vmax": 1.0,
            }

            plots["Background model B-scan"] = {
                "type": "image",
                "array": _normalise_display(_agc(_background_remove(_dewow(bg2)))),
                "extent": extent_diff,
                "origin": "upper",
                "title": "Background-only B-scan",
                "xlabel": "x [m]",
                "ylabel": "time [ns]",
                "clabel": f"{component} normalised amplitude",
                "colorbar": True,
                "vmin": -1.0,
                "vmax": 1.0,
            }

            energy_full = float(np.sqrt(np.nanmean(raw2**2)))
            energy_diff = float(np.sqrt(np.nanmean(diff**2)))
            detectability_ratio = energy_diff / max(energy_full, 1e-30)

            plots["Detectability metric"] = {
                "type": "text",
                "title": "GPR detectability metric",
                "text": (
                    "GPR anomaly detectability metric\n\n"
                    f"RMS(full B-scan) = {energy_full:.6g}\n"
                    f"RMS(full - background) = {energy_diff:.6g}\n"
                    f"difference/full ratio = {detectability_ratio:.6g}\n\n"
                    "Interpretation:\n"
                    "< 0.02  : very weak / likely hidden\n"
                    "0.02-0.10: detectable only after processing\n"
                    "> 0.10  : clearly detectable\n\n"
                    "Use the 'Anomaly-only difference B-scan' tab to judge whether the target response exists."
                ),
            }

        centre_idx = int(np.argmin(np.abs(info.x_positions)))
        centre_trace = raw[:, centre_idx]

        plots["A-scan centre trace"] = {
            "type": "line",
            "array": np.column_stack([time_ns, centre_trace]),
            "title": f"A-scan centre trace at x={info.x_positions[centre_idx]:.2f} m",
            "xlabel": "time [ns]",
            "ylabel": f"{component} amplitude",
            "labels": {1: component},
        }

    else:
        time_ns = np.linspace(0.0, info.time_window_s * 1e9, 700)
        synthetic = _synthetic_preview_bscan(scenario, info.x_positions, time_ns)
        processed = _normalise_display(_agc(_background_remove(_dewow(synthetic))))

        x_extent = _safe_extent_x(info.x_positions)
        extent_data = [x_extent[0], x_extent[1], float(time_ns[-1]), float(time_ns[0])]

        plots["Synthetic preview B-scan"] = {
            "type": "image",
            "array": processed,
            "extent": extent_data,
            "origin": "upper",
            "title": "Fast synthetic GPR B-scan preview",
            "xlabel": "x [m]",
            "ylabel": "time [ns]",
            "clabel": "normalised amplitude",
            "colorbar": True,
            "vmin": -1.0,
            "vmax": 1.0,
        }

    t_wave, w = _ricker_wavelet(info.frequency_hz, info.time_window_s)

    plots["Source wavelet"] = {
        "type": "line",
        "array": np.column_stack([t_wave * 1e9, w]),
        "title": f"Ricker source wavelet, f={info.frequency_hz / 1e6:.1f} MHz",
        "xlabel": "time [ns]",
        "ylabel": "amplitude",
        "labels": {1: "Ricker wavelet"},
    }

    plots["gprMax input"] = {
        "type": "text",
        "title": "Generated gprMax input file",
        "text": info.input_text,
    }

    return plots


def run_gpr(scenario: dict) -> dict:
    _progress(scenario, "Preparing GPR model...")

    x, z, eps, sigma = _make_property_models(scenario)
    gpr = scenario.get("survey", {}).get("gpr", {})

    component = str(gpr.get("component", "Ey"))
    run_actual = bool(gpr.get("run_actual_gprmax", True))
    run_background = bool(gpr.get("run_background_difference", True))

    full_info = _write_gprmax_input(
        scenario=scenario,
        include_anomalies=True,
        base_name="generated_gprmax_model",
    )

    actual_full = None
    actual_background = None

    if run_actual:
        _validate_gprmax_sampling(scenario, eps)

        _progress(scenario, "Running actual gprMax full model with anomaly...")
        actual_full = _run_actual_gprmax(full_info, component=component, scenario=scenario)

        if run_background and scenario.get("anomalies", []):
            bg_info = _write_gprmax_input(
                scenario=scenario,
                include_anomalies=False,
                base_name="generated_gprmax_background",
            )

            _progress(scenario, "Running actual gprMax background model without anomaly...")
            actual_background = _run_actual_gprmax(bg_info, component=component, scenario=scenario)

            _progress(scenario, "Computing full-minus-background anomaly difference B-scan...")

    plots = _build_plots(
        scenario=scenario,
        x=x,
        z=z,
        eps=eps,
        sigma=sigma,
        info=full_info,
        actual_full=actual_full,
        actual_background=actual_background,
        component=component,
    )

    info_lines = [
        "GPR forward model completed",
        "Implementation: actual gprMax FDTD, optional background model, anomaly-difference B-scan, dewow/background-removal/AGC display.",
        _target_detectability_notes(scenario, eps, sigma),
        f"gprmax_input={full_info.input_path}",
        f"frequency_hz={full_info.frequency_hz:.6g}",
        f"frequency_mhz={full_info.frequency_hz / 1e6:.6g}",
        f"trace_spacing_m={full_info.trace_spacing:.6g}",
        f"n_traces={full_info.n_traces}",
        f"model_length_m={full_info.length:.6g}",
        f"model_depth_m={full_info.depth:.6g}",
        f"dx_m={full_info.dx:.6g}",
        f"dz_m={full_info.dz:.6g}",
        f"component={component}",
        f"topography_mode={gpr.get('topography_mode', 'horizontal_interfaces')}",
        f"topography_csv={scenario.get('files', {}).get('elevation_csv', '')}",
        f"run_actual_gprmax={run_actual}",
        f"run_background_difference={run_background}",
        f"epsilon_min={float(np.nanmin(eps)):.6g}",
        f"epsilon_max={float(np.nanmax(eps)):.6g}",
        f"conductivity_min={float(np.nanmin(sigma)):.6g}",
        f"conductivity_max={float(np.nanmax(sigma)):.6g}",
    ]

    if actual_full is not None:
        info_lines.append(f"actual_full_output={actual_full['source_path']}")

    if actual_background is not None:
        info_lines.append(f"actual_background_output={actual_background['source_path']}")

    return {
        "model": eps,
        "data": None,
        "plots": plots,
        "info": "\n".join(info_lines),
    }
