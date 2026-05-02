from __future__ import annotations

from pathlib import Path

import numpy as np

from devito import configuration
configuration["log-level"] = "WARNING"

from examples.seismic import Model, TimeAxis, RickerSource, Receiver, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

try:
    from matplotlib.path import Path as MplPath
except Exception:
    MplPath = None


def _progress(scenario: dict, text: str):
    cb = scenario.get("_progress_callback")
    if cb is not None:
        cb(str(text))


def _to_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _survey_seismic(scenario: dict) -> dict:
    return scenario.get("survey", {}).get("seismic", scenario.get("seismic", {}))


def _read_topography_csv(path: str | None):
    if not path:
        return None

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Seismic topography CSV not found: {p}")

    last_error = None

    for delimiter in [";", ",", None]:
        try:
            arr = np.genfromtxt(
                p,
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

            x = np.atleast_1d(np.asarray(arr[x_col], dtype=float))
            z = np.atleast_1d(np.asarray(arr[z_col], dtype=float))

            mask = np.isfinite(x) & np.isfinite(z)
            x = x[mask]
            z = z[mask]

            if len(x) < 2:
                continue

            order = np.argsort(x)
            x = x[order]
            z = z[order]

            x_unique, idx = np.unique(x, return_index=True)
            z_unique = z[idx]

            if len(x_unique) < 2:
                continue

            return x_unique, z_unique, str(p)

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Could not parse seismic topography CSV: {p}. "
        "Expected columns like Distance/Altitude or x/z."
    ) from last_error


def _make_topography_function(scenario: dict, length: float):
    """
    Devito z-axis is positive downward.

    CSV altitude is positive upward, so we convert:
        high elevation -> shallow surface depth
        low elevation  -> deeper surface depth

    Surface depth is relative to the highest point.
    """
    csv_path = scenario.get("files", {}).get("elevation_csv", "")
    csv_topo = _read_topography_csv(csv_path) if csv_path else None

    if csv_topo is not None:
        raw_x, raw_alt, source = csv_topo

        span = float(np.nanmax(raw_x) - np.nanmin(raw_x))
        if span <= 0:
            raise ValueError("Topography CSV distance range is zero.")

        x_local = (raw_x - np.nanmin(raw_x)) / span * length
        alt_rel = raw_alt - np.nanmin(raw_alt)
        relief = float(np.nanmax(alt_rel) - np.nanmin(alt_rel))

        # Convert elevation to depth below highest point.
        topo_depth = relief - alt_rel

    else:
        topo = scenario.get("topography", [])

        if topo and len(topo) >= 2:
            pts = np.asarray(topo, dtype=float)
            x_raw = pts[:, 0]
            z_raw = pts[:, 1]

            # Accept either centred [-L/2, L/2] or [0, L].
            if np.nanmin(x_raw) < 0:
                x_local = x_raw + 0.5 * length
            else:
                x_local = x_raw

            z_rel = z_raw - np.nanmin(z_raw)
            relief = float(np.nanmax(z_rel) - np.nanmin(z_rel))

            # Treat scenario topography as elevation-like.
            topo_depth = relief - z_rel
            source = "scenario.topography"

        else:
            x_local = np.asarray([0.0, length], dtype=float)
            topo_depth = np.asarray([0.0, 0.0], dtype=float)
            relief = 0.0
            source = "flat default"

    order = np.argsort(x_local)
    x_local = np.asarray(x_local[order], dtype=float)
    topo_depth = np.asarray(topo_depth[order], dtype=float)

    x_unique, idx = np.unique(x_local, return_index=True)
    topo_depth = topo_depth[idx]
    x_local = x_unique

    if PchipInterpolator is not None and len(x_local) >= 3:
        interp = PchipInterpolator(x_local, topo_depth, extrapolate=True)

        def topo_fn(x):
            return np.asarray(interp(x), dtype=float)

    else:
        def topo_fn(x):
            return np.interp(np.asarray(x, dtype=float), x_local, topo_depth)

    return topo_fn, x_local, topo_depth, relief, source


def _layers(scenario: dict):
    layers = scenario.get("layers", [])

    if not layers:
        layers = [
            {"thickness": 10.0, "velocity": 1200.0},
            {"thickness": 15.0, "velocity": 2500.0},
            {"thickness": 999.0, "velocity": 3500.0},
        ]

    clean = []

    for layer in layers:
        vp = _to_float(layer.get("velocity", layer.get("vp", 1500.0)), 1500.0)
        clean.append(
            {
                "thickness": max(_to_float(layer.get("thickness", 999.0), 999.0), 0.0),
                "velocity": max(vp, 300.0),
                "gradient": _to_float(layer.get("gradient", layer.get("vp_gradient", 0.0)), 0.0),
            }
        )

    return clean


def _build_velocity_model(scenario: dict):
    domain = scenario.get("domain", {})
    seis = _survey_seismic(scenario)

    length = _to_float(domain.get("length", 80.0), 80.0)
    depth = _to_float(domain.get("depth", 40.0), 40.0)

    dx = _to_float(domain.get("dx", seis.get("dx", 1.0)), 1.0)
    dz = _to_float(domain.get("dz", seis.get("dz", dx)), dx)

    length = max(length, 2.0)
    depth = max(depth, 1.0)
    dx = max(dx, 0.1)
    dz = max(dz, 0.1)

    topo_fn, topo_x, topo_z, topo_relief, topo_source = _make_topography_function(scenario, length)
    topography_mode = str(seis.get("topography_mode", "horizontal_interfaces")).strip().lower()
    follow_topography = topography_mode in {"parallel_to_topography", "follow_topography", "layers_follow_topography"}

    # Extra depth because topography is represented as air above the ground surface.
    model_depth = depth + topo_relief

    nx = int(round(length / dx)) + 1
    nz = int(round(model_depth / dz)) + 1

    max_cells = nx * nz
    if max_cells > 900_000:
        raise ValueError(
            "Seismic Devito model is too large for interactive GUI use.\n"
            f"Grid would be {nx} x {nz} = {max_cells:,} cells.\n"
            "Increase dx/dz or reduce model length/depth."
        )

    x = np.linspace(0.0, length, nx)
    z = np.linspace(0.0, model_depth, nz)
    topo_grid = topo_fn(x)

    layers = _layers(scenario)
    min_geo_vp = min(layer["velocity"] for layer in layers)
    vp_air = _to_float(seis.get("vp_air", 0.8 * min_geo_vp), 0.8 * min_geo_vp)

    vp = np.empty((nx, nz), dtype=np.float32)

    cumulative = np.cumsum([layer["thickness"] for layer in layers])

    for ix, xval in enumerate(x):
        surface = float(topo_grid[ix])

        for iz, zval in enumerate(z):
            if zval < surface:
                vp[ix, iz] = vp_air
                continue

            if follow_topography:
                depth_for_layering = zval - surface
            else:
                depth_for_layering = zval - topo_relief

            layer_idx = len(layers) - 1
            for k, bottom in enumerate(cumulative):
                if depth_for_layering < bottom:
                    layer_idx = k
                    break

            layer_top = 0.0 if layer_idx == 0 else cumulative[layer_idx - 1]
            depth_in_layer = max(0.0, depth_for_layering - layer_top)

            layer = layers[layer_idx]
            vp[ix, iz] = layer["velocity"] + layer["gradient"] * depth_in_layer

    X, Z = np.meshgrid(x, z, indexing="ij")
    anomalies = scenario.get("anomalies", [])

    body_outlines = []

    for anomaly in anomalies:
        typ = str(anomaly.get("type", "circle")).lower().strip()
        avp = _to_float(anomaly.get("velocity", anomaly.get("vp", 1800.0)), 1800.0)

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -5.0])
            cx = _to_float(cx, 0.0) + 0.5 * length
            surface = float(topo_fn([cx])[0])
            zc = surface + abs(_to_float(cz, -5.0))
            r = max(_to_float(anomaly.get("radius", 2.0), 2.0), dx)

            mask = (X - cx) ** 2 + (Z - zc) ** 2 <= r ** 2
            vp[mask] = avp

            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            body_outlines.append(
                np.column_stack([cx + r * np.cos(theta), zc + r * np.sin(theta)])
            )

        elif typ in {"polygon", "rectangle", "block"}:
            points = anomaly.get("points", [])

            if len(points) >= 3:
                px_local = np.asarray([_to_float(p[0], 0.0) for p in points], dtype=float)
                px = px_local + 0.5 * length

                pz = np.asarray(
                    [
                        float(topo_fn([px[k]])[0]) + abs(_to_float(points[k][1], 0.0))
                        for k in range(len(points))
                    ],
                    dtype=float,
                )

                if MplPath is not None:
                    path = MplPath(np.column_stack([px, pz]))
                    grid_points = np.column_stack([X.ravel(), Z.ravel()])
                    mask = path.contains_points(grid_points).reshape(X.shape)
                else:
                    mask = (
                        (X >= np.nanmin(px)) & (X <= np.nanmax(px)) &
                        (Z >= np.nanmin(pz)) & (Z <= np.nanmax(pz))
                    )

                vp[mask] = avp
                body_outlines.append(np.column_stack([px, pz]))

    return {
        "x": x,
        "z": z,
        "vp": vp,
        "dx": dx,
        "dz": dz,
        "length": length,
        "depth": depth,
        "model_depth": model_depth,
        "topo_grid": topo_grid,
        "topo_x": topo_x,
        "topo_z": topo_z,
        "topo_relief": topo_relief,
        "topo_source": topo_source,
        "topography_mode": topography_mode,
        "follow_topography": follow_topography,
        "vp_air": vp_air,
        "layers": layers,
        "body_outlines": body_outlines,
    }


def _safe_clip(data: np.ndarray, clip: float = 0.02):
    vmax = float(clip * np.nanmax(np.abs(data))) if data.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax



def _first_arrival_curve(data: np.ndarray, time_ms: np.ndarray, rec_x: np.ndarray):
    """Simple first-arrival picker using thresholded envelope."""
    if data.size == 0:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(data, dtype=float)
    env = np.abs(arr)
    max_per_trace = np.nanmax(env, axis=0)
    picks = []

    for j in range(env.shape[1]):
        threshold = 0.08 * max(max_per_trace[j], 1e-30)
        idx = np.where(env[:, j] >= threshold)[0]
        if len(idx):
            picks.append([rec_x[j], time_ms[int(idx[0])]])

    return np.asarray(picks, dtype=float)


def _normalise_shot_display(data: np.ndarray, clip_percentile: float = 99.0):
    arr = np.asarray(data, dtype=float)
    scale = np.nanpercentile(np.abs(arr), clip_percentile) if arr.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(arr)) if arr.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(arr / scale, -1.0, 1.0)



def _pick_first_arrivals(data: np.ndarray, time_ms: np.ndarray, threshold: float = 0.08):
    """Simple first-break picker using per-trace amplitude threshold."""
    arr = np.asarray(data, dtype=float)
    t = np.asarray(time_ms, dtype=float)

    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([]), np.asarray([])

    nt, nr = arr.shape
    picks = np.full(nr, np.nan, dtype=float)
    pick_amp = np.full(nr, np.nan, dtype=float)
    mute_samples = max(2, int(0.002 * nt))

    for ir in range(nr):
        tr = arr[:, ir]
        scale = np.nanmax(np.abs(tr))
        if not np.isfinite(scale) or scale <= 0:
            continue

        idx = np.where(np.abs(tr[mute_samples:]) >= threshold * scale)[0]
        if idx.size:
            k = int(idx[0] + mute_samples)
            picks[ir] = t[min(k, len(t) - 1)]
            pick_amp[ir] = tr[k]

    return picks, pick_amp


def _apparent_velocity_from_picks(x_m: np.ndarray, t_ms: np.ndarray):
    """Compute local apparent velocity dx/dt from first-arrival picks."""
    x = np.asarray(x_m, dtype=float)
    t = np.asarray(t_ms, dtype=float) / 1000.0

    valid = np.isfinite(x) & np.isfinite(t)
    if valid.sum() < 3:
        return np.asarray([]), np.asarray([])

    xv = x[valid]
    tv = t[valid]

    dx = np.gradient(xv)
    dt = np.gradient(tv)

    with np.errstate(divide="ignore", invalid="ignore"):
        vapp = dx / dt

    vapp[~np.isfinite(vapp)] = np.nan
    vapp[np.abs(vapp) > 20000] = np.nan

    return xv, vapp

def run_seismic(scenario: dict) -> dict:
    _progress(scenario, "Preparing seismic velocity model with topography correction...")

    m = _build_velocity_model(scenario)

    x = m["x"]
    z = m["z"]
    vp = m["vp"]
    dx = m["dx"]
    dz = m["dz"]
    length = m["length"]
    model_depth = m["model_depth"]
    topo_grid = m["topo_grid"]

    seis = _survey_seismic(scenario)

    freq = _to_float(
        seis.get("source_frequency", seis.get("freq", 20.0)),
        20.0,
    )
    tn_s = _to_float(
        seis.get("duration", seis.get("recording_time_ms", 1000.0) / 1000.0),
        1.0,
    )
    tn_ms = tn_s * 1000.0 if tn_s < 100.0 else tn_s

    space_order = int(seis.get("space_order", 8))
    nbl = int(seis.get("nbl", 40))
    free_surface = bool(seis.get("free_surface", False))

    show_refraction_overlay = bool(seis.get("show_refraction_overlay", True))
    show_apparent_velocity = bool(seis.get("show_apparent_velocity", True))
    show_first_arrival_difference = bool(seis.get("show_first_arrival_difference", True))
    show_refraction_summary = bool(seis.get("show_refraction_summary", True))
    run_background = bool(seis.get("run_background_difference", True))
    show_first_arrivals = bool(seis.get("show_first_arrivals", True))
    save_wavefield_snapshot = bool(seis.get("save_wavefield_snapshot", False))
    acquisition_mode = str(seis.get("acquisition_mode", "single_shot")).strip().lower()
    nshots = int(seis.get("nshots", 1))
    shot_spacing = _to_float(seis.get("shot_spacing", 5.0), 5.0)

    _progress(
        scenario,
        (
            "Seismic model grid:\n"
            f"  nx={vp.shape[0]}, nz={vp.shape[1]}\n"
            f"  dx={dx:g} m, dz={dz:g} m\n"
            f"  model_length={length:g} m\n"
            f"  model_depth_with_topography={model_depth:g} m\n"
            f"  topography_source={m['topo_source']}\n"
            f"  topography_relief={m['topo_relief']:.3f} m\n"
            f"  layer_topography_mode={m['topography_mode']}"
        ),
    )

    # Devito expects km/s.
    model = Model(
        vp=vp / 1000.0,
        origin=(0.0, 0.0),
        spacing=(dx, dz),
        shape=vp.shape,
        nbl=nbl,
        space_order=space_order,
        bcs="damp",
        fs=free_surface,
    )

    time_axis = TimeAxis(start=0.0, stop=tn_ms, step=model.critical_dt)

    nrec = int(seis.get("nrec", seis.get("receivers", max(24, min(201, vp.shape[0])))))
    x_rec_start = _to_float(seis.get("x_rec_start", 0.0), 0.0)
    x_rec_end = _to_float(seis.get("x_rec_end", length), length)
    z_rec = _to_float(seis.get("z_rec", seis.get("receiver_depth", 2.0)), 2.0)

    x_src = _to_float(seis.get("x_src", 0.5 * length), 0.5 * length)
    z_src = _to_float(seis.get("z_src", seis.get("source_depth", 2.0)), 2.0)

    x_src = min(max(x_src, 0.0), length)
    src_surface = float(np.interp(x_src, x, topo_grid))
    src_z_abs = src_surface + z_src

    rec_x = np.linspace(x_rec_start, x_rec_end, nrec)
    rec_x = np.clip(rec_x, 0.0, length)
    rec_surface = np.interp(rec_x, x, topo_grid)
    rec_z_abs = rec_surface + z_rec

    src = RickerSource(
        name="src",
        grid=model.grid,
        f0=freq / 1000.0,
        time_range=time_axis,
    )
    src.coordinates.data[0, 0] = x_src
    src.coordinates.data[0, 1] = src_z_abs

    rec = Receiver(
        name="rec",
        grid=model.grid,
        npoint=nrec,
        time_range=time_axis,
    )
    rec.coordinates.data[:, 0] = rec_x
    rec.coordinates.data[:, 1] = rec_z_abs

    _progress(
        scenario,
        (
            "Running Devito acoustic forward simulation...\n"
            f"  source_x={x_src:.3f} m\n"
            f"  source_z_abs={src_z_abs:.3f} m\n"
            f"  receiver_count={nrec}\n"
            f"  receiver_depth_below_surface={z_rec:g} m\n"
            f"  f0={freq:g} Hz\n"
            f"  tn={tn_ms:g} ms\n"
            f"  critical_dt={model.critical_dt:.5g} ms"
        ),
    )

    geometry = AcquisitionGeometry(
        model,
        rec.coordinates.data.copy(),
        src.coordinates.data.copy(),
        t0=0.0,
        tn=tn_ms,
        f0=freq / 1000.0,
        src_type="Ricker",
    )

    solver = AcousticWaveSolver(model, geometry, space_order=space_order)
    rec_out, u, _ = solver.forward(save=save_wavefield_snapshot)

    data = np.asarray(rec_out.data, dtype=float)

    time_values_ms = np.asarray(time_axis.time_values, dtype=float)
    first_arrival_ms, first_arrival_amp = _pick_first_arrivals(data, time_values_ms)
    appvel_x, appvel = _apparent_velocity_from_picks(rec_x, first_arrival_ms)
    background_difference = None

    if run_background and scenario.get("anomalies", []):
        _progress(scenario, "Running seismic background model without anomaly...")
        bg_scenario = dict(scenario)
        bg_scenario["anomalies"] = []
        bg_m = _build_velocity_model(bg_scenario)
        bg_vp = bg_m["vp"]

        bg_model = Model(
            vp=bg_vp / 1000.0,
            origin=(0.0, 0.0),
            spacing=(dx, dz),
            shape=bg_vp.shape,
            nbl=nbl,
            space_order=space_order,
            bcs="damp",
            fs=free_surface,
        )

        bg_geometry = AcquisitionGeometry(
            bg_model,
            rec.coordinates.data.copy(),
            src.coordinates.data.copy(),
            t0=0.0,
            tn=tn_ms,
            f0=freq / 1000.0,
            src_type="Ricker",
        )

        bg_solver = AcousticWaveSolver(bg_model, bg_geometry, space_order=space_order)
        bg_rec, _, _ = bg_solver.forward(save=False)

        nt = min(data.shape[0], bg_rec.data.shape[0])
        nr = min(data.shape[1], bg_rec.data.shape[1])
        background_difference = data[:nt, :nr] - np.asarray(bg_rec.data[:nt, :nr], dtype=float)

    _progress(scenario, "Devito seismic simulation completed. Building plots...")

    vmin, vmax = _safe_clip(data, clip=0.02)

    extent_model = [0.0, length, model_depth, 0.0]
    extent_data = [float(rec_x[0]), float(rec_x[-1]), tn_ms, 0.0]

    plots = {
        "Velocity model": {
            "type": "image",
            "array": vp.T,
            "extent": extent_model,
            "origin": "upper",
            "title": "Seismic P-wave velocity model with topography",
            "xlabel": "x [m]",
            "ylabel": "z [m]",
            "clabel": "Vp [m/s]",
            "colorbar": True,
        },
        "Shot gather": {
            "type": "image",
            "array": data,
            "extent": extent_data,
            "origin": "upper",
            "title": f"Seismic shot gather, source x={x_src:.1f} m",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "amplitude",
            "colorbar": True,
            "vmin": vmin,
            "vmax": vmax,
        },
        "Source wavelet": {
            "type": "line",
            "array": np.column_stack([np.asarray(time_axis.time_values), np.asarray(src.data[:, 0])]),
            "title": f"Ricker source wavelet, f0={freq:g} Hz",
            "xlabel": "time [ms]",
            "ylabel": "amplitude",
            "labels": {1: "Ricker"},
        },
        "Topography profile": {
            "type": "line",
            "array": np.column_stack([x, topo_grid]),
            "title": "Seismic topography converted to Devito z-depth",
            "xlabel": "x [m]",
            "ylabel": "surface z [m, positive downward]",
            "labels": {1: "surface"},
        },
    }

    time_values = np.asarray(time_axis.time_values, dtype=float)

    if background_difference is not None:
        diff_display = _normalise_shot_display(background_difference)
        plots["Background-difference shot gather"] = {
            "type": "image",
            "array": diff_display,
            "extent": extent_data,
            "origin": "upper",
            "title": "Seismic anomaly-only response: full model minus background",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "normalised difference",
            "colorbar": True,
            "vmin": -1.0,
            "vmax": 1.0,
        }

        rms_full = float(np.sqrt(np.nanmean(data ** 2)))
        rms_diff = float(np.sqrt(np.nanmean(background_difference ** 2)))
        ratio = rms_diff / max(rms_full, 1e-30)

        plots["Seismic detectability metric"] = {
            "type": "text",
            "title": "Seismic detectability metric",
            "text": (
                "Seismic anomaly detectability metric\n\n"
                f"RMS(full shot gather) = {rms_full:.6g}\n"
                f"RMS(full - background) = {rms_diff:.6g}\n"
                f"difference/full ratio = {ratio:.6g}\n\n"
                "Interpretation:\n"
                "< 0.02  : very weak / likely hidden\n"
                "0.02-0.10: weak but possibly visible after processing\n"
                "> 0.10  : likely visible"
            ),
        }

    if show_first_arrivals:
        picks = _first_arrival_curve(data, time_values, rec_x)
        if picks.size:
            plots["First-arrival / refraction picks"] = {
                "type": "line",
                "array": picks,
                "title": "Simple first-arrival / refraction pick curve",
                "xlabel": "receiver x [m]",
                "ylabel": "first arrival time [ms]",
                "labels": {1: "first arrival"},
            }


    if show_apparent_velocity and appvel_x.size:
        plots["Apparent velocity from first arrivals"] = {
            "type": "line",
            "array": np.column_stack([appvel_x, appvel]),
            "title": "Local apparent velocity from first arrivals",
            "xlabel": "receiver x [m]",
            "ylabel": "apparent velocity [m/s]",
            "labels": {1: "v_app"},
        }

    if show_refraction_summary:
        finite_picks = first_arrival_ms[np.isfinite(first_arrival_ms)]
        finite_appvel = appvel[np.isfinite(appvel)] if appvel.size else np.asarray([])

        txt = [
            "Seismic refraction summary",
            "",
            f"Topography correction: active ({m['topo_source']})",
            f"Topography relief: {m['topo_relief']:.3f} m",
            f"Layers follow topography: {m.get('layers_follow_topography', False)}",
            f"Source frequency: {freq:g} Hz",
            f"Receiver count: {nrec}",
            f"Receiver spacing: {float(np.median(np.diff(rec_x))) if len(rec_x) > 1 else 0:.3g} m",
            f"Picked first arrivals: {len(finite_picks)} / {len(rec_x)}",
        ]

        if finite_picks.size:
            txt += [
                f"First-arrival min: {float(np.nanmin(finite_picks)):.3f} ms",
                f"First-arrival max: {float(np.nanmax(finite_picks)):.3f} ms",
            ]

        if finite_appvel.size:
            txt += [
                f"Apparent velocity median abs: {float(np.nanmedian(np.abs(finite_appvel))):.1f} m/s",
                f"Apparent velocity max abs: {float(np.nanmax(np.abs(finite_appvel))):.1f} m/s",
            ]

        txt += [
            "",
            "Use for fieldwork prediction:",
            "Shot gather = visual QC.",
            "First-arrival picks = main refraction evidence.",
            "Apparent velocity = lateral velocity disturbance check.",
            "Background-difference gather = anomaly visibility check.",
        ]

        plots["Seismic refraction summary"] = {
            "type": "text",
            "title": "Seismic refraction summary",
            "text": "\n".join(txt),
        }

    info = "\n".join(
        [
            "Seismic forward model completed",
            "Implementation: Devito acoustic finite-difference simulation with topography-following source/receivers.",
            f"topography_csv={scenario.get('files', {}).get('elevation_csv', '')}",
            f"topography_source={m['topo_source']}",
            f"topography_relief_m={m['topo_relief']:.6g}",
            f"layer_topography_mode={m['topography_mode']}",
            f"model_length_m={length:.6g}",
            f"geological_depth_m={m['depth']:.6g}",
            f"model_depth_with_topography_m={model_depth:.6g}",
            f"dx_m={dx:.6g}",
            f"dz_m={dz:.6g}",
            f"nx={vp.shape[0]}",
            f"nz={vp.shape[1]}",
            f"vp_min_m_s={float(np.nanmin(vp)):.6g}",
            f"vp_max_m_s={float(np.nanmax(vp)):.6g}",
            f"vp_air_m_s={m['vp_air']:.6g}",
            f"source_frequency_hz={freq:.6g}",
            f"recording_time_ms={tn_ms:.6g}",
            f"critical_dt_ms={model.critical_dt:.6g}",
            f"source_x_m={x_src:.6g}",
            f"source_z_abs_m={src_z_abs:.6g}",
            f"receivers={nrec}",
            f"receiver_depth_below_surface_m={z_rec:.6g}",
            f"free_surface={free_surface}",
            f"show_refraction_overlay={show_refraction_overlay}",
            f"show_apparent_velocity={show_apparent_velocity}",
            f"show_first_arrival_difference={show_first_arrival_difference}",
            f"show_refraction_summary={show_refraction_summary}",
            f"topography_mode={m.get('topography_mode', '')}",
            f"layers_follow_topography={m.get('layers_follow_topography', False)}",
            f"run_background_difference={run_background}",
            f"show_first_arrivals={show_first_arrivals}",
            f"save_wavefield_snapshot={save_wavefield_snapshot}",
            f"acquisition_mode={acquisition_mode}",
            f"nshots={nshots}",
            f"shot_spacing_m={shot_spacing}",
        ]
    )

    return {
        "model": vp,
        "data": data.T,
        "plots": plots,
        "info": info,
    }
