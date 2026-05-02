from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

try:
    from matplotlib.path import Path as MplPath
except Exception:
    MplPath = None

import devito as dv
from examples.seismic import TimeAxis, Receiver
from examples.seismic.source import PointSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
            if value.lower().endswith("hz"):
                return float(value[:-2].strip())
        return float(value)
    except Exception:
        return float(default)


def _survey_seismic(scenario: dict) -> dict:
    return scenario.get("survey", {}).get("seismic", {})


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
                "distance",
                "distanz",
                "x",
                "chainage",
                "profile_x",
                "dist",
                "abstand",
            ]
            z_candidates = [
                "altitude",
                "elevation",
                "height",
                "hoehe",
                "höhe",
                "z",
                "topography",
                "topo",
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

            x_unique, idx = np.unique(x, return_index=True)
            z_unique = z[idx]

            if len(x_unique) < 2:
                continue

            return x_unique, z_unique

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Could not parse seismic topography CSV: {p}. "
        "Expected columns like Distance/Altitude or x/z."
    ) from last_error


def _make_topography_function(scenario: dict, length: float):
    csv_path = scenario.get("files", {}).get("elevation_csv", "")
    csv_topo = _read_topography_csv(csv_path) if csv_path else None

    if csv_topo is not None:
        raw_x, raw_z = csv_topo
        raw_span = float(np.nanmax(raw_x) - np.nanmin(raw_x))

        if raw_span <= 0:
            raise ValueError("Topography CSV distance range is zero.")

        x_local = (raw_x - np.nanmin(raw_x)) / raw_span * length
        elev = raw_z - np.nanmin(raw_z)

        # Devito z is positive downward. Highest elevation becomes shallowest surface.
        relief = float(np.nanmax(elev) - np.nanmin(elev))
        topo_depth = relief - elev
        source = str(csv_path)

    else:
        topo = scenario.get("topography", [])

        if topo and len(topo) >= 2:
            pts = np.asarray(topo, dtype=float)
            x_local = pts[:, 0].copy()
            topo_depth = pts[:, 1].copy()

            if np.nanmin(x_local) < 0:
                x_local = x_local + 0.5 * length

            topo_depth = topo_depth - np.nanmin(topo_depth)
            relief = float(np.nanmax(topo_depth) - np.nanmin(topo_depth))
            source = "scenario.topography"
        else:
            x_local = np.asarray([0.0, length], dtype=float)
            topo_depth = np.asarray([0.0, 0.0], dtype=float)
            relief = 0.0
            source = "flat default"

    order = np.argsort(x_local)
    x_local = x_local[order]
    topo_depth = topo_depth[order]

    x_unique, idx = np.unique(x_local, return_index=True)
    topo_depth = topo_depth[idx]
    x_local = x_unique

    if PchipInterpolator is not None and len(x_local) >= 3:
        interp = PchipInterpolator(x_local, topo_depth, extrapolate=True)

        def topo_fn(x):
            xx = np.asarray(x, dtype=float)
            return np.asarray(interp(xx), dtype=float)

    else:
        def topo_fn(x):
            xx = np.asarray(x, dtype=float)
            return np.interp(xx, x_local, topo_depth)

    return topo_fn, x_local, topo_depth, relief, source


def _layers(scenario: dict):
    layers = scenario.get("layers", [])

    if not layers:
        layers = [
            {"thickness": 10.0, "vp": 1200.0, "vs": 500.0, "rho": 1800.0},
            {"thickness": 15.0, "vp": 2500.0, "vs": 1100.0, "rho": 2100.0},
            {"thickness": 999.0, "vp": 3500.0, "vs": 1800.0, "rho": 2300.0},
        ]

    clean = []

    for layer in layers:
        vp = _to_float(layer.get("vp", layer.get("velocity", 1500.0)), 1500.0)

        # If Vs is not given, use a simple isotropic approximation.
        # This is not a material calibration; it is a reasonable modelling default.
        vs = _to_float(layer.get("vs", layer.get("s_velocity", vp / math.sqrt(3.0))), vp / math.sqrt(3.0))

        rho = _to_float(layer.get("rho", layer.get("density", 2000.0)), 2000.0)
        grad_vp = _to_float(layer.get("gradient", layer.get("vp_gradient", 0.0)), 0.0)
        grad_vs = _to_float(layer.get("vs_gradient", 0.0), 0.0)

        clean.append(
            {
                "thickness": max(_to_float(layer.get("thickness", 999.0), 999.0), 0.0),
                "vp": max(vp, 300.0),
                "vs": max(vs, 100.0),
                "rho": max(rho, 500.0),
                "vp_gradient": grad_vp,
                "vs_gradient": grad_vs,
            }
        )

    return clean


def _build_elastic_model(scenario: dict, include_anomalies: bool = True):
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

    model_depth = depth + topo_relief

    nx = int(round(length / dx)) + 1
    nz = int(round(model_depth / dz)) + 1

    max_cells = nx * nz
    if max_cells > 750_000:
        raise ValueError(
            "Elastic seismic model is too large for interactive GUI use.\n"
            f"Grid would be {nx} x {nz} = {max_cells:,} cells.\n"
            "Increase dx/dz or reduce model length/depth."
        )

    x = np.linspace(0.0, length, nx)
    z = np.linspace(0.0, model_depth, nz)
    topo_grid = topo_fn(x)

    layers = _layers(scenario)
    cumulative = np.cumsum([layer["thickness"] for layer in layers])

    topography_mode = str(seis.get("topography_mode", "horizontal_interfaces")).strip().lower()
    layers_follow_topography = topography_mode in {
        "parallel_to_topography",
        "follow_topography",
        "layers_follow_topography",
    }

    min_geo_vp = min(layer["vp"] for layer in layers)
    min_geo_vs = min(layer["vs"] for layer in layers)

    vp_air = _to_float(seis.get("vp_air", 0.8 * min_geo_vp), 0.8 * min_geo_vp)
    vs_air = _to_float(seis.get("vs_air", max(0.5 * min_geo_vs, 150.0)), max(0.5 * min_geo_vs, 150.0))
    rho_air = _to_float(seis.get("rho_air", 1000.0), 1000.0)

    vp = np.empty((nx, nz), dtype=np.float32)
    vs = np.empty((nx, nz), dtype=np.float32)
    rho = np.empty((nx, nz), dtype=np.float32)

    for ix, xval in enumerate(x):
        surface = float(topo_grid[ix])

        for iz, zval in enumerate(z):
            if zval < surface:
                vp[ix, iz] = vp_air
                vs[ix, iz] = vs_air
                rho[ix, iz] = rho_air
                continue

            if layers_follow_topography:
                depth_for_layering = zval - surface
            else:
                depth_for_layering = zval

            layer_idx = len(layers) - 1
            for k, bottom in enumerate(cumulative):
                if depth_for_layering < bottom:
                    layer_idx = k
                    break

            layer_top = 0.0 if layer_idx == 0 else cumulative[layer_idx - 1]
            depth_in_layer = max(0.0, depth_for_layering - layer_top)
            layer = layers[layer_idx]

            vp[ix, iz] = layer["vp"] + layer["vp_gradient"] * depth_in_layer
            vs[ix, iz] = layer["vs"] + layer["vs_gradient"] * depth_in_layer
            rho[ix, iz] = layer["rho"]

    X, Z = np.meshgrid(x, z, indexing="ij")
    body_outlines = []

    if include_anomalies:
        anomalies = scenario.get("anomalies", [])

        for anomaly in anomalies:
            typ = str(anomaly.get("type", "circle")).lower().strip()

            avp = _to_float(anomaly.get("vp", anomaly.get("velocity", 1000.0)), 1000.0)
            avs = _to_float(anomaly.get("vs", avp / math.sqrt(3.0)), avp / math.sqrt(3.0))
            arho = _to_float(anomaly.get("rho", anomaly.get("density", 1800.0)), 1800.0)

            avp = max(avp, 300.0)
            avs = max(avs, 100.0)
            arho = max(arho, 500.0)

            if typ == "circle":
                cx, cz = anomaly.get("center", [0.0, -5.0])
                cx = _to_float(cx, 0.0) + 0.5 * length
                surface = float(topo_fn([cx])[0])
                zc = surface + abs(_to_float(cz, -5.0))
                r = max(_to_float(anomaly.get("radius", 2.0), 2.0), dx)

                mask = (X - cx) ** 2 + (Z - zc) ** 2 <= r**2

                theta = np.linspace(0.0, 2.0 * np.pi, 200)
                body_outlines.append(
                    np.column_stack([cx + r * np.cos(theta), zc + r * np.sin(theta)])
                )

            elif typ in {"polygon", "rectangle", "block"}:
                points = anomaly.get("points", [])

                if len(points) < 3:
                    continue

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
                        (X >= np.nanmin(px))
                        & (X <= np.nanmax(px))
                        & (Z >= np.nanmin(pz))
                        & (Z <= np.nanmax(pz))
                    )

                body_outlines.append(np.column_stack([px, pz]))

            else:
                continue

            vp[mask] = avp
            vs[mask] = avs
            rho[mask] = arho

    # Elastic parameters
    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    # Keep lambda numerically safe.
    lam = np.maximum(lam, 0.05 * mu)

    return {
        "x": x,
        "z": z,
        "vp": vp,
        "vs": vs,
        "rho": rho,
        "lam": lam.astype(np.float32),
        "mu": mu.astype(np.float32),
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
        "vp_air": vp_air,
        "vs_air": vs_air,
        "rho_air": rho_air,
        "layers": layers,
        "body_outlines": body_outlines,
        "layers_follow_topography": layers_follow_topography,
    }


def _make_damping(shape, nbl: int, damping_strength: float = 35.0, free_surface: bool = False):
    nx, nz = shape
    damp = np.zeros(shape, dtype=np.float32)

    if nbl <= 0:
        return damp

    for ix in range(nx):
        left = max(0, nbl - ix)
        right = max(0, ix - (nx - nbl - 1))
        dist_x = max(left, right)

        for iz in range(nz):
            top = 0 if free_surface else max(0, nbl - iz)
            bottom = max(0, iz - (nz - nbl - 1))
            dist_z = max(top, bottom)

            dist = max(dist_x, dist_z)

            if dist > 0:
                r = dist / float(nbl)
                damp[ix, iz] = damping_strength * r * r

    return damp


def _ricker_wavelet(time_s: np.ndarray, f0_hz: float):
    t0 = 1.0 / max(f0_hz, 1e-6)
    arg = math.pi * f0_hz * (time_s - t0)
    w = (1.0 - 2.0 * arg**2) * np.exp(-arg**2)
    scale = np.nanmax(np.abs(w))
    if np.isfinite(scale) and scale > 0:
        w = w / scale
    return w


def _safe_clip(data: np.ndarray, clip: float = 0.02):
    vmax = float(clip * np.nanmax(np.abs(data))) if data.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def _normalise_display(data: np.ndarray):
    scale = np.nanpercentile(np.abs(data), 99.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(data))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(data / scale, -1.0, 1.0)


def _first_arrival_curve(data: np.ndarray, time_ms: np.ndarray, rec_x: np.ndarray):
    if data.size == 0:
        return np.empty((0, 2))

    picks = []

    for j in range(data.shape[1]):
        tr = np.abs(data[:, j])
        maxamp = float(np.nanmax(tr))

        if not np.isfinite(maxamp) or maxamp <= 0:
            picks.append(np.nan)
            continue

        threshold = 0.05 * maxamp
        idx = np.where(tr >= threshold)[0]

        if len(idx) == 0:
            picks.append(np.nan)
        else:
            picks.append(float(time_ms[idx[0]]))

    picks = np.asarray(picks, dtype=float)
    return np.column_stack([rec_x, picks])


def _apparent_velocity_from_picks(picks: np.ndarray):
    if picks.size == 0:
        return np.empty((0, 2))

    x = picks[:, 0]
    t_s = picks[:, 1] / 1000.0

    mask = np.isfinite(x) & np.isfinite(t_s)
    x = x[mask]
    t_s = t_s[mask]

    if len(x) < 3:
        return np.empty((0, 2))

    dx = np.gradient(x)
    dt = np.gradient(t_s)

    with np.errstate(divide="ignore", invalid="ignore"):
        vapp = dx / dt

    # Remove obviously unstable slopes around turning points.
    vapp[~np.isfinite(vapp)] = np.nan
    vapp[np.abs(vapp) > 10000.0] = np.nan

    return np.column_stack([x, vapp])


def _run_elastic_once(m: dict, scenario: dict, source_x: float, rec_x: np.ndarray):
    seis = _survey_seismic(scenario)

    freq = _to_float(seis.get("source_frequency", seis.get("freq", 20.0)), 20.0)
    tn_ms = _to_float(seis.get("recording_time_ms", 1000.0), 1000.0)
    tn_s = tn_ms / 1000.0

    space_order = int(seis.get("space_order", 8))
    nbl = int(seis.get("nbl", 40))
    free_surface = bool(seis.get("free_surface", False))

    x = m["x"]
    dx = m["dx"]
    dz = m["dz"]
    length = m["length"]
    model_depth = m["model_depth"]
    topo_grid = m["topo_grid"]

    z_src = _to_float(seis.get("z_src", seis.get("source_depth", 2.0)), 2.0)
    z_rec = _to_float(seis.get("z_rec", seis.get("receiver_depth", 2.0)), 2.0)

    source_x = min(max(source_x, 0.0), length)
    src_surface = float(np.interp(source_x, x, topo_grid))
    src_z_abs = min(max(src_surface + z_src, dz), model_depth - dz)

    rec_x = np.clip(np.asarray(rec_x, dtype=float), 0.0, length)
    rec_surface = np.interp(rec_x, x, topo_grid)
    rec_z_abs = np.clip(rec_surface + z_rec, dz, model_depth - dz)

    vpmax = float(np.nanmax(m["vp"]))
    vsmax = float(np.nanmax(m["vs"]))
    vmax = max(vpmax, vsmax, 1.0)
    hmin = min(dx, dz)

    # Conservative elastic CFL.
    dt_s = 0.35 * hmin / (math.sqrt(2.0) * vmax)
    max_dt_s = _to_float(seis.get("max_dt_ms", 0.5), 0.5) / 1000.0
    dt_s = min(dt_s, max_dt_s)

    time_axis = TimeAxis(start=0.0, stop=tn_s, step=dt_s)
    time_s = np.asarray(time_axis.time_values, dtype=float)
    time_ms = time_s * 1000.0

    grid = dv.Grid(
        shape=m["vp"].shape,
        extent=(length, model_depth),
        origin=(0.0, 0.0),
    )

    rho = dv.Function(name="rho", grid=grid, space_order=space_order)
    lam = dv.Function(name="lam", grid=grid, space_order=space_order)
    mu = dv.Function(name="mu", grid=grid, space_order=space_order)
    buoy = dv.Function(name="buoy", grid=grid, space_order=space_order)
    damp = dv.Function(name="damp", grid=grid, space_order=space_order)

    rho.data[:] = m["rho"]
    lam.data[:] = m["lam"]
    mu.data[:] = m["mu"]
    buoy.data[:] = 1.0 / np.maximum(m["rho"], 1.0)
    damp.data[:] = _make_damping(m["vp"].shape, nbl=nbl, free_surface=free_surface)

    vx = dv.TimeFunction(name="vx", grid=grid, time_order=1, space_order=space_order)
    vz = dv.TimeFunction(name="vz", grid=grid, time_order=1, space_order=space_order)

    sxx = dv.TimeFunction(name="sxx", grid=grid, time_order=1, space_order=space_order)
    szz = dv.TimeFunction(name="szz", grid=grid, time_order=1, space_order=space_order)
    sxz = dv.TimeFunction(name="sxz", grid=grid, time_order=1, space_order=space_order)

    eq_vx = dv.Eq(vx.forward, vx + dt_s * (buoy * (sxx.dx + sxz.dy) - damp * vx))
    eq_vz = dv.Eq(vz.forward, vz + dt_s * (buoy * (sxz.dx + szz.dy) - damp * vz))

    eq_sxx = dv.Eq(
        sxx.forward,
        sxx + dt_s * ((lam + 2.0 * mu) * vx.forward.dx + lam * vz.forward.dy - damp * sxx),
    )
    eq_szz = dv.Eq(
        szz.forward,
        szz + dt_s * (lam * vx.forward.dx + (lam + 2.0 * mu) * vz.forward.dy - damp * szz),
    )
    eq_sxz = dv.Eq(
        sxz.forward,
        sxz + dt_s * (mu * (vx.forward.dy + vz.forward.dx) - damp * sxz),
    )

    src = PointSource(name="src", grid=grid, time_range=time_axis, npoint=1)
    src.coordinates.data[0, 0] = source_x
    src.coordinates.data[0, 1] = src_z_abs

    wavelet = _ricker_wavelet(time_s, freq)
    stress_scale = _to_float(seis.get("elastic_source_scale", 1.0e6), 1.0e6)
    src.data[:, 0] = stress_scale * wavelet

    rec_vx = Receiver(name="rec_vx", grid=grid, npoint=len(rec_x), time_range=time_axis)
    rec_vz = Receiver(name="rec_vz", grid=grid, npoint=len(rec_x), time_range=time_axis)

    rec_vx.coordinates.data[:, 0] = rec_x
    rec_vx.coordinates.data[:, 1] = rec_z_abs

    rec_vz.coordinates.data[:, 0] = rec_x
    rec_vz.coordinates.data[:, 1] = rec_z_abs

    src_terms = []
    src_terms += src.inject(field=sxx.forward, expr=src * dt_s)
    src_terms += src.inject(field=szz.forward, expr=src * dt_s)

    rec_terms = []
    rec_terms += rec_vx.interpolate(expr=vx.forward)
    rec_terms += rec_vz.interpolate(expr=vz.forward)

    op = dv.Operator(
        [eq_vx, eq_vz, eq_sxx, eq_szz, eq_sxz] + src_terms + rec_terms,
        subs=grid.spacing_map,
    )

    op(time=time_axis.num - 2, dt=dt_s)

    return {
        "vx": np.asarray(rec_vx.data, dtype=float),
        "vz": np.asarray(rec_vz.data, dtype=float),
        "time_s": time_s,
        "time_ms": time_ms,
        "wavelet": wavelet,
        "rec_x": rec_x,
        "rec_z_abs": rec_z_abs,
        "source_x": source_x,
        "source_z_abs": src_z_abs,
        "dt_s": dt_s,
        "dt_ms": dt_s * 1000.0,
        "tn_ms": tn_ms,
        "freq": freq,
    }




# ============================================================
# Robust first-arrival / refraction QC helpers
# ============================================================

def _moving_average_1d(y, n=5):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if y.size < n or n <= 1:
        return y
    kernel = np.ones(n, dtype=float) / float(n)
    out = np.convolve(y, kernel, mode="same")
    out[: n // 2] = y[: n // 2]
    out[-n // 2 :] = y[-n // 2 :]
    return out


def _first_arrivals_envelope(data, time_ms, rec_x, source_x=None, threshold=0.04, mute_ms=5.0, near_source_m=3.0):
    """
    Envelope/onset-based first-arrival picker.
    Returns Nx2 array: receiver_x, first_arrival_ms.
    More stable than raw amplitude threshold picking.
    """
    import numpy as np

    data = np.asarray(data, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    rec_x = np.asarray(rec_x, dtype=float)

    if data.ndim != 2 or data.size == 0:
        return np.empty((0, 2), dtype=float)

    # data may be nt x nrec
    if data.shape[0] != time_ms.size and data.shape[1] == time_ms.size:
        data = data.T

    nt, nrec = data.shape
    picks = np.full(nrec, np.nan, dtype=float)

    dt = float(np.nanmedian(np.diff(time_ms))) if time_ms.size > 1 else 1.0
    mute_n = max(0, int(round(mute_ms / max(dt, 1e-9))))

    # simple envelope proxy: abs + short smoothing
    env = np.abs(data)
    win = max(3, int(round(4.0 / max(dt, 1e-9))))
    if win % 2 == 0:
        win += 1

    for ir in range(nrec):
        tr = env[:, ir].astype(float)
        if not np.isfinite(tr).any() or np.nanmax(tr) <= 0:
            continue

        tr = _moving_average_1d(tr, win)
        tr[:mute_n] = 0.0

        if source_x is not None and abs(float(rec_x[ir]) - float(source_x)) < near_source_m:
            continue

        amp = np.nanmax(tr)
        if amp <= 0:
            continue

        idx = np.where(tr >= threshold * amp)[0]
        if idx.size:
            picks[ir] = time_ms[int(idx[0])]

    # split left/right branches and smooth separately
    if source_x is not None:
        left = rec_x < source_x
        right = rec_x > source_x
        for mask in (left, right):
            good = mask & np.isfinite(picks)
            if np.count_nonzero(good) >= 5:
                picks[good] = _moving_average_1d(picks[good], 5)
    else:
        good = np.isfinite(picks)
        if np.count_nonzero(good) >= 5:
            picks[good] = _moving_average_1d(picks[good], 5)

    good = np.isfinite(picks)
    return np.column_stack([rec_x[good], picks[good]])


def _split_apparent_velocity_from_picks(picks, source_x=None):
    """
    Apparent velocity from first arrivals, split around source position.
    Avoids misleading negative velocities across the source.
    """
    import numpy as np

    picks = np.asarray(picks, dtype=float)
    if picks.ndim != 2 or picks.shape[0] < 4:
        return np.empty((0, 2), dtype=float)

    x = picks[:, 0]
    t = picks[:, 1] / 1000.0

    out_x = []
    out_v = []

    masks = []
    if source_x is None:
        masks = [np.ones_like(x, dtype=bool)]
    else:
        masks = [x < source_x, x > source_x]

    for mask in masks:
        xx = x[mask]
        tt = t[mask]
        if xx.size < 4:
            continue

        order = np.argsort(xx)
        xx = xx[order]
        tt = tt[order]

        dx = np.gradient(xx)
        dt = np.gradient(tt)

        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.abs(dx / dt)

        good = np.isfinite(v) & (v > 100.0) & (v < 10000.0)
        out_x.extend(xx[good])
        out_v.extend(v[good])

    if not out_x:
        return np.empty((0, 2), dtype=float)

    return np.column_stack([np.asarray(out_x), np.asarray(out_v)])


def _normalised_difference(full, background):
    import numpy as np
    diff = np.asarray(full, dtype=float) - np.asarray(background, dtype=float)
    scale = np.nanmax(np.abs(diff))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return diff / scale


def _run_seismic_elastic_core(scenario: dict) -> dict:
    _progress(scenario, "Preparing elastic seismic model with topography correction...")

    seis = _survey_seismic(scenario)

    m_full = _build_elastic_model(scenario, include_anomalies=True)

    x = m_full["x"]
    length = m_full["length"]
    model_depth = m_full["model_depth"]

    receiver_spacing = _to_float(seis.get("receiver_spacing", 2.0), 2.0)
    receiver_spacing = max(receiver_spacing, m_full["dx"])

    nrec_default = int(round(length / receiver_spacing)) + 1
    nrec = int(seis.get("nrec", seis.get("receivers", nrec_default)))
    nrec = max(2, min(nrec, 512))

    x_rec_start = _to_float(seis.get("x_rec_start", 0.0), 0.0)
    x_rec_end = _to_float(seis.get("x_rec_end", length), length)

    rec_x_input = np.linspace(x_rec_start, x_rec_end, nrec)
    rec_x_input = np.clip(rec_x_input, 0.0, length)

    acquisition_mode = str(seis.get("acquisition_mode", "single_shot")).strip().lower()
    nshots = int(seis.get("nshots", 1))
    shot_spacing = _to_float(seis.get("shot_spacing", 5.0), 5.0)

    if acquisition_mode in {"multi_shot", "cmp"}:
        nshots = max(1, min(nshots, 25))
    else:
        nshots = 1

    centre_x = _to_float(seis.get("x_src", 0.5 * length), 0.5 * length)

    if acquisition_mode in {"multi_shot", "cmp"}:
        start_x = centre_x - 0.5 * (nshots - 1) * shot_spacing
        shot_xs = start_x + np.arange(nshots) * shot_spacing
        shot_xs = np.clip(shot_xs, 0.0, length)
    else:
        shot_xs = np.asarray([centre_x], dtype=float)

    _progress(
        scenario,
        (
            "Elastic seismic model grid:\n"
            f"  nx={m_full['vp'].shape[0]}, nz={m_full['vp'].shape[1]}\n"
            f"  dx={m_full['dx']:g} m, dz={m_full['dz']:g} m\n"
            f"  model_length={length:g} m\n"
            f"  model_depth_with_topography={model_depth:g} m\n"
            f"  topography_source={m_full['topo_source']}\n"
            f"  topography_relief={m_full['topo_relief']:.3f} m\n"
            f"  layers_follow_topography={m_full['layers_follow_topography']}\n"
            f"  acquisition_mode={acquisition_mode}\n"
            f"  nshots={len(shot_xs)}"
        ),
    )

    full_runs = []

    for i, sx in enumerate(shot_xs, start=1):
        _progress(scenario, f"Running elastic shot {i}/{len(shot_xs)} at x={sx:.3f} m...")
        full_runs.append(_run_elastic_once(m_full, scenario, source_x=float(sx), rec_x=rec_x_input))

    vz_stack = np.mean([r["vz"] for r in full_runs], axis=0)
    vx_stack = np.mean([r["vx"] for r in full_runs], axis=0)
    first = full_runs[0]

    time_ms = first["time_ms"]
    rec_x = first["rec_x"]

    run_background = bool(seis.get("run_background_difference", True))
    bg_vz_stack = None
    bg_vx_stack = None
    diff_vz = None
    diff_vx = None

    if run_background and scenario.get("anomalies", []):
        _progress(scenario, "Running elastic background model without anomaly...")

        m_bg = _build_elastic_model(scenario, include_anomalies=False)
        bg_runs = []

        for i, sx in enumerate(shot_xs, start=1):
            _progress(scenario, f"Running elastic background shot {i}/{len(shot_xs)} at x={sx:.3f} m...")
            bg_runs.append(_run_elastic_once(m_bg, scenario, source_x=float(sx), rec_x=rec_x_input))

        bg_vz_stack = np.mean([r["vz"] for r in bg_runs], axis=0)
        bg_vx_stack = np.mean([r["vx"] for r in bg_runs], axis=0)

        nt = min(vz_stack.shape[0], bg_vz_stack.shape[0])
        nxr = min(vz_stack.shape[1], bg_vz_stack.shape[1])

        vz_stack = vz_stack[:nt, :nxr]
        vx_stack = vx_stack[:nt, :nxr]
        bg_vz_stack = bg_vz_stack[:nt, :nxr]
        bg_vx_stack = bg_vx_stack[:nt, :nxr]
        time_ms = time_ms[:nt]
        rec_x = rec_x[:nxr]

        diff_vz = vz_stack - bg_vz_stack
        diff_vx = vx_stack - bg_vx_stack

    extent_model = [0.0, length, model_depth, 0.0]
    extent_data = [float(rec_x[0]), float(rec_x[-1]), float(time_ms[-1]), 0.0]

    vz_vmin, vz_vmax = _safe_clip(vz_stack, clip=0.02)
    vx_vmin, vx_vmax = _safe_clip(vx_stack, clip=0.02)

    show_first_arrivals = bool(seis.get("show_first_arrivals", True))


    show_stacked_anomaly_response = bool(seis.get("show_stacked_anomaly_response", True))
    show_apparent_velocity = bool(seis.get("show_apparent_velocity", False))

    first_arrival_picks = _first_arrivals_envelope(
        vz_stack,
        time_ms,
        rec_x,
        source_x=float(first["source_x"]),
    )
    apparent_velocity = _split_apparent_velocity_from_picks(
        first_arrival_picks,
        source_x=float(first["source_x"]),
    )

    plots = {
        "Vp model": {
            "type": "image",
            "array": m_full["vp"].T,
            "extent": extent_model,
            "origin": "upper",
            "title": "Elastic seismic Vp model with topography",
            "xlabel": "x [m]",
            "ylabel": "z [m]",
            "clabel": "Vp [m/s]",
            "colorbar": True,
        },
        "Vs model": {
            "type": "image",
            "array": m_full["vs"].T,
            "extent": extent_model,
            "origin": "upper",
            "title": "Elastic seismic Vs model with topography",
            "xlabel": "x [m]",
            "ylabel": "z [m]",
            "clabel": "Vs [m/s]",
            "colorbar": True,
        },
        "Vertical-component shot gather": {
            "type": "image",
            "array": vz_stack,
            "extent": extent_data,
            "origin": "upper",
            "title": "Elastic seismic gather, vertical particle velocity Vz",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "Vz amplitude",
            "colorbar": True,
            "vmin": vz_vmin,
            "vmax": vz_vmax,
        },
        "Horizontal-component shot gather": {
            "type": "image",
            "array": vx_stack,
            "extent": extent_data,
            "origin": "upper",
            "title": "Elastic seismic gather, horizontal particle velocity Vx",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "Vx amplitude",
            "colorbar": True,
            "vmin": vx_vmin,
            "vmax": vx_vmax,
        },
        "Source wavelet": {
            "type": "line",
            "array": np.column_stack([first["time_ms"], first["wavelet"]]),
            "title": f"Elastic Ricker stress source, f0={first['freq']:g} Hz",
            "xlabel": "time [ms]",
            "ylabel": "normalised amplitude",
            "labels": {1: "Ricker"},
        },
        "Topography profile": {
            "type": "line",
            "array": np.column_stack([x, m_full["topo_grid"]]),
            "title": "Elastic seismic topography converted to Devito z-depth",
            "xlabel": "x [m]",
            "ylabel": "surface z [m, positive downward]",
            "labels": {1: "surface"},
        },
    }

    if show_first_arrivals and first_arrival_picks.size:
        plots["First-arrival picks"] = {
            "type": "line",
            "array": first_arrival_picks,
            "title": "Elastic first-arrival / refraction picks from Vz gather",
            "xlabel": "receiver x [m]",
            "ylabel": "first arrival time [ms]",
            "labels": {1: "first arrival"},
        }

    if show_apparent_velocity and apparent_velocity.size:
        plots["Apparent velocity"] = {
            "type": "line",
            "array": apparent_velocity,
            "title": "Elastic apparent velocity from first arrivals",
            "xlabel": "receiver x [m]",
            "ylabel": "apparent velocity [m/s]",
            "labels": {1: "v_app"},
        }

    if diff_vz is not None:
        plots["Elastic anomaly-only Vz difference"] = {
            "type": "image",
            "array": _normalise_display(diff_vz),
            "extent": extent_data,
            "origin": "upper",
            "title": "Elastic anomaly-only Vz response: full model minus background",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "normalised difference",
            "colorbar": True,
            "vmin": -1.0,
            "vmax": 1.0,
        }

        plots["Elastic anomaly-only Vx difference"] = {
            "type": "image",
            "array": _normalise_display(diff_vx),
            "extent": extent_data,
            "origin": "upper",
            "title": "Elastic anomaly-only Vx response: full model minus background",
            "xlabel": "receiver x [m]",
            "ylabel": "time [ms]",
            "clabel": "normalised difference",
            "colorbar": True,
            "vmin": -1.0,
            "vmax": 1.0,
        }

        rms_full = float(np.sqrt(np.nanmean(vz_stack ** 2)))
        rms_diff = float(np.sqrt(np.nanmean(diff_vz ** 2)))
        ratio = rms_diff / max(rms_full, 1e-30)

        plots["Elastic detectability metric"] = {
            "type": "text",
            "title": "Elastic seismic detectability metric",
            "text": (
                "Elastic anomaly detectability metric\n\n"
                f"RMS(full Vz gather) = {rms_full:.6g}\n"
                f"RMS(full - background Vz) = {rms_diff:.6g}\n"
                f"difference/full ratio = {ratio:.6g}\n\n"
                "Interpretation:\n"
                "< 0.02  : very weak / likely hidden\n"
                "0.02-0.10: weak but possibly visible after processing\n"
                "> 0.10  : likely visible\n\n"
                "Use the anomaly-only Vz/Vx difference gathers to judge whether the target produces a measurable elastic response."
            ),
        }

    info_lines = [
        "Elastic seismic forward model completed",
        "Implementation: isotropic elastic velocity-stress finite-difference simulation in Devito.",
        "Purpose: P-wave, S-wave, converted-wave, and surface-wave style feasibility assessment.",
        f"topography_csv={scenario.get('files', {}).get('elevation_csv', '')}",
        f"topography_source={m_full['topo_source']}",
        f"topography_relief_m={m_full['topo_relief']:.6g}",
        f"layers_follow_topography={m_full['layers_follow_topography']}",
        f"model_length_m={length:.6g}",
        f"geological_depth_m={m_full['depth']:.6g}",
        f"model_depth_with_topography_m={model_depth:.6g}",
        f"dx_m={m_full['dx']:.6g}",
        f"dz_m={m_full['dz']:.6g}",
        f"nx={m_full['vp'].shape[0]}",
        f"nz={m_full['vp'].shape[1]}",
        f"vp_min_m_s={float(np.nanmin(m_full['vp'])):.6g}",
        f"vp_max_m_s={float(np.nanmax(m_full['vp'])):.6g}",
        f"vs_min_m_s={float(np.nanmin(m_full['vs'])):.6g}",
        f"vs_max_m_s={float(np.nanmax(m_full['vs'])):.6g}",
        f"rho_min_kg_m3={float(np.nanmin(m_full['rho'])):.6g}",
        f"rho_max_kg_m3={float(np.nanmax(m_full['rho'])):.6g}",
        f"source_frequency_hz={first['freq']:.6g}",
        f"recording_time_ms={first['tn_ms']:.6g}",
        f"dt_ms={first['dt_ms']:.6g}",
        f"source_x_m={first['source_x']:.6g}",
        f"source_z_abs_m={first['source_z_abs']:.6g}",
        f"receivers={len(rec_x)}",
        f"acquisition_mode={acquisition_mode}",
        f"nshots={len(shot_xs)}",
        f"shot_spacing_m={shot_spacing:.6g}",
        f"run_background_difference={run_background}",
        f"first_arrival_picks={len(first_arrival_picks)}",
    ]


    if show_stacked_anomaly_response:
        stack = None
        extent = None

        for key, plot in plots.items():
            key_low = str(key).lower()
            if "anomaly-only" in key_low and ("vz" in key_low or "vx" in key_low):
                arr = np.abs(np.asarray(plot.get("array"), dtype=float))
                if arr.size:
                    if stack is None:
                        stack = arr.copy()
                        extent = plot.get("extent")
                    elif arr.shape == stack.shape:
                        stack = np.sqrt(stack**2 + arr**2)

        if stack is not None and stack.size:
            stack_max = float(np.nanmax(stack))
            if np.isfinite(stack_max) and stack_max > 0.0:
                stack = stack / stack_max

            plots["Stacked absolute anomaly response"] = {
                "type": "image",
                "array": stack,
                "extent": extent,
                "origin": "upper",
                "title": "Stacked absolute elastic anomaly response |Vz,Vx|",
                "xlabel": "receiver x [m]",
                "ylabel": "time [ms]",
                "clabel": "normalised |anomaly response|",
                "colorbar": True,
                "vmin": 0.0,
                "vmax": 1.0,
            }


    return {
        "model": m_full["vp"],
        "data": vz_stack,
        "plots": plots,
        "info": "\n".join(info_lines),
    }


def run_seismic(scenario: dict) -> dict:
    return run_seismic_elastic(scenario)



# ============================================================
# Display/QC post-processing wrapper
# ============================================================

def _elastic_agc(data, window_samples):
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 2 or window_samples <= 1:
        return arr

    window_samples = int(max(3, window_samples))
    pad = window_samples // 2
    out = np.zeros_like(arr, dtype=float)

    for j in range(arr.shape[1]):
        tr = arr[:, j]
        sq = tr * tr
        csum = np.cumsum(np.pad(sq, (pad, pad), mode="edge"))
        rms = np.sqrt((csum[window_samples:] - csum[:-window_samples]) / float(window_samples))
        if rms.size != tr.size:
            rms = np.interp(
                np.arange(tr.size),
                np.linspace(0, tr.size - 1, rms.size),
                rms,
            )
        scale = np.nanmedian(rms[rms > 0]) if np.any(rms > 0) else 1.0
        out[:, j] = tr / np.maximum(rms, scale * 1e-3)

    return out


def _elastic_direct_mute(data, extent, source_x, mute_velocity, mute_padding_ms):
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 2 or not extent or len(extent) < 4:
        return arr

    x0, x1, tmax, tmin = map(float, extent[:4])
    nt, nx = arr.shape
    xs = np.linspace(x0, x1, nx)
    ts = np.linspace(tmin, tmax, nt)

    mute_velocity = max(float(mute_velocity), 1.0)
    mute_padding_ms = max(float(mute_padding_ms), 0.0)

    for ix, x in enumerate(xs):
        t_mute = 1000.0 * abs(float(x) - float(source_x)) / mute_velocity + mute_padding_ms
        arr[ts <= t_mute, ix] = 0.0

    return arr


def _elastic_clip_limits(arr, percentile, signed=True):
    a = np.asarray(arr, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None, None

    percentile = float(percentile)
    percentile = min(max(percentile, 90.0), 100.0)

    if signed:
        vmax = np.nanpercentile(np.abs(finite), percentile)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(finite))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return -float(vmax), float(vmax)

    vmax = np.nanpercentile(finite, percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(finite)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return 0.0, float(vmax)


def _postprocess_elastic_display(result: dict, scenario: dict) -> dict:
    seis = scenario.get("survey", {}).get("seismic", {})

    apply_agc = bool(seis.get("display_agc", True))
    agc_window_ms = float(seis.get("display_agc_window_ms", 80.0))
    mute_direct = bool(seis.get("display_mute_direct", True))
    mute_velocity = float(seis.get("display_mute_velocity", 900.0))
    mute_padding_ms = float(seis.get("display_mute_padding_ms", 40.0))
    clip_percentile = float(seis.get("display_clip_percentile", 99.0))

    source_x = float(seis.get("x_src", scenario.get("domain", {}).get("length", 0.0) * 0.5))

    plots = result.get("plots", {})
    changed = []

    for name, plot in list(plots.items()):
        if not isinstance(plot, dict):
            continue
        if plot.get("type") != "image":
            continue

        title = str(plot.get("title", name)).lower()

        # Do not process physical property models.
        if "vp model" in title or "vs model" in title or "velocity model" in title:
            continue

        # Process gathers and anomaly-response maps only.
        if not any(k in title for k in ["gather", "anomaly", "response"]):
            continue

        arr = np.asarray(plot.get("array"), dtype=float)
        if arr.ndim != 2:
            continue

        extent = plot.get("extent", None)
        signed = "absolute" not in title and "stacked" not in title

        arr2 = arr.copy()

        if mute_direct:
            arr2 = _elastic_direct_mute(
                arr2,
                extent,
                source_x=source_x,
                mute_velocity=mute_velocity,
                mute_padding_ms=mute_padding_ms,
            )

        if apply_agc:
            if extent and len(extent) >= 4:
                tmax = abs(float(extent[2]) - float(extent[3]))
                dt_ms = tmax / max(arr2.shape[0] - 1, 1)
            else:
                dt_ms = 1.0
            win = int(round(agc_window_ms / max(dt_ms, 1e-9)))
            arr2 = _elastic_agc(arr2, win)

        if "absolute" in title or "stacked" in title:
            arr_abs = np.abs(arr2)
            mx = np.nanmax(arr_abs) if arr_abs.size else 1.0
            if np.isfinite(mx) and mx > 0:
                arr2 = arr_abs / mx
            plot["vmin"], plot["vmax"] = 0.0, 1.0
        else:
            vmin, vmax = _elastic_clip_limits(arr2, clip_percentile, signed=signed)
            if vmin is not None:
                plot["vmin"], plot["vmax"] = vmin, vmax

        plot["array"] = arr2
        plot["title"] = str(plot.get("title", name)) + " [display processed]"
        changed.append(name)

    extra = [
        "",
        "Elastic display/QC processing:",
        f"display_agc={apply_agc}",
        f"display_agc_window_ms={agc_window_ms:g}",
        f"display_mute_direct={mute_direct}",
        f"display_mute_velocity_m_s={mute_velocity:g}",
        f"display_mute_padding_ms={mute_padding_ms:g}",
        f"display_clip_percentile={clip_percentile:g}",
        "display_processed_plots=" + ", ".join(changed) if changed else "display_processed_plots=none",
    ]

    result["info"] = str(result.get("info", "")).rstrip() + "\n" + "\n".join(extra)
    return result


def run_seismic_elastic(scenario: dict) -> dict:
    result = _run_seismic_elastic_core(scenario)
    return _postprocess_elastic_display(result, scenario)
