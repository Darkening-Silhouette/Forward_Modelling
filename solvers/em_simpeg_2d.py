
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _progress(scenario: dict, text: str):
    cb = scenario.get("_progress_callback")
    if cb is not None:
        cb(str(text))


def _to_float(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _domain(scenario):
    domain = scenario.get("domain", {})
    em = scenario.get("survey", {}).get("em", {})

    length = max(_to_float(domain.get("length", 80.0), 80.0), 2.0)
    depth = max(_to_float(domain.get("depth", 40.0), 40.0), 1.0)

    # IMPORTANT:
    # Do not inherit fine ERT/GPR dx=0.5 for SimPEG EM 2D/3D.
    # EM finite-volume modelling is much heavier, so use coarse defaults
    # unless explicit EM2D values are given in YAML.
    dx = max(_to_float(em.get("em2d_dx", 4.0), 4.0), 1.0)
    dy = max(_to_float(em.get("em2d_dy", 4.0), 4.0), 1.0)
    dz = max(_to_float(em.get("em2d_dz", 4.0), 4.0), 1.0)

    return length, depth, dx, dy, dz


def _layer_sigma(layer):
    if "conductivity" in layer:
        return max(_to_float(layer.get("conductivity"), 0.01), 1e-8)

    rho = _to_float(layer.get("resistivity", 100.0), 100.0)
    return max(1.0 / max(rho, 1e-12), 1e-8)


def _layers(scenario):
    layers = scenario.get("layers", [])
    if layers:
        return layers

    return [
        {"thickness": 2.0, "conductivity": 0.01},
        {"thickness": 999.0, "conductivity": 0.02},
    ]


def _read_topography_csv(path: str | None):
    if not path:
        return None

    p = Path(path).expanduser()
    if not p.exists():
        return None

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
            lower = {n.lower().strip(): n for n in names}

            x_col = next(
                (lower[k] for k in ["distance", "distanz", "x", "chainage", "dist", "abstand"] if k in lower),
                None,
            )
            z_col = next(
                (lower[k] for k in ["altitude", "elevation", "height", "hoehe", "höhe", "z"] if k in lower),
                None,
            )

            if x_col is None or z_col is None:
                continue

            x = np.asarray(arr[x_col], dtype=float)
            z = np.asarray(arr[z_col], dtype=float)

            mask = np.isfinite(x) & np.isfinite(z)
            x = x[mask]
            z = z[mask]

            if len(x) < 2:
                continue

            order = np.argsort(x)
            x = x[order]
            z = z[order]

            xu, idx = np.unique(x, return_index=True)
            zu = z[idx]

            if len(xu) < 2:
                continue

            return xu, zu

        except Exception:
            continue

    return None


def _topography_function(scenario, length):
    csv_path = scenario.get("files", {}).get("elevation_csv", "")
    csv = _read_topography_csv(csv_path)

    if csv is not None:
        raw_x, raw_z = csv
        span = float(np.nanmax(raw_x) - np.nanmin(raw_x))
        if span <= 0:
            return lambda x: np.zeros_like(np.asarray(x, dtype=float)), "flat fallback", 0.0

        x_local = (raw_x - np.nanmin(raw_x)) / span * length - 0.5 * length
        z_up = raw_z - np.nanmin(raw_z)
        relief = float(np.nanmax(z_up) - np.nanmin(z_up))

        def surface_z_up(x):
            return np.interp(np.asarray(x, dtype=float), x_local, z_up)

        return surface_z_up, str(csv_path), relief

    topo = scenario.get("topography", [])
    if topo and len(topo) >= 2:
        pts = np.asarray(topo, dtype=float)
        x = pts[:, 0]
        z = pts[:, 1]

        order = np.argsort(x)
        x = x[order]
        z = z[order]
        z = z - np.nanmin(z)

        if np.nanmin(x) >= -1e-9 and np.nanmax(x) <= length + 1e-9:
            x = x - 0.5 * length

        relief = float(np.nanmax(z) - np.nanmin(z))

        def surface_z_up(xq):
            return np.interp(np.asarray(xq, dtype=float), x, z)

        return surface_z_up, "scenario.topography", relief

    return lambda x: np.zeros_like(np.asarray(x, dtype=float)), "flat default", 0.0


def _anomaly_sigma(anomaly, default=1e-6):
    if "conductivity" in anomaly:
        return max(_to_float(anomaly.get("conductivity"), default), 1e-8)
    if "resistivity" in anomaly:
        rho = _to_float(anomaly.get("resistivity"), 1e6)
        return max(1.0 / max(rho, 1e-12), 1e-8)
    return default


def _apply_anomalies_to_sigma(sigma, x, y, z, scenario, surface_fn):
    """
    Apply GUI anomalies to the EM 2D/3D conductivity model.

    Coordinate convention:
      GUI x     : centred profile coordinate [m]
      GUI z/depth: negative in polygon points, positive in target_specs custom_depth
      EM mask   : depth-positive-down below local topographic surface

    The anomaly is extruded in y, i.e. a 2.5D approximation. This is intentional:
    the GUI target is a 2D cross-section, while SimPEG needs a 3D volume.
    """
    anomalies = list(scenario.get("anomalies", []) or [])
    target_specs = list(scenario.get("target_specs", []) or [])

    try:
        from matplotlib.path import Path as MplPath
    except Exception:
        MplPath = None

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    depth = surface_fn(X) - Z  # positive downward below local surface

    dx_cell = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dz_cell = float(np.median(np.diff(z))) if len(z) > 1 else 1.0

    outlines = []
    total_cells = 0

    def apply_mask(mask, sig):
        nonlocal sigma, total_cells
        n = int(np.count_nonzero(mask))
        if n > 0:
            sigma[mask] = sig
            total_cells += n
        return n

    def padded_box_mask(xmin, xmax, dmin, dmax, pad_cells=1.5):
        pad_x = pad_cells * abs(dx_cell)
        pad_z = pad_cells * abs(dz_cell)
        return (
            (X >= xmin - pad_x) & (X <= xmax + pad_x) &
            (depth >= dmin - pad_z) & (depth <= dmax + pad_z)
        )

    # --------------------------------------------------------
    # A) Apply explicit anomaly polygons/circles from scenario
    # --------------------------------------------------------
    for anomaly in anomalies:
        typ = str(anomaly.get("type", "polygon")).strip().lower()
        sig = _anomaly_sigma(anomaly, 1e-6)
        mask = None

        if typ == "circle":
            cx, cz = anomaly.get("center", [0.0, -5.0])
            cx = _to_float(cx, 0.0)
            cd = abs(_to_float(cz, -5.0))
            r = max(_to_float(anomaly.get("radius", 2.0), 2.0), 0.25)

            mask = ((X - cx) ** 2 + (depth - cd) ** 2) <= r ** 2

            if np.count_nonzero(mask) == 0:
                mask = padded_box_mask(cx - r, cx + r, cd - r, cd + r)

            theta = np.linspace(0, 2 * np.pi, 200)
            outlines.append(np.column_stack([cx + r * np.cos(theta), cd + r * np.sin(theta)]))

        elif typ in {"ellipse", "elliptical"}:
            cx, cz = anomaly.get("center", [0.0, -5.0])
            cx = _to_float(cx, 0.0)
            cd = abs(_to_float(cz, -5.0))
            rx = max(_to_float(anomaly.get("width", 4.0), 4.0) * 0.5, 0.25)
            rz = max(_to_float(anomaly.get("height", 2.0), 2.0) * 0.5, 0.25)

            mask = ((X - cx) / rx) ** 2 + ((depth - cd) / rz) ** 2 <= 1.0

            if np.count_nonzero(mask) == 0:
                mask = padded_box_mask(cx - rx, cx + rx, cd - rz, cd + rz)

            theta = np.linspace(0, 2 * np.pi, 200)
            outlines.append(np.column_stack([cx + rx * np.cos(theta), cd + rz * np.sin(theta)]))

        elif typ in {"polygon", "rectangle", "block"}:
            pts = anomaly.get("points", [])
            if len(pts) < 3:
                continue

            px = np.asarray([_to_float(p[0], 0.0) for p in pts], dtype=float)
            pd = np.asarray([abs(_to_float(p[1], 0.0)) for p in pts], dtype=float)

            if MplPath is not None:
                path = MplPath(np.column_stack([px, pd]))
                test = np.column_stack([X.ravel(), depth.ravel()])
                mask = path.contains_points(test).reshape(X.shape)
            else:
                mask = padded_box_mask(float(np.nanmin(px)), float(np.nanmax(px)),
                                       float(np.nanmin(pd)), float(np.nanmax(pd)))

            # Coarse meshes can miss a small polygon completely; fallback to padded bbox.
            if np.count_nonzero(mask) == 0:
                mask = padded_box_mask(float(np.nanmin(px)), float(np.nanmax(px)),
                                       float(np.nanmin(pd)), float(np.nanmax(pd)))

            outlines.append(np.column_stack([px, pd]))

        if mask is not None:
            apply_mask(mask, sig)

    # --------------------------------------------------------
    # B) Safety fallback from target_specs if no anomaly cells
    # --------------------------------------------------------
    if total_cells == 0 and target_specs:
        for t in target_specs:
            sig = max(_to_float(t.get("conductivity", 1e-6), 1e-6), 1e-8)
            cx = _to_float(t.get("custom_x", t.get("x", 0.0)), 0.0)
            cd = abs(_to_float(t.get("custom_depth", t.get("depth", 5.0)), 5.0))
            width = max(_to_float(t.get("width", 2.0 * t.get("radius", 2.0)), 4.0), 0.5)
            height = max(_to_float(t.get("height", 2.0 * t.get("radius", 2.0)), 4.0), 0.5)

            rx = 0.5 * width
            rz = 0.5 * height

            mask = ((X - cx) / rx) ** 2 + ((depth - cd) / rz) ** 2 <= 1.0

            if np.count_nonzero(mask) == 0:
                mask = padded_box_mask(cx - rx, cx + rx, cd - rz, cd + rz)

            apply_mask(mask, sig)

            theta = np.linspace(0, 2 * np.pi, 200)
            outlines.append(np.column_stack([cx + rx * np.cos(theta), cd + rz * np.sin(theta)]))

    scenario["_em2d_anomaly_cells"] = int(total_cells)
    return sigma, outlines


def _make_tensor_mesh(m):
    from discretize import TensorMesh

    x = m["x"]
    y = m["y"]
    z = m["z"]

    hx = np.ones(len(x)) * m["dx"]
    hy = np.ones(len(y)) * m["dy"]
    hz = np.ones(len(z)) * m["dz"]

    # TensorMesh cell centers are x0 + cumulative cell widths. Use x0 so centers approximately match x/y/z arrays.
    x0 = [
        float(x[0] - 0.5 * m["dx"]),
        float(y[0] - 0.5 * m["dy"]),
        float(z[0] - 0.5 * m["dz"]),
    ]

    return TensorMesh([hx, hy, hz], x0=x0)


def _primary_bz_vertical_dipole(moment, spacing):
    # Approximate free-space primary Bz at receiver in horizontal plane of a vertical magnetic dipole.
    r = max(abs(float(spacing)), 1e-9)
    return -MU0 * float(moment) / (4.0 * np.pi * r ** 3)


def _run_simpeg_profile(scenario, model_full, model_background):
    try:
        import simpeg.electromagnetics.frequency_domain as fdem
        from simpeg import maps
    except Exception as exc:
        raise ImportError(
            "SimPEG is required for EM 2D. Install it with:\n"
            "  pip install -U simpeg discretize pymatsolver"
        ) from exc

    # Use SolverLU by default.
    # Do NOT use Pardiso unless pydiso is installed, because pymatsolver.Pardiso
    # can import successfully but still fail at runtime.
    try:
        from pymatsolver import SolverLU as Solver
    except Exception:
        Solver = None

    em = scenario.get("survey", {}).get("em", {})

    frequencies = em.get("frequencies", [em.get("freq", 10000.0)])
    frequencies = np.asarray([float(f) for f in frequencies], dtype=float)
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]

    if frequencies.size == 0:
        frequencies = np.asarray([10000.0], dtype=float)

    # To keep GUI runtime sane, default to highest frequency only.
    if not bool(em.get("em2d_all_frequencies", False)):
        frequencies = np.asarray([float(np.nanmax(frequencies))], dtype=float)

    length = model_full["length"]
    surface_fn = model_full["surface_fn"]
    coil_spacing = _to_float(em.get("coil_spacing", em.get("spacing", 3.66)), 3.66)
    height = _to_float(em.get("height", 0.5), 0.5)
    moment = _to_float(em.get("moment", 1.0), 1.0)

    nstations = int(em.get("em2d_stations", min(25, max(11, int(length / max(2.0 * coil_spacing, 1.0))))))
    nstations = max(7, min(nstations, 81))

    # Use midpoint profile positions; source is left, receiver is right.
    margin = max(0.5 * coil_spacing + 1.0, 2.0)
    mids = np.linspace(-0.5 * length + margin, 0.5 * length - margin, nstations)

    src_locs = []
    rx_locs = []
    valid_mids = []

    for xm in mids:
        xs = xm - 0.5 * coil_spacing
        xr = xm + 0.5 * coil_spacing

        if xs < -0.5 * length or xr > 0.5 * length:
            continue

        zs = float(surface_fn(xs)) + height
        zr = float(surface_fn(xr)) + height

        src_locs.append([xs, 0.0, zs])
        rx_locs.append([xr, 0.0, zr])
        valid_mids.append(xm)

    mids = np.asarray(valid_mids, dtype=float)
    src_locs = np.asarray(src_locs, dtype=float)
    rx_locs = np.asarray(rx_locs, dtype=float)

    if len(mids) < 3:
        raise ValueError("Not enough valid EM 2D profile stations. Reduce coil spacing or increase model length.")

    mesh = _make_tensor_mesh(model_full)
    sigma_map = maps.IdentityMap(nP=mesh.nC)

    sigma_full = np.asarray(model_full["sigma"], dtype=float).ravel(order="F")
    sigma_bg = np.asarray(model_background["sigma"], dtype=float).ravel(order="F")

    def predict(sigma_vec, label):
        _progress(
            scenario,
            f"Running SimPEG EM 2D/3D {label}: {len(mids)} stations, {len(frequencies)} frequency/frequencies, cells={mesh.nC:,}"
        )

        source_list = []

        for freq in frequencies:
            for i in range(len(mids)):
                rx_real = fdem.receivers.PointMagneticFluxDensitySecondary(
                    rx_locs[i, :], "z", "real"
                )
                rx_imag = fdem.receivers.PointMagneticFluxDensitySecondary(
                    rx_locs[i, :], "z", "imag"
                )

                source_list.append(
                    fdem.sources.MagDipole(
                        [rx_real, rx_imag],
                        float(freq),
                        src_locs[i, :],
                        orientation="z",
                        moment=moment,
                    )
                )

        survey = fdem.Survey(source_list)

        kwargs = {
            "survey": survey,
            "sigmaMap": sigma_map,
            "forward_only": True,
        }
        if Solver is not None:
            kwargs["solver"] = Solver

        sim = fdem.simulation.Simulation3DMagneticFluxDensity(mesh, **kwargs)
        d = np.asarray(sim.dpred(sigma_vec), dtype=float)

        # Shape: frequency, station, real/imag
        out = d.reshape((len(frequencies), len(mids), 2))
        return out[:, :, 0], out[:, :, 1]

    real_bg, imag_bg = predict(sigma_bg, "background")
    real_full, imag_full = predict(sigma_full, "full model")

    primary = abs(_primary_bz_vertical_dipole(moment, coil_spacing))
    scale = 1e6 / max(primary, 1e-30)

    # Convert secondary Bz to approximate ppm relative to primary field.
    ip_bg = real_bg * scale
    qp_bg = imag_bg * scale
    ip_full = real_full * scale
    qp_full = imag_full * scale

    return {
        "x_mid": mids,
        "frequencies": frequencies,
        "ip_background": ip_bg,
        "qp_background": qp_bg,
        "ip_full": ip_full,
        "qp_full": qp_full,
        "delta_ip": ip_full - ip_bg,
        "delta_qp": qp_full - qp_bg,
        "coil_spacing": coil_spacing,
        "height": height,
        "moment": moment,
        "mesh_cells": mesh.nC,
    }


def _sigma_xz_slice(model):
    x = model["x"]
    y = model["y"]
    z = model["z"]
    sigma = model["sigma"]

    iy = int(np.argmin(np.abs(y)))
    section = sigma[:, iy, :]  # x,z

    # Plot only depth-positive-down part. Air above ground is removed.
    depth_axis = -np.asarray(z, dtype=float)
    keep = depth_axis >= 0.0

    depth = depth_axis[keep]
    sec = section[:, keep]

    return x, depth, sec.T


def _safe_log10(a):
    return np.log10(np.maximum(np.asarray(a, dtype=float), 1e-8))




def _build_conductivity_model(scenario, include_anomaly=True):
    """
    Build EM 2D/3D SimPEG conductivity model.

    Uses x-y-z grid internally:
      x = profile coordinate [m]
      y = cross-line extrusion [m]
      z = elevation coordinate [m], surface near z=0, depth is -z

    The returned sigma array has shape (nx, ny, nz).
    """
    em = scenario.get("survey", {}).get("em", {})

    length, depth_max, dx, dy, dz = _domain(scenario)

    y_width = max(_to_float(em.get("em2d_y_width", 16.0), 16.0), 8.0)

    # Build topography before defining z mesh.
    # Source/receiver z = local surface elevation + sensor height.
    # Therefore air above datum must include topographic relief + sensor height + safety buffer.
    surface_fn, topo_source, relief = _topography_function(scenario, length)
    sensor_height = _to_float(em.get("height", 0.5), 0.5)
    air_buffer = max(_to_float(em.get("em2d_air_height", 4.0), 4.0), 2.0 * dz)
    z_min = -depth_max
    z_max = float(relief) + max(sensor_height, 0.0) + air_buffer

    x = np.arange(-0.5 * length, 0.5 * length + 0.5 * dx, dx)
    y = np.arange(-0.5 * y_width, 0.5 * y_width + 0.5 * dy, dy)
    z = np.arange(z_min, z_max + 0.5 * dz, dz)

    max_cells = int(em.get("em2d_max_cells", 120000))
    ncell = len(x) * len(y) * len(z)

    if ncell > max_cells:
        scale = float((ncell / max_cells) ** (1.0 / 3.0))
        dx *= scale * 1.10
        dy *= scale * 1.10
        dz *= scale * 1.10

        _progress(
            scenario,
            "EM 2D mesh auto-coarsened for GUI safety: "
            f"old grid={len(x)}x{len(y)}x{len(z)} ({ncell:,} cells), "
            f"new dx/dy/dz≈{dx:.2f}/{dy:.2f}/{dz:.2f} m"
        )

        # Recompute vertical extent after coarsening, preserving enough air above topography.
        air_buffer = max(_to_float(em.get("em2d_air_height", 4.0), 4.0), 2.0 * dz)
        z_max = float(relief) + max(sensor_height, 0.0) + air_buffer

        x = np.arange(-0.5 * length, 0.5 * length + 0.5 * dx, dx)
        y = np.arange(-0.5 * y_width, 0.5 * y_width + 0.5 * dy, dy)
        z = np.arange(z_min, z_max + 0.5 * dz, dz)
        ncell = len(x) * len(y) * len(z)

        if ncell > max_cells:
            raise ValueError(
                "EM 2D/3D SimPEG mesh is still too large after auto-coarsening.\n"
                f"Grid would be {len(x)} x {len(y)} x {len(z)} = {ncell:,} cells.\n"
                "Set em2d_dx/em2d_dy/em2d_dz larger, e.g. 6–8 m."
            )

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    depth = surface_fn(X) - Z

    air_sigma = max(_to_float(em.get("air_conductivity", 1e-8), 1e-8), 1e-10)
    sigma = np.full(X.shape, air_sigma, dtype=float)

    layers = scenario.get("layers", []) or []
    if not layers:
        layers = [{"thickness": 999.0, "conductivity": 0.001, "resistivity": 1000.0}]

    # Geological cells: depth >= 0
    geological = depth >= 0.0
    cumulative_top = 0.0

    for i, layer in enumerate(layers):
        thick = _to_float(layer.get("thickness", 999.0), 999.0)
        sig = _layer_sigma(layer)

        if i == len(layers) - 1:
            mask = geological & (depth >= cumulative_top)
        else:
            mask = geological & (depth >= cumulative_top) & (depth < cumulative_top + thick)

        sigma[mask] = sig
        cumulative_top += thick

    outlines = []
    if include_anomaly:
        sigma, outlines = _apply_anomalies_to_sigma(sigma, x, y, z, scenario, surface_fn)

    return {
        "x": x,
        "y": y,
        "z": z,
        "sigma": sigma,
        "surface_fn": surface_fn,
        "outlines": outlines,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "length": length,
        "depth": depth_max,
        "mesh_cells": int(ncell),
        "topo_source": topo_source,
        "relief": float(relief),
    }

def run_em_2d(scenario):
    _progress(scenario, "Preparing EM 2D/3D SimPEG model...")

    full = _build_conductivity_model(scenario, include_anomaly=True)
    background = _build_conductivity_model(scenario, include_anomaly=False)

    anomaly_cells = int(scenario.get("_em2d_anomaly_cells", 0))
    sigma_diff_cells = int(np.count_nonzero(np.abs(full["sigma"] - background["sigma"]) > 0.0))

    if sigma_diff_cells == 0:
        raise ValueError(
            "EM 2D target was not inserted into the conductivity mesh. "
            "Increase target size or reduce EM2D dx/dy/dz."
        )

    _progress(
        scenario,
        f"EM 2D target inserted: anomaly_cells={anomaly_cells}, changed_sigma_cells={sigma_diff_cells}"
    )

    result = _run_simpeg_profile(scenario, full, background)

    x_mid = result["x_mid"]
    freqs = result["frequencies"]

    # Use highest/single frequency for main profile plots.
    idx = int(np.argmax(freqs))
    f = float(freqs[idx])

    delta_ip = result["delta_ip"][idx, :]
    delta_qp = result["delta_qp"][idx, :]
    amp = np.sqrt(delta_ip ** 2 + delta_qp ** 2)

    bg_amp = np.sqrt(result["ip_background"][idx, :] ** 2 + result["qp_background"][idx, :] ** 2)
    ratio = float(np.sqrt(np.nanmean(amp ** 2)) / max(np.sqrt(np.nanmean(bg_amp ** 2)), 1e-30))

    x, depth, sigma_sec = _sigma_xz_slice(full)
    extent_model = [float(x[0]), float(x[-1]), float(depth[-1]), float(depth[0])]

    _, _, sigma_bg_sec = _sigma_xz_slice(background)
    target_mask_sec = (np.abs(sigma_sec - sigma_bg_sec) > 0.0).astype(float)

    plots = {
        "EM 2D conductivity model": {
            "type": "image",
            "array": _safe_log10(sigma_sec),
            "extent": extent_model,
            "origin": "upper",
            "title": "EM 2D/3D SimPEG conductivity model at y=0",
            "xlabel": "x [m]",
            "ylabel": "depth [m]",
            "clabel": "log10 conductivity [S/m]",
            "colorbar": True,
        },
        "EM 2D target mask": {
            "type": "image",
            "array": target_mask_sec,
            "extent": extent_model,
            "origin": "upper",
            "title": "EM 2D inserted target mask at y=0",
            "xlabel": "x [m]",
            "ylabel": "depth [m]",
            "clabel": "target cells",
            "colorbar": True,
        },
        "EM 2D anomaly profile": {
            "type": "line",
            "array": np.column_stack([x_mid, delta_ip, delta_qp]),
            "title": f"EM 2D spatial anomaly profile at {f:g} Hz",
            "xlabel": "profile midpoint x [m]",
            "ylabel": "secondary field anomaly [ppm approx.]",
            "labels": {1: "ΔIP/Bp ppm", 2: "ΔQP/Bp ppm"},
        },
        "EM 2D anomaly amplitude": {
            "type": "line",
            "array": np.column_stack([x_mid, amp]),
            "title": f"EM 2D anomaly amplitude at {f:g} Hz",
            "xlabel": "profile midpoint x [m]",
            "ylabel": "sqrt(ΔIP² + ΔQP²) [ppm approx.]",
            "labels": {1: "anomaly amplitude"},
        },
        "EM 2D detectability metric": {
            "type": "text",
            "title": "EM 2D/3D SimPEG detectability metric",
            "text": (
                "EM 2D/3D SimPEG detectability metric\n\n"
                f"frequency = {f:g} Hz\n"
                f"RMS anomaly amplitude = {float(np.sqrt(np.nanmean(amp**2))):.6g} ppm approx.\n"
                f"RMS background amplitude = {float(np.sqrt(np.nanmean(bg_amp**2))):.6g} ppm approx.\n"
                f"anomaly/background ratio = {ratio:.6g}\n\n"
                "Interpretation:\n"
                "< 0.01  : very weak\n"
                "0.01-0.05: weak but possibly measurable\n"
                "> 0.05  : potentially detectable\n\n"
                "This is a coarse SimPEG 3D FDEM forward model used as a practical 2D profile approximation. "
                "Use it for spatial detectability trends, then refine mesh/survey settings for final quantitative modelling."
            ),
        },
    }

    data = np.column_stack([x_mid, delta_ip, delta_qp, amp])

    info = "\n".join(
        [
            "EM 2D forward model completed",
            "Implementation: SimPEG frequency-domain EM, 3D finite-volume model used as a 2D profile approximation.",
            "Survey: vertical magnetic dipole source and vertical secondary B-field receiver along profile.",
            "Data scaling: secondary Bz normalised by approximate free-space primary Bz, reported in ppm approx.",
            f"frequencies_hz={freqs.tolist()}",
            f"used_frequency_hz={f:.6g}",
            f"stations={len(x_mid)}",
            f"coil_spacing_m={result['coil_spacing']:.6g}",
            f"height_m={result['height']:.6g}",
            f"mesh_cells={result['mesh_cells']}",
            f"anomaly_cells={anomaly_cells}",
            f"changed_sigma_cells={sigma_diff_cells}",
            f"topography_source={full['topo_source']}",
            f"topography_relief_m={full['relief']:.6g}",
            f"model_length_m={full['length']:.6g}",
            f"model_depth_m={full['depth']:.6g}",
            f"dx_m={full['dx']:.6g}",
            f"dy_m={full['dy']:.6g}",
            f"dz_m={full['dz']:.6g}",
            f"delta_ip_ppm_approx={delta_ip.tolist()}",
            f"delta_qp_ppm_approx={delta_qp.tolist()}",
            f"anomaly_amplitude_ppm_approx={amp.tolist()}",
            f"detectability_ratio={ratio:.6g}",
        ]
    )

    return {
        "model": full["sigma"],
        "data": data,
        "plots": plots,
        "info": info,
    }


# Compatibility alias.
def run_em(scenario):
    return run_em_2d(scenario)
