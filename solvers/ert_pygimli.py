from __future__ import annotations

from pathlib import Path
import copy
import math

import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

def _progress(scenario: dict, text: str):
    cb = scenario.get("_progress_callback")
    if cb is not None:
        cb(str(text))


from matplotlib.path import Path as MplPath

try:
    from scipy.interpolate import CubicSpline
except Exception:
    CubicSpline = None


def _progress(scenario: dict, text: str) -> None:
    cb = scenario.get("_progress_callback")
    if cb:
        cb(str(text))


def _normalise_ert_scheme(name: str) -> str:
    mapping = {
        "wenner": "wa",
        "wenner-alpha": "wa",
        "wenner_alpha": "wa",
        "wa": "wa",
        "wenner-beta": "wb",
        "wenner_beta": "wb",
        "wb": "wb",
        "dipole-dipole": "dd",
        "dipole_dipole": "dd",
        "dd": "dd",
        "schlumberger": "slm",
        "slm": "slm",
        "gradient": "gr",
        "gr": "gr",
    }

    key = str(name).strip().lower()

    if key in {"combined", "combo", "all", "wa+slm+dd", "wenner+schlumberger+dipole-dipole"}:
        return "combined"

    if key not in mapping:
        raise ValueError(
            f"Unknown ERT array '{name}'. Use: combined, wenner-alpha, "
            "wenner-beta, dipole-dipole, schlumberger, gradient."
        )

    return mapping[key]


def _read_elevation_csv(path: str | None, domain_length: float | None = None):
    if not path:
        return None

    p = Path(path).expanduser()

    if not p.exists():
        # Try relative to project root.
        project_root = Path(__file__).resolve().parents[1]
        p2 = project_root / path
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"Elevation CSV not found: {path}")

    # Swiss topo CSV from map.geo.admin.ch is normally semicolon-separated.
    try:
        df = pd.read_csv(p, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(p, sep=";")

    df.columns = [c.lower().strip() for c in df.columns]

    x_col = next(
        (c for c in ["x", "distance", "distanz", "chainage", "profile_x"] if c in df.columns),
        None,
    )
    z_col = next(
        (c for c in ["z", "elevation", "altitude", "height", "hoehe", "topography", "topo"] if c in df.columns),
        None,
    )

    if x_col is None or z_col is None:
        raise ValueError(
            "Elevation CSV must contain columns like Distance/Altitude, x/z, or x/elevation."
        )

    x = df[x_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    good = np.isfinite(x) & np.isfinite(z)
    x = x[good]
    z = z[good]

    order = np.argsort(x)
    x = x[order]
    z = z[order]

    # Convert CSV profile to local centred coordinates, matching GUI convention.
    x = x - np.nanmin(x)
    if domain_length is None:
        domain_length = float(np.nanmax(x) - np.nanmin(x))
    x = x - 0.5 * float(domain_length)

    # Normalize topography so minimum elevation is zero.
    z = z - np.nanmin(z)

    return x, z


def _spline_or_interp(x: np.ndarray, z: np.ndarray):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    order = np.argsort(x)
    x = x[order]
    z = z[order]

    unique_x, idx = np.unique(x, return_index=True)
    unique_z = z[idx]

    if len(unique_x) >= 3 and CubicSpline is not None:
        cs = CubicSpline(unique_x, unique_z, extrapolate=True)
        return lambda xx: np.asarray(cs(xx), dtype=float)

    return lambda xx: np.interp(np.asarray(xx, dtype=float), unique_x, unique_z)


def _surface_function(scenario: dict, xmin: float, xmax: float, length: float):
    csv_path = scenario.get("files", {}).get("elevation_csv", "")
    csv_topo = _read_elevation_csv(csv_path, domain_length=length)

    if csv_topo is not None:
        tx, tz = csv_topo
    else:
        topo = scenario.get("topography", [])
        if topo:
            tx = np.asarray([p[0] for p in topo], dtype=float)
            tz = np.asarray([p[1] for p in topo], dtype=float)
        else:
            tx = np.asarray([xmin, xmax], dtype=float)
            tz = np.asarray([0.0, 0.0], dtype=float)

    # Ensure surface covers full model width.
    if np.nanmin(tx) > xmin:
        tx = np.r_[xmin, tx]
        tz = np.r_[tz[0], tz]
    if np.nanmax(tx) < xmax:
        tx = np.r_[tx, xmax]
        tz = np.r_[tz, tz[-1]]

    f = _spline_or_interp(tx, tz)
    return tx, tz, f


def _layer_interface_splines(scenario: dict, layers: list[dict], xmin: float, xmax: float, depth: float):
    explicit = scenario.get("layer_interfaces", None)

    splines = []

    if explicit:
        for interface in explicit:
            pts = np.asarray(interface, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
                continue
            splines.append(_spline_or_interp(pts[:, 0], pts[:, 1]))
        return splines

    survey = scenario.get("survey", {}).get("ert", {})
    topography_mode = str(survey.get("topography_mode", "horizontal_interfaces")).strip().lower()
    follow_topography = topography_mode in {
        "parallel_to_topography",
        "layers_follow_topography",
        "follow_topography",
    }

    length = float(xmax - xmin)
    _, _, surface_f = _surface_function(scenario, xmin, xmax, length)

    cumulative = 0.0

    for layer in layers[:-1]:
        cumulative += float(layer.get("thickness", 1.0))

        if cumulative < depth:
            x_pts = np.linspace(xmin, xmax, 80)

            if follow_topography:
                # pyGIMLi z is positive upward. Interface = local surface minus depth.
                z_pts = surface_f(x_pts) - cumulative
            else:
                z_pts = np.full_like(x_pts, -cumulative, dtype=float)

            splines.append(_spline_or_interp(x_pts, z_pts))

    return splines

def _make_world_plc(
    xmin: float,
    xmax: float,
    z_bottom: float,
    surface_f,
    layer_splines: list,
    electrode_x: np.ndarray,
):
    world = pg.Mesh(dim=2, isGeometry=True)

    surface_x = np.unique(
        np.concatenate(
            [
                np.linspace(xmin, xmax, 80),
                np.asarray(electrode_x, dtype=float),
            ]
        )
    )
    surface_z = surface_f(surface_x)

    surface_nodes = [
        world.createNode(pg.Pos(float(x), float(z)))
        for x, z in zip(surface_x, surface_z)
    ]

    iface_x = np.linspace(xmin, xmax, 80)
    iface_chains = []

    for spline in layer_splines:
        zz = spline(iface_x)
        chain = [
            world.createNode(pg.Pos(float(x), float(z)))
            for x, z in zip(iface_x, zz)
        ]
        iface_chains.append(chain)

    n_bl = world.createNode(pg.Pos(float(xmin), float(z_bottom)))
    n_br = world.createNode(pg.Pos(float(xmax), float(z_bottom)))

    # Surface: marker -1 = Neumann / earth-air boundary.
    for i in range(len(surface_nodes) - 1):
        world.createEdge(surface_nodes[i], surface_nodes[i + 1], -1)

    # Right side: marker -2 = mixed far boundary.
    right_chain = [surface_nodes[-1]] + [chain[-1] for chain in iface_chains] + [n_br]
    for i in range(len(right_chain) - 1):
        world.createEdge(right_chain[i], right_chain[i + 1], -2)

    # Bottom.
    world.createEdge(n_br, n_bl, -2)

    # Left side.
    left_chain = [n_bl] + [chain[0] for chain in reversed(iface_chains)] + [surface_nodes[0]]
    for i in range(len(left_chain) - 1):
        world.createEdge(left_chain[i], left_chain[i + 1], -2)

    # Internal layer boundaries.
    for chain in iface_chains:
        for i in range(len(chain) - 1):
            world.createEdge(chain[i], chain[i + 1], 0)

    # Region markers.
    x_mid = 0.5 * (xmin + xmax)
    z_surface_mid = float(surface_f([x_mid])[0])

    if not layer_splines:
        world.addRegionMarker(pg.Pos(x_mid, 0.5 * (z_surface_mid + z_bottom)), marker=1)
        return world

    z_first = float(layer_splines[0]([x_mid])[0])
    world.addRegionMarker(pg.Pos(x_mid, 0.5 * (z_surface_mid + z_first)), marker=1)

    for i in range(len(layer_splines) - 1):
        z_top = float(layer_splines[i]([x_mid])[0])
        z_bot = float(layer_splines[i + 1]([x_mid])[0])
        world.addRegionMarker(pg.Pos(x_mid, 0.5 * (z_top + z_bot)), marker=i + 2)

    z_last = float(layer_splines[-1]([x_mid])[0])
    world.addRegionMarker(pg.Pos(x_mid, 0.5 * (z_last + z_bottom)), marker=len(layer_splines) + 1)

    return world


def _ellipse_polygon(cx: float, cz: float, rx: float, rz: float, n: int = 72):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(theta), cz + rz * np.sin(theta)])


def _anomaly_to_polygon(anomaly: dict):
    typ = str(anomaly.get("type", "circle")).strip().lower()

    if typ in {"circle", "ellipse"}:
        if "center" in anomaly:
            cx, cz = anomaly.get("center", [0.0, -5.0])
        else:
            cx = anomaly.get("x", 0.0)
            cz = anomaly.get("z", -5.0)

        if typ == "circle":
            r = float(anomaly.get("radius", anomaly.get("radius_x", 2.0)))
            rx = r
            rz = r
        else:
            rx = float(anomaly.get("radius_x", anomaly.get("rx", anomaly.get("width", 4.0) / 2.0)))
            rz = float(anomaly.get("radius_z", anomaly.get("rz", anomaly.get("height", 4.0) / 2.0)))

        return _ellipse_polygon(float(cx), float(cz), float(rx), float(rz))

    if typ == "polygon":
        pts = np.asarray(anomaly.get("points", []), dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
            return pts

    return None


def _make_body_plcs(anomalies: list[dict]):
    body_plcs = []
    body_defs = []

    for i, anomaly in enumerate(anomalies):
        poly = _anomaly_to_polygon(anomaly)
        if poly is None:
            continue

        marker = 100 + i

        body = mt.createPolygon(
            poly.tolist(),
            isClosed=True,
            marker=marker,
            boundaryMarker=10 + i,
            area=float(anomaly.get("area", 0.1)),
        )

        body_plcs.append(body)
        body_defs.append(
            {
                "polygon": poly,
                "resistivity": float(anomaly.get("resistivity", 1000.0)),
                "label": anomaly.get("label", f"anomaly_{i + 1}"),
            }
        )

    return body_plcs, body_defs


def _layer_resistivity_at_cell(cell, layer_splines: list, layers: list[dict], z_bottom: float):
    x_c = float(cell.center().x())
    z_c = float(cell.center().y())

    if not layers:
        return 100.0

    if not layer_splines:
        return float(layers[0].get("resistivity", 100.0))

    boundaries = [float(s([x_c])[0]) for s in layer_splines] + [float(z_bottom)]

    for k, z_boundary in enumerate(boundaries):
        if z_c > z_boundary:
            return float(layers[min(k, len(layers) - 1)].get("resistivity", 100.0))

    return float(layers[-1].get("resistivity", 100.0))


def _build_cell_resistivity(mesh, layers: list[dict], layer_splines: list, body_defs: list[dict], z_bottom: float):
    cells = list(mesh.cells())
    res = np.zeros(mesh.cellCount(), dtype=float)

    for cell in cells:
        res[cell.id()] = _layer_resistivity_at_cell(cell, layer_splines, layers, z_bottom)

    centers = np.asarray([[c.center().x(), c.center().y()] for c in cells], dtype=float)
    ids = np.asarray([c.id() for c in cells], dtype=int)

    override_info = []

    for bd in body_defs:
        path = MplPath(np.asarray(bd["polygon"], dtype=float))
        inside = path.contains_points(centers)

        n_inside = int(np.sum(inside))
        if n_inside > 0:
            res[ids[inside]] = float(bd["resistivity"])

        override_info.append((str(bd["label"]), n_inside, float(bd["resistivity"])))

    return res, override_info


def _make_scheme(survey: dict, electrodes: np.ndarray):
    scheme_name_input = survey.get("scheme", survey.get("array", "combined"))
    scheme_name = _normalise_ert_scheme(scheme_name_input)

    if scheme_name != "combined":
        return ert.createData(elecs=electrodes, schemeName=scheme_name), scheme_name_input, scheme_name, [scheme_name]

    requested = survey.get("combined_arrays", ["wa", "slm", "dd"])
    requested = [_normalise_ert_scheme(a) for a in requested]

    schemes = []
    used = []

    for arr in requested:
        if arr == "combined":
            continue
        try:
            s = ert.createData(elecs=electrodes, schemeName=arr)
            if s.size() > 0:
                schemes.append(s)
                used.append(arr)
        except Exception:
            pass

    if not schemes:
        raise ValueError("No valid ERT arrays could be created for combined scheme.")

    combined = pg.DataContainerERT(schemes[0])

    for s in schemes[1:]:
        for i in range(s.size()):
            combined.createFourPointData(
                combined.size(),
                int(s("a")[i]),
                int(s("b")[i]),
                int(s("m")[i]),
                int(s("n")[i]),
            )

    return combined, scheme_name_input, "combined", used


def _copy_data_with_rhoa(data, rhoa_values):
    copied = pg.DataContainerERT(data)
    copied["rhoa"] = np.asarray(rhoa_values, dtype=float)
    return copied


def run_ert(scenario: dict) -> dict:
    _progress(scenario, "Preparing ERT model...")
    _progress(scenario, "Preparing Jan-style pyGIMLi ERT model...")

    domain = scenario.get("domain", {})
    survey = scenario.get("survey", {}).get("ert", {})

    length = float(domain.get("length", 92.7))
    depth = float(domain.get("depth", 50.0))
    xmin = float(domain.get("xmin", -0.5 * length))
    xmax = float(domain.get("xmax", 0.5 * length))
    z_bottom = -abs(depth)

    layers = copy.deepcopy(scenario.get("layers", []))
    if not layers:
        layers = [
            {"thickness": 0.3, "resistivity": 100.0},
            {"thickness": 25.7, "resistivity": 10000.0},
            {"thickness": 999.0, "resistivity": 250.0},
        ]

    n_elecs = int(survey.get("electrodes", survey.get("n_electrodes", 41)))

    if "electrode_x_start" in survey and "electrode_x_end" in survey:
        elec_x_start = float(survey["electrode_x_start"])
        elec_x_end = float(survey["electrode_x_end"])
        elec_x = np.linspace(elec_x_start, elec_x_end, n_elecs)
        spacing = float(abs(elec_x[1] - elec_x[0])) if n_elecs > 1 else 1.0
    else:
        spacing = float(
            survey.get(
                "spacing",
                survey.get("electrode_spacing", length / max(n_elecs - 1, 1)),
            )
        )
        profile_length = spacing * (n_elecs - 1)
        elec_x = np.linspace(-0.5 * profile_length, 0.5 * profile_length, n_elecs)

    xmin = min(xmin, float(elec_x.min()) - 2.0 * spacing)
    xmax = max(xmax, float(elec_x.max()) + 2.0 * spacing)

    topo_x, topo_z, surface_f = _surface_function(scenario, xmin, xmax, length)
    elec_z = surface_f(elec_x)
    electrodes = np.column_stack([elec_x, elec_z])

    layer_splines = _layer_interface_splines(scenario, layers, xmin, xmax, depth)

    _progress(scenario, f"ERT electrodes={n_elecs}, spacing={spacing:.3f} m")
    _progress(scenario, f"ERT domain x=[{xmin:.2f}, {xmax:.2f}], z=[{z_bottom:.2f}, {float(np.max(topo_z)):.2f}]")

    world = _make_world_plc(
        xmin=xmin,
        xmax=xmax,
        z_bottom=z_bottom,
        surface_f=surface_f,
        layer_splines=layer_splines,
        electrode_x=elec_x,
    )

    anomalies = copy.deepcopy(scenario.get("anomalies", []))
    body_plcs, body_defs = _make_body_plcs(anomalies)

    geom = world
    for body in body_plcs:
        geom = geom + body

    # Electrode refinement nodes, Jan-style: 10% of electrode spacing below each electrode.
    refinement_depth = float(survey.get("electrode_refinement_fraction", 0.10)) * spacing
    for x, z in electrodes:
        geom.createNode(pg.Pos(float(x), float(z) - refinement_depth))

    quality = float(survey.get("mesh_quality", 34))
    mesh_area = float(survey.get("mesh_area", 0.5))

    _progress(scenario, f"Creating ERT mesh: quality={quality}, area={mesh_area}...")
    _progress(scenario, "Creating ERT mesh...")

    mesh = mt.createMesh(geom, quality=quality, area=mesh_area)

    res, override_info = _build_cell_resistivity(
        mesh=mesh,
        layers=layers,
        layer_splines=layer_splines,
        body_defs=body_defs,
        z_bottom=z_bottom,
    )

    _progress(scenario, f"ERT mesh created: cells={mesh.cellCount()}, nodes={mesh.nodeCount()}")

    scheme, scheme_name_input, scheme_name, used_arrays = _make_scheme(survey, electrodes)

    noise_percent = float(
        scenario.get("noise", {}).get(
            "relative_percent",
            survey.get("noise_rel", 1.0),
        )
    )
    noise_abs = float(survey.get("noise_abs", 1e-6))
    seed = int(survey.get("seed", 1337))

    _progress(scenario, f"Simulating ERT data: scheme={scheme_name}, measurements={scheme.size()}...")

    data = ert.simulate(
        mesh=mesh,
        scheme=scheme,
        res=pg.RVector(res),
        noiseLevel=noise_percent,
        noiseAbs=noise_abs,
        seed=seed,
        verbose=bool(survey.get("verbose", False)),
    )

    n_before = data.size()
    _progress(scenario, "ERT simulation finished. Cleaning negative apparent resistivity values...")

    data.remove(data["rhoa"] < 0)
    n_removed = n_before - data.size()

    rhoa = np.asarray(data["rhoa"], dtype=float)
    valid = np.asarray(data["valid"], dtype=float) if data.haveData("valid") else np.ones_like(rhoa)
    idx = np.arange(len(rhoa), dtype=float)
    output_data = np.column_stack([idx, rhoa, valid])

    cmin = max(float(np.nanmin(res)) * 0.5, 1e-6)
    cmax = float(np.nanmax(res)) * 1.5

    xlim = [float(elec_x.min()) - 5.0, float(elec_x.max()) + 5.0]
    ylim = [z_bottom, float(np.max(topo_z)) + 1.0]

    plots = {
        "Measured apparent resistivity pseudosection": {
            "type": "ert_show",
            "data": data,
            "title": "Measured apparent resistivity pseudosection",
            "cmap": "Spectral_r",
            "logScale": True,
        },
        "True resistivity model": {
            "type": "pg_show",
            "mesh": mesh,
            "values": pg.RVector(res),
            "title": "True resistivity model",
            "xlabel": "x [m]",
            "ylabel": "z [m]",
            "cmap": "Spectral_r",
            "logScale": True,
            "cMin": cmin,
            "cMax": cmax,
            "xlim": xlim,
            "ylim": ylim,
        },
    }

    inversion_info = ""

    if bool(survey.get("run_inversion", False)):
        _progress(scenario, "Running ERT inversion...")

        try:
            _progress(scenario, "Starting ERT inversion...")

            mgr = ert.ERTManager(data)

            _progress(scenario, "Running pyGIMLi ERT inversion. This can take a while...")

            inv_model = mgr.invert(
                lam=float(survey.get("lam", 2000.0)),
                lambdaFactor=float(survey.get("lambda_factor", 0.8)),
                zWeight=float(survey.get("z_weight", 1.0)),
                maxIter=int(survey.get("max_iter", 20)),
                dPhi=float(survey.get("d_phi", 1.0)),
                stopAtChi1=bool(survey.get("stop_at_chi1", True)),
                robustData=bool(survey.get("robust_data", False)),
                blockyModel=bool(survey.get("blocky_model", False)),
                verbose=bool(survey.get("verbose_inversion", False)),
            )

            _progress(scenario, f"ERT inversion completed: chi2={mgr.inv.chi2():.4g}, relrms={mgr.inv.relrms():.4g}")

            response = np.asarray(mgr.inv.response, dtype=float)
            response_data = _copy_data_with_rhoa(data, response)

            mesh_pd = pg.Mesh(mgr.paraDomain)
            inv_unstructured = np.asarray(inv_model, dtype=float)

            plots["Model response pseudosection"] = {
                "type": "ert_show",
                "data": response_data,
                "title": "Model response pseudosection",
                "cmap": "Spectral_r",
                "logScale": True,
            }

            plots["Inverted resistivity model"] = {
                "type": "pg_show",
                "mesh": mesh_pd,
                "values": inv_unstructured,
                "title": f"Inverted resistivity model: chi2={mgr.inv.chi2():.2f}",
                "xlabel": "x [m]",
                "ylabel": "z [m]",
                "cmap": "Spectral_r",
                "logScale": True,
                "cMin": cmin,
                "cMax": cmax,
                "xlim": xlim,
                "ylim": ylim,
            }

            try:
                coverage = np.asarray(mgr.coverage(), dtype=float)
                plots["ERT model coverage"] = {
                    "type": "pg_show",
                    "mesh": mesh_pd,
                    "values": coverage,
                    "title": "ERT model coverage / cumulative sensitivity",
                    "xlabel": "x [m]",
                    "ylabel": "z [m]",
                    "cmap": "hot_r",
                    "logScale": False,
                    "xlim": xlim,
                    "ylim": ylim,
                }
            except Exception:
                pass

            inversion_info = (
                "\nERT inversion completed."
                f"\nchi2={mgr.inv.chi2():.6g}"
                f"\nrelrms={mgr.inv.relrms():.6g}"
                f"\nabsrms={mgr.inv.absrms():.6g}"
            )

        except Exception as exc:
            inversion_info = f"\nERT inversion failed: {exc}"

    override_lines = "\n".join(
        f"  {label}: cells={n_inside}, rho={rho:g} ohm m"
        for label, n_inside, rho in override_info
    )

    info = (
        "ERT forward model completed using Jan-style pyGIMLi workflow\n"
        "Features: spline/topographic surface, explicit PLC, electrode surface nodes, "
        "10%-spacing electrode refinement, mesh-conforming anomaly outlines, "
        "per-cell body resistivity override, combined ERT arrays, optional inversion.\n"
        f"array_input={scheme_name_input}\n"
        f"pygimli_scheme={scheme_name}\n"
        f"used_arrays={used_arrays}\n"
        f"electrodes={n_elecs}\n"
        f"spacing_m={spacing:.6g}\n"
        f"measurements_before_filter={n_before}\n"
        f"negative_rhoa_removed={n_removed}\n"
        f"measurements_after_filter={len(rhoa)}\n"
        f"noise_percent={noise_percent}\n"
        f"noise_abs={noise_abs}\n"
        f"mesh_cells={mesh.cellCount()}\n"
        f"mesh_nodes={mesh.nodeCount()}\n"
        f"rho_min={float(np.nanmin(res)):.6g}\n"
        f"rho_max={float(np.nanmax(res)):.6g}\n"
        f"rhoa_min={float(np.nanmin(rhoa)):.6g}\n"
        f"rhoa_max={float(np.nanmax(rhoa)):.6g}\n"
        f"rhoa_mean={float(np.nanmean(rhoa)):.6g}\n"
        f"body_overrides:\n{override_lines if override_lines else '  none'}"
        f"{inversion_info}"
    )

    return {
        "model": pg.RVector(res),
        "data": output_data,
        "plots": plots,
        "info": info,
    }
