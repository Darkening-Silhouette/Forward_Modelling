
from solvers.ert_pygimli import run_ert
from solvers.gpr_gprmax import run_gpr
from solvers.em_fdem import run_em as run_em_1d
from solvers.em_simpeg_2d import run_em_2d

try:
    from solvers.seismic_acoustic import run_seismic as run_seismic_acoustic
except Exception:
    from solvers.seismic_devito import run_seismic as run_seismic_acoustic

try:
    from solvers.seismic_elastic import run_seismic_elastic
except Exception:
    run_seismic_elastic = None


def run_forward(method: str, scenario: dict) -> dict:
    key = str(method).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())

    if key == "ert":
        return run_ert(scenario)

    if key in {"seismic", "seismic acoustic", "acoustic seismic"}:
        return run_seismic_acoustic(scenario)

    if key in {"seismic elastic", "elastic seismic"}:
        if run_seismic_elastic is None:
            raise ValueError("Seismic elastic solver is not available.")
        return run_seismic_elastic(scenario)

    if key == "gpr":
        return run_gpr(scenario)

    if key in {"em", "em 1d", "fdem 1d"}:
        return run_em_1d(scenario)

    if key in {"em 2d", "fdem 2d", "em simpeg 2d"}:
        return run_em_2d(scenario)

    raise ValueError(f"Unknown method: {method}")
