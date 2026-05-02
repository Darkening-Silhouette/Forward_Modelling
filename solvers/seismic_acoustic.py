from __future__ import annotations

# Existing acoustic Devito solver.
# Kept as a wrapper so the GUI/dispatch can call "Seismic acoustic"
# without breaking the current seismic_devito.py implementation.

from solvers.seismic_devito import run_seismic as run_seismic_acoustic


def run_seismic(scenario: dict) -> dict:
    return run_seismic_acoustic(scenario)
