"""Backward-compatibility shim redirecting to src/experiments/run_compact_state_experiment.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "experiments" / "run_compact_state_experiment.py"
runpy.run_path(str(_target), run_name="__main__")
