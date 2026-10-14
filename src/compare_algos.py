"""Backward-compatibility shim redirecting to src/experiments/compare_algos.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "experiments" / "compare_algos.py"
runpy.run_path(str(_target), run_name="__main__")
