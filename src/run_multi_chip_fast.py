"""Backward-compatibility shim redirecting to src/experiments/run_multi_chip_fast.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "experiments" / "run_multi_chip_fast.py"
runpy.run_path(str(_target), run_name="__main__")
