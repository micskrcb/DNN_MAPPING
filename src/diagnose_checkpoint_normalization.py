"""Backward-compatibility shim redirecting to src/analysis/diagnose_checkpoint_normalization.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "analysis" / "diagnose_checkpoint_normalization.py"
runpy.run_path(str(_target), run_name="__main__")
