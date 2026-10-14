"""Backward-compatibility shim redirecting to src/analysis/analyze_ppo_asa_holdout.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "analysis" / "analyze_ppo_asa_holdout.py"
runpy.run_path(str(_target), run_name="__main__")
