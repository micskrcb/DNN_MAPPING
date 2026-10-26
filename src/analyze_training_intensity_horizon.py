"""Validate the 16-versus-32-step learner comparison at 200k proposals."""
import argparse
import json
from pathlib import Path

from analyze_online_rapid import summarize

METHODS = ["train_more", "train_32", "fixed_sa"]


def analyze(root, seed):
    root = Path(root)
    result = summarize(root, seed, METHODS)
    reports = {
        method: json.loads((root / f"seed{seed}-{method}.json").read_text())
        for method in METHODS
    }
    reference, candidate = reports["train_more"], reports["train_32"]
    changed = {
        key for key in reference["config"].keys() | candidate["config"].keys()
        if reference["config"].get(key) != candidate["config"].get(key)
    }
    allowed = {"ppo_update_epochs", "report", "asa_diagnostics"}
    if changed - allowed:
        raise ValueError("Learning arms differ beyond optimizer-step intensity: " +
                         str(sorted(changed - allowed)))
    base = reference["algorithm_metadata"]
    more = candidate["algorithm_metadata"]
    if base["update_count"] != more["update_count"]:
        raise ValueError("Update-event counts differ")
    if base["bandit_optimizer_steps_this_invocation"] * 2 != more["bandit_optimizer_steps_this_invocation"]:
        raise ValueError("Expected twofold optimizer-step ratio")
    result["optimizer_step_ratio"] = 2.0
    result["candidate_beats_both_controls"] = all(
        candidate["best_cost"] < reports[method]["best_cost"]
        for method in ("train_more", "fixed_sa"))
    result["interpretation"] = (
        "One fresh-seed training-intensity screen. It does not establish "
        "convergence, a general 32-step benefit, or published-DDPG superiority.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()
    summary = analyze(args.directory, args.seed)
    (args.directory / "training-intensity-horizon-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
