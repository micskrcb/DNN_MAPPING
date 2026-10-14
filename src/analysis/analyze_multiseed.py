"""Summarize multi-seed results and diagnose whether DDPG actually learned.

Pass one or more output directories created by run_multiseed_experiment.py.
The resulting JSON compares aggregate costs and, for each DDPG seed, separates
the best noisy search result from the deterministic actor's final behavior.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def describe(values):
    values = finite(values)
    return {"count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"No diagnostic records in {path}")
    return records


def diagnose(records):
    tail_count = max(1, math.ceil(len(records) * 0.1))
    tail = records[-tail_count:]
    best_record = min(records, key=lambda item: item["best_cost"])
    final = records[-1]
    best_cost = float(best_record["best_cost"])
    deterministic_final = float(final["deterministic_cost"])
    return {
        "episodes": len(records),
        "best_noisy_cost": best_cost,
        "best_first_seen_episode": next(item["episode"] for item in records
                                          if float(item["best_cost"]) == best_cost),
        "final_noisy_cost": float(final["current_cost"]),
        "final_deterministic_cost": deterministic_final,
        "deterministic_gap_over_best_fraction": ((deterministic_final / best_cost) - 1.0)
                                                   if best_cost else None,
        "tail_noisy_cost": describe(item["current_cost"] for item in tail),
        "tail_deterministic_cost": describe(item["deterministic_cost"] for item in tail),
        "tail_collision_repairs": describe(item["collision_repairs"] for item in tail),
        "tail_deterministic_collision_repairs": describe(
            item.get("deterministic_collision_repairs") for item in tail),
        "tail_unique_intended_cores": describe(
            item.get("unique_intended_cores") for item in tail),
        "tail_deterministic_unique_intended_cores": describe(
            item.get("deterministic_unique_intended_cores") for item in tail),
        "tail_actor_loss": describe(item.get("actor_loss_mean") for item in tail),
        "tail_critic_loss": describe(item.get("critic_loss_mean") for item in tail),
        "final_noise_scale": final["noise_scale"],
        "assessment": "deterministic_policy_near_best"
                      if best_cost and deterministic_final <= 1.01 * best_cost
                      else "best_result_not_reproduced_by_deterministic_policy",
    }


def analyze_directory(directory):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    architecture = summary.get("config", {}).get("agent_arch", "unknown")
    diagnostics = []
    for run in summary.get("runs", []):
        if run.get("algorithm") != "ddpg":
            continue
        path = directory / f"ddpg-seed{run['seed']}.jsonl"
        if not path.exists() and run.get("diagnostics"):
            candidate = Path(run["diagnostics"])
            path = candidate if candidate.is_absolute() else directory / candidate.name
        diagnostics.append({"seed": run["seed"], "path": str(path),
                            **diagnose(load_jsonl(path))})
    aggregates = summary.get("aggregates", {})
    ddpg_mean = aggregates.get("ddpg", {}).get("mean_best_cost")
    comparisons = {}
    if ddpg_mean is not None:
        for baseline in ("bs", "random", "sa"):
            baseline_mean = aggregates.get(baseline, {}).get("mean_best_cost")
            if baseline_mean:
                comparisons[f"ddpg_reduction_vs_{baseline}_mean_fraction"] = 1.0 - ddpg_mean / baseline_mean
    return {"directory": str(directory), "agent_arch": architecture,
            "aggregates": aggregates, "comparisons": comparisons,
            "ddpg_diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/experiment-analysis.json"))
    args = parser.parse_args()
    analyses = [analyze_directory(path) for path in args.directories]
    output = {"experiments": analyses}
    if len(analyses) > 1:
        means = {item["agent_arch"]: item["aggregates"].get("ddpg", {}).get("mean_best_cost")
                 for item in analyses}
        output["ddpg_mean_best_cost_by_architecture"] = means
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
