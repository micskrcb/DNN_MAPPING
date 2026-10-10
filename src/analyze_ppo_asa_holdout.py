"""Summarize frozen PPO-ASA holdout evaluation against uniform and ASA."""

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sample_summary(values):
    return {
        "mean_microseconds": statistics.mean(values),
        "sample_std_microseconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0),
        "minimum_microseconds": min(values),
        "maximum_microseconds": max(values),
        "individual_microseconds": values,
    }


def paired_summary(reference, candidate):
    """Positive differences mean that candidate has lower latency."""
    differences = [left - right for left, right in zip(reference, candidate)]
    mean = statistics.mean(differences)
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    # Student-t 97.5% quantiles for the supported small-sample evaluations.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    }.get(len(differences) - 1, 1.96)
    half_width = critical * std / math.sqrt(len(differences))
    return {
        "candidate_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "mean_improvement_microseconds": mean,
        "mean_improvement_percent": mean / statistics.mean(reference) * 100,
        "exploratory_paired_95_percent_interval_microseconds": [
            mean - half_width, mean + half_width],
        "individual_improvements_microseconds": differences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    trained_paths = sorted(glob.glob(str(
        args.directory / "seed-*-trained-report.json")))
    if not trained_paths:
        raise SystemExit("no holdout reports found")

    seeds = [int(Path(path).name.split("-")[1]) for path in trained_paths]
    rows = []
    for seed, trained_path in zip(seeds, trained_paths):
        trained = load(trained_path)
        control = load(args.directory / f"seed-{seed}-control-report.json")
        asa = load(args.directory / f"seed-{seed}-asa-report.json")
        trained_meta = trained["algorithm_metadata"]
        control_meta = control["algorithm_metadata"]
        asa_meta = asa["algorithm_metadata"]
        mechanics = {
            "initial_costs_match": (
                trained_meta["initial_cost"] == control_meta["initial_cost"] ==
                asa_meta["initial_cost"]),
            "matched_objective_evaluations": (
                trained_meta["total_objective_evaluations"] ==
                control_meta["total_objective_evaluations"] ==
                asa_meta["candidate_evaluations"] + 1),
            "trained_checkpoint_loaded": (
                trained_meta.get("model_initialization") ==
                "loaded_checkpoint"),
            "uniform_control_initialized_fresh": (
                control_meta.get("model_initialization") ==
                "zero_score_uniform"),
            "both_policies_frozen": (
                not trained_meta["learning_enabled"] and
                not control_meta["learning_enabled"] and
                trained_meta["update_count"] == 0 and
                control_meta["update_count"] == 0),
            "shared_proposal_focus": (
                trained_meta["proposal_focus"] ==
                control_meta["proposal_focus"]),
            "single_uninterrupted_chain": (
                trained_meta["independent_chains"] ==
                control_meta["independent_chains"] == 1 and
                trained_meta["restart_interval"] ==
                control_meta["restart_interval"] == 0),
        }
        rows.append({
            "seed": seed,
            "mechanics": mechanics,
            "mechanics_pass": all(mechanics.values()),
            "trained_microseconds": trained["best_cost"] * 1e6,
            "uniform_control_microseconds": control["best_cost"] * 1e6,
            "asa_microseconds": asa["best_cost"] * 1e6,
            "objective_evaluations": (
                trained_meta["total_objective_evaluations"]),
            "source_training_evaluations": trained_meta.get(
                "source_training_evaluations"),
        })

    trained_values = [row["trained_microseconds"] for row in rows]
    control_values = [row["uniform_control_microseconds"] for row in rows]
    asa_values = [row["asa_microseconds"] for row in rows]
    versus_control = paired_summary(control_values, trained_values)
    versus_asa = paired_summary(asa_values, trained_values)
    required_wins = math.ceil(len(rows) * 0.8)
    policy_pass = (
        versus_control["mean_improvement_microseconds"] > 0 and
        versus_control["candidate_wins"] >= required_wins)
    optimizer_pass = (
        versus_asa["mean_improvement_microseconds"] > 0 and
        versus_asa["candidate_wins"] >= required_wins)
    mechanics_pass = all(row["mechanics_pass"] for row in rows)
    if not mechanics_pass:
        verdict = "MECHANICS_FAILURE"
    elif policy_pass and optimizer_pass:
        verdict = "HOLDOUT_POLICY_AND_OPTIMIZER_PASS"
    elif policy_pass:
        verdict = "HOLDOUT_POLICY_ONLY"
    else:
        verdict = "NO_HOLDOUT_POLICY_SIGNAL"
    summary = {
        "verdict": verdict,
        "mechanics_pass": mechanics_pass,
        "holdout_seeds": seeds,
        "required_pair_wins": required_wins,
        "policy_generalization_pass": policy_pass,
        "optimizer_pass_against_asa": optimizer_pass,
        "trained_policy": sample_summary(trained_values),
        "uniform_control": sample_summary(control_values),
        "ordinary_asa": sample_summary(asa_values),
        "trained_versus_uniform_control": versus_control,
        "trained_versus_asa": versus_asa,
        "pairs": rows,
    }
    destination = args.directory / "holdout-summary.json"
    with destination.open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
