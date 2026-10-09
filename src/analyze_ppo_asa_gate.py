"""Validate and summarize a paired PPO-guided ASA experiment."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    with path.open() as stream:
        return json.load(stream)


def finite_metrics(metadata):
    metrics = metadata.get("final_update_metrics") or {}
    return bool(metrics) and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in metrics.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    trained = load(args.directory / "ppo-asa-trained-report.json")
    control = load(args.directory / "ppo-asa-control-report.json")
    asa = load(args.directory / "asa-report.json")
    trained_meta = trained["algorithm_metadata"]
    control_meta = control["algorithm_metadata"]
    asa_meta = asa["algorithm_metadata"]

    initial_match = math.isclose(
        trained_meta["initial_cost"], control_meta["initial_cost"],
        rel_tol=0.0, abs_tol=0.0)
    matched_totals = (
        trained_meta["total_objective_evaluations"] ==
        control_meta["total_objective_evaluations"] ==
        asa_meta["candidate_evaluations"] + 1)
    mechanics = {
        "initial_placements_match_by_cost": initial_match,
        "matched_total_objective_evaluations": matched_totals,
        "trained_updates_are_finite": finite_metrics(trained_meta),
        "trained_update_count_positive": trained_meta["update_count"] > 0,
        "control_update_count_zero": control_meta["update_count"] == 0,
        "control_learning_disabled": not control_meta["learning_enabled"],
        "fixed_metropolis_acceptance": (
            trained_meta["acceptance_rule"] == "fixed_metropolis" and
            control_meta["acceptance_rule"] == "fixed_metropolis"),
        "shared_temperature_controller": (
            trained_meta["temperature_controller"] ==
            control_meta["temperature_controller"] ==
            "existing_adaptive_sa"),
    }
    mechanics_pass = all(mechanics.values())
    trained_cost = trained["best_cost"]
    control_cost = control["best_cost"]
    asa_cost = asa["best_cost"]
    learning_pass = trained_cost < control_cost
    optimizer_pass = trained_cost < asa_cost
    summary = {
        "verdict": (
            "ONE_SEED_LEARNING_AND_OPTIMIZER_PASS"
            if mechanics_pass and learning_pass and optimizer_pass else
            "ONE_SEED_LEARNING_ONLY" if mechanics_pass and learning_pass else
            "NO_LEARNING_SIGNAL" if mechanics_pass else
            "MECHANICS_FAILURE"),
        "mechanics_pass": mechanics_pass,
        "mechanics": mechanics,
        "costs_microseconds": {
            "ppo_asa_trained": trained_cost * 1e6,
            "ppo_asa_uniform_control": control_cost * 1e6,
            "ordinary_asa": asa_cost * 1e6,
        },
        "trained_improvement_over_control_percent": (
            (control_cost - trained_cost) / control_cost * 100),
        "trained_improvement_over_asa_percent": (
            (asa_cost - trained_cost) / asa_cost * 100),
        "learning_pass_against_uniform_control": learning_pass,
        "optimizer_pass_against_ordinary_asa": optimizer_pass,
        "objective_evaluations": {
            "ppo_asa_trained": trained_meta["total_objective_evaluations"],
            "ppo_asa_control": control_meta["total_objective_evaluations"],
            "ordinary_asa": asa_meta["candidate_evaluations"] + 1,
        },
        "runtime_seconds": {
            "ppo_asa_trained": trained["seconds_elapsed"],
            "ppo_asa_control": control["seconds_elapsed"],
            "ordinary_asa": asa["seconds_elapsed"],
        },
        "proposal_diagnostics": {
            "trained_updates": trained_meta["update_count"],
            "trained_improving_moves": trained_meta["improving_moves"],
            "control_improving_moves": control_meta["improving_moves"],
            "asa_improving_moves": asa_meta["improving_moves"],
            "trained_neutral_proposals": trained_meta["neutral_proposals"],
            "control_neutral_proposals": control_meta["neutral_proposals"],
            "trained_final_update_metrics": trained_meta.get(
                "final_update_metrics"),
        },
    }
    destination = args.directory / "gate-summary.json"
    with destination.open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
