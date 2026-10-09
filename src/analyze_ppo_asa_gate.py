"""Validate and summarize a paired PPO-guided ASA experiment."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    with path.open() as stream:
        return json.load(stream)


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_metrics(metadata):
    metrics = metadata.get("final_update_metrics") or {}
    return bool(metrics) and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in metrics.values())


def objective_evaluation_count(row, multiple_chains):
    """Include initialization calls omitted by diagnostic proposal counts."""
    initializations = row.get("chain", 1) if multiple_chains else 1
    return row["evaluations"] + initializations


def align_curves_by_evaluations(left, right, left_multichain=False,
                                right_multichain=False):
    """Return rows recorded at identical true-objective evaluation counts."""
    left_by_evaluation = {
        objective_evaluation_count(row, left_multichain): row for row in left
    }
    right_by_evaluation = {
        objective_evaluation_count(row, right_multichain): row for row in right
    }
    shared = sorted(left_by_evaluation.keys() & right_by_evaluation.keys())
    return [
        (left_by_evaluation[evaluation], right_by_evaluation[evaluation])
        for evaluation in shared
    ]


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
        "shared_proposal_focus": (
            trained_meta.get("proposal_focus") ==
            control_meta.get("proposal_focus")),
        "shared_restart_protocol": (
            trained_meta.get("restart_interval", 0) ==
            control_meta.get("restart_interval", 0) and
            trained_meta.get("independent_chains", 1) ==
            control_meta.get("independent_chains", 1) and
            trained_meta.get("initial_placement_evaluations", 1) ==
            control_meta.get("initial_placement_evaluations", 1)),
    }
    mechanics_pass = all(mechanics.values())
    trained_cost = trained["best_cost"]
    control_cost = control["best_cost"]
    asa_cost = asa["best_cost"]
    learning_pass = trained_cost < control_cost
    optimizer_pass = trained_cost < asa_cost
    trained_curve = load_jsonl(args.directory / "ppo-asa-trained.jsonl")
    control_curve = load_jsonl(args.directory / "ppo-asa-control.jsonl")
    asa_curve = load_jsonl(args.directory / "asa.jsonl")
    trained_control_pairs = align_curves_by_evaluations(
        trained_curve, control_curve,
        trained_meta.get("restart_interval", 0) > 0,
        control_meta.get("restart_interval", 0) > 0)
    trained_asa_pairs = align_curves_by_evaluations(
        trained_curve, asa_curve,
        trained_meta.get("restart_interval", 0) > 0, False)
    entropies = [
        row["entropy"] for row in trained_curve
        if isinstance(row.get("entropy"), (int, float))
        and math.isfinite(row["entropy"])
    ]
    candidate_count = trained_meta["proposal_candidates_per_step"]
    uniform_entropy = math.log(candidate_count)
    policy_curve = {
        "trained_control_aligned_evaluation_points": len(
            trained_control_pairs),
        "trained_asa_aligned_evaluation_points": len(trained_asa_pairs),
        "trained_better_than_control_points": sum(
            trained_row["best_cost"] < control_row["best_cost"]
            for trained_row, control_row in trained_control_pairs),
        "trained_better_than_asa_points": sum(
            trained_row["best_cost"] < asa_row["best_cost"]
            for trained_row, asa_row in trained_asa_pairs),
        "trained_neutral_proposal_percent": (
            trained_meta["neutral_proposals"] /
            trained_meta["candidate_evaluations"] * 100),
        "control_neutral_proposal_percent": (
            control_meta["neutral_proposals"] /
            control_meta["candidate_evaluations"] * 100),
        "trained_improving_move_percent": (
            trained_meta["improving_moves"] /
            trained_meta["candidate_evaluations"] * 100),
        "control_improving_move_percent": (
            control_meta["improving_moves"] /
            control_meta["candidate_evaluations"] * 100),
    }
    if entropies:
        policy_curve.update({
            "uniform_candidate_entropy": uniform_entropy,
            "first_reported_entropy": entropies[0],
            "final_reported_entropy": entropies[-1],
            "final_entropy_gap_from_uniform_percent": (
                (uniform_entropy - entropies[-1]) /
                uniform_entropy * 100),
        })
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
            "trained_independent_chains": trained_meta.get(
                "independent_chains", 1),
            "control_independent_chains": control_meta.get(
                "independent_chains", 1),
            "trained_improving_moves": trained_meta["improving_moves"],
            "control_improving_moves": control_meta["improving_moves"],
            "asa_improving_moves": asa_meta["improving_moves"],
            "trained_neutral_proposals": trained_meta["neutral_proposals"],
            "control_neutral_proposals": control_meta["neutral_proposals"],
            "trained_final_update_metrics": trained_meta.get(
                "final_update_metrics"),
            "policy_curve": policy_curve,
        },
    }
    destination = args.directory / "gate-summary.json"
    with destination.open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
