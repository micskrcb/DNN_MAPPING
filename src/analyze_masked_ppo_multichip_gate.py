#!/usr/bin/env python3
"""Analyze the one-seed multi-chip PPO/control/RS/ASA gate."""

import argparse
import json
import math
import os
import statistics


def load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = args.directory

    trained_records = load_jsonl(os.path.join(directory, "ppo-trained.jsonl"))
    control_records = load_jsonl(os.path.join(directory, "ppo-control.jsonl"))
    reports = {
        name: load_json(os.path.join(directory, f"{name}-report.json"))
        for name in ("ppo-trained", "ppo-control", "random", "asa", "bs")
    }
    trained_meta = reports["ppo-trained"]["algorithm_metadata"]
    control_meta = reports["ppo-control"]["algorithm_metadata"]
    trained_late = [float(row["deterministic_cost"])
                    for row in trained_records[-min(5, len(trained_records)):]]
    control_late = [float(row["deterministic_cost"])
                    for row in control_records[-min(5, len(control_records)):]]
    trained_late_mean = statistics.fmean(trained_late)
    control_late_mean = statistics.fmean(control_late)
    finite = all(
        value is None or math.isfinite(float(value))
        for row in trained_records
        for key in ("policy_loss", "value_loss", "entropy", "approximate_kl",
                    "clip_fraction", "gradient_norm")
        for value in (row.get(key),))
    mechanics = (
        math.isclose(float(trained_meta["initial_deterministic_cost"]),
                     float(control_meta["initial_deterministic_cost"]),
                     rel_tol=0.0, abs_tol=1e-15)
        and trained_meta["collision_repairs"] == 0
        and control_meta["collision_repairs"] == 0
        and finite
    )
    best = {name: float(report["best_cost"]) for name, report in reports.items()}
    learning_pass = trained_late_mean < control_late_mean
    optimizer_pass = best["ppo-trained"] < min(best["random"], best["asa"])
    if not mechanics:
        verdict = "MECHANICS_FAIL"
    elif learning_pass and optimizer_pass:
        verdict = "ONE_SEED_MULTICHIP_GATE_PASS"
    elif learning_pass:
        verdict = "POLICY_LEARNS_BUT_OPTIMIZER_GATE_FAILS"
    else:
        verdict = "NO_MULTICHIP_LEARNING_SIGNAL"
    result = {
        "verdict": verdict,
        "mechanics_pass": mechanics,
        "learning_pass": learning_pass,
        "optimizer_pass_against_random_and_asa": optimizer_pass,
        "initial_deterministic_cost": trained_meta["initial_deterministic_cost"],
        "trained_late_deterministic_mean": trained_late_mean,
        "control_late_deterministic_mean": control_late_mean,
        "late_deterministic_improvement": control_late_mean - trained_late_mean,
        "late_deterministic_improvement_percent":
            100.0 * (control_late_mean - trained_late_mean) / control_late_mean,
        "best_costs": best,
        "actual_complete_placement_evaluations": {
            "ppo-trained": reports["ppo-trained"]["complete_placement_evaluations"],
            "ppo-control": reports["ppo-control"]["complete_placement_evaluations"],
            "random": reports["random"]["complete_placement_evaluations"],
            "asa": reports["asa"]["complete_placement_evaluations"] + 1,
            "bs": 1,
        },
        "note": "ASA reports proposal count; its initialization is one additional complete placement.",
    }
    output = os.path.join(directory, "gate-summary.json")
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
