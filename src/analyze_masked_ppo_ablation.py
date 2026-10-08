#!/usr/bin/env python3
"""Analyze paired trained/frozen masked-PPO experiments."""

import argparse
import glob
import json
import math
import os
import statistics


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(records):
    costs = [float(row["deterministic_cost"]) for row in records]
    late = costs[-min(5, len(costs)):]
    finite = all(
        value is None or math.isfinite(float(value))
        for row in records
        for key in ("policy_loss", "value_loss", "entropy", "approximate_kl",
                    "clip_fraction", "gradient_norm")
        for value in (row.get(key),))
    return {
        "final_deterministic_cost": costs[-1],
        "best_deterministic_cost": min(costs),
        "late_deterministic_mean": statistics.fmean(late),
        "zero_collision_repairs": all(
            row.get("collision_repairs") == 0 and
            row.get("deterministic_collision_repairs") == 0 for row in records),
        "finite_training_diagnostics": finite,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    trained_paths = sorted(glob.glob(os.path.join(args.directory, "seed*-trained.jsonl")))
    if not trained_paths:
        raise SystemExit(f"No seed*-trained.jsonl files found in {args.directory}")

    pairs = []
    for trained_path in trained_paths:
        prefix = trained_path.removesuffix("-trained.jsonl")
        control_path = prefix + "-control.jsonl"
        trained_report_path = prefix + "-trained-report.json"
        control_report_path = prefix + "-control-report.json"
        required = (control_path, trained_report_path, control_report_path)
        if not all(os.path.exists(path) for path in required):
            raise SystemExit(f"Missing paired files for {trained_path}")
        trained_records, control_records = read_jsonl(trained_path), read_jsonl(control_path)
        trained_report, control_report = load(trained_report_path), load(control_report_path)
        if not trained_records or not control_records:
            raise SystemExit(f"Empty diagnostics in pair for {trained_path}")
        trained_meta = trained_report["algorithm_metadata"]
        control_meta = control_report["algorithm_metadata"]
        trained, control = summarize(trained_records), summarize(control_records)
        delta = control["late_deterministic_mean"] - trained["late_deterministic_mean"]
        initial_match = math.isclose(
            float(trained_meta["initial_deterministic_cost"]),
            float(control_meta["initial_deterministic_cost"]),
            rel_tol=0.0, abs_tol=1e-15)
        seed = int(os.path.basename(prefix).split("-")[0][4:])
        pairs.append({
            "seed": seed, "initial_policy_costs_match": initial_match,
            "trained": trained, "control": control,
            "late_deterministic_improvement": delta,
            "late_deterministic_improvement_percent":
                100.0 * delta / control["late_deterministic_mean"],
        })

    deltas = [pair["late_deterministic_improvement"] for pair in pairs]
    mechanics = all(
        pair["initial_policy_costs_match"] and
        pair["trained"]["zero_collision_repairs"] and
        pair["control"]["zero_collision_repairs"] and
        pair["trained"]["finite_training_diagnostics"] for pair in pairs)
    aggregate = {
        "paired_seeds": len(pairs), "mechanics_pass": mechanics,
        "trained_better_seed_count": sum(delta > 0 for delta in deltas),
        "mean_late_deterministic_improvement": statistics.fmean(deltas),
    }
    if len(deltas) >= 2:
        critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(deltas), 1.96)
        half = critical * statistics.stdev(deltas) / math.sqrt(len(deltas))
        mean = aggregate["mean_late_deterministic_improvement"]
        aggregate["exploratory_paired_95_interval"] = [mean - half, mean + half]
    if not mechanics:
        verdict = "MECHANICS_FAIL"
    elif len(pairs) < 3:
        verdict = "ONE_SEED_POSITIVE_SIGNAL" if deltas[0] > 0 else "ONE_SEED_NO_LEARNING_SIGNAL"
    elif (aggregate["trained_better_seed_count"] >= math.ceil(0.8 * len(pairs)) and
          aggregate["mean_late_deterministic_improvement"] > 0 and
          aggregate.get("exploratory_paired_95_interval", [-1])[0] > 0):
        verdict = "MULTISEED_LEARNING_SIGNAL"
    else:
        verdict = "MULTISEED_INCONCLUSIVE_OR_NEGATIVE"
    result = {"verdict": verdict, "aggregate": aggregate, "pairs": pairs}
    output_path = os.path.join(args.directory, "paired-summary.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
