#!/usr/bin/env python3
"""Compare paired guided-DDPG and no-learning diagnostic logs."""

import argparse
import glob
import json
import math
import os
import statistics


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(records):
    deterministic = [float(row["deterministic_cost"]) for row in records]
    late = deterministic[-min(5, len(deterministic)):]
    q_values = [float(row["q_abs_max"]) for row in records
                if row.get("q_abs_max") is not None]
    saturation = [float(row["proto_action_saturated_fraction"])
                  for row in records[-min(5, len(records)):]
                  if row.get("proto_action_saturated_fraction") is not None]
    finite_training = all(
        value is None or math.isfinite(float(value))
        for row in records
        for key in ("actor_loss_mean", "critic_loss_mean", "q_abs_max",
                    "critic_grad_norm_max", "actor_grad_norm_max")
        for value in (row.get(key),)
    )
    return {
        "final_deterministic_cost": deterministic[-1],
        "best_deterministic_cost": min(deterministic),
        "late_deterministic_mean": statistics.fmean(late),
        "maximum_absolute_q": max(q_values) if q_values else None,
        "late_proto_saturation_mean":
            statistics.fmean(saturation) if saturation else None,
        "zero_collision_repairs": all(
            row.get("collision_repairs") == 0 and
            row.get("deterministic_collision_repairs") == 0
            for row in records),
        "finite_training_diagnostics": finite_training,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    trained_paths = sorted(glob.glob(os.path.join(
        args.directory, "seed*-trained.jsonl")))
    if not trained_paths:
        raise SystemExit(f"No seed*-trained.jsonl files found in {args.directory}")

    pairs = []
    for trained_path in trained_paths:
        control_path = trained_path.replace("-trained.jsonl", "-control.jsonl")
        if not os.path.exists(control_path):
            raise SystemExit(f"Missing paired control: {control_path}")
        trained, control = read_jsonl(trained_path), read_jsonl(control_path)
        if not trained or not control:
            raise SystemExit(f"Empty diagnostics in pair for {trained_path}")
        seed = int(os.path.basename(trained_path).split("-")[0][4:])
        trained_by_episode = {row["episode"]: row for row in trained}
        control_by_episode = {row["episode"]: row for row in control}
        warmup_episodes = sorted(
            episode for episode in trained_by_episode.keys() & control_by_episode.keys()
            if (trained_by_episode[episode].get("rollout_source") == "uniform_random" and
                control_by_episode[episode].get("rollout_source") == "uniform_random"))
        warmup_match = bool(warmup_episodes) and all(
            math.isclose(float(trained_by_episode[episode]["current_cost"]),
                         float(control_by_episode[episode]["current_cost"]),
                         rel_tol=0.0, abs_tol=1e-15)
            for episode in warmup_episodes)
        trained_summary, control_summary = summarize(trained), summarize(control)
        delta = (control_summary["late_deterministic_mean"] -
                 trained_summary["late_deterministic_mean"])
        pair = {
            "seed": seed,
            "warmup_current_costs_match": warmup_match,
            "trained": trained_summary,
            "control": control_summary,
            "late_deterministic_improvement": delta,
            "late_deterministic_improvement_percent":
                100.0 * delta / control_summary["late_deterministic_mean"],
        }
        pairs.append(pair)

    deltas = [pair["late_deterministic_improvement"] for pair in pairs]
    mechanics_pass = all(
        pair["warmup_current_costs_match"] and
        pair["trained"]["zero_collision_repairs"] and
        pair["control"]["zero_collision_repairs"] and
        pair["trained"]["finite_training_diagnostics"]
        for pair in pairs)
    stability_pass = all(
        pair["trained"]["finite_training_diagnostics"] and
        (pair["trained"]["maximum_absolute_q"] is None or
         pair["trained"]["maximum_absolute_q"] <= 100.0) and
        (pair["trained"]["late_proto_saturation_mean"] is None or
         pair["trained"]["late_proto_saturation_mean"] < 0.95)
        for pair in pairs)
    aggregate = {
        "paired_seeds": len(pairs),
        "mechanics_pass": mechanics_pass,
        "stability_pass": stability_pass,
        "trained_better_seed_count": sum(delta > 0 for delta in deltas),
        "mean_late_deterministic_improvement": statistics.fmean(deltas),
    }
    if len(deltas) >= 2:
        critical_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                       6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        t_value = critical_95.get(len(deltas), 1.96)
        half_width = t_value * statistics.stdev(deltas) / math.sqrt(len(deltas))
        mean = aggregate["mean_late_deterministic_improvement"]
        aggregate["exploratory_paired_95_interval"] = [
            mean - half_width, mean + half_width]

    if not mechanics_pass:
        verdict = "MECHANICS_FAIL"
    elif not stability_pass:
        verdict = "TRAINING_UNSTABLE_OR_COLLAPSED"
    elif len(pairs) < 3:
        verdict = ("ONE_SEED_POSITIVE_SIGNAL" if deltas[0] > 0 else
                   "ONE_SEED_NO_LEARNING_SIGNAL")
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
