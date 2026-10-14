"""Summarize the oracle, planted-signal, and real-reward PPO-ASA gates."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_interval(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "interval_95": [None, None], "count": 0}
    mean = statistics.mean(values)
    half_width = (1.96 * statistics.stdev(values) / math.sqrt(len(values))
                  if len(values) > 1 else 0.0)
    return {
        "mean": mean,
        "interval_95": [mean - half_width, mean + half_width],
        "count": len(values),
    }


def late_summary(path):
    rows = load_jsonl(path)
    max_update = max(int(row.get("update_count") or 0) for row in rows)
    threshold = max_update * 0.8
    late = [row for row in rows
            if int(row.get("update_count") or 0) >= threshold]
    proxy = mean_interval([
        row.get("mean_proxy_advantage_over_uniform") for row in late])
    last = rows[-1]
    return {
        "maximum_update_count": max_update,
        "late_window_count": len(late),
        "late_proxy_advantage_over_uniform": proxy,
        "final_entropy": last.get("entropy"),
        "final_logit_std": last.get("mean_logit_std"),
        "final_logit_range": last.get("mean_logit_range"),
        "final_probability_l1_from_uniform": last.get(
            "mean_probability_l1_from_uniform"),
        "final_uniform_action_match_fraction": last.get(
            "uniform_action_match_fraction"),
        "final_score_head_weight_norm": last.get("score_head_weight_norm"),
        "final_policy_head_gradient_norm": last.get(
            "policy_head_gradient_norm"),
        "final_value_head_gradient_norm": last.get(
            "value_head_gradient_norm"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-split-pass", action="store_true")
    args = parser.parse_args()
    directory = args.directory

    summary = {"oracle": load_json(directory / "oracle-summary.json")}
    planted = {}
    for style in ("shared", "split"):
        trained = late_summary(
            directory / f"planted-{style}-trained.jsonl")
        control = late_summary(
            directory / f"planted-{style}-control.jsonl")
        trained_proxy = trained["late_proxy_advantage_over_uniform"]
        control_proxy = control["late_proxy_advantage_over_uniform"]
        passed = bool(
            trained_proxy["interval_95"][0] is not None and
            trained_proxy["interval_95"][0] > 0 and
            trained_proxy["mean"] > control_proxy["mean"])
        planted[style] = {
            "trained": trained,
            "control": control,
            "pass": passed,
            "criterion": (
                "trained late proxy advantage has a positive window-level "
                "95% interval and exceeds its uniform control"),
        }
    summary["planted_signal"] = planted

    real_diagnostics = directory / "real-split-trained.jsonl"
    if real_diagnostics.exists():
        summary["real_split_pilot"] = late_summary(real_diagnostics)
    holdout = directory / "real-holdout" / "holdout-summary.json"
    if holdout.exists():
        summary["real_split_holdout"] = load_json(holdout)

    destination = directory / "diagnostic-gate-summary.json"
    with destination.open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved {destination}")

    if args.require_split_pass and not planted["split"]["pass"]:
        print("Split PPO failed the planted-signal gate; real-reward pilot "
              "skipped.", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
