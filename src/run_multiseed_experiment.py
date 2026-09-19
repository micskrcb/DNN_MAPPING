"""Run BS plus matched-budget DDPG, random-search, and SA across seeds.

Example (on a CUDA host):
  python src/run_multiseed_experiment.py --device cuda --seeds 0,1,2,3,4

DDPG evaluates ``--epochs * --placements_per_epoch`` complete placements.
RS and SA use ``--search_budget`` or the same count when it is omitted. BS is a
single deterministic sequential placement.
DDPG additionally uses ``--baseline_trials`` random placements only to form its
fixed reward normalizer; that work is reported separately and is not treated as
an evaluation in the comparison.  Outputs are JSON reports, DDPG JSONL
diagnostics, text logs, and a summary.json in --output_dir.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/multiseed")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated integer seeds")
    parser.add_argument("--epochs", type=positive, default=1000,
                        help="Declared DDPG epochs per seed")
    parser.add_argument("--placements_per_epoch", type=positive, default=1,
                        help="Complete DDPG placements per epoch; paper states 30")
    parser.add_argument("--baseline_trials", type=positive, default=1000,
                        help="DDPG-only random placements used to normalize sparse reward")
    parser.add_argument("--search_budget", type=positive, default=None,
                        help="RS/SA placements per seed; defaults to the DDPG complete-placement budget")
    parser.add_argument("--model", default="alexnet")
    parser.add_argument("--channels_per_partition", type=positive, default=512)
    parser.add_argument("--partition_mode", choices=["uniform", "paper_targets"],
                        default="uniform")
    parser.add_argument("--workload_region", choices=["all", "conv", "fc"], default="all")
    parser.add_argument("--timing_model", choices=["full_frame", "paper_pipeline"],
                        default="full_frame")
    parser.add_argument("--routing_model", choices=["legacy_distance", "paper_xy"],
                        default="legacy_distance")
    parser.add_argument("--conv_blocks", type=positive, default=4)
    parser.add_argument("--chips_x", type=positive, default=4)
    parser.add_argument("--chips_y", type=positive, default=4)
    parser.add_argument("--rows", type=positive, default=16)
    parser.add_argument("--cols", type=positive, default=16)
    parser.add_argument("--batch_z", type=positive, default=3)
    parser.add_argument("--train_every", type=positive, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--agent_arch", choices=["mlp", "cnn", "paper_cnn"], default="mlp")
    parser.add_argument("--reward_mode", choices=["sparse", "potential"], default="sparse")
    args = parser.parse_args()
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    except ValueError as error:
        parser.error(f"--seeds must be comma-separated integers: {error}")
    if not seeds:
        parser.error("--seeds must contain at least one integer")

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = ["--use_cnn", "--model", args.model,
              "--channels_per_partition", str(args.channels_per_partition),
              "--partition_mode", args.partition_mode,
              "--timing_model", args.timing_model,
              "--routing_model", args.routing_model,
              "--workload_region", args.workload_region,
              "--conv_blocks", str(args.conv_blocks),
              "--chips_x", str(args.chips_x), "--chips_y", str(args.chips_y),
              "--rows", str(args.rows), "--cols", str(args.cols)]
    results = []
    placement_budget = args.epochs * args.placements_per_epoch
    search_budget = args.search_budget or placement_budget
    for seed in seeds:
        for algorithm in ("bs", "ddpg", "random", "sa"):
            stem = f"{algorithm}-seed{seed}"
            report = output_dir / f"{stem}.json"
            command = [sys.executable, "src/run_multi_chip.py", "--algo", algorithm,
                       "--seed", str(seed), "--report", str(report), *common]
            if algorithm == "ddpg":
                command.extend(["--device", args.device, "--epochs", str(args.epochs),
                                "--placements_per_epoch", str(args.placements_per_epoch),
                                "--baseline_trials", str(args.baseline_trials),
                                "--batch_z", str(args.batch_z), "--train_every", str(args.train_every),
                                "--agent_arch", args.agent_arch,
                                "--reward_mode", args.reward_mode,
                                "--diagnostics", str(output_dir / f"{stem}.jsonl"),
                                "--save_checkpoint", str(output_dir / f"{stem}.pt")])
            elif algorithm != "bs":
                command.extend(["--iters", str(search_budget)])
            print("Running:", " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            (output_dir / f"{stem}.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{algorithm} seed {seed} failed; see {output_dir / (stem + '.log')}")
            payload = json.loads(report.read_text())
            results.append({"algorithm": algorithm, "seed": seed,
                            "best_cost": payload["best_cost"],
                            "seconds_elapsed": payload["seconds_elapsed"],
                            "report": str(report),
                            "diagnostics": payload.get("ddpg_metadata", {}).get("diagnostics_path")
                            if payload.get("ddpg_metadata") else None})

    aggregates = {}
    for algorithm in ("bs", "ddpg", "random", "sa"):
        costs = [item["best_cost"] for item in results if item["algorithm"] == algorithm]
        aggregates[algorithm] = {"runs": len(costs), "mean_best_cost": sum(costs) / len(costs),
                                 "sample_std_best_cost": statistics.stdev(costs) if len(costs) > 1 else 0.0,
                                 "min_best_cost": min(costs), "max_best_cost": max(costs)}
    bs_cost = aggregates["bs"]["mean_best_cost"]
    if bs_cost > 0:
        for values in aggregates.values():
            values["mean_objective_normalized_to_bs"] = values["mean_best_cost"] / bs_cost
            values["inverse_objective_ratio_to_bs"] = bs_cost / values["mean_best_cost"]
    summary = {"config": vars(args), "comparison": {
        "ddpg_complete_placement_evaluations": placement_budget,
        "random_and_sa_placement_evaluations": search_budget,
        "ddpg_reward_normalizer_random_trials": args.baseline_trials,
        "note": "DDPG baseline trials are reported separately and do not make budgets identical."},
        "runs": results, "aggregates": aggregates}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
