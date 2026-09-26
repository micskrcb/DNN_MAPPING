"""Run selected BS/DDPG/RS/SA/ASA/DDPG+ASA methods across seeds.

Example (20 CPU threads split across two independent jobs):
  python src/run_multiseed_experiment.py --device cpu --jobs 2 \
      --cpu_threads 10 --seeds 0,1,2,3,4

DDPG evaluates ``--epochs * --placements_per_epoch`` complete placements.
RS, SA, and ASA use ``--search_budget`` or the same count when it is omitted.
DDPG+ASA splits the DDPG placement budget between both phases so the hybrid and
plain DDPG use the same number of optimization candidates. BS is one placement.
DDPG additionally uses ``--baseline_trials`` random placements only to form its
fixed reward normalizer; that work is reported separately and is not treated as
an evaluation in the comparison.  Outputs are JSON reports, DDPG JSONL
diagnostics, text logs, and a summary.json in --output_dir.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=positive, default=1,
                        help="Independent seed/algorithm subprocesses to run concurrently")
    parser.add_argument("--cpu_threads", type=positive, default=None,
                        help="PyTorch threads per job (default: logical CPUs divided by --jobs)")
    parser.add_argument("--diagnostics_every", type=positive, default=100)
    parser.add_argument("--algorithms", default="bs,ddpg,random,sa,asa,ddpg_asa",
                        help="Comma-separated subset of bs,ddpg,random,sa,asa,ddpg_asa")
    parser.add_argument("--hybrid_ddpg_fraction", type=float, default=0.8,
                        help="Fraction of the matched hybrid budget assigned to DDPG; ASA gets the remainder")
    parser.add_argument("--agent_arch", choices=["mlp", "cnn", "paper_cnn"], default="mlp")
    parser.add_argument("--reward_mode", choices=["sparse", "potential"], default="sparse")
    args = parser.parse_args()
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    except ValueError as error:
        parser.error(f"--seeds must be comma-separated integers: {error}")
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    algorithms = [part.strip() for part in args.algorithms.split(",") if part.strip()]
    supported = {"bs", "ddpg", "random", "sa", "asa", "ddpg_asa"}
    if not algorithms or len(set(algorithms)) != len(algorithms) or not set(algorithms) <= supported:
        parser.error(f"--algorithms must be a unique comma-separated subset of {sorted(supported)}")
    if not 0 < args.hybrid_ddpg_fraction < 1:
        parser.error("--hybrid_ddpg_fraction must be in (0,1)")

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
    hybrid_ddpg_budget = max(1, min(placement_budget - 1,
                                    round(placement_budget * args.hybrid_ddpg_fraction)))
    hybrid_asa_budget = placement_budget - hybrid_ddpg_budget
    if "ddpg_asa" in algorithms and placement_budget < 2:
        parser.error("DDPG+ASA needs a matched placement budget of at least 2")
    cpu_threads = args.cpu_threads or max(1, (os.cpu_count() or 1) // args.jobs)

    def build_run(seed, algorithm):
        stem = f"{algorithm}-seed{seed}"
        report = output_dir / f"{stem}.json"
        command = [sys.executable, "src/run_multi_chip.py", "--algo", algorithm,
                   "--seed", str(seed), "--report", str(report), *common]
        if algorithm in ("ddpg", "ddpg_asa"):
            ddpg_budget = placement_budget if algorithm == "ddpg" else hybrid_ddpg_budget
            command.extend(["--device", args.device, "--epochs", str(ddpg_budget),
                            "--placements_per_epoch", "1",
                            "--baseline_trials", str(args.baseline_trials),
                            "--batch_z", str(args.batch_z), "--train_every", str(args.train_every),
                            "--agent_arch", args.agent_arch, "--reward_mode", args.reward_mode,
                            "--diagnostics_every", str(args.diagnostics_every),
                            "--diagnostics", str(output_dir / f"{stem}-ddpg.jsonl"),
                            "--save_checkpoint", str(output_dir / f"{stem}.pt")])
            if args.device == "cpu":
                command.extend(["--cpu_threads", str(cpu_threads), "--cpu_interop_threads", "1"])
            if algorithm == "ddpg_asa":
                command.extend(["--iters", str(hybrid_asa_budget),
                                "--asa_diagnostics", str(output_dir / f"{stem}-asa.jsonl")])
        elif algorithm in ("random", "sa", "asa"):
            command.extend(["--iters", str(search_budget)])
            if algorithm == "asa":
                command.extend(["--asa_diagnostics", str(output_dir / f"{stem}-asa.jsonl")])
        return seed, algorithm, stem, report, command

    def execute(spec):
        seed, algorithm, stem, report, command = spec
        log = output_dir / f"{stem}.log"
        print("Running:", " ".join(command), flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=root, text=True,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{algorithm} seed {seed} failed; see {log}")
        payload = json.loads(report.read_text())
        ddpg = payload.get("ddpg_metadata") or {}
        return {"algorithm": algorithm, "seed": seed,
                "best_cost": payload["best_cost"],
                "seconds_elapsed": payload["seconds_elapsed"],
                "report": str(report), "diagnostics": ddpg.get("diagnostics_path")}

    specs = [build_run(seed, algorithm) for seed in seeds for algorithm in algorithms]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["seed"], algorithms.index(item["algorithm"])))

    aggregates = {}
    for algorithm in algorithms:
        costs = [item["best_cost"] for item in results if item["algorithm"] == algorithm]
        aggregates[algorithm] = {"runs": len(costs), "mean_best_cost": sum(costs) / len(costs),
                                 "sample_std_best_cost": statistics.stdev(costs) if len(costs) > 1 else 0.0,
                                 "min_best_cost": min(costs), "max_best_cost": max(costs)}
    bs_cost = aggregates.get("bs", {}).get("mean_best_cost")
    if bs_cost is not None and bs_cost > 0:
        for values in aggregates.values():
            values["mean_objective_normalized_to_bs"] = values["mean_best_cost"] / bs_cost
            values["inverse_objective_ratio_to_bs"] = bs_cost / values["mean_best_cost"]
    summary = {"config": vars(args), "comparison": {
        "ddpg_complete_placement_evaluations": placement_budget,
        "random_sa_and_asa_placement_evaluations": search_budget,
        "ddpg_asa_matched_total_evaluations": placement_budget,
        "ddpg_asa_split": {"ddpg": hybrid_ddpg_budget, "asa": hybrid_asa_budget},
        "ddpg_reward_normalizer_random_trials": args.baseline_trials,
        "note": "DDPG reward-normalizer trials are reported separately from optimization budgets."},
        "runs": results, "aggregates": aggregates}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
