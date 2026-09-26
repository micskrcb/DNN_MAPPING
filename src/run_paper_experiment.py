"""Run the closest documented paper-mode CONV and FC experiment suite.

Defaults follow the published placement budgets: 30 placements per epoch,
300,000 DDPG placements, a one-million-placement reward baseline, and one
million RS/SA placements. Per-layer partitions, physical regions, GRS gateway
placement and CONV block count include documented reconstruction assumptions;
see NEXT_STEPS.md and PROJECT_STATE.md before interpreting results.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def git_provenance(root):
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            text=True, capture_output=True).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def build_commands(args, output_dir):
    commands = []
    for region in ("conv", "fc"):
        commands.append([
            sys.executable, "src/run_multiseed_experiment.py",
            "--device", args.device,
            "--jobs", str(args.jobs),
            "--agent_arch", "paper_cnn",
            "--reward_mode", "sparse",
            "--seeds", args.seeds,
            "--epochs", str(args.epochs),
            "--placements_per_epoch", str(args.placements_per_epoch),
            "--baseline_trials", str(args.baseline_trials),
            "--search_budget", str(args.search_budget),
            "--model", args.model,
            "--partition_mode", "paper_targets",
            "--workload_region", region,
            "--timing_model", "paper_pipeline",
            "--routing_model", "paper_xy",
            "--conv_blocks", str(args.conv_blocks),
            "--chips_x", "4", "--chips_y", "4",
            "--rows", "16", "--cols", "16",
            "--batch_z", str(args.batch_z),
            "--train_every", "1",
            "--diagnostics_every", str(args.diagnostics_every),
            "--algorithms", args.algorithms,
            "--hybrid_ddpg_fraction", str(args.hybrid_ddpg_fraction),
            "--output_dir", str(output_dir / f"{args.model}-{region}"),
        ])
        if args.cpu_threads is not None:
            commands[-1].extend(["--cpu_threads", str(args.cpu_threads)])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/paper-reproduction")
    parser.add_argument("--model", choices=["alexnet", "vgg16", "resnet50"],
                        default="alexnet")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=positive, default=1,
                        help="Concurrent multiseed subprocesses")
    parser.add_argument("--cpu_threads", type=positive, default=None,
                        help="Threads per CPU subprocess; defaults to CPUs divided by jobs")
    parser.add_argument("--diagnostics_every", type=positive, default=100)
    parser.add_argument("--algorithms", default="bs,ddpg,random,sa,asa,ddpg_asa")
    parser.add_argument("--hybrid_ddpg_fraction", type=float, default=0.8)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=positive, default=10_000)
    parser.add_argument("--placements_per_epoch", type=positive, default=30)
    parser.add_argument("--baseline_trials", type=positive, default=1_000_000)
    parser.add_argument("--search_budget", type=positive, default=1_000_000)
    parser.add_argument("--batch_z", type=positive, default=3,
                        help="Unpublished by the paper; default retains the documented project assumption")
    parser.add_argument("--conv_blocks", type=positive, default=4,
                        help="Unpublished for evaluated workloads; default follows Figure 7's illustration")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hybrid_ddpg_fraction < 1:
        parser.error("--hybrid_ddpg_fraction must be in (0,1)")

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args, output_dir)
    manifest = {
        "config": vars(args),
        "git": git_provenance(root),
        "python_version": sys.version,
        "ddpg_complete_placements_per_seed": args.epochs * args.placements_per_epoch,
        "commands": commands,
        "published": {
            "placements_per_epoch": 30,
            "ddpg_reported_convergence_placements": "approximately 300000-400000",
            "random_search_placements": 1_000_000,
            "simulated_annealing_placements": "approximately 1000000",
        },
        "research_extension": {
            "asa": "temperature and neighborhood adapt from acceptance and stagnation",
            "ddpg_asa": "DDPG best placement warm-starts ASA under a matched combined budget",
            "status": "extension; not claimed as part of the reproduced source paper",
        },
        "reconstruction_assumptions": [
            "per-layer M/N grids reconstructed from Figure-6 aggregate counts and MAC balance",
            "CONV and FC regions use the minimum number of contiguous chip-major whole chips",
            "inter-chip traffic uses a lower-left Figure-3 chip-periphery gateway",
            "CONV block count is configurable because workload-specific values are unpublished",
            "residual addition executes in the destination transformation/VVA path",
            "batch_z, OU parameters, replay capacity, padding, LRN and target-update details are unpublished",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for command in commands:
        print("Running:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
