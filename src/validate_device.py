"""Bounded device validation and component timings, not a convergence benchmark.

Example: python src/validate_device.py --device cuda --output runs/h100.json
Runs the maintained test suites, then profiles real mapper/replay/DDPG operations.
CUDA timers synchronize each component; timings include synchronization overhead.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run(args, report):
    import torch
    import run_multi_chip as rm
    from compute_model import compute_seconds, edge_bytes
    from multi_chip_environment import MultiChipEnvironment

    report["torch_version"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback performed")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["device"] = {"name": props.name, "visible_memory_bytes": props.total_memory,
                            "compute_capability": [props.major, props.minor]}
    else:
        report["device"] = {"name": "cpu", "visible_memory_bytes": None}

    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True)
    report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True)
    report["working_tree_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    report["test_suites"] = []
    for suffix in [["src/test_multi_chip.py"],
                   ["-m", "unittest", "discover", "-s", "src", "-p", "test_reconciliation.py", "-v"]]:
        print("Running", " ".join(suffix), flush=True)
        result = subprocess.run([sys.executable] + suffix, cwd=root, text=True,
                                capture_output=True, timeout=args.test_timeout,
                                env={**os.environ, "OMP_NUM_THREADS": str(args.cpu_threads)})
        report["test_suites"].append({"command": suffix, "returncode": result.returncode,
                                      "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("A test suite failed; see test_suites in the report")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    setup_started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()) as extraction_log:
        graph, n, labels, ops, kinds = rm.extract_model_task_graph(
            args.model, args.channels_per_partition, return_work=True)
    report["extraction_log"] = extraction_log.getvalue()
    env = MultiChipEnvironment(args.chips_x, args.chips_y, args.rows, args.cols,
                               on_chip_latency=1/64e9, off_chip_latency=1/100e9,
                               task_graph=edge_bytes(graph, kinds), num_tasks=n,
                               compute_latency=compute_seconds(ops, kinds), timing_units="seconds")
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = rm.run_random(env, args.baseline_trials)
    mapper = rm.MultiChipCoreMapper(env, baseline_latency=baseline, batch_z=args.batch_z)
    batch_z = min(args.batch_z, n)
    mapper.batch_z = batch_z
    agent = rm.DDPGAgent(env.total_cores+n, 2*batch_z, device=args.device)
    replay = rm.ReplayBuffer(args.replay_capacity)
    initial_actor = [p.detach().clone() for p in agent.actor.parameters()]
    initial_critic = [p.detach().clone() for p in agent.critic.parameters()]
    report["setup_seconds"] = time.perf_counter() - setup_started
    report["tasks"] = n
    report["physical_cores"] = env.total_cores
    report["baseline_cost_seconds"] = baseline
    report["objective"] = "full_frame maximum task service-time approximation"
    report["timing_method"] = "Synchronized component wall time; warmup updates separate; includes instrumentation overhead"
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = {key: [] for key in ["action", "placement", "state", "replay_add", "warmup_update", "update"]}
    def measured(key, fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings[key].append(time.perf_counter() - start)
        return value

    updates, steps = 0, 0
    best_cost = baseline
    started = time.perf_counter()
    for episode in range(args.episodes):
        state = mapper.reset()
        agent.ou_state.fill(0)
        done = False
        while not done:
            action = measured("action", lambda: agent.select_action(state, noise_scale=0.1))
            if not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                raise RuntimeError("Invalid action")
            reward, done, _, cost = measured("placement", lambda: mapper.step(action))
            next_state = measured("state", mapper._occ_map)
            measured("replay_add", lambda: replay.add(state, action, reward, next_state, done))
            steps += 1
            if len(replay) >= 64 and steps % args.train_every == 0:
                key = "warmup_update" if updates < args.warmup_updates else "update"
                measured(key, lambda: agent.train(replay, batch_size=64))
                updates += 1
            state = next_state
        placement = mapper.get_placement()
        if len(set(placement.tolist())) != n or np.any(placement < 0) or np.any(placement >= env.total_cores):
            raise RuntimeError("Placement is not a valid one-task-per-core assignment")
        if not math.isfinite(cost) or cost < 0:
            raise RuntimeError("Invalid objective")
        if not math.isclose(reward, math.sqrt(baseline) - math.sqrt(cost), abs_tol=1e-12):
            raise RuntimeError("Terminal reward does not match objective")
        best_cost = min(best_cost, cost)
        print(f"Episode {episode+1}/{args.episodes}: {cost:.6g} seconds, updates={updates}", flush=True)
    report["training_wall_seconds"] = time.perf_counter() - started
    report["steps"] = steps
    report["updates"] = updates
    report["best_cost_seconds"] = best_cost
    report["component_timings"] = {
        key: {"count": len(values), "total_seconds": sum(values),
              "mean_seconds": sum(values)/len(values) if values else None,
              "p95_seconds": float(np.percentile(values, 95)) if values else None}
        for key, values in timings.items()}
    report["replay_entries"] = len(replay)
    report["replay_array_bytes"] = sum(
        item.nbytes for transition in replay.buffer for item in transition if isinstance(item, np.ndarray))
    if device.type == "cuda":
        report["cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    else:
        report["cuda_memory"] = None
    if updates <= args.warmup_updates:
        raise RuntimeError("Insufficient updates after warmup; increase --episodes for this workload")
    for name, before, network in [("actor", initial_actor, agent.actor), ("critic", initial_critic, agent.critic)]:
        if not all(torch.isfinite(p).all().item() for p in network.parameters()):
            raise RuntimeError(f"Non-finite {name} parameters")
        if not any(not torch.equal(b, p) for b, p in zip(before, network.parameters())):
            raise RuntimeError(f"{name} parameters did not change")
        if any(p.device.type != args.device for p in network.parameters()):
            raise RuntimeError(f"{name} is on the wrong device")

    # Agent network/optimizer round-trip. Whole-run resume is a separate check.
    with tempfile.TemporaryDirectory(prefix="dnn-agent-check-") as tmp:
        checkpoint = Path(tmp) / "agent.pt"
        torch.save(agent.state_dict(), checkpoint)
        restored = rm.DDPGAgent(env.total_cores+n, 2*batch_z, device=args.device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
        for original, loaded in [(agent.actor, restored.actor), (agent.critic, restored.critic)]:
            for a, b in zip(original.parameters(), loaded.parameters()):
                if not torch.equal(a, b):
                    raise RuntimeError("Checkpoint parameter mismatch")
    report["agent_checkpoint_roundtrip"] = "passed"
    report["limitations"] = ["Not a convergence or paper-result test", "Peak memory covers this short workload only, excludes checkpoint roundtrip",
                             "Replay byte count excludes Python overhead", "No parallel environments", "Environment remains CPU-side"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", default="runs/device-validation.json")
    parser.add_argument("--model", default="simple")
    parser.add_argument("--channels_per_partition", type=positive, default=128)
    parser.add_argument("--episodes", type=positive, default=12)
    parser.add_argument("--baseline_trials", type=positive, default=10)
    parser.add_argument("--batch_z", type=positive, default=3)
    parser.add_argument("--train_every", type=positive, default=1)
    parser.add_argument("--replay_capacity", type=positive, default=1024)
    parser.add_argument("--warmup_updates", type=positive, default=2)
    parser.add_argument("--cpu_threads", type=positive, default=2)
    parser.add_argument("--test_timeout", type=positive, default=300)
    parser.add_argument("--seed", type=int, default=0)
    for key, default in [("chips_x", 2), ("chips_y", 2), ("rows", 4), ("cols", 4)]:
        parser.add_argument("--"+key, type=positive, default=default)
    args = parser.parse_args()
    if args.replay_capacity < 64:
        parser.error("replay_capacity must be >= 64 to allow training")
    report = {"status": "failed", "config": vars(args)}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Validation {report['status']}; report: {output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
