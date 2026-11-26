import subprocess
import sys
import os
import json
import time

PYTHON = sys.executable
BASE_FLAGS = [
    "--use_cnn",
    "--partition_mode", "paper_targets",
    "--chips_x", "4", "--chips_y", "4", "--rows", "16", "--cols", "16",
    "--routing_model", "paper_xy",
    "--timing_model", "paper_pipeline",
    "--vva_ops_per_cycle", "16",
    "--partition_vva_ops_per_cycle", "1",
    "--seed", "0",
    "--allow_flat_objective"
]

RESULTS_DIR = "paper/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

runs = [
    # (label, model, region, algo, iters, outfile)
    ("AlexNet-CONV RS (10k)",   "alexnet",  "conv", "random", 10000,  "alexnet_conv_rs_10k.json"),
    ("AlexNet-CONV SA (100k)",  "alexnet",  "conv", "sa",     100000, "alexnet_conv_sa_100k.json"),
    ("ResNet50-CONV RS (5k)",   "resnet50", "conv", "random", 5000,   "resnet50_conv_rs_5k.json"),
    ("ResNet50-CONV SA (100k)", "resnet50", "conv", "sa",     100000, "resnet50_conv_sa_100k.json"),
    ("ResNet50-FC RS (10k)",    "resnet50", "fc",   "random", 10000,  "resnet50_fc_rs_10k.json"),
    ("ResNet50-FC SA (100k)",   "resnet50", "fc",   "sa",     100000, "resnet50_fc_sa_100k.json"),
]

env = os.environ.copy()
env["PYTHONPATH"] = f"src{os.pathsep}."

summary_file = os.path.join(RESULTS_DIR, "long_budget_summary.json")
results_summary = []

print("="*75, flush=True)
print(" STARTING LONG-BUDGET BENCHMARK SUITE (100,000 EVALS PER WORKLOAD)", flush=True)
print("="*75, flush=True)

for idx, (label, model, region, algo, iters, outfile) in enumerate(runs, 1):
    out_path = os.path.join(RESULTS_DIR, outfile)
    print(f"\n[{idx}/{len(runs)}] Running {label} (model={model}, region={region}, iters={iters})...", flush=True)
    
    cmd = [
        PYTHON, "src/run_multi_chip.py",
        "--algo", algo,
        "--model", model,
        "--workload_region", region,
        "--iters", str(iters),
        "--report", out_path
    ] + BASE_FLAGS
    
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    
    if res.returncode != 0:
        print(f"FAILED (code {res.returncode}):\n{res.stderr[-400:]}", flush=True)
    else:
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                d = json.load(f)
            best_us = round(d["best_cost"] * 1e6, 5)
            tasks = d.get("tasks", 0)
            evals = d.get("complete_placement_evaluations", iters)
            print(f"    --> COMPLETED in {dt:.1f}s ({dt/60:.2f} min)", flush=True)
            print(f"    --> Best Cost: {best_us:.5f} us | Tasks: {tasks} | Evaluations: {evals}", flush=True)
            
            entry = {
                "label": label,
                "model": model,
                "region": region,
                "algo": algo,
                "iters": iters,
                "best_cost_us": best_us,
                "tasks": tasks,
                "evals": evals,
                "runtime_s": round(dt, 2)
            }
            results_summary.append(entry)
            
            # Save progress after each run
            with open(summary_file, "w") as sf:
                json.dump(results_summary, sf, indent=2)

print("\n" + "="*80, flush=True)
print(" LONG-BUDGET RESULTS SUMMARY (100k EVALUATIONS)", flush=True)
print("="*80, flush=True)
print(f"{'Workload':<25} {'Algo':<8} {'Tasks':>6} {'Evaluations':>12} {'Latency (us)':>15} {'Time (s)':>10}", flush=True)
print("-"*80, flush=True)
for r in results_summary:
    print(f"{r['label']:<25} {r['algo']:<8} {r['tasks']:>6} {r['evals']:>12} {r['best_cost_us']:>15.5f} {r['runtime_s']:>10.1f}", flush=True)
print("="*80, flush=True)
