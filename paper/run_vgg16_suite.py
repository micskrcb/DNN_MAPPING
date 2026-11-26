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
    # (label, model, region, algo, iters, outfile, logfile)
    ("VGG16-CONV RS (5k)",  "vgg16", "conv", "random", 5000,  "vgg16_conv_rs_5k.json",  "vgg16_conv_rs_5k.log"),
    ("VGG16-CONV SA (10k)", "vgg16", "conv", "sa",     10000, "vgg16_conv_sa_10k.json", "vgg16_conv_sa_10k.log"),
    ("VGG16-FC RS (5k)",    "vgg16", "fc",   "random", 5000,  "vgg16_fc_rs_5k.json",    "vgg16_fc_rs_5k.log"),
    ("VGG16-FC SA (10k)",   "vgg16", "fc",   "sa",     10000, "vgg16_fc_sa_10k.json",   "vgg16_fc_sa_10k.log"),
]

env = os.environ.copy()
env["PYTHONPATH"] = f"src{os.pathsep}."

print("="*70, flush=True)
print(" RUNNING VGG16 BENCHMARKS (VGG16-CONV & VGG16-FC)", flush=True)
print("="*70, flush=True)

for label, model, region, algo, iters, outfile, logfile in runs:
    out_path = os.path.join(RESULTS_DIR, outfile)
    log_path = os.path.join(RESULTS_DIR, logfile)
    print(f"\n>>> Running: {label} ({algo}, iters={iters})...", flush=True)
    
    cmd = [
        PYTHON, "src/run_multi_chip.py",
        "--algo", algo,
        "--model", model,
        "--workload_region", region,
        "--iters", str(iters),
        "--report", out_path
    ] + BASE_FLAGS
    
    t0 = time.time()
    with open(log_path, "w") as lf:
        res = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    
    if res.returncode != 0:
        print(f"FAILED (code {res.returncode}): check {log_path}", flush=True)
    else:
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                d = json.load(f)
            best_us = round(d["best_cost"] * 1e6, 5)
            tasks = d.get("tasks", 0)
            evals = d.get("complete_placement_evaluations", iters)
            print(f"    --> COMPLETED in {dt:.1f}s -> Best Cost: {best_us:.5f} us (tasks={tasks}, evals={evals})", flush=True)

print("\n" + "="*70, flush=True)
print(" VGG16 SUITE RUNS FINISHED", flush=True)
print("="*70, flush=True)
