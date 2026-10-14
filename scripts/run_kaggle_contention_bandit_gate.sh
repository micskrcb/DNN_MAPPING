#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-contention-bandit-${mode}}"
archive="${output_dir%/}.zip"
model="${MODEL_OVERRIDE:-vgg16}"
region="${WORKLOAD_REGION_OVERRIDE:-conv}"

case "${mode}" in
  short)
    budget=20000
    seeds=(0 1 2)
    ;;
  extended)
    budget=100000
    seeds=(0 1 2)
    ;;
  *)
    echo "Usage: $0 [short|extended] [output-directory]" >&2
    exit 2
    ;;
esac

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a seeds <<<"${SEEDS_OVERRIDE}"
fi

mkdir -p "${output_dir}"

archive_results() {
  [[ -d "${output_dir}" ]] || return 0
  python - "${output_dir}" "${archive}" <<'PY'
import os
import sys
import zipfile

source, destination = sys.argv[1:]
temporary = destination + ".tmp"
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            bundle.write(path, os.path.relpath(path, os.path.dirname(source)))
os.replace(temporary, destination)
print(f"Saved recoverable results to {destination}")
PY
}
trap archive_results EXIT

PYTHONPATH=src python -m unittest src.test_reconciliation

gpu_count="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if (( gpu_count < 1 )); then
  echo "At least one CUDA GPU is required." >&2
  exit 1
fi

python - "${output_dir}/experiment-manifest.json" "${mode}" "${budget}" \
  "${seeds[*]}" "${model}" "${region}" <<'PY'
import json
import subprocess
import sys

destination, mode, budget, seeds, model, region = sys.argv[1:]
manifest = {
    "experiment": "contention-aware contextual-bandit fixed-SA gate",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "mode": mode,
    "budget_per_method_per_seed": int(budget),
    "seeds": [int(value) for value in seeds.split()],
    "model": model,
    "region": region,
    "primary_comparisons": [
        "contention bandit versus shuffled-label bandit",
        "contention bandit versus proxy-greedy",
        "contention bandit versus fixed SA",
        "contention bandit versus random search",
    ],
    "new_configuration": {
        "candidate_features": "contention12",
        "focus_fraction": 0.5,
        "focus_stage_slack": 0.05,
        "exploration": "epsilon_top_m",
        "final_top_m_epsilon": 0.25,
        "top_m": 4,
        "global_uniform_probability": 0.05,
    },
}
with open(destination, "w") as stream:
    json.dump(manifest, stream, indent=2)
PY

common=(
  --use_cnn --model "${model}"
  --partition_mode paper_targets --workload_region "${region}"
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1
  --sensitivity_trials 256
)

bandit_common=(
  --algo ppo_asa --ppo_asa_annealing fixed
  --ppo_asa_learner bandit
  --ppo_asa_candidates 16
  --ppo_asa_focus_fraction 0.5
  --ppo_asa_focus_stage_slack 0.05
  --ppo_asa_bandit_feature_set contention12
  --ppo_asa_bandit_exploration epsilon_top_m
  --ppo_asa_bandit_epsilon_final 0.25
  --ppo_asa_bandit_epsilon_decay 1.0
  --ppo_asa_bandit_top_m 4
  --ppo_asa_bandit_global_epsilon 0.05
  --ppo_asa_update_style reference_separate
  --ppo_asa_reward_mode proposed_delta
  --ppo_asa_rollout_steps 256
  --ppo_update_epochs 4
  --ppo_minibatch_size 256
  --ppo_hidden_dim 128
  --ppo_learning_rate 3e-4
  --ppo_asa_weight_decay 1e-4
  --ppo_gamma 0
  --ppo_gae_lambda 0
  --ppo_entropy_coef 0.001
)

for seed in "${seeds[@]}"; do
  seed_dir="${output_dir}/seed-${seed}"
  mkdir -p "${seed_dir}"
  second_gpu=0
  if (( gpu_count > 1 )); then
    second_gpu=1
  fi

  echo "===== Seed ${seed}: learned versus shuffled-label control ====="
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" "${bandit_common[@]}" --device cuda \
    --seed "${seed}" --iters "${budget}" \
    --asa_diagnostics "${seed_dir}/learned.jsonl" \
    --save_checkpoint "${seed_dir}/learned.pt" \
    --checkpoint_every "${budget}" \
    --report "${seed_dir}/learned-report.json" \
    >"${seed_dir}/learned.log" 2>&1 &
  learned_pid=$!

  CUDA_VISIBLE_DEVICES="${second_gpu}" PYTHONUNBUFFERED=1 \
    python src/run_multi_chip.py \
    "${common[@]}" "${bandit_common[@]}" --device cuda \
    --ppo_asa_bandit_shuffle_labels \
    --seed "${seed}" --iters "${budget}" \
    --asa_diagnostics "${seed_dir}/shuffled.jsonl" \
    --report "${seed_dir}/shuffled-report.json" \
    >"${seed_dir}/shuffled.log" 2>&1 &
  shuffled_pid=$!

  status=0
  wait "${learned_pid}" || status=1
  wait "${shuffled_pid}" || status=1
  archive_results
  if (( status != 0 )); then
    echo "Learned/shuffled pair failed; inspect archived logs." >&2
    exit 1
  fi

  echo "===== Seed ${seed}: proxy and bottleneck-only controls ====="
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" "${bandit_common[@]}" --device cuda \
    --ppo_asa_disable_learning --ppo_asa_bandit_control proxy_greedy \
    --seed "${seed}" --iters "${budget}" \
    --report "${seed_dir}/proxy-report.json" \
    >"${seed_dir}/proxy.log" 2>&1 &
  proxy_pid=$!

  CUDA_VISIBLE_DEVICES="${second_gpu}" PYTHONUNBUFFERED=1 \
    python src/run_multi_chip.py \
    "${common[@]}" "${bandit_common[@]}" --device cuda \
    --ppo_asa_disable_learning --ppo_asa_bandit_control bottleneck_random \
    --seed "${seed}" --iters "${budget}" \
    --report "${seed_dir}/bottleneck-random-report.json" \
    >"${seed_dir}/bottleneck-random.log" 2>&1 &
  bottleneck_pid=$!

  status=0
  wait "${proxy_pid}" || status=1
  wait "${bottleneck_pid}" || status=1
  archive_results
  if (( status != 0 )); then
    echo "Non-learned control pair failed; inspect archived logs." >&2
    exit 1
  fi

  echo "===== Seed ${seed}: paper context baselines ====="
  PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" --algo sa --device cpu \
    --seed "${seed}" --iters "${budget}" \
    --report "${seed_dir}/fixed-sa-report.json" \
    >"${seed_dir}/fixed-sa.log" 2>&1

  PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" --algo random --device cpu \
    --seed "${seed}" --iters "${budget}" \
    --report "${seed_dir}/random-report.json" \
    >"${seed_dir}/random.log" 2>&1
  archive_results
done

python - "${output_dir}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = (
    "learned", "shuffled", "proxy", "bottleneck-random",
    "fixed-sa", "random",
)
rows = []
for seed_dir in sorted(root.glob("seed-*")):
    row = {"seed": int(seed_dir.name.split("-")[-1])}
    for method in methods:
        path = seed_dir / f"{method}-report.json"
        with path.open() as stream:
            report = json.load(stream)
        row[f"{method}_microseconds"] = report["best_cost"] * 1e6
        row[f"{method}_seconds_elapsed"] = report.get("seconds_elapsed")
    rows.append(row)

def improvement(reference, learned):
    return (reference - learned) / reference * 100.0

for row in rows:
    learned = row["learned_microseconds"]
    for comparator in methods[1:]:
        row[f"learned_improvement_over_{comparator}_percent"] = improvement(
            row[f"{comparator}_microseconds"], learned)

aggregates = {}
for method in methods:
    values = [row[f"{method}_microseconds"] for row in rows]
    aggregates[method] = {
        "mean_microseconds": statistics.mean(values),
        "sample_std_microseconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0),
        "individual_microseconds": values,
    }

comparisons = {}
learned_wins = {}
for comparator in methods[1:]:
    values = [
        row[f"learned_improvement_over_{comparator}_percent"]
        for row in rows
    ]
    comparisons[f"learned_vs_{comparator}"] = {
        "mean_improvement_percent": statistics.mean(values),
        "individual_improvement_percent": values,
    }
    learned_wins[comparator] = sum(value > 0 for value in values)

required = ("shuffled", "proxy", "fixed-sa")
gate_pass = all(learned_wins[name] >= 2 for name in required)
summary = {
    "rows": rows,
    "aggregates": aggregates,
    "comparisons": comparisons,
    "learned_wins": learned_wins,
    "extended_gate_pass": gate_pass,
    "gate_rule": (
        "Learned must beat shuffled labels, proxy-greedy, and fixed SA in "
        "at least two of three seeds."
    ),
}
with (root / "summary.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
PY

archive_results
trap - EXIT
echo "FINISHED: ${archive}"
sha256sum "${archive}"
