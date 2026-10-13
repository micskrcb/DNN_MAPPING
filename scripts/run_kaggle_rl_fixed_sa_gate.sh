#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-rl-fixed-sa-${mode}}"
archive="${output_dir%/}.zip"
learner="ppo"
skip_context_baselines="${SKIP_CONTEXT_BASELINES:-0}"

case "${mode}" in
  short)
    budget=20000
    seeds=(0)
    annealing="fixed"
    ;;
  extended)
    budget=100000
    seeds=(0 1 2)
    annealing="fixed"
    ;;
  adaptive-short)
    budget=20000
    seeds=(0)
    annealing="adaptive"
    ;;
  adaptive-extended)
    budget=100000
    seeds=(0 1 2)
    annealing="adaptive"
    ;;
  bandit-short)
    budget=20000
    seeds=(0)
    annealing="fixed"
    learner="bandit"
    ;;
  bandit-extended)
    budget=100000
    seeds=(0 1 2)
    annealing="fixed"
    learner="bandit"
    ;;
  *)
    echo "Usage: $0 [short|extended|adaptive-short|adaptive-extended|bandit-short|bandit-extended] [output-directory]" >&2
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
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            archive.write(path, os.path.relpath(path, os.path.dirname(source)))
os.replace(temporary, destination)
print(f"Saved recoverable results to {destination}")
PY
}
trap archive_results EXIT

echo "Running reconciliation tests."
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
  echo "At least one CUDA GPU is required for this RL gate." >&2
  exit 1
fi

python - "${output_dir}/experiment-manifest.json" "${mode}" \
  "${annealing}" "${learner}" "${budget}" "${seeds[*]}" \
  "${skip_context_baselines}" <<'PY'
import json
import subprocess
import sys

destination, mode, annealing, learner, budget, seeds, skip_baselines = sys.argv[1:]
manifest = {
    "experiment": f"{learner} learned proposals inside {annealing} SA",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "mode": mode,
    "budget_per_method_per_seed": int(budget),
    "seeds": [int(value) for value in seeds.split()],
    "context_baselines_included": skip_baselines != "1",
    "primary_comparison": f"learned versus frozen proposals inside matched {annealing} SA",
    "configuration": {
        "model": "alexnet",
        "region": "fc",
        "vva_additions_per_cycle": 16,
        "partition_vva_additions_per_cycle": 1,
        "annealing_schedule": annealing,
        "proposal_learner": learner,
        "reward": (
            "signed immediate objective gain with uniform replay"
            if learner == "bandit" else
            "immediate improvement-only positive objective delta"),
        "bandit_final_epsilon": 0.5 if learner == "bandit" else None,
        "bandit_epsilon_decay_fraction": 1.0 if learner == "bandit" else None,
        "ppo_gamma": 0.0 if learner == "ppo" else None,
        "candidate_pool": 16,
    },
    "interpretation": (
        "The frozen policy is the causal control. Plain fixed SA and ASA are "
        "context baselines; a one-seed short win is only a signal for the extended gate."
    ),
}
with open(destination, "w") as stream:
    json.dump(manifest, stream, indent=2)
PY

common=(
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1
  --sensitivity_trials 256
)

ppo_common=(
  --algo ppo_asa --ppo_asa_annealing "${annealing}"
  --ppo_asa_learner "${learner}"
  --ppo_asa_candidates 16 --ppo_asa_focus_fraction 0.5
  --ppo_asa_update_style reference_separate
  --ppo_asa_reward_mode positive_delta
  --ppo_asa_rollout_steps 256 --ppo_update_epochs 4
  --ppo_minibatch_size 256 --ppo_hidden_dim 128
  --ppo_learning_rate 3e-4 --ppo_asa_weight_decay 1e-4
  --ppo_gamma 0 --ppo_gae_lambda 0
  --ppo_entropy_coef 0.001
)

for seed in "${seeds[@]}"; do
  seed_dir="${output_dir}/seed-${seed}"
  mkdir -p "${seed_dir}"
  control_gpu=0
  if (( gpu_count > 1 )); then
    control_gpu=1
  fi

  echo "===== Seed ${seed}: learned versus frozen ${annealing}-SA proposals ====="
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" "${ppo_common[@]}" --device cuda \
    --seed "${seed}" --iters "${budget}" \
    --asa_diagnostics "${seed_dir}/learned.jsonl" \
    --save_checkpoint "${seed_dir}/learned.pt" \
    --checkpoint_every "${budget}" \
    --report "${seed_dir}/learned-report.json" \
    >"${seed_dir}/learned.log" 2>&1 &
  learned_pid=$!

  CUDA_VISIBLE_DEVICES="${control_gpu}" PYTHONUNBUFFERED=1 \
    python src/run_multi_chip.py \
    "${common[@]}" "${ppo_common[@]}" --device cuda \
    --ppo_asa_disable_learning \
    --seed "${seed}" --iters "${budget}" \
    --asa_diagnostics "${seed_dir}/control.jsonl" \
    --report "${seed_dir}/control-report.json" \
    >"${seed_dir}/control.log" 2>&1 &
  control_pid=$!

  status=0
  wait "${learned_pid}" || status=1
  wait "${control_pid}" || status=1
  archive_results
  if (( status != 0 )); then
    echo "Learned/control run failed for seed ${seed}; inspect archived logs." >&2
    exit 1
  fi

  if [[ "${skip_context_baselines}" != "1" ]]; then
    echo "===== Seed ${seed}: context baselines ====="
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      "${common[@]}" --algo sa --device cpu \
      --seed "${seed}" --iters "${budget}" \
      --report "${seed_dir}/fixed-sa-report.json" \
      >"${seed_dir}/fixed-sa.log" 2>&1
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      "${common[@]}" --algo asa --device cpu \
      --seed "${seed}" --iters "${budget}" \
      --report "${seed_dir}/asa-report.json" \
      >"${seed_dir}/asa.log" 2>&1
  fi
  archive_results
done

python - "${output_dir}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for seed_dir in sorted(out.glob("seed-*")):
    reports = {}
    for method in ("learned", "control", "fixed-sa", "asa"):
        path = seed_dir / f"{method}-report.json"
        if path.exists():
            with path.open() as stream:
                reports[method] = json.load(stream)
    learned = reports["learned"]["best_cost"]
    control = reports["control"]["best_cost"]
    learned_metadata = reports["learned"]["algorithm_metadata"]
    control_metadata = reports["control"]["algorithm_metadata"]
    rows.append({
        "seed": int(seed_dir.name.split("-")[-1]),
        "learned_microseconds": learned * 1e6,
        "frozen_control_microseconds": control * 1e6,
        "fixed_sa_microseconds": (
            reports["fixed-sa"]["best_cost"] * 1e6
            if "fixed-sa" in reports else None),
        "asa_microseconds": (
            reports["asa"]["best_cost"] * 1e6
            if "asa" in reports else None),
        "learned_improvement_over_control_percent": (
            (control - learned) / control * 100),
        "learned_wins": learned < control,
        "learned_updates": learned_metadata["update_count"],
        "learned_schedule": learned_metadata["annealing_schedule"],
        "proposal_learner": learned_metadata["proposal_learner"],
        "control_schedule": control_metadata["annealing_schedule"],
    })

improvements = [
    row["learned_improvement_over_control_percent"] for row in rows]
schedule = rows[0]["learned_schedule"] if rows else "unknown"
summary = {
    "protocol": f"Learned proposal selection inside matched {schedule} SA",
    "seeds": len(rows),
    "learned_win_count": sum(row["learned_wins"] for row in rows),
    "mean_learned_improvement_over_control_percent": statistics.mean(improvements),
    "individual_results": rows,
    "next_decision": (
        "Run extended only if the short learned policy beats its frozen control "
        "and performs at least one finite learner update."
        if len(rows) == 1 else
        "Treat learning as supported only if the mean improvement is positive "
        "and learned proposals win at least two of three seeds."
    ),
}
with (out / "summary.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
PY

archive_results
trap - EXIT
echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
