#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-masked-ppo-${mode}}"
mkdir -p "${output_dir}"
protocol_id="masked-ppo-ablation-v1"

case "${mode}" in
  short)
    seeds=(0)
    epochs=10
    baseline_trials=256
    sensitivity_trials=32
    ;;
  extensive)
    seeds=(0 1 2 3 4)
    epochs=100
    baseline_trials=1000
    sensitivity_trials=64
    ;;
  *)
    echo "Usage: $0 short|extensive [output-directory]" >&2
    exit 2
    ;;
esac

target_placements=$((epochs * 30))
archive="${output_dir%/}.zip"

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

# Refresh the archive when the shell exits as well as after each completed pair.
trap archive_results EXIT

condition_complete() {
  local seed="$1"
  local condition="$2"
  local prefix="${output_dir}/seed${seed}-${condition}"
  python - "${prefix}-complete.json" "${prefix}-report.json" \
    "${protocol_id}" "${target_placements}" "${seed}" "${condition}" <<'PY'
import json
import os
import sys

marker_path, report_path, protocol, target, seed, condition = sys.argv[1:]
prefix = marker_path.removesuffix("-complete.json")
required = (marker_path, report_path, prefix + ".jsonl", prefix + ".log")
if not all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in required):
    raise SystemExit(1)
try:
    with open(marker_path, encoding="utf-8") as stream:
        marker = json.load(stream)
    with open(report_path, encoding="utf-8") as stream:
        report = json.load(stream)
except (OSError, ValueError):
    raise SystemExit(1)
metadata = report.get("algorithm_metadata") or {}
expected_learning = condition == "trained"
complete = (
    marker.get("protocol") == protocol
    and marker.get("target_placements") == int(target)
    and marker.get("seed") == int(seed)
    and marker.get("condition") == condition
    and metadata.get("complete_placement_evaluations") == int(target)
    and metadata.get("random_seed") == int(seed)
    and metadata.get("learning_enabled") is expected_learning
)
raise SystemExit(0 if complete else 1)
PY
}

mark_complete() {
  local seed="$1"
  local condition="$2"
  local prefix="${output_dir}/seed${seed}-${condition}"
  python - "${prefix}-complete.json" "${protocol_id}" \
    "${target_placements}" "${seed}" "${condition}" <<'PY'
import json
import os
import sys

path, protocol, target, seed, condition = sys.argv[1:]
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump({"protocol": protocol, "target_placements": int(target),
               "seed": int(seed), "condition": condition}, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY
}

clear_incomplete_case() {
  local seed="$1"
  local condition="$2"
  local prefix="${output_dir}/seed${seed}-${condition}"
  rm -f "${prefix}.pt" "${prefix}.jsonl" \
    "${prefix}-report.json" "${prefix}.log" "${prefix}-complete.json"
}

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${gpu_count}" -lt 1 ]]; then
  echo "CUDA is unavailable. Enable a Kaggle GPU accelerator first." >&2
  exit 1
fi

run_case() {
  local seed="$1"
  local condition="$2"
  local gpu="$3"
  local prefix="${output_dir}/seed${seed}-${condition}"
  local learning_flag=()
  if [[ "${condition}" == "control" ]]; then
    learning_flag+=(--ppo_disable_learning)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py \
      --algo ppo_masked --device cuda \
      --use_cnn --model alexnet \
      --partition_mode paper_targets --workload_region conv \
      --timing_model paper_pipeline --routing_model paper_xy \
      --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
      --reward_mode sparse \
      --epochs "${epochs}" --placements_per_epoch 30 \
      --baseline_trials "${baseline_trials}" \
      --ppo_learning_rate 3e-4 --ppo_rollout_placements 10 \
      --ppo_update_epochs 4 --ppo_minibatch_size 512 \
      --ppo_gamma 1.0 --ppo_gae_lambda 1.0 --ppo_clip_ratio 0.2 \
      --ppo_entropy_coef 0.01 --ppo_value_coef 0.5 \
      --ppo_max_grad_norm 0.5 --ppo_hidden_dim 256 \
      --diagnostics_every 30 --checkpoint_every 100 \
      --sensitivity_trials "${sensitivity_trials}" --seed "${seed}" \
      --save_checkpoint "${prefix}.pt" \
      --diagnostics "${prefix}.jsonl" \
      --report "${prefix}-report.json" \
      "${learning_flag[@]}" >"${prefix}.log" 2>&1
  mark_complete "${seed}" "${condition}"
}

echo "Running ${mode} paired masked-PPO ablation on ${gpu_count} visible GPU(s)."
for seed in "${seeds[@]}"; do
  trained_pending=1
  control_pending=1
  if condition_complete "${seed}" trained; then
    trained_pending=0
    echo "Seed ${seed} trained condition is already complete; skipping it."
  else
    clear_incomplete_case "${seed}" trained
  fi
  if condition_complete "${seed}" control; then
    control_pending=0
    echo "Seed ${seed} control condition is already complete; skipping it."
  else
    clear_incomplete_case "${seed}" control
  fi

  if [[ "${gpu_count}" -ge 2 ]]; then
    pair_status=0
    trained_pid=""
    control_pid=""
    if [[ "${trained_pending}" -eq 1 ]]; then
      run_case "${seed}" trained 0 &
      trained_pid=$!
    fi
    if [[ "${control_pending}" -eq 1 ]]; then
      run_case "${seed}" control 1 &
      control_pid=$!
    fi
    if [[ -n "${trained_pid}" ]] && ! wait "${trained_pid}"; then
      pair_status=1
    fi
    if [[ -n "${control_pid}" ]] && ! wait "${control_pid}"; then
      pair_status=1
    fi
    if [[ "${pair_status}" -ne 0 ]]; then
      echo "Seed ${seed} failed; see its logs. Partial files were archived." >&2
      exit 1
    fi
  else
    if [[ "${trained_pending}" -eq 1 ]]; then
      run_case "${seed}" trained 0
    fi
    if [[ "${control_pending}" -eq 1 ]]; then
      run_case "${seed}" control 0
    fi
  fi
  echo "Completed paired seed ${seed}."
  tail -n 4 "${output_dir}/seed${seed}-trained.log"
  tail -n 4 "${output_dir}/seed${seed}-control.log"
  python src/analyze_masked_ppo_ablation.py "${output_dir}" \
    | tee "${output_dir}/analysis.txt"
  archive_results
done
trap - EXIT
