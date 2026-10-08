#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-guided-${mode}}"
mkdir -p "${output_dir}"
protocol_id="guided-ablation-v2"

case "${mode}" in
  short)
    seeds=(0)
    epochs=10
    baseline_trials=256
    demo_iterations=1000
    pretrain_updates=200
    random_starts=60
    bc_decay=240
    exploration_decay=240
    sensitivity_trials=32
    ;;
  extensive)
    seeds=(0 1 2 3 4)
    epochs=100
    baseline_trials=1000
    demo_iterations=5000
    pretrain_updates=500
    random_starts=100
    bc_decay=2000
    exploration_decay=2400
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
  rm -f "${prefix}.pt" "${prefix}.jsonl" "${prefix}-asa.jsonl" \
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
    learning_flag+=(--guided_disable_learning)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py \
      --algo ddpg_guided --device cuda \
      --use_cnn --model alexnet \
      --partition_mode paper_targets --workload_region conv \
      --timing_model paper_pipeline --routing_model paper_xy \
      --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
      --agent_arch paper_cnn --reward_mode sparse \
      --epochs "${epochs}" --placements_per_epoch 30 \
      --baseline_trials "${baseline_trials}" --batch_z 3 --train_every 1 \
      --exploration_decay_placements "${exploration_decay}" \
      --guided_top_k 8 --guided_demo_iterations "${demo_iterations}" \
      --guided_pretrain_updates "${pretrain_updates}" \
      --guided_bc_decay_placements "${bc_decay}" \
      --guided_random_start_placements "${random_starts}" \
      --diagnostics_every 30 --checkpoint_every 100 \
      --sensitivity_trials "${sensitivity_trials}" --seed "${seed}" \
      --save_checkpoint "${prefix}.pt" \
      --diagnostics "${prefix}.jsonl" \
      --asa_diagnostics "${prefix}-asa.jsonl" \
      --report "${prefix}-report.json" \
      "${learning_flag[@]}" >"${prefix}.log" 2>&1
  mark_complete "${seed}" "${condition}"
}

echo "Running ${mode} paired guided-DDPG ablation on ${gpu_count} visible GPU(s)."
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
  python src/analyze_guided_ablation.py "${output_dir}" \
    | tee "${output_dir}/analysis.txt"
  archive_results
done
trap - EXIT
