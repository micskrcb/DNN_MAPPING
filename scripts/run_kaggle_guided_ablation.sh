#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-guided-${mode}}"
mkdir -p "${output_dir}"

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
      --save_checkpoint "${prefix}.pt" --load_checkpoint "${prefix}.pt" \
      --diagnostics "${prefix}.jsonl" \
      --asa_diagnostics "${prefix}-asa.jsonl" \
      --report "${prefix}-report.json" \
      "${learning_flag[@]}" >"${prefix}.log" 2>&1
}

echo "Running ${mode} paired guided-DDPG ablation on ${gpu_count} visible GPU(s)."
for seed in "${seeds[@]}"; do
  if [[ "${gpu_count}" -ge 2 ]]; then
    run_case "${seed}" trained 0 &
    trained_pid=$!
    run_case "${seed}" control 1 &
    control_pid=$!
    wait "${trained_pid}"
    wait "${control_pid}"
  else
    run_case "${seed}" trained 0
    run_case "${seed}" control 0
  fi
  echo "Completed paired seed ${seed}."
  tail -n 4 "${output_dir}/seed${seed}-trained.log"
  tail -n 4 "${output_dir}/seed${seed}-control.log"
done

python src/analyze_guided_ablation.py "${output_dir}" | tee "${output_dir}/analysis.txt"
archive="${output_dir%/}.zip"
python - "${output_dir}" "${archive}" <<'PY'
import os
import sys
import zipfile

source, destination = sys.argv[1:]
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            archive.write(path, os.path.relpath(path, os.path.dirname(source)))
print(f"Saved {destination}")
PY
