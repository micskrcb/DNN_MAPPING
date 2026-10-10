#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-ppo-asa-sufficient-scale}"
training_dir="${output_dir}/training"
holdout_dir="${output_dir}/holdout"
archive="${output_dir%/}.zip"
mkdir -p "${training_dir}"

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

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "This experiment requires two visible GPUs for paired holdout evaluation." >&2
  exit 1
fi

echo "Training PPO-ASA for 1,000 PPO updates on deployment-matched chains."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py \
    --algo ppo_asa --device cuda \
    --use_cnn --model alexnet \
    --partition_mode paper_targets --workload_region fc \
    --timing_model paper_pipeline --routing_model paper_xy \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
    --sensitivity_trials 64 --seed 0 --iters 1000000 \
    --asa_calibration_trials 32 --asa_adapt_window 100 \
    --ppo_asa_candidates 16 --ppo_asa_focus_fraction 0.5 \
    --ppo_asa_restart_interval 4100 \
    --ppo_asa_rollout_steps 1000 \
    --ppo_learning_rate 2e-4 --ppo_asa_weight_decay 0.01 \
    --ppo_update_epochs 10 --ppo_minibatch_size 1024 \
    --ppo_gamma 0.9 --ppo_gae_lambda 0.9 \
    --ppo_clip_ratio 0.25 --ppo_entropy_coef 0 \
    --ppo_value_coef 0.5 --ppo_max_grad_norm 0.5 --ppo_hidden_dim 128 \
    --asa_diagnostics "${training_dir}/trained.jsonl" \
    --save_checkpoint "${training_dir}/trained.pt" \
    --report "${training_dir}/trained-report.json" \
    >"${training_dir}/trained.log" 2>&1 &
training_pid=$!

(
  while kill -0 "${training_pid}" 2>/dev/null; do
    sleep 60
    if kill -0 "${training_pid}" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) sufficient-scale training still running"
      tail -n 1 "${training_dir}/trained.log" 2>/dev/null || true
    fi
  done
) &
monitor_pid=$!
training_status=0
if ! wait "${training_pid}"; then training_status=1; fi
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
if [[ "${training_status}" -ne 0 ]]; then
  echo "Training failed; partial files were archived. Inspect trained.log." >&2
  exit 1
fi
archive_results

actual_updates="$(python - "${training_dir}/trained-report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
print(report["algorithm_metadata"]["update_count"])
PY
)"
if [[ "${actual_updates}" -ne 1000 ]]; then
  echo "Expected exactly 1,000 PPO updates, observed ${actual_updates}." >&2
  exit 1
fi

echo "Running ten-seed frozen holdout with matched 4,101-call budgets."
bash scripts/run_kaggle_ppo_asa_holdout.sh \
  4100 "${holdout_dir}" "${training_dir}/trained.pt" 10

archive_results
trap - EXIT
echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
