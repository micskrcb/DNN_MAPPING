#!/usr/bin/env bash
set -euo pipefail

proposals="${1:-4100}"
output_dir="${2:-runs/kaggle-ppo-asa-holdout-${proposals}}"
checkpoint="${3:-runs/kaggle-ppo-asa-multichain-5ea4c65-2026-10-09/ppo-asa-trained.pt}"
seed_count="${4:-5}"
archive="${output_dir%/}.zip"
mkdir -p "${output_dir}"

if [[ ! "${seed_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Holdout seed count must be a positive integer." >&2
  exit 1
fi

if [[ ! -f "${checkpoint}" ]]; then
  echo "Trained PPO-ASA checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

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
  echo "This paired holdout requires two visible GPUs." >&2
  exit 1
fi

common=(
  --device cuda
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 64 --iters "${proposals}"
  --asa_calibration_trials 32 --asa_adapt_window 100
  --ppo_asa_candidates 16 --ppo_asa_rollout_steps 4096
  --ppo_asa_focus_fraction 0.5 --ppo_asa_disable_learning
  --ppo_hidden_dim 128
)

for seed in $(seq 1 "${seed_count}"); do
  echo "===== Holdout seed ${seed}: trained checkpoint versus uniform ====="
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
      --seed "${seed}" --load_checkpoint "${checkpoint}" \
      --asa_diagnostics "${output_dir}/seed-${seed}-trained.jsonl" \
      --report "${output_dir}/seed-${seed}-trained-report.json" \
      >"${output_dir}/seed-${seed}-trained.log" 2>&1 &
  trained_pid=$!
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
      --seed "${seed}" \
      --asa_diagnostics "${output_dir}/seed-${seed}-control.jsonl" \
      --report "${output_dir}/seed-${seed}-control-report.json" \
      >"${output_dir}/seed-${seed}-control.log" 2>&1 &
  control_pid=$!
  pair_status=0
  if ! wait "${trained_pid}"; then pair_status=1; fi
  if ! wait "${control_pid}"; then pair_status=1; fi
  if [[ "${pair_status}" -ne 0 ]]; then
    echo "Policy pair failed for seed ${seed}; inspect its logs." >&2
    exit 1
  fi

  echo "===== Holdout seed ${seed}: matched ordinary ASA ====="
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py --algo asa "${common[@]}" \
      --device cpu --seed "${seed}" \
      --asa_diagnostics "${output_dir}/seed-${seed}-asa.jsonl" \
      --report "${output_dir}/seed-${seed}-asa-report.json" \
      >"${output_dir}/seed-${seed}-asa.log" 2>&1
  archive_results
  echo "Completed holdout seed ${seed}."
done

python src/analyze_ppo_asa_holdout.py "${output_dir}" \
  | tee "${output_dir}/analysis.txt"
archive_results
trap - EXIT

echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
