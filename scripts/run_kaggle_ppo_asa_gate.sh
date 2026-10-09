#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-ppo-asa-alexnet-fc-${mode}}"
case "${mode}" in
  short)
    proposals=1000
    ;;
  extended)
    proposals=4100
    ;;
  *)
    echo "Usage: $0 short|extended [output-directory]" >&2
    exit 2
    ;;
esac
archive="${output_dir%/}.zip"
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

gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "This paired gate requires two visible GPUs." >&2
  exit 1
fi

common=(
  --device cuda
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 64 --seed 0 --iters "${proposals}"
  --asa_calibration_trials 32 --asa_adapt_window 100
  --ppo_asa_candidates 16 --ppo_asa_rollout_steps 128
  --ppo_learning_rate 1e-3 --ppo_update_epochs 4
  --ppo_minibatch_size 128 --ppo_gamma 0.98 --ppo_gae_lambda 0.95
  --ppo_clip_ratio 0.2 --ppo_entropy_coef 0.01
  --ppo_value_coef 0.5 --ppo_max_grad_norm 0.5 --ppo_hidden_dim 128
)

run_condition() {
  local condition="$1"
  local gpu="$2"
  local prefix="${output_dir}/ppo-asa-${condition}"
  local control_flag=()
  if [[ "${condition}" == "control" ]]; then
    control_flag+=(--ppo_asa_disable_learning)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
      --asa_diagnostics "${prefix}.jsonl" \
      --save_checkpoint "${prefix}.pt" \
      --report "${prefix}-report.json" \
      "${control_flag[@]}" >"${prefix}.log" 2>&1
}

echo "Running ${mode} paired PPO-guided ASA on two GPUs."
run_condition trained 0 &
trained_pid=$!
run_condition control 1 &
control_pid=$!
pair_status=0
if ! wait "${trained_pid}"; then pair_status=1; fi
if ! wait "${control_pid}"; then pair_status=1; fi
if [[ "${pair_status}" -ne 0 ]]; then
  echo "PPO-ASA pair failed; partial files were archived. Inspect logs." >&2
  exit 1
fi
archive_results

echo "Running ordinary ASA with the same initialization plus ${proposals} proposals."
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo asa "${common[@]}" --device cpu \
  --asa_diagnostics "${output_dir}/asa.jsonl" \
  --report "${output_dir}/asa-report.json" \
  >"${output_dir}/asa.log" 2>&1

python src/analyze_ppo_asa_gate.py "${output_dir}" \
  | tee "${output_dir}/analysis.txt"
archive_results
trap - EXIT

echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
