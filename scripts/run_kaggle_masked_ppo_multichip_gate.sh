#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-masked-ppo-alexnet-fc-${mode}}"
case "${mode}" in
  short)
    epochs=10
    baseline_trials=256
    matched_evaluations=567
    asa_proposals=566
    ;;
  extended)
    epochs=100
    baseline_trials=1000
    matched_evaluations=4101
    asa_proposals=4100
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
  echo "This gate requires two visible GPUs for the paired PPO run." >&2
  exit 1
fi

common=(
  --device cuda
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 64 --seed 0
)

run_ppo() {
  local condition="$1"
  local gpu="$2"
  local prefix="${output_dir}/ppo-${condition}"
  local learning_flag=()
  if [[ "${condition}" == "control" ]]; then
    learning_flag+=(--ppo_disable_learning)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python src/run_multi_chip.py \
      --algo ppo_masked "${common[@]}" \
      --reward_mode sparse --epochs "${epochs}" --placements_per_epoch 30 \
      --baseline_trials "${baseline_trials}" \
      --ppo_learning_rate 3e-4 --ppo_rollout_placements 10 \
      --ppo_update_epochs 4 --ppo_minibatch_size 512 \
      --ppo_gamma 1.0 --ppo_gae_lambda 1.0 --ppo_clip_ratio 0.2 \
      --ppo_entropy_coef 0.01 --ppo_value_coef 0.5 \
      --ppo_max_grad_norm 0.5 --ppo_hidden_dim 256 \
      --diagnostics_every 30 --checkpoint_every 100 \
      --save_checkpoint "${prefix}.pt" \
      --diagnostics "${prefix}.jsonl" \
      --report "${prefix}-report.json" \
      "${learning_flag[@]}" >"${prefix}.log" 2>&1
}

echo "Running ${mode} paired AlexNet-FC masked PPO on two GPUs."
run_ppo trained 0 &
trained_pid=$!
run_ppo control 1 &
control_pid=$!
pair_status=0
if ! wait "${trained_pid}"; then pair_status=1; fi
if ! wait "${control_pid}"; then pair_status=1; fi
if [[ "${pair_status}" -ne 0 ]]; then
  echo "PPO pair failed; partial files were archived. Inspect ppo-*.log." >&2
  exit 1
fi
archive_results

echo "Running matched ${matched_evaluations}-evaluation random search."
python src/run_multi_chip.py --algo random "${common[@]}" \
  --iters "${matched_evaluations}" --report "${output_dir}/random-report.json" \
  >"${output_dir}/random.log" 2>&1

echo "Running matched ${matched_evaluations}-evaluation adaptive SA " \
  "(initialization + ${asa_proposals} proposals)."
python src/run_multi_chip.py --algo asa "${common[@]}" \
  --iters "${asa_proposals}" --asa_diagnostics "${output_dir}/asa.jsonl" \
  --report "${output_dir}/asa-report.json" >"${output_dir}/asa.log" 2>&1

echo "Running the deterministic sequential baseline."
python src/run_multi_chip.py --algo bs "${common[@]}" \
  --report "${output_dir}/bs-report.json" >"${output_dir}/bs.log" 2>&1

python src/analyze_masked_ppo_multichip_gate.py "${output_dir}" \
  | tee "${output_dir}/analysis.txt"
archive_results
trap - EXIT
