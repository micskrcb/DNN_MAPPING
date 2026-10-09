#!/usr/bin/env bash
set -euo pipefail

mode="${1:-short}"
output_dir="${2:-runs/kaggle-ppo-asa-alexnet-fc-${mode}}"
focus_flag=()
restart_interval=0
rollout_steps=128
learning_rate=1e-3
update_epochs=4
minibatch_size=128
case "${mode}" in
  short)
    proposals=1000
    ;;
  extended)
    proposals=4100
    ;;
  focused-short)
    proposals=1000
    focus_flag+=(--ppo_asa_focus_bottleneck)
    ;;
  focused-extended)
    proposals=4100
    focus_flag+=(--ppo_asa_focus_bottleneck)
    ;;
  multichain-preflight)
    proposals=10000
    restart_interval=128
    focus_flag+=(--ppo_asa_focus_fraction 0.5)
    rollout_steps=4096
    learning_rate=2e-4
    update_epochs=10
    minibatch_size=1024
    ;;
  multichain-train)
    proposals=250000
    restart_interval=128
    focus_flag+=(--ppo_asa_focus_fraction 0.5)
    rollout_steps=4096
    learning_rate=2e-4
    update_epochs=10
    minibatch_size=1024
    ;;
  *)
    echo "Usage: $0 short|extended|focused-short|focused-extended|multichain-preflight|multichain-train [output-directory]" >&2
    exit 2
    ;;
esac
if [[ "${restart_interval}" -gt 0 ]]; then
  focus_flag+=(--ppo_asa_restart_interval "${restart_interval}")
  chain_count=$(( (proposals - 1) / restart_interval + 1 ))
else
  chain_count=1
fi
asa_proposals=$(( proposals + chain_count - 1 ))
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
  --sensitivity_trials 64 --seed 0
  --asa_calibration_trials 32 --asa_adapt_window 100
  --ppo_asa_candidates 16 --ppo_asa_rollout_steps "${rollout_steps}"
  --ppo_learning_rate "${learning_rate}" --ppo_update_epochs "${update_epochs}"
  --ppo_minibatch_size "${minibatch_size}" --ppo_gamma 0.98 --ppo_gae_lambda 0.95
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
      "${focus_flag[@]}" \
      --iters "${proposals}" \
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
(
  while kill -0 "${trained_pid}" 2>/dev/null || kill -0 "${control_pid}" 2>/dev/null; do
    sleep 60
    if kill -0 "${trained_pid}" 2>/dev/null || kill -0 "${control_pid}" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) PPO-ASA pair still running"
      tail -n 1 "${output_dir}/ppo-asa-trained.log" 2>/dev/null || true
      tail -n 1 "${output_dir}/ppo-asa-control.log" 2>/dev/null || true
    fi
  done
) &
monitor_pid=$!
pair_status=0
if ! wait "${trained_pid}"; then pair_status=1; fi
if ! wait "${control_pid}"; then pair_status=1; fi
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
if [[ "${pair_status}" -ne 0 ]]; then
  echo "PPO-ASA pair failed; partial files were archived. Inspect logs." >&2
  exit 1
fi
archive_results

echo "Running ordinary ASA with ${asa_proposals} proposals for ${proposals} proposal and ${chain_count} initialization evaluations."
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo asa "${common[@]}" --device cpu \
  --iters "${asa_proposals}" \
  --asa_diagnostics "${output_dir}/asa.jsonl" \
  --report "${output_dir}/asa-report.json" \
  >"${output_dir}/asa.log" 2>&1 &
asa_pid=$!
(
  while kill -0 "${asa_pid}" 2>/dev/null; do
    sleep 60
    if kill -0 "${asa_pid}" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) ordinary ASA still running"
    fi
  done
) &
asa_monitor_pid=$!
asa_status=0
if ! wait "${asa_pid}"; then asa_status=1; fi
kill "${asa_monitor_pid}" 2>/dev/null || true
wait "${asa_monitor_pid}" 2>/dev/null || true
if [[ "${asa_status}" -ne 0 ]]; then
  echo "Ordinary ASA failed; partial files were archived. Inspect asa.log." >&2
  exit 1
fi

python src/analyze_ppo_asa_gate.py "${output_dir}" \
  | tee "${output_dir}/analysis.txt"
archive_results
trap - EXIT

echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
