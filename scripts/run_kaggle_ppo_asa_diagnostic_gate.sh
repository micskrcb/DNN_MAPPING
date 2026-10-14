#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-ppo-asa-diagnostic-gate}"
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
  echo "This diagnostic requires a Kaggle T4 x2 session." >&2
  exit 1
fi

echo "Running reconciliation tests."
PYTHONPATH=src python -m unittest src.test_reconciliation

python - "${output_dir}/experiment-manifest.json" <<'PY'
import json
import subprocess
import sys

manifest = {
    "experiment": "PPO-ASA oracle, planted-signal, and split pilot gate",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "stages": {
        "oracle": {"pools": 5000, "candidates_per_pool": 16},
        "planted": {
            "proposal_steps_per_condition": 200000,
            "expected_updates": 200,
            "conditions": [
                "shared-trained", "shared-control",
                "split-trained", "split-control"],
        },
        "real_split_pilot": {
            "proposal_steps": 300000,
            "expected_updates": 300,
            "oracle_sampled_pools": 2000,
            "holdout_seeds": 5,
        },
    },
}
with open(sys.argv[1], "w") as stream:
    json.dump(manifest, stream, indent=2)
PY

common=(
  --device cuda
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 64 --seed 0
  --asa_calibration_trials 32 --asa_adapt_window 100
  --ppo_asa_candidates 16 --ppo_asa_focus_fraction 0.5
  --ppo_asa_restart_interval 4100
  --ppo_asa_rollout_steps 1000
  --ppo_learning_rate 2e-4 --ppo_asa_weight_decay 0.01
  --ppo_update_epochs 10 --ppo_minibatch_size 1024
  --ppo_gamma 0.9 --ppo_gae_lambda 0.9
  --ppo_clip_ratio 0.25 --ppo_entropy_coef 0
  --ppo_value_coef 0.5 --ppo_max_grad_norm 0.5 --ppo_hidden_dim 128
)

monitor_pair() {
  local first_pid="$1"
  local second_pid="$2"
  local first_log="$3"
  local second_log="$4"
  while kill -0 "${first_pid}" 2>/dev/null || kill -0 "${second_pid}" 2>/dev/null; do
    sleep 60
    echo "$(date -u +%FT%TZ) stage still running"
    tail -n 1 "${first_log}" 2>/dev/null || true
    tail -n 1 "${second_log}" 2>/dev/null || true
  done
}

echo "Stage E0: auditing true cost for every candidate in 5,000 pools."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
    --iters 5032 --ppo_asa_update_style reference_separate \
    --ppo_asa_disable_learning \
    --ppo_asa_oracle_pools 5000 --ppo_asa_oracle_stride 1 \
    --ppo_asa_oracle_diagnostics "${output_dir}/oracle-summary.json" \
    --asa_diagnostics "${output_dir}/oracle.jsonl" \
    --report "${output_dir}/oracle-report.json" \
    >"${output_dir}/oracle.log" 2>&1 &
oracle_pid=$!
while kill -0 "${oracle_pid}" 2>/dev/null; do
  sleep 60
  echo "$(date -u +%FT%TZ) oracle audit still running"
  tail -n 1 "${output_dir}/oracle.log" 2>/dev/null || true
done
wait "${oracle_pid}"
archive_results

run_planted_pair() {
  local label="$1"
  local style="$2"
  local trained_log="${output_dir}/planted-${label}-trained.log"
  local control_log="${output_dir}/planted-${label}-control.log"
  echo "Stage E1: ${label} planted-signal trained/control pair."
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
      --iters 200032 --ppo_asa_update_style "${style}" \
      --ppo_asa_reward_mode planted_proxy \
      --asa_diagnostics "${output_dir}/planted-${label}-trained.jsonl" \
      --save_checkpoint "${output_dir}/planted-${label}-trained.pt" \
      --report "${output_dir}/planted-${label}-trained-report.json" \
      >"${trained_log}" 2>&1 &
  trained_pid=$!
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
      --iters 200032 --ppo_asa_update_style "${style}" \
      --ppo_asa_reward_mode planted_proxy --ppo_asa_disable_learning \
      --asa_diagnostics "${output_dir}/planted-${label}-control.jsonl" \
      --report "${output_dir}/planted-${label}-control-report.json" \
      >"${control_log}" 2>&1 &
  control_pid=$!
  monitor_pair "${trained_pid}" "${control_pid}" \
    "${trained_log}" "${control_log}"
  pair_status=0
  if ! wait "${trained_pid}"; then pair_status=1; fi
  if ! wait "${control_pid}"; then pair_status=1; fi
  if [[ "${pair_status}" -ne 0 ]]; then
    echo "${label} planted-signal pair failed; inspect its logs." >&2
    exit 1
  fi
  archive_results
}

run_planted_pair shared shared_gae
run_planted_pair split reference_separate

echo "Analyzing E0 and E1. The real-reward pilot runs only if split PPO passes."
python src/analysis/analyze_ppo_asa_diagnostic_gate.py \
  "${output_dir}" --require-split-pass \
  | tee "${output_dir}/diagnostic-gate-analysis.txt"
archive_results

echo "Stage E2: 300-update split PPO with the real post-acceptance reward."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
    --iters 300032 --ppo_asa_update_style reference_separate \
    --ppo_asa_reward_mode realized \
    --ppo_asa_oracle_pools 2000 --ppo_asa_oracle_stride 150 \
    --ppo_asa_oracle_diagnostics "${output_dir}/real-oracle-summary.json" \
    --asa_diagnostics "${output_dir}/real-split-trained.jsonl" \
    --save_checkpoint "${output_dir}/real-split-trained.pt" \
    --report "${output_dir}/real-split-trained-report.json" \
    >"${output_dir}/real-split-trained.log" 2>&1 &
real_pid=$!
while kill -0 "${real_pid}" 2>/dev/null; do
  sleep 60
  echo "$(date -u +%FT%TZ) real-reward pilot still running"
  tail -n 1 "${output_dir}/real-split-trained.log" 2>/dev/null || true
done
wait "${real_pid}"

actual_updates="$(python - "${output_dir}/real-split-trained-report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)["algorithm_metadata"]["update_count"])
PY
)"
if [[ "${actual_updates}" -ne 300 ]]; then
  echo "Expected 300 real-reward updates, observed ${actual_updates}." >&2
  exit 1
fi
archive_results

echo "Running five-seed frozen holdout for the real-reward checkpoint."
bash scripts/run_kaggle_ppo_asa_holdout.sh \
  4100 "${output_dir}/real-holdout" \
  "${output_dir}/real-split-trained.pt" 5 reference_separate

python src/analysis/analyze_ppo_asa_diagnostic_gate.py "${output_dir}" \
  | tee "${output_dir}/final-analysis.txt"
archive_results
trap - EXIT

echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
