#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-improvement-ranker-gate}"
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

if ! python -c 'import torch; assert torch.cuda.is_available()'; then
  echo "This experiment requires a Kaggle GPU session." >&2
  exit 1
fi

echo "Running reconciliation tests."
PYTHONPATH=src python -m unittest src.test_reconciliation

python - "${output_dir}/experiment-manifest.json" <<'PY'
import json
import subprocess
import sys

manifest = {
    "experiment": "fresh full-chain improvement-only ranking gate",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "prespecified_before_data_generation": True,
    "dataset": {
        "chains": 20,
        "proposals_per_chain": 4100,
        "oracle_stride": 5,
        "expected_pools": 16394,
        "candidates_per_pool": 16,
    },
    "training": {
        "target": "max(exact_gain, 0)",
        "training_chains": 10,
        "validation_chains": 10,
        "maximum_epochs": 300,
    },
    "gate": {
        "scope": "held-out-chain late half",
        "minimum_argmax_capture": 0.10,
        "minimum_chain_bootstrap_lower_bound": 0.05,
        "minimum_improvement_rate_ratio": 1.5,
        "minimum_positive_chains": 8,
        "minimum_sampled_capture": 0.05,
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
  --ppo_asa_rollout_steps 4100
  --ppo_asa_update_style reference_separate
  --ppo_hidden_dim 128
)

echo "Stage G0: collecting 20 full 4,100-proposal chains at oracle stride 5."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
    --iters 82000 --ppo_asa_disable_learning \
    --ppo_asa_oracle_pools 16394 --ppo_asa_oracle_stride 5 \
    --ppo_asa_oracle_diagnostics "${output_dir}/oracle-summary.json" \
    --ppo_asa_oracle_dataset "${output_dir}/oracle-dataset.npz" \
    --asa_diagnostics "${output_dir}/oracle-collection.jsonl" \
    --report "${output_dir}/oracle-collection-report.json" \
    >"${output_dir}/oracle-collection.log" 2>&1
archive_results

echo "Stage G1: training on ten chains and testing ten unseen chains."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/train_oracle_candidate_ranker.py \
    "${output_dir}/oracle-dataset.npz" \
    --checkpoint "${output_dir}/improvement-ranker.pt" \
    --report "${output_dir}/improvement-ranking-report.json" \
    --device cuda --hidden-dim 128 --epochs 300 --batch-size 256 \
    --learning-rate 3e-4 --weight-decay 1e-4 \
    --target-mode improvement_only --target-temperature 0.5 \
    --validation-fraction 0.5 --patience 40 \
    --strict-chain-gate --bootstrap-samples 5000 --seed 20261011 \
    >"${output_dir}/improvement-ranking.log" 2>&1
archive_results

python - "${output_dir}/improvement-ranking-report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
print(json.dumps({
    "verdict": report["verdict"],
    "pass": report["pass"],
    "criterion": report["criterion"],
    "validation_chain_metrics": report["validation_chain_metrics"],
}, indent=2))
PY

archive_results
trap - EXIT
echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
