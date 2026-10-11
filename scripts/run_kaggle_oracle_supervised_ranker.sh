#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-oracle-supervised-ranker}"
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
  echo "This experiment requires a Kaggle T4 x2 session." >&2
  exit 1
fi

echo "Running reconciliation tests."
PYTHONPATH=src python -m unittest src.test_reconciliation

python - "${output_dir}/experiment-manifest.json" <<'PY'
import json
import subprocess
import sys

manifest = {
    "experiment": "oracle-supervised PPO-ASA candidate ranking gate",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "stages": {
        "dataset": {"pools": 8000, "candidates_per_pool": 16},
        "supervised_training": {
            "training_fraction": 0.8,
            "maximum_epochs": 300,
            "early_stopping_patience": 40,
        },
        "conditional": {
            "frozen_holdout_seeds": 5,
            "ppo_finetuning_proposals": 200000,
            "finetuned_holdout_seeds": 5,
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
  --ppo_asa_rollout_steps 1000
  --ppo_asa_update_style reference_separate
  --ppo_hidden_dim 128
)

echo "Stage S0: collecting exact gains for 8,000 candidate pools."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
    --iters 8032 --ppo_asa_disable_learning \
    --ppo_asa_restart_interval 832 \
    --ppo_asa_oracle_pools 8000 --ppo_asa_oracle_stride 1 \
    --ppo_asa_oracle_diagnostics "${output_dir}/oracle-summary.json" \
    --ppo_asa_oracle_dataset "${output_dir}/oracle-dataset.npz" \
    --asa_diagnostics "${output_dir}/oracle-collection.jsonl" \
    --report "${output_dir}/oracle-collection-report.json" \
    >"${output_dir}/oracle-collection.log" 2>&1
archive_results

echo "Stage S1: training on 80% of pools and evaluating unseen pools."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/train_oracle_candidate_ranker.py \
    "${output_dir}/oracle-dataset.npz" \
    --checkpoint "${output_dir}/oracle-supervised.pt" \
    --report "${output_dir}/supervised-ranking-report.json" \
    --device cuda --hidden-dim 128 --epochs 300 --batch-size 256 \
    --learning-rate 3e-4 --weight-decay 1e-4 \
    --target-temperature 0.5 --validation-fraction 0.2 \
    --patience 40 --seed 20261011 \
    >"${output_dir}/supervised-ranking.log" 2>&1
archive_results

ranking_pass="$(python - "${output_dir}/supervised-ranking-report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    print("yes" if json.load(stream)["pass"] else "no")
PY
)"

if [[ "${ranking_pass}" != "yes" ]]; then
  echo "SUPERVISED RANKING GATE STOPPED: current features do not generalize."
  echo "The ZIP is complete; do not spend another long run on PPO fine-tuning."
  archive_results
  trap - EXIT
  echo "Download ${archive} from the Kaggle Output/Files panel."
  sha256sum "${archive}"
  exit 0
fi

echo "Stage S2: five-seed frozen holdout for the supervised actor."
bash scripts/run_kaggle_ppo_asa_holdout.sh \
  4100 "${output_dir}/supervised-holdout" \
  "${output_dir}/oracle-supervised.pt" 5 reference_separate
archive_results

echo "Stage S3: conservative PPO fine-tuning from the supervised actor."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/run_multi_chip.py --algo ppo_asa "${common[@]}" \
    --iters 200032 --load_checkpoint "${output_dir}/oracle-supervised.pt" \
    --ppo_asa_restart_interval 4100 \
    --ppo_asa_finetune_loaded --ppo_asa_reward_mode proposed_delta \
    --ppo_learning_rate 5e-5 --ppo_asa_weight_decay 1e-4 \
    --ppo_update_epochs 4 --ppo_minibatch_size 1024 \
    --ppo_gamma 0.9 --ppo_gae_lambda 0.9 --ppo_clip_ratio 0.15 \
    --ppo_entropy_coef 0 --ppo_value_coef 0.5 --ppo_max_grad_norm 0.5 \
    --asa_diagnostics "${output_dir}/finetuned.jsonl" \
    --save_checkpoint "${output_dir}/finetuned.pt" \
    --report "${output_dir}/finetuned-report.json" \
    >"${output_dir}/finetuned.log" 2>&1
archive_results

echo "Stage S4: five-seed frozen holdout for the fine-tuned actor."
bash scripts/run_kaggle_ppo_asa_holdout.sh \
  4100 "${output_dir}/finetuned-holdout" \
  "${output_dir}/finetuned.pt" 5 reference_separate

python - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
with (directory / "supervised-ranking-report.json").open() as stream:
    ranking = json.load(stream)
with (directory / "supervised-holdout" / "holdout-summary.json").open() as stream:
    supervised = json.load(stream)
with (directory / "finetuned-holdout" / "holdout-summary.json").open() as stream:
    finetuned = json.load(stream)
summary = {
    "supervised_ranking": ranking,
    "supervised_frozen_holdout": supervised,
    "ppo_finetuned_holdout": finetuned,
}
with (directory / "final-summary.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
PY

archive_results
trap - EXIT
echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
