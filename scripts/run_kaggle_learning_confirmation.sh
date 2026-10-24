#!/usr/bin/env bash
set -euo pipefail
# Fresh seed; retain the selected learner unchanged to isolate repeatability.
export SEED_OVERRIDE=22
export BUDGET_OVERRIDE=100000
bash scripts/run_kaggle_learning_intensity.sh "${1:-runs/kaggle-learning-confirmation-seed22}"
