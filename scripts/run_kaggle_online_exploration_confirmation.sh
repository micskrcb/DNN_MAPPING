#!/usr/bin/env bash
set -euo pipefail
# Run seed13 and seed14 in separate sessions: about six hours each.
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-13}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-100000}"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-vgg16}"
exec bash scripts/run_kaggle_online_exploration.sh "${1:-runs/kaggle-online-exploration-confirmation}"
