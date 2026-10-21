#!/usr/bin/env bash
set -euo pipefail
# Separate sessions avoid putting two ~5.7-hour seeds near a notebook limit.
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-9}"
export BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-100000}"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-vgg16}"
exec bash scripts/run_kaggle_online_pool_gate.sh "${1:-runs/kaggle-online-pool-confirmation}"
