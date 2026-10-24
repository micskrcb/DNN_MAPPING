#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MODEL_OVERRIDE=vgg16 WORKLOAD_REGION_OVERRIDE=fc SEEDS_OVERRIDE=24
export BUDGET_OVERRIDE=100000
export METHODS_OVERRIDE='train_more attraction attraction_uniform fixed_sa'
bash scripts/run_kaggle_online_rapid_stage.sh "${1:-runs/kaggle-attraction-bandit-seed24}"
