#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="${1:-runs/kaggle-alexnet-pool-transfer}"
seed="${2:-34}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer"; exit 2; }
# Keep the lock outside the result directory, which the stage creates.
mkdir -p "$(dirname "$out")"
exec 9>"${out}.lock"
flock -n 9 || { echo 'This output already has an active run'; exit 2; }
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_pool_context test_pool_transfer
RESUME_STAGE=1 MODEL_OVERRIDE=alexnet WORKLOAD_REGION_OVERRIDE=fc \
SEEDS_OVERRIDE="$seed" BUDGET_OVERRIDE=200000 \
METHODS_OVERRIDE='pool_context pool_frozen fixed_sa' \
bash scripts/run_kaggle_online_rapid_stage.sh "$out"
python src/analyze_pool_transfer.py "$out" --seed "$seed"
# Refresh the archive to include the stricter feature comparison summary.
python - "$out" <<'PY'
import os, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
with zipfile.ZipFile(str(root)+'.zip.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            z.write(path, path.relative_to(root.parent))
os.replace(str(root)+'.zip.tmp', str(root)+'.zip')
PY
