#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="${1:-runs/kaggle-pool-context}"
# Keep the lock outside the result directory, which the stage creates.
mkdir -p "$(dirname "$out")"
exec 9>"${out}.lock"
flock -n 9 || { echo 'This output already has an active run'; exit 2; }
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_pool_context
RESUME_STAGE=1 MODEL_OVERRIDE=vgg16 WORKLOAD_REGION_OVERRIDE=fc \
SEEDS_OVERRIDE=29 BUDGET_OVERRIDE=200000 \
METHODS_OVERRIDE='train_more pool_context fixed_sa' \
bash scripts/run_kaggle_online_rapid_stage.sh "$out"
python src/analyze_pool_context.py "$out" --seed 29
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
