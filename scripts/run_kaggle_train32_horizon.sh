#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="${1:-runs/kaggle-train32-horizon}"
seed="${SEED_OVERRIDE:-28}"
mkdir -p "$out"
revision="$(git rev-parse HEAD)"
if [[ -f "$out/commit.txt" ]] && [[ "$(cat "$out/commit.txt")" != "$revision" ]]; then
  echo 'Output belongs to another commit; choose a new directory.'
  exit 2
fi
printf '%s\n' "$revision" > "$out/commit.txt"

archive() {
  local status=$?
  printf '%s\n' "$status" > "$out/exit-status.txt"
  python - "$out" <<'PY'
from pathlib import Path
import os, sys, zipfile
root = Path(sys.argv[1])
temporary = str(root) + '.zip.tmp'
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.name.endswith(('.zip', '.tmp')):
            archive.write(path, path.relative_to(root.parent))
os.replace(temporary, str(root) + '.zip')
PY
}
trap archive EXIT

printf 'running\n' > "$out/run-status.txt"
stage="$out/seed$seed"
RESUME_STAGE=1 MODEL_OVERRIDE=vgg16 WORKLOAD_REGION_OVERRIDE=fc \
SEEDS_OVERRIDE="$seed" BUDGET_OVERRIDE=200000 \
METHODS_OVERRIDE='train_more train_32 fixed_sa' \
bash scripts/run_kaggle_online_rapid_stage.sh "$stage"
python src/analyze_training_intensity_horizon.py "$stage" --seed "$seed"
cp "$stage/summary.json" "$out/seed${seed}-summary.json"
printf 'complete\n' > "$out/run-status.txt"
