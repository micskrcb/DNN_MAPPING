#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="${1:-runs/kaggle-learning-horizon}"
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo "This output directory already has an active run."; exit 2; }
revision="$(git rev-parse HEAD)"
if [[ -f "$out/commit.txt" ]] && [[ "$(cat "$out/commit.txt")" != "$revision" ]]; then
 echo 'Output belongs to another commit; choose a new directory.'; exit 2
fi
printf '%s\n' "$revision" > "$out/commit.txt"
archive() {
 local status=$?
 printf '%s\n' "$status" > "$out/exit-status.txt"
 python - "$out" <<'PY'
from pathlib import Path
import os,sys,zipfile
p=Path(sys.argv[1]);tmp=str(p)+'.zip.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file() and not f.name.endswith(('.zip','.tmp')):z.write(f,f.relative_to(p.parent))
os.replace(tmp,str(p)+'.zip')
PY
}
trap archive EXIT
printf 'running\n' > "$out/run-status.txt"
for seed in 25 26 27; do
 stage="$out/seed$seed"
 if [[ -f "$stage/run-status.txt" ]] && [[ "$(cat "$stage/run-status.txt")" == complete ]]; then
  test "$(cat "$stage/commit.txt")" = "$revision"
  python src/analyze_online_rapid.py "$stage" --seed "$seed" --methods train_more frozen fixed_sa
  echo "Validated completed seed $seed; skipping."
 else
  # Completed arms are retained; an unfinished arm restarts, not checkpoint-resumes.
  RESUME_STAGE=1 MODEL_OVERRIDE=vgg16 WORKLOAD_REGION_OVERRIDE=fc SEEDS_OVERRIDE="$seed" BUDGET_OVERRIDE=200000 \
  METHODS_OVERRIDE='train_more frozen fixed_sa' bash scripts/run_kaggle_online_rapid_stage.sh "$stage"
 fi
 python src/analyze_learning_horizon.py "$out" --seeds 25 26 27
 archive
done
printf 'complete\n' > "$out/run-status.txt"
