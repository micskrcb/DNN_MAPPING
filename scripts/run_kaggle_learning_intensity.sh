#!/usr/bin/env bash
set -euo pipefail
# Bound native-library threads as well as PyTorch on shared Kaggle CPUs.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out="${1:-runs/kaggle-learning-intensity}"
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory'; exit 2; }
mkdir -p "$out"
finish() {
 status=$?
 printf '%s\n' "$status" > "$out/exit-status.txt"
 python - "$out" <<'PYZIP'
import sys,zipfile,os
from pathlib import Path
p=Path(sys.argv[1]);tmp=str(p)+'.zip.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file() and not f.name.endswith('.zip'):z.write(f,f.relative_to(p.parent))
os.replace(tmp,str(p)+'.zip')
print('Saved',str(p)+'.zip')
PYZIP
}
trap finish EXIT
# Separate exploratory outcomes; no pooled cross-workload percentage.
seed="${SEED_OVERRIDE:-21}"
for model in vgg16 alexnet; do
 MODEL_OVERRIDE="$model" WORKLOAD_REGION_OVERRIDE=fc SEEDS_OVERRIDE="$seed" \
 BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-100000}" \
 METHODS_OVERRIDE="exploit train_more fixed_sa" \
 bash scripts/run_kaggle_online_rapid_stage.sh "$out/$model-fc"
 finish
done
python src/analyze_learning_intensity.py "$out" --seed "$seed"
printf 'complete\n' > "$out/run-status.txt"
