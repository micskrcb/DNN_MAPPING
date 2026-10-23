#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-online-rapid}"
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
SEEDS_OVERRIDE=15 BUDGET_OVERRIDE=25000 METHODS_OVERRIDE="reference exploit topology pool_regression ranking" bash scripts/run_kaggle_online_rapid_stage.sh "$out/screen"
finish
winner=$(python - "$out/screen/summary.json" <<'PYSELECT'
import json,sys
print(json.load(open(sys.argv[1]))['selected_challenger'])
PYSELECT
)
printf '%s\n' "$winner" > "$out/selected-challenger.txt"
# Test the best challenger at full budget even if its short screen loses.
# This avoids assuming short-run ordering predicts long-run performance.
full_methods="reference $winner fixed_sa"
if [[ "$winner" == "ranking" ]]; then full_methods="$full_methods pool_regression"; fi
SEEDS_OVERRIDE=16 BUDGET_OVERRIDE=100000 METHODS_OVERRIDE="$full_methods" bash scripts/run_kaggle_online_rapid_stage.sh "$out/full"
printf 'complete\n' > "$out/run-status.txt"
