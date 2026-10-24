#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-learning-followup}"
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
# Main research method first; comparator debugging is short and secondary.
SEEDS_OVERRIDE=19 BUDGET_OVERRIDE=100000 METHODS_OVERRIDE="exploit fast_exploit fixed_sa" bash scripts/run_kaggle_online_rapid_stage.sh "$out/bandit"
finish
bash scripts/run_kaggle_ddpg_stability.sh "$out/ddpg-stability"
printf 'complete\n' > "$out/run-status.txt"
