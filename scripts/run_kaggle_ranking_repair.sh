#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-ranking-repair}"
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory'; exit 2; }
mkdir -p "$out"
archive_results() {
  local status=$?
  printf '%s\n' "$status" > "$out/exit-status.txt"
  python - "$out" <<'PY'
import sys,zipfile,os
from pathlib import Path
p=Path(sys.argv[1]); tmp=str(p)+'.zip.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file(): z.write(f,f.relative_to(p.parent))
os.replace(tmp,str(p)+'.zip')
print('Saved',str(p)+'.zip')
PY
}
trap archive_results EXIT
 git rev-parse HEAD > "$out/commit.txt"
 git status --short > "$out/git-status.txt"
python tests/run_all_tests.py > "$out/tests.log" 2>&1
PYTHONPATH=src python -m unittest test_ranking_repair >> "$out/tests.log" 2>&1
for seed in 30 31 32 33 34 35; do
 echo "Collecting Proxy-SA chain $seed"
 python -u src/run_multi_chip.py --algo ppo_asa --ppo_asa_learner bandit \
 --ppo_asa_disable_learning --ppo_asa_bandit_control proxy_greedy \
 --device cpu --cpu_threads 2 --use_cnn --model vgg16 --workload_region conv \
 --partition_mode paper_targets --timing_model paper_pipeline --routing_model paper_xy \
 --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
 --vva_ops_per_cycle 1 --partition_vva_ops_per_cycle 1 --sensitivity_trials 16 \
 --ppo_asa_annealing fixed --ppo_asa_candidates 16 \
 --ppo_asa_bandit_feature_set topology16 --ppo_asa_state_encoding relative_cost \
 --ppo_asa_focus_fraction .5 --ppo_asa_focus_stage_slack .05 \
 --ppo_asa_bandit_exploration epsilon_top_m --ppo_asa_bandit_top_m 4 \
 --ppo_asa_bandit_global_epsilon .05 --ppo_asa_bandit_epsilon_final .25 \
 --seed "$seed" --iters 4100 --ppo_asa_oracle_pools 256 --ppo_asa_oracle_stride 16 \
 --ppo_asa_oracle_dataset "$out/seed${seed}.npz" \
 --ppo_asa_oracle_diagnostics "$out/seed${seed}-oracle.json" \
 --report "$out/seed${seed}.json" > "$out/seed${seed}.log" 2>&1
 # Reports count audited pools separately; retain the complete accounting.
 python - "$out/seed${seed}.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));m=r['algorithm_metadata']
assert r['complete_placement_evaluations'] == 4101 + m['oracle_additional_objective_evaluations']
print('seed',r['config']['seed'],'reported total calls',r['complete_placement_evaluations'],
      'oracle calls',m['oracle_candidate_evaluations'])
PY
done
python src/diagnose_bandit_ranking.py "$out" --epochs 100
