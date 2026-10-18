#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-pairwise-rollout}"
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory'; exit 2; }
mkdir -p "$out"
archive_results() {
 local status=$?
 printf '%s\n' "$status" > "$out/exit-status.txt"
 python - "$out" <<'PY'
import sys,zipfile,os
from pathlib import Path
p=Path(sys.argv[1]);tmp=str(p)+'.zip.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(p.parent))
os.replace(tmp,str(p)+'.zip')
print('Saved',str(p)+'.zip')
PY
}
trap archive_results EXIT
git rev-parse HEAD > "$out/commit.txt"
git status --short > "$out/git-status.txt"
(cd artifacts/ranking-repair-20261018 && sha256sum -c SHA256SUMS) > "$out/data-integrity.log"
python tests/run_all_tests.py > "$out/tests.log" 2>&1
PYTHONPATH=src python -m unittest test_ranking_repair >> "$out/tests.log" 2>&1
python src/diagnose_bandit_ranking.py artifacts/ranking-repair-20261018 \
 --epochs 100 --export-signed "$out/signed-ranker.pt" > "$out/training.log" 2>&1
for seed in 40 41 42; do
 for method in checkpoint proxy_greedy uniform; do
  extra=()
  [[ "$method" != checkpoint ]] || extra+=(--load_checkpoint "$out/signed-ranker.pt")
  echo "VGG16-CONV fresh seed $seed, $method"
  python -u src/run_multi_chip.py --algo ppo_asa --ppo_asa_learner bandit \
   --ppo_asa_disable_learning --ppo_asa_bandit_control "$method" "${extra[@]}" \
   --device cpu --cpu_threads 2 --use_cnn --model vgg16 --workload_region conv \
   --partition_mode paper_targets --timing_model paper_pipeline --routing_model paper_xy \
   --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
   --vva_ops_per_cycle 1 --partition_vva_ops_per_cycle 1 --sensitivity_trials 16 \
   --ppo_asa_annealing fixed --ppo_asa_candidates 16 --ppo_hidden_dim 128 \
   --ppo_asa_bandit_feature_set topology16 --ppo_asa_state_encoding relative_cost \
   --ppo_asa_bandit_scoring direct --ppo_asa_focus_fraction .5 --ppo_asa_focus_stage_slack .05 \
   --ppo_asa_bandit_exploration epsilon_top_m --ppo_asa_bandit_top_m 4 \
   --ppo_asa_bandit_global_epsilon .05 --ppo_asa_bandit_epsilon_final .25 \
   --seed "$seed" --iters 4100 --asa_diagnostics "$out/seed${seed}-${method}.jsonl" \
   --report "$out/seed${seed}-${method}.json" > "$out/seed${seed}-${method}.log" 2>&1
 done
done
python - "$out" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);pairs=[]
for seed in (40,41,42):
 rows={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in ('checkpoint','proxy_greedy','uniform')}
 assert all(r['complete_placement_evaluations']==4101 and r['algorithm_metadata']['update_count']==0 for r in rows.values())
 assert len({r['algorithm_metadata']['initial_cost'] for r in rows.values()})==1
 m=rows['checkpoint']['algorithm_metadata']
 assert m['model_initialization']=='loaded_checkpoint' and m['source_training_updates']==1600
 assert m['proposal_policy']=='frozen_checkpoint_contextual_bandit'
 cost={k:r['best_cost'] for k,r in rows.items()}
 pairs.append({'seed':seed,'cost_seconds':cost,'runtime_seconds':{k:r['seconds_elapsed'] for k,r in rows.items()},
 'gain_over_proxy_percent':100*(cost['proxy_greedy']-cost['checkpoint'])/cost['proxy_greedy'],
 'gain_over_uniform_percent':100*(cost['uniform']-cost['checkpoint'])/cost['uniform']})
gains=[p['gain_over_proxy_percent'] for p in pairs]
summary={'protocol':'frozen signed pairwise scorer + fixed SA, 4101-call rollout screen',
 'training_seeds':[30,31,32,33],'model_selection_seeds':[34,35],'rollout_seeds':[40,41,42],
 'pairs':pairs,'mean_paired_gain_over_proxy_percent':statistics.mean(gains),
 'wins_over_proxy':sum(g>0 for g in gains),
 'advance_to_longer_rollouts':sum(g>0 for g in gains)>=2 and statistics.mean(gains)>=1 and sum(p['gain_over_uniform_percent']>0 for p in pairs)>=2,
 'limitation':'Supervised proposal learning, not online PPO or DDPG. Three-seed developmental screen; training and oracle collection are additional costs.'}
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
PY
