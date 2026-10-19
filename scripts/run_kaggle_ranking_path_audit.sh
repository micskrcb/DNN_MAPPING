#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-ranking-path-audit}"
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory'; exit 2; }
mkdir -p "$out"
archive_results() {
 local status=$?
 if [[ "$status" -ne 0 ]]; then
  echo "Run failed; recent diagnostic logs:"
  for log in "$out"/*.log; do
   [[ ! -f "$log" ]] || tail -n 35 "$log"
  done
 fi
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
for seed in 40 41; do
 for selection in exploratory greedy; do
 for method in checkpoint proxy_greedy; do
  extra=()
  [[ "$method" != checkpoint ]] || extra+=(--load_checkpoint "$out/signed-ranker.pt")
  mode_args=()
  [[ "$selection" != greedy ]] || mode_args+=(--ppo_asa_bandit_warmup 0 --ppo_asa_bandit_top_m 1 --ppo_asa_bandit_global_epsilon 0)
  name="seed${seed}-${method}-${selection}"
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
   "${mode_args[@]}" --seed "$seed" --iters 4100 \
   --ppo_asa_oracle_pools 64 --ppo_asa_oracle_stride 64 \
   --ppo_asa_oracle_dataset "$out/$name.npz" \
   --ppo_asa_oracle_diagnostics "$out/$name-oracle.json" \
   --asa_diagnostics "$out/$name.jsonl" \
   --report "$out/$name.json" > "$out/$name.log" 2>&1
 done
done
done
python src/analyze_ranking_path.py "$out"
