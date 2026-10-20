#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-online-followup}"
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory'; exit 2; }
mkdir -p "$out"
printf "running\n" > "$out/run-status.txt"
archive_results() {
 local status=$?
 if [[ "$status" -ne 0 ]]; then
  printf "failed\n" > "$out/run-status.txt"
  for log in "$out"/*.log; do [[ ! -f "$log" ]] || tail -n 35 "$log"; done
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

python tests/run_all_tests.py > "$out/tests.log" 2>&1
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_ranking_repair test_online_followup >> "$out/tests.log" 2>&1
model="${MODEL_OVERRIDE:-vgg16}"
budget="${BUDGET_OVERRIDE:-100000}"
read -r -a seeds <<< "${SEEDS_OVERRIDE:-6 7}"
for seed in "${seeds[@]}"; do
 for method in original uniform proxy fixed_sa; do
  extra=(--algo ppo_asa --ppo_asa_learner bandit --ppo_asa_annealing fixed
    --ppo_asa_candidates 16 --ppo_asa_focus_fraction .5 --ppo_asa_focus_stage_slack 0
    --ppo_asa_reward_mode proposed_delta --ppo_asa_state_encoding legacy_clipped
    --ppo_asa_bandit_feature_set legacy9 --ppo_asa_bandit_scoring direct
    --ppo_asa_bandit_exploration epsilon_all --ppo_asa_bandit_epsilon_final .5
    --ppo_asa_bandit_epsilon_decay 1 --ppo_asa_bandit_replay 50000
    --ppo_asa_bandit_warmup 512 --ppo_asa_rollout_steps 256
    --ppo_update_epochs 4 --ppo_minibatch_size 256 --ppo_hidden_dim 128
    --ppo_learning_rate .0003 --ppo_asa_weight_decay .0001
    --ppo_gamma 0 --ppo_gae_lambda 0 --ppo_entropy_coef .001
    --ppo_asa_update_style reference_separate)
  case "$method" in
   uniform) extra+=(--ppo_asa_disable_learning --ppo_asa_bandit_control uniform);;
   proxy) extra+=(--ppo_asa_disable_learning --ppo_asa_bandit_control proxy_greedy);;
   fixed_sa) extra=(--algo sa);;
  esac
  echo "Online follow-up: seed $seed, $method, $budget proposals"
  python -u src/run_multi_chip.py "${extra[@]}" --device cpu --cpu_threads 2 \
   --use_cnn --model "$model" --workload_region fc --partition_mode paper_targets \
   --logic_task_order output_group_interleaved --timing_model paper_pipeline \
   --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
   --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1 --sensitivity_trials 256 \
   --seed "$seed" --iters "$budget" \
   --asa_diagnostics "$out/seed${seed}-${method}.jsonl" \
   --report "$out/seed${seed}-${method}.json" > "$out/seed${seed}-${method}.log" 2>&1
  printf 'running; completed seed %s method %s\n' "$seed" "$method" > "$out/run-status.txt"
  archive_results
 done
done
python src/analyze_online_followup.py "$out" --seeds "${seeds[@]}"
printf 'complete\n' > "$out/run-status.txt"
