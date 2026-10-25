#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-online-rapid-stage}"
if [[ -e "$out" ]]; then
 [[ "${RESUME_STAGE:-0}" == 1 && -f "$out/commit.txt" ]] || { echo 'Use a fresh output directory'; exit 2; }
 [[ "$(cat "$out/commit.txt")" == "$(git rev-parse HEAD)" ]] || { echo 'Commit mismatch'; exit 2; }
fi
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
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_ranking_repair test_online_followup test_online_pool test_online_pool_summary test_recent_replay test_online_exploration test_online_ranking test_learning_intensity test_structured_moves test_attraction_moves test_improvement_priority >> "$out/tests.log" 2>&1
model="${MODEL_OVERRIDE:-vgg16}"
region="${WORKLOAD_REGION_OVERRIDE:-fc}"
budget="${BUDGET_OVERRIDE:-25000}"
read -r -a seeds <<< "${SEEDS_OVERRIDE:-15}"
python - "$out" "${seeds[@]}" <<'PYENV'
import json,platform,sys,torch,numpy
from pathlib import Path
(Path(sys.argv[1])/'run-environment.json').write_text(json.dumps({
 'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
 'numpy':numpy.__version__,'seeds':[int(x) for x in sys.argv[2:]],
 'device':'cpu','requested_cpu_threads':2},indent=2))
PYENV
read -r -a methods <<< "${METHODS_OVERRIDE:-reference exploit topology pool_regression ranking}"
for seed in "${seeds[@]}"; do
 for method in "${methods[@]}"; do
  extra=(--algo ppo_asa --ppo_asa_learner bandit --ppo_asa_annealing fixed
    --ppo_asa_candidates 16 --ppo_asa_focus_fraction .5 --ppo_asa_focus_stage_slack 0
    --ppo_asa_reward_mode proposed_delta --ppo_asa_state_encoding legacy_clipped
    --ppo_asa_bandit_feature_set legacy9 --ppo_asa_bandit_scoring direct
    --ppo_asa_bandit_exploration epsilon_all --ppo_asa_bandit_epsilon_final .1 --ppo_asa_pool_focus_fraction 0
    --ppo_asa_bandit_epsilon_decay 1 --ppo_asa_bandit_replay 50000
    --ppo_asa_bandit_warmup 512 --ppo_asa_rollout_steps 256
    --ppo_update_epochs 4 --ppo_minibatch_size 256 --ppo_hidden_dim 128
    --ppo_learning_rate .0003 --ppo_asa_weight_decay .0001
    --ppo_gamma 0 --ppo_gae_lambda 0 --ppo_entropy_coef .001
    --ppo_asa_update_style reference_separate)
  case "$method" in
   reference) ;;
   fast_exploit) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_asa_bandit_epsilon_decay .5);;
   attraction) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_pool_attraction_fraction .5);;
   attraction_uniform) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_pool_attraction_fraction .5 --ppo_asa_disable_learning);;
   structured) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_pool_structured_fraction .5);;
   structured_uniform) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_pool_structured_fraction .5 --ppo_asa_disable_learning);;
   priority) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_reward_mode improvement_priority);;
   frozen) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16 --ppo_asa_disable_learning);;
   train_more) extra+=(--ppo_asa_bandit_epsilon_final .02 --ppo_update_epochs 16);;
   exploit) extra+=(--ppo_asa_bandit_epsilon_final .02);;
   topology) extra+=(--ppo_asa_bandit_feature_set topology16);;
   pool_regression) extra+=(--ppo_asa_bandit_pool_supervision_stride 64);;
   ranking) extra+=(--ppo_asa_bandit_pool_supervision_stride 64 --ppo_asa_bandit_ranking_weight .25);;
   fixed_sa) extra=(--algo sa);;
   *) echo "Unknown method: $method"; exit 2;;
  esac
  echo "Rapid development: seed $seed, $method, $budget proposals"
  command=(python -u src/run_multi_chip.py "${extra[@]}" --device cpu --cpu_threads 2 \
   --use_cnn --model "$model" --workload_region "$region" --partition_mode paper_targets \
   --logic_task_order output_group_interleaved --timing_model paper_pipeline \
   --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
   --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1 --sensitivity_trials 256 \
   --seed "$seed" --iters "$budget" \
   --asa_diagnostics "$out/seed${seed}-${method}.jsonl" \
   --report "$out/seed${seed}-${method}.json")
  manifest="$out/seed${seed}-${method}.command"
  printf '%q ' "${command[@]}" > "${manifest}.pending"
  if [[ -f "$manifest" ]]; then
   cmp -s "$manifest" "${manifest}.pending" || { echo 'Saved command differs; use fresh output'; exit 2; }
  fi
  mv "${manifest}.pending" "$manifest"
  if [[ -f "$out/seed${seed}-${method}.done" ]]; then
   python - "$out/seed${seed}-${method}.json" "$budget" <<'PYVALID'
import json,sys,math
r=json.load(open(sys.argv[1]))
assert r['complete_placement_evaluations']==int(sys.argv[2])+1
assert math.isfinite(r['best_cost']) and r['best_cost']>0
PYVALID
   echo "Skipping completed seed $seed method $method"
   continue
  fi
  "${command[@]}" > "$out/seed${seed}-${method}.log" 2>&1
  touch "$out/seed${seed}-${method}.done"
  printf 'running; completed seed %s method %s\n' "$seed" "$method" > "$out/run-status.txt"
  archive_results
 done
 # Save a validated per-seed summary before starting the next seed.
 python src/analyze_online_rapid.py "$out" --seed "$seed" --methods "${methods[@]}"
 cp "$out/summary.json" "$out/seed${seed}-summary.json"
 archive_results
done

printf 'complete\n' > "$out/run-status.txt"
