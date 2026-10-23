#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-ddpg-horizon}"
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
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_ranking_repair test_online_followup test_online_pool test_online_pool_summary test_recent_replay test_online_exploration test_online_ranking test_ddpg_comparison test_ddpg_horizon >> "$out/tests.log" 2>&1
seed="${SEED_OVERRIDE:-18}"
epochs="${DDPG_EPOCHS_OVERRIDE:-6}"
placements=$((epochs * 30))
baseline=1000
interval=30
budget=$((baseline + placements + (placements + interval - 1) / interval))
device=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
common=(--use_cnn --model vgg16 --workload_region fc --partition_mode paper_targets
 --logic_task_order output_group_interleaved --timing_model paper_pipeline
 --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16
 --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1 --sensitivity_trials 256
 --seed "$seed" --cpu_threads 2)
for method in default_gamma long_gamma frozen; do
 gamma=.98
 extra=()
 [[ "$method" != long_gamma ]] || gamma=.999
 [[ "$method" != frozen ]] || extra+=(--ddpg_disable_learning)
 echo "DDPG horizon: $method, $placements placements, device $device"
 python -u src/run_multi_chip.py "${common[@]}" --algo ddpg --device "$device" \
  --agent_arch paper_cnn --reward_mode sparse --batch_z 3 --train_every 1 \
  --epochs "$epochs" --placements_per_epoch 30 --baseline_trials "$baseline" \
  --ddpg_gamma "$gamma" "${extra[@]}" --diagnostics_every "$interval" \
  --retain_deterministic_candidates --diagnostics "$out/${method}.jsonl" \
  --save_checkpoint "$out/${method}.pt" --checkpoint_every 30 \
  --report "$out/${method}.json" > "$out/${method}.log" 2>&1
 archive_results
done
for method in exploit fixed_sa random; do
  extra=(--algo ppo_asa --ppo_asa_learner bandit --ppo_asa_annealing fixed
    --ppo_asa_candidates 16 --ppo_asa_focus_fraction .5 --ppo_asa_focus_stage_slack 0
    --ppo_asa_reward_mode proposed_delta --ppo_asa_state_encoding legacy_clipped
    --ppo_asa_bandit_feature_set legacy9 --ppo_asa_bandit_scoring direct
    --ppo_asa_bandit_exploration epsilon_all --ppo_asa_bandit_epsilon_final .02 --ppo_asa_pool_focus_fraction 0
    --ppo_asa_bandit_epsilon_decay 1 --ppo_asa_bandit_replay 50000
    --ppo_asa_bandit_warmup 512 --ppo_asa_rollout_steps 256
    --ppo_update_epochs 4 --ppo_minibatch_size 256 --ppo_hidden_dim 128
    --ppo_learning_rate .0003 --ppo_asa_weight_decay .0001
    --ppo_gamma 0 --ppo_gae_lambda 0 --ppo_entropy_coef .001
    --ppo_asa_update_style reference_separate)

 case "$method" in
  fixed_sa) extra=(--algo sa);;
  random) extra=(--algo random);;
 esac
 iters=$((budget - 1))
 [[ "$method" != random ]] || iters=$budget
 echo "Matched comparison: $method, $budget evaluations"
 python -u src/run_multi_chip.py "${common[@]}" "${extra[@]}" --device cpu \
  --iters "$iters" --asa_diagnostics "$out/${method}.jsonl" \
  --report "$out/${method}.json" > "$out/${method}.log" 2>&1
 archive_results
done
python src/analyze_ddpg_horizon.py "$out"
printf 'complete\n' > "$out/run-status.txt"
