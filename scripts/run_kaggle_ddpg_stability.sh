#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/kaggle-ddpg-stability}"
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
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m unittest test_ranking_repair test_online_followup test_online_pool test_online_pool_summary test_recent_replay test_online_exploration test_online_ranking test_ddpg_comparison test_ddpg_horizon test_ddpg_stability >> "$out/tests.log" 2>&1
seed=20
device=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
common=(--use_cnn --model vgg16 --workload_region fc --partition_mode paper_targets
 --logic_task_order output_group_interleaved --timing_model paper_pipeline
 --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16
 --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1 --sensitivity_trials 256
 --seed "$seed" --cpu_threads 2)
for method in unclipped clipped; do
 clip=0
 [[ "$method" != clipped ]] || clip=1
 echo "DDPG stability: $method, 30 placements"
 python -u src/run_multi_chip.py "${common[@]}" --algo ddpg --device "$device" \
  --agent_arch paper_cnn --reward_mode sparse --batch_z 3 --train_every 1 \
  --epochs 1 --placements_per_epoch 30 --baseline_trials 128 \
  --ddpg_gamma .999 --ddpg_grad_clip "$clip" --diagnostics_every 10 \
  --retain_deterministic_candidates --diagnostics "$out/${method}.jsonl" \
  --report "$out/${method}.json" > "$out/${method}.log" 2>&1
 archive_results
done
python - "$out" <<'PYREPORT'
import json,sys,math
from pathlib import Path
p=Path(sys.argv[1]);result={}
for name in ('unclipped','clipped'):
 r=json.loads((p/f'{name}.json').read_text());m=r['algorithm_metadata']
 assert m['search_evaluations_including_baseline']==161
 assert m['ddpg_grad_clip']==(1 if name=='clipped' else 0)
 rows=[json.loads(x) for x in (p/f'{name}.jsonl').read_text().splitlines()]
 result[name]={'best_cost_microseconds':r['best_cost']*1e6,'best_source':m['best_candidate_source'],'diagnostics':rows,'runtime_seconds':r['seconds_elapsed']}
result['note']='30-placement numerical diagnostic only. Not convergence, published comparison or proof that clipping fixes DDPG.'
(p/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
PYREPORT
printf 'complete\n' > "$out/run-status.txt"
