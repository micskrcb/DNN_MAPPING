import pathlib, subprocess, sys, json, os
root=pathlib.Path.cwd(); out=root/'runs/local-cpu-5seed-2026-09-30'
common=['--device','cpu','--cpu_threads','2','--cpu_interop_threads','1','--use_cnn','--model','alexnet','--partition_mode','paper_targets','--workload_region','conv','--timing_model','paper_pipeline','--routing_model','paper_xy','--chips_x','4','--chips_y','4','--rows','16','--cols','16','--sensitivity_trials','64']
for seed in range(5):
 for algo in ['ddpg','bs','random','sa','asa']:
  stem=f'{algo}-seed{seed}' if algo in ['ddpg','bs'] else f'{algo}-budget82-seed{seed}'
  report=out/(stem+'.json')
  if report.exists(): continue
  cmd=[sys.executable,'src/run_multi_chip.py','--algo',algo,'--seed',str(seed),'--report',str(report),*common]
  if algo=='ddpg':
   diag=out/(stem+'-ddpg.jsonl')
   if diag.exists(): diag.rename(out/(stem+'-interrupted.jsonl'))
   cmd+=['--epochs','12','--baseline_trials','64','--batch_z','3','--train_every','10','--agent_arch','paper_cnn','--reward_mode','potential','--diagnostics_every','2','--exploration_decay_placements','1000','--retain_deterministic_candidates','--checkpoint_every','2','--save_checkpoint',str(out/(stem+'.pt')),'--diagnostics',str(diag)]
  elif algo!='bs': cmd+=['--iters','82']
  print('Starting',stem,flush=True)
  with (out/(stem+'.log')).open('w') as f:
   subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('Completed',stem,json.loads(report.read_text())['best_cost'],flush=True)
print('ALL COMPLETE',flush=True)
