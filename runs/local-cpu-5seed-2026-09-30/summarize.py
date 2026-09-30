from pathlib import Path
import json,statistics,math
p=Path('runs/local-cpu-5seed-2026-09-30')
methods={'Trained DDPG':'ddpg-seed{}','Untrained control':'untrained-seed{}','Random search':'random-budget82-seed{}','Fixed SA':'sa-budget82-seed{}','Adaptive SA':'asa-budget82-seed{}','Sequential':'bs-seed{}'}
data={m:[json.loads((p/(pat.format(i)+'.json')).read_text()) for i in range(5)] for m,pat in methods.items()}
values={m:[x['best_cost'] for x in v] for m,v in data.items()}
agg={m:{'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)} for m,v in values.items()}
diff=[(u-t)*1e6 for u,t in zip(values['Untrained control'],values['Trained DDPG'])]
mean=statistics.mean(diff); half=2.776445105*statistics.stdev(diff)/math.sqrt(5)
summary={'seeds':list(range(5)),'best_cost_seconds':values,'aggregates':agg,'paired_untrained_minus_trained_microseconds':diff,'paired_mean_microseconds':mean,'approximate_t95_interval_microseconds':[mean-half,mean+half],'note':'Exploratory five-seed short diagnostic, not a convergence or paper reproduction study.'}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text='# Local CPU five-seed diagnostic\n\nCompleted 2026-09-30 on the CPU branch, implementation commit `6bee27f`. Lower cost is better.\n\n'
text+='| Method | Mean latency (µs) | Sample SD (µs) |\n|---|---:|---:|\n'
for m,a in agg.items():text+=f"| {m} | {a['mean']*1e6:.5f} | {a['sample_sd']*1e6:.5f} |\n"
text+='\n| Seed | Trained best (µs) | Untrained best (µs) |\n|---|---:|---:|\n'
for i,(t,u) in enumerate(zip(values['Trained DDPG'],values['Untrained control'])):text+=f'| {i} | {t*1e6:.5f} | {u*1e6:.5f} |\n'
text+=f'\nThe paired mean benefit of training was {mean:.5f} µs. The approximate paired 95% t interval was [{mean-half:.5f}, {mean+half:.5f}] µs; five seeds and a discrete objective make this interval exploratory.\n'
text+='''
## Protocol and limits

- AlexNet CONV, 183 tasks, paper-target reconstruction, paper CNN, potential shaping and deterministic retention. These optional extensions are not the frozen sparse-reward paper condition.
- Per DDPG/control seed: 12 complete training rollouts plus six deterministic evaluations, with 64 random baseline samples also eligible as the saved best: 82 candidates. The control uses `train_every=100000000` and performs zero gradient updates. Trained runs use `train_every=10`.
- RS uses 82 trials. SA/ASA use 82 proposals plus one initial placement: 83 candidates. ASA calibration is included in the proposal budget. This is approximately matched candidate count, not equal compute or runtime.
- Seed 0 DDPG and sequential runs were preserved from the earlier ten-thread run using two preflight samples and an explicit override. Other completed runs use two threads and 64 preflight samples. Preflight preserves RNG state and does not contribute candidates to optimization. Thread count can affect floating-point reproducibility; timings are not directly comparable.
- The interrupted seed-1 trace is historical only and excluded from aggregates. Fresh seed-1 training completed without checkpoint resumption. Large checkpoints remain local and are excluded from Git.
- The short fixed-SA schedule is not temperature-calibrated to seconds. ASA barely exceeds its calibration/adaptation window. These are weak short-budget comparisons, not evidence against well-tuned annealing.
- The untrained control already obtains strong layouts. Beating RS/SA here does not establish that DDPG learned. Test longer runs and policy quality versus the untrained control before a paper-scale launch.

`run_remaining.py` recreates missing completed runs from the repository root, skipping existing reports. `run_untrained_controls.py` reproduces the zero-update control. Use the project Python environment. Raw reports and JSONL traces preserve actual settings and runtimes.
'''
(p/'STUDY.md').write_text(text)
print(text[:2300])
