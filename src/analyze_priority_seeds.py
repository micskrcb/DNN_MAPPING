"""Three-seed development summary with all outcomes and budget checkpoints."""
import argparse
import json
import statistics
from pathlib import Path
from analyze_online_rapid import summarize

METHODS=['train_more','priority','fixed_sa']

def analyze(root,seeds,methods=None,challenger="priority",controls=None,allowed_changes=None):
    methods = methods or METHODS
    controls = controls or ["train_more", "fixed_sa"]
    allowed_changes = set(allowed_changes or ["ppo_asa_reward_mode", "report", "asa_diagnostics"])
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seed')
    pairs=[];signature=None
    for seed in seeds:
        folder=Path(root)/f'seed{seed}'
        if not (folder/'run-status.txt').exists():continue
        if (folder/'run-status.txt').read_text().strip()!='complete':continue
        s=summarize(folder,seed,methods)
        reports={m:json.loads((folder/f'seed{seed}-{m}.json').read_text()) for m in methods}
        configs=[reports[m]['config'] for m in methods[:2]]
        changed={k for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)}
        if changed-allowed_changes:raise ValueError('Learning arms differ beyond permitted ablation')
        cfg=configs[0]
        sig=tuple(cfg.get(k) for k in ('model','workload_region','partition_mode','timing_model','routing_model','logic_task_order','chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','iters'))
        if signature is not None and signature!=sig:raise ValueError('Seed workloads/budgets differ')
        signature=sig
        s['checkpoints']={}
        for method in methods:
            path=folder/f'seed{seed}-{method}.jsonl'
            rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
            points=[]
            for fraction in (.25,.5,.75,1.):
                cutoff=int(cfg['iters']*fraction)
                eligible=[r for r in rows if r.get('evaluations',cutoff+1)<=cutoff]
                row=max(eligible,key=lambda r:r['evaluations']) if eligible else None
                points.append({'requested_evaluations':cutoff,'logged_evaluations':row['evaluations'] if row else None,
                               'best_cost_microseconds':row['best_cost']*1e6 if row else None,
                               'selected_move_outcomes':row.get('selected_move_outcomes') if row else None})
            s['checkpoints'][method]=points
        pairs.append(s)
    aggregate={}
    for control in controls:
        gains=[100*(1-p['best_cost_microseconds'][challenger]/p['best_cost_microseconds'][control]) for p in pairs]
        aggregate[control]={'mean_paired_gain_percent':statistics.mean(gains) if gains else None,
            'sample_std_gain_percent':statistics.stdev(gains) if len(gains)>1 else None,
            'wins':sum(g>0 for g in gains),'ties':sum(g==0 for g in gains),'losses':sum(g<0 for g in gains),
            'individual_gains_percent':gains}
    return {'requested_seeds':seeds,'completed_seeds':[p['seed'] for p in pairs],
            'provisional':len(pairs)!=len(seeds),'comparisons':aggregate,'pairs':pairs,
            'note':'Three-seed development evidence; no significance or published-DDPG claim. Missing checkpoints are null; logged budgets are explicit.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--seeds',nargs='+',type=int,required=True);a=p.parse_args()
    r=analyze(a.directory,a.seeds);(a.directory/'three-seed-summary.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
