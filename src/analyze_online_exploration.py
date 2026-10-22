"""Matched online exploration experiment; retain the working reference."""
import argparse
import json
import math
import statistics
from pathlib import Path

METHODS=('pool_online','low_exploration','pool_uniform','pool_proxy','fixed_sa')

def summarize(root,seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply nonempty unique seeds")
    pairs=[]
    workload_signatures=set()
    for seed in seeds:
        reports={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in METHODS}
        assert len({r['complete_placement_evaluations'] for r in reports.values()})==1
        assert len({r['algorithm_metadata']['initial_cost'] for r in reports.values()})==1
        temperatures=[]
        for method,r in reports.items():
            assert r['config']['seed']==seed
            cfg=r['config']
            workload_signatures.add(tuple(cfg.get(k) for k in (
                'model','workload_region','partition_mode','timing_model','routing_model',
                'chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','iters')))
            assert r['complete_placement_evaluations']==r['config']['iters']+1
            assert math.isfinite(r['best_cost']) and r['best_cost']>0
            if method=='fixed_sa':continue
            meta=r['algorithm_metadata']
            assert meta['reward_mode']=='proposed_delta'
            assert meta['pool_focus_fraction']==0
            assert meta['calibration_focus_fraction']==.5
            assert meta['learning_enabled']==(method in ('low_exploration','pool_online'))
            assert (meta['update_count']>0)==(method in ('low_exploration','pool_online'))
            assert meta['pool_small_move_fraction']==0
            assert meta['bandit_recent_fraction']==0
            assert meta['bandit_epsilon_final']==(.1 if method=='low_exploration' else .5)
            assert meta['bandit_epsilon_decay_fraction']==1
            assert meta['bandit_recent_window']==10000
            temperatures.append(meta['initial_temperature'])
        assert len(set(temperatures))==1, 'Calibration must not change with pool focus'
        assert reports['pool_online']['algorithm_metadata']['update_count']==reports['low_exploration']['algorithm_metadata']['update_count']
        cost={m:r['best_cost'] for m,r in reports.items()}
        gains={m:100*(1-cost['low_exploration']/cost[m]) for m in METHODS if m!='low_exploration'}
        pairs.append({'seed':seed,'best_cost_microseconds':{m:c*1e6 for m,c in cost.items()},
            'low_exploration_gain_percent':gains,'directional_screen_positive':all(g>0 for g in gains.values()),
            'final_exploration_epsilon':{m:reports[m]['algorithm_metadata']['final_exploration_epsilon'] for m in ('pool_online','low_exploration')},
            'final_learning_metrics':{m:reports[m]['algorithm_metadata']['final_update_metrics'] for m in ('pool_online','low_exploration')},
            'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()}})
    assert len(workload_signatures)==1, 'Workload and budgets must match across seeds'
    aggregate={m:{'mean_paired_gain_percent':statistics.mean(p['low_exploration_gain_percent'][m] for p in pairs),
        'wins':sum(p['low_exploration_gain_percent'][m]>0 for p in pairs),'seeds':len(pairs)}
        for m in METHODS if m!='low_exploration'}
    return {'aggregate':aggregate,'protocol':'online signed bandit, epsilon endpoint only; unfocused pool and calibration fixed',
        'pairs':pairs,'limitation':'Small-sample comparison on one workload; Exploratory seed12 is not independent confirmation of a selected method. No significance or broad-generalization claim. Plain SA retains distinct calibration and selection mechanics.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[12]);args=p.parse_args()
    result=summarize(args.directory,args.seeds)
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
