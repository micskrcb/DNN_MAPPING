"""Developmental screen: selection is exploratory, never a confirmation claim."""
import argparse,json,math
from pathlib import Path

def summarize(root,seed,methods):
    assert 'reference' in methods and len(set(methods))==len(methods)
    reports={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in methods}
    assert len({r['complete_placement_evaluations'] for r in reports.values()})==1
    assert len({r['algorithm_metadata']['initial_cost'] for r in reports.values()})==1
    signatures={tuple(r['config'].get(k) for k in ('model','workload_region','partition_mode','logic_task_order','timing_model','routing_model','chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','iters')) for r in reports.values()}
    assert len(signatures)==1, 'Workloads/budgets differ'
    temperatures=[]
    for m,r in reports.items():
        meta=r['algorithm_metadata'];cfg=r['config']
        assert cfg['seed']==seed and r['complete_placement_evaluations']==cfg['iters']+1
        assert math.isfinite(r['best_cost']) and r['best_cost']>0
        if m=='fixed_sa':continue
        assert meta['proposal_learner']=='bandit' and meta['annealing_schedule']=='fixed'
        assert meta['bandit_scoring']=='direct' and meta['state_encoding']=='legacy_clipped'
        assert meta['bandit_exploration']=='epsilon_all' and not meta['bandit_shuffle_labels']
        assert meta['bandit_replay_capacity']==50000 and meta['proposal_candidates_per_step']==16
        assert meta['learning_enabled'] and meta['update_count']>0
        assert meta['bandit_epsilon_final']==(.02 if m=='exploit' else .1)
        assert meta['bandit_feature_set']==('topology16' if m=='topology' else 'legacy9')
        assert meta['bandit_pool_supervision_stride']==(64 if m in ('ranking','pool_regression') else 0)
        assert meta['bandit_ranking_weight']==(.25 if m=='ranking' else 0)
        assert meta['pool_focus_fraction']==0 and meta['pool_small_move_fraction']==0
        assert meta['bandit_recent_fraction']==0 and meta['reward_mode']=='proposed_delta'
        temperatures.append(meta['initial_temperature'])
    assert len(set(temperatures))==1
    candidates=[m for m in methods if m not in ('reference','fixed_sa')]
    winner=min(candidates,key=lambda m:reports[m]['best_cost'])
    costs={m:r['best_cost']*1e6 for m,r in reports.items()}
    return {'seed':seed,'best_cost_microseconds':costs,'selected_challenger':winner,
        'gain_over_reference_percent':{m:100*(1-r['best_cost']/reports['reference']['best_cost']) for m,r in reports.items()},
        'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()},
        'updates':{m:r['algorithm_metadata'].get('update_count',0) for m,r in reports.items()},
        'note':'Exploratory selection. Full-budget rerun uses a fresh seed. Pool labels consume objective budget; update counts and search steps may differ. No published-DDPG comparison.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--methods',nargs='+',required=True);a=p.parse_args()
    result=summarize(a.directory,a.seed,a.methods)
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
