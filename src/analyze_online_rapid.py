"""Developmental screen: selection is exploratory, never a confirmation claim."""
import argparse,json,math
from pathlib import Path

def summarize(root,seed,methods):
    reference=next((m for m in ('reference','exploit','train_more','pool_context') if m in methods), None)
    assert reference in methods and len(set(methods))==len(methods)
    reports={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in methods}
    assert len({r['complete_placement_evaluations'] for r in reports.values()})==1
    assert len({r['algorithm_metadata']['initial_cost'] for r in reports.values()})==1
    signatures={tuple(r['config'].get(k) for k in ('model','workload_region','partition_mode','logic_task_order','timing_model','routing_model','chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','iters')) for r in reports.values()}
    assert len(signatures)==1, 'Workloads/budgets differ'
    temperatures=[]
    for m,r in reports.items():
        meta=r['algorithm_metadata'];cfg=r['config']
        combined = m == 'pool_context32'
        assert cfg['seed']==seed and r['complete_placement_evaluations']==cfg['iters']+1
        assert math.isfinite(r['best_cost']) and r['best_cost']>0
        if m=='fixed_sa':continue
        assert meta['proposal_learner']=='bandit' and meta['annealing_schedule']=='fixed'
        assert meta['bandit_scoring']=='direct' and meta['state_encoding']=='legacy_clipped'
        assert meta['bandit_exploration']=='epsilon_all' and not meta['bandit_shuffle_labels']
        assert meta['bandit_replay_capacity']==50000 and meta['proposal_candidates_per_step']==16
        assert meta.get('bandit_neutral_weight', 1.0) == (.25 if m == 'neutral_weight' else 1.0)
        assert meta['learning_enabled']==(m not in ('structured_uniform','attraction_uniform','frozen'))
        assert (meta['update_count']==0 if m in ('structured_uniform','attraction_uniform','frozen') else meta['update_count']>0)
        assert meta['bandit_epsilon_final']==(.02 if combined or m in ('neutral_weight','stage_context','exploit','fast_exploit','train_more','train_32','pool_context','structured','structured_uniform','attraction','attraction_uniform','priority','frozen') else .1)
        assert meta['bandit_epsilon_decay_fraction']==(.5 if m=='fast_exploit' else 1)
        assert meta['bandit_feature_set']==('stage24' if m=='stage_context' else 'pool18' if combined or m in ('pool_context','neutral_weight') else 'topology16' if m=='topology' else 'legacy9')
        assert meta['bandit_pool_supervision_stride']==(64 if m in ('ranking','pool_regression') else 0)
        assert meta['bandit_ranking_weight']==(.25 if m=='ranking' else 0)
        assert meta['pool_focus_fraction']==0 and meta['pool_small_move_fraction']==0
        assert meta['bandit_recent_fraction']==0 and meta['reward_mode']==('improvement_priority' if m=='priority' else 'proposed_delta')
        assert meta.get('pool_structured_fraction',0)==(.5 if m in ('structured','structured_uniform') else 0)
        if m in ('structured','structured_uniform'):
            assert meta['structured_candidates_per_pool']==8 and meta['structured_group_count']>0
            assert meta['selected_structured_proposals']>0
        assert meta.get('pool_attraction_fraction',0)==(.5 if m in ('attraction','attraction_uniform') else 0)
        if m in ('attraction','attraction_uniform'):
            assert meta['attraction_candidates_per_pool']==8 and meta['attraction_pair_count']>0
            assert sum(meta['attraction_generated'][k] for k in ('relocation','swap'))>0
        assert cfg['ppo_update_epochs']==(32 if combined or m == 'train_32' else 16 if m in ('neutral_weight','stage_context','train_more','pool_context','structured','structured_uniform','attraction','attraction_uniform','priority','frozen') else 4)
        if m=='train_more':
            assert meta['bandit_update_epochs']==16
        if combined or m == 'train_32':
            assert meta['bandit_update_epochs'] == 32
        if 'bandit_optimizer_steps_this_invocation' in meta:
            assert meta['bandit_optimizer_steps_this_invocation']==meta['update_count']*cfg['ppo_update_epochs']
        temperatures.append(meta['initial_temperature'])
    assert len(set(temperatures))==1
    candidates=[m for m in methods if m not in (reference,'fixed_sa','structured_uniform','attraction_uniform','frozen')]
    winner=min(candidates,key=lambda m:reports[m]['best_cost']) if candidates else None
    costs={m:r['best_cost']*1e6 for m,r in reports.items()}
    return {'seed':seed,'reference_method':reference,'best_cost_microseconds':costs,'selected_challenger':winner,
        'gain_over_reference_percent':{m:100*(1-r['best_cost']/reports[reference]['best_cost']) for m,r in reports.items()},
        'workload':{k:reports[reference]['config'][k] for k in ('model','workload_region')},
        'gain_over_fixed_sa_percent':({m:100*(1-r['best_cost']/reports['fixed_sa']['best_cost']) for m,r in reports.items()} if 'fixed_sa' in reports else None),
        'gain_over_structured_uniform_percent':({m:100*(1-r['best_cost']/reports['structured_uniform']['best_cost']) for m,r in reports.items()} if 'structured_uniform' in reports else None),
        'gain_over_attraction_uniform_percent':({m:100*(1-r['best_cost']/reports['attraction_uniform']['best_cost']) for m,r in reports.items()} if 'attraction_uniform' in reports else None),
        'selected_move_outcomes':{m:r['algorithm_metadata'].get('selected_move_outcomes') for m,r in reports.items()},
        'attraction_selected':{m:r['algorithm_metadata'].get('attraction_selected') for m,r in reports.items()},
        'structured_selected_proposals':{m:r['algorithm_metadata'].get('selected_structured_proposals',0) for m,r in reports.items()},
        'optimizer_steps':{m:r['algorithm_metadata'].get('bandit_optimizer_steps_this_invocation') for m,r in reports.items()},
        'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()},
        'updates':{m:r['algorithm_metadata'].get('update_count',0) for m,r in reports.items()},
        'note':'Exploratory selection. Full-budget rerun uses a fresh seed. Pool labels consume objective budget; update counts and search steps may differ. No published-DDPG comparison.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--methods',nargs='+',required=True);a=p.parse_args()
    result=summarize(a.directory,a.seed,a.methods)
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
