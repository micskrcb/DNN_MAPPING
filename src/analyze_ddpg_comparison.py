"""Inclusive-budget comparison against a local DDPG reconstruction, not published SOTA."""
import argparse,json,math
from pathlib import Path

def summarize(root):
    reports={m:json.loads((root/f'{m}.json').read_text()) for m in ('ddpg','exploit','fixed_sa','random')}
    keys=('model','workload_region','partition_mode','logic_task_order','timing_model','routing_model','chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','seed','sensitivity_trials')
    assert len({tuple(r['config'].get(k) for k in keys) for r in reports.values()})==1
    d=reports['ddpg']['algorithm_metadata'];cfg=reports['ddpg']['config']
    assert d['agent_arch']=='paper_cnn' and d['reward_mode']=='sparse'
    assert cfg['train_every']==1 and cfg['batch_z']==3 and d['start_episode']==0
    assert d['retain_deterministic_candidates'] and d['invocation_gradient_updates']>0
    budget=d['search_evaluations_including_baseline']
    assert budget==d['baseline_search_evaluations']+d['training_candidate_evaluations']+d['deterministic_candidate_evaluations']
    for m,r in reports.items():
        assert math.isfinite(r['best_cost']) and r['best_cost']>0
        if m!='ddpg':assert r['complete_placement_evaluations']==budget
    b=reports['exploit']['algorithm_metadata']
    assert b['proposal_learner']=='bandit' and b['learning_enabled']
    assert b['bandit_epsilon_final']==.02 and b['pool_focus_fraction']==0
    assert b['bandit_feature_set']=='legacy9' and b['annealing_schedule']=='fixed'
    assert b['bandit_pool_supervision_stride']==0 and b['update_count']>0
    assert b['bandit_ranking_weight']==0 and b['bandit_recent_fraction']==0
    gain=100*(1-reports['exploit']['best_cost']/reports['ddpg']['best_cost'])
    return {'protocol':'Single-seed preliminary local reconstruction comparison; not published-method superiority',
      'inclusive_search_evaluations_per_method':budget,
      'best_cost_microseconds':{m:r['best_cost']*1e6 for m,r in reports.items()},
      'bandit_gain_over_ddpg_percent':gain,'numerical_five_percent_threshold_met':gain>=5,
      'published_five_percent_claim_supported':False,
      'ddpg_gradient_updates':d['invocation_gradient_updates'],'ddpg_best_source':d['best_candidate_source'],
      'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()},
      'limitations':['DDPG is a reconstruction with unvalidated numerical reproduction, not author code.',
        'At most3000 training placements after timing is not a convergence study or the paper budget.',
        'DDPG baseline and deterministic diagnostics are charged; common sensitivity probes and reporting evaluations are outside search budget.',
        'SA calibration is within its proposal budget. DDPG constructs placements; SA modifies them, so equal objective counts do not imply equal runtime.',
        'DDPG checkpoint replay is not saved; resuming is not uninterrupted training.',
        'No frozen-DDPG control or multi-seed confirmation in this preliminary comparison.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
