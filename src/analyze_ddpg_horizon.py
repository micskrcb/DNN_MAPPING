"""Discount-horizon ablation; all claims are about the local reconstruction."""
import argparse,json,math
from pathlib import Path

def summarize(root):
    methods=('default_gamma','long_gamma','frozen','exploit','fixed_sa','random')
    rs={m:json.loads((root/f'{m}.json').read_text()) for m in methods}
    keys=('model','workload_region','partition_mode','logic_task_order','timing_model','routing_model','chips_x','chips_y','rows','cols','vva_ops_per_cycle','partition_vva_ops_per_cycle','seed','sensitivity_trials')
    assert len({tuple(r['config'].get(k) for k in keys) for r in rs.values()})==1
    budget=rs['default_gamma']['algorithm_metadata']['search_evaluations_including_baseline']
    diagnostics={}
    for m,r in rs.items():
        assert math.isfinite(r['best_cost']) and r['best_cost']>0
        meta=r['algorithm_metadata']
        if m in methods[:3]:
            assert meta['search_evaluations_including_baseline']==budget
            assert meta['gamma']==(.999 if m=='long_gamma' else .98)
            assert meta['learning_enabled']==(m!='frozen')
            assert (meta['invocation_gradient_updates']>0)==(m!='frozen')
            assert meta['agent_arch']=='paper_cnn' and meta['reward_mode']=='sparse'
            assert r['config']['train_every']==1 and meta['start_episode']==0
            assert r['config']['batch_z']==3 and meta['retain_deterministic_candidates']
            assert budget==meta['baseline_search_evaluations']+meta['training_candidate_evaluations']+meta['deterministic_candidate_evaluations']
            rows=[json.loads(x) for x in (root/f'{m}.jsonl').read_text().splitlines()]
            diagnostics[m]={'first':rows[0],'last':rows[-1], 'gamma':meta['gamma'],
                'first_step_terminal_discount':meta['first_step_terminal_discount'],
                'last_collision_fraction':rows[-1]['collision_repairs']/r['tasks'],
                'gradient_updates':meta['invocation_gradient_updates'],'best_source':meta['best_candidate_source']}
        else:assert r['complete_placement_evaluations']==budget
    assert len({rs[m]['algorithm_metadata']['baseline_cost'] for m in methods[:3]})==1
    cost={m:r['best_cost'] for m,r in rs.items()}
    return {'inclusive_budget':budget,'best_cost_microseconds':{m:c*1e6 for m,c in cost.items()},
        'long_gamma_gain_over_default_percent':100*(1-cost['long_gamma']/cost['default_gamma']),
        'bandit_gain_over_long_gamma_percent':100*(1-cost['exploit']/cost['long_gamma']),
        'ddpg_diagnostics':diagnostics,'runtime_seconds':{m:r['seconds_elapsed'] for m,r in rs.items()},
        'published_superiority_claim_supported':False,
        'note':'One-seed short-horizon learning diagnosis. Frozen control tests initialization/repair advantage. Gamma .999 is an extension, not a verified paper setting. No convergence claim.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=summarize(a.directory)
    (a.directory/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
