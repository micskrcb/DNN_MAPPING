"""Fresh-seed comparison for the unchanged online bandit; seed 3 is not pooled."""
import argparse
import json
import math
import statistics
from pathlib import Path

METHODS=('original','uniform','proxy','fixed_sa')


def summarize(root,seeds):
    pairs=[]
    for seed in seeds:
        reports={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in METHODS}
        counts={r['complete_placement_evaluations'] for r in reports.values()}
        assert len(counts)==1, 'Unequal evaluation budgets'
        assert len({r['algorithm_metadata']['initial_cost'] for r in reports.values()})==1, 'Unequal initial costs'
        for method,r in reports.items():
            assert r['config']['seed']==seed
            assert r['complete_placement_evaluations']==r['config']['iters']+1
            assert math.isfinite(r['best_cost']) and r['best_cost']>0
            if method!='fixed_sa':
                meta=r['algorithm_metadata']
                assert meta['reward_mode']=='proposed_delta'
                assert meta['learning_enabled']==(method=='original')
                assert (meta['update_count']>0)==(method=='original')
                assert meta['candidate_feature_dimension']==9
        costs={m:r['best_cost'] for m,r in reports.items()}
        pairs.append({'seed':seed,'calls_per_method':counts.pop(),
            'best_cost_microseconds':{m:c*1e6 for m,c in costs.items()},
            'gain_percent':{m:100*(1-costs['original']/costs[m]) for m in METHODS[1:]},
            'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()}})
    assert pairs
    return {'protocol':'unchanged online signed bandit, additional seeds; seed 3 reported separately',
        'pairs':pairs,'aggregate':{m:{'mean_paired_gain_percent':statistics.mean(p['gain_percent'][m] for p in pairs),
            'wins':sum(p['gain_percent'][m]>0 for p in pairs),'seeds':len(pairs)} for m in METHODS[1:]},
        'limitation':'Small seed sample on the same workload. No random-search run, no broad generalization or statistical-significance claim.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[6,7]);args=parser.parse_args()
    result=summarize(args.directory,args.seeds)
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
