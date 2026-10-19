"""Restore the online bandit reference before accepting a replacement."""
import json
import sys
from pathlib import Path


def main(directory):
    root=Path(directory);pairs=[]
    for path in sorted(root.glob('seed*-original.json')):
        prefix=path.name.removesuffix('-original.json')
        reports={m:json.loads((root/f'{prefix}-{m}.json').read_text()) for m in ('historical','original','relative','uniform','proxy','fixed_sa')}
        counts={r['complete_placement_evaluations'] for r in reports.values()}
        assert len(counts)==1
        for m in ('original','relative','uniform','proxy'):
            meta=reports[m]['algorithm_metadata']
            assert meta['reward_mode']=='proposed_delta'
            assert meta['learning_enabled']==(m in ('original','relative'))
            if m in ('original','relative'): assert meta['update_count']>0
            else: assert meta['update_count']==0
        assert len({reports[m]['algorithm_metadata']['initial_cost'] for m in ('original','relative','uniform','proxy')})==1
        cost={m:r['best_cost'] for m,r in reports.items()}
        pairs.append({'model':reports['original']['config']['model'],'seed':reports['original']['config']['seed'],'objective_calls_per_arm':counts.pop(),
            'best_cost_microseconds':{m:c*1e6 for m,c in cost.items()},
            'original_gain_over_controls_percent':{m:100*(1-cost['original']/cost[m]) for m in ('uniform','proxy','fixed_sa')},
            'current_gain_over_historical_percent':100*(1-cost['original']/cost['historical']),
            'relative_gain_over_original_percent':100*(1-cost['relative']/cost['original']),
            'runtime_seconds':{m:r['seconds_elapsed'] for m,r in reports.items()}})
    assert pairs
    result={'protocol':'historical online signed bandit restoration, FC VVA16/1 (see model per row)',
        'limitation':'Reused historical seeds, developmental reproduction. Relative-cost encoding is an ablation, not a replacement. No comparison to newer CONV/VVA1 results.',
        'historical_seed3_100001_calls_microseconds':{'original':4.45568,'uniform':4.51376,'fixed_sa':4.4464,'random':5.69456},
        'pairs':pairs}
    (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1])
