"""Matched frozen productive/signed/proxy SA screen; no online-RL claim."""
import json
import sys
import statistics
from pathlib import Path
import numpy as np


def main(directory):
    root=Path(directory); pairs=[]
    for seed in (43,44,45):
        rows={m:json.loads((root/f'seed{seed}-{m}.json').read_text()) for m in ('productive','signed','proxy_greedy')}
        assert len({r['algorithm_metadata']['initial_cost'] for r in rows.values()})==1
        diagnostics={}
        for method,r in rows.items():
            meta=r['algorithm_metadata']
            assert meta['update_count']==0
            assert r['complete_placement_evaluations']==4581
            assert meta['oracle_additional_objective_evaluations']==480
            if method!='proxy_greedy':
                assert meta['source_training_updates']==1600 and meta['model_initialization']=='loaded_checkpoint'
            with np.load(root/f'seed{seed}-{method}.npz') as d:
                g=d['true_gains']; ix=np.arange(len(g))
                selected=g[ix,d['policy_scores'].argmax(1)]
                actual=g[ix,d['selected_indices']]
                diagnostics[method]={'greedy_improving_fraction':float(np.mean(selected>1e-15)),
                    'greedy_neutral_fraction':float(np.mean(np.abs(selected)<=1e-15)),
                    'actual_improving_fraction':float(np.mean(actual>1e-15))}
        cost={m:r['best_cost'] for m,r in rows.items()}
        pairs.append({'seed':seed,'best_cost_seconds':cost,'audit':diagnostics,
            'gain_over_proxy_percent':100*(1-cost['productive']/cost['proxy_greedy']),
            'gain_over_signed_percent':100*(1-cost['productive']/cost['signed'])})
    gains=[r['gain_over_proxy_percent'] for r in pairs]
    result={'protocol':'productive target, frozen scorer + fixed SA, three fresh seeds',
        'pairs':pairs,'mean_paired_gain_over_proxy_percent':statistics.mean(gains),
        'advance':sum(g>0 for g in gains)>=2 and statistics.mean(gains)>=1 and sum(r['gain_over_signed_percent']>0 for r in pairs)>=2,
        'limitation':'Developmental screen, not publication confirmation. Supervised learning, not online RL. Training cost excluded from deployment comparison and disclosed separately.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main(sys.argv[1])
