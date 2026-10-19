"""Descriptive same-pool audit; reused seeds are not confirmation data."""
import json
import sys
from pathlib import Path
import numpy as np


def pool_metrics(gains, scores, probabilities, selected, features):
    idx = np.arange(len(gains))
    greedy = gains[idx, scores.argmax(1)]
    proxy = gains[idx, features[:, :, 7].argmax(1)]
    expected = (probabilities * gains).sum(1)
    return {
        'pools': len(gains),
        'greedy_minus_proxy_ns': float(np.mean(greedy-proxy)*1e9),
        'expected_minus_greedy_ns': float(np.mean(expected-greedy)*1e9),
        'realized_minus_proxy_ns': float(np.mean(gains[idx, selected]-proxy)*1e9),
        'greedy_worse_than_proxy_fraction': float(np.mean(greedy < proxy-1e-15)),
    }


def main(directory):
    root = Path(directory)
    rows = []
    for seed in (40, 41):
        initial = []
        for selection in ('exploratory', 'greedy'):
            for method in ('checkpoint', 'proxy_greedy'):
                name = f'seed{seed}-{method}-{selection}'
                r = json.loads((root/f'{name}.json').read_text())
                m = r['algorithm_metadata']
                assert m['update_count'] == 0
                assert r['complete_placement_evaluations'] == 4101+m['oracle_additional_objective_evaluations']
                initial.append(m['initial_cost'])
                with np.load(root/f'{name}.npz', allow_pickle=False) as d:
                    g,s,p,i,x = [d[k] for k in ('true_gains','policy_scores','selection_probabilities','selected_indices','candidate_features')]
                    assert np.isfinite(g).all() and np.isfinite(s).all()
                    assert np.allclose(p.sum(1),1)
                    if selection == 'greedy':
                        assert np.allclose(p.max(1),1) and np.allclose(s[np.arange(len(s)),i],s.max(1))
                    metrics = {label:pool_metrics(g[mask],s[mask],p[mask],i[mask],x[mask])
                        for label,mask in [('all',np.ones(len(g),bool)),('late',d['global_states'][:,0]>=.6)]}
                rows.append(dict(seed=seed,method=method,selection=selection,
                    best_cost_seconds=r['best_cost'],objective_calls=r['complete_placement_evaluations'],audit=metrics))
        assert len(set(initial)) == 1
    result = {'protocol':'two-seed developmental exploration and on-trajectory ranking audit',
        'limitation':'Seeds 40–41 are reused for diagnosis, not independent confirmation. Oracle labels are diagnostic only and charged separately; no training uses them.',
        'rows':rows}
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main(sys.argv[1])
