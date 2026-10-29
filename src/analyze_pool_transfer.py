"""Validate matched learned/frozen pool18 policies on a fresh workload."""
import argparse
import json
from pathlib import Path
from analyze_online_rapid import summarize


def analyze(root, seed):
    root = Path(root)
    methods = ['pool_context', 'pool_frozen', 'fixed_sa']
    result = summarize(root, seed, methods)
    reports = {m: json.loads((root / f'seed{seed}-{m}.json').read_text()) for m in methods}
    a, b = (reports[m]['config'] for m in methods[:2])
    changes = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if changes - {'ppo_asa_disable_learning', 'report', 'asa_diagnostics'}:
        raise ValueError('Learner and frozen control differ beyond learning toggle')
    if a['ppo_asa_disable_learning'] or not b['ppo_asa_disable_learning']:
        raise ValueError('Incorrect learning toggles')
    if reports['pool_frozen']['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] != 0:
        raise ValueError('Frozen policy was trained')
    costs = {m: reports[m]['best_cost'] for m in methods}
    gain = 100 * (1 - costs['pool_context'] / costs['pool_frozen'])
    result['gain_over_matched_frozen_percent'] = gain
    result['comparison_outcome'] = ('learned_beats_both' if gain > 0 and costs['pool_context'] < costs['fixed_sa']
                                    else 'no_win_over_both_controls')
    result['complete_placement_evaluations'] = {m: reports[m]['complete_placement_evaluations'] for m in methods}
    result['note'] = ('Fresh training on this workload, not transfer of pretrained weights. '
                      'One-seed development screen; no published-DDPG comparison.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, default=34)
    a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory / 'pool-transfer-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
