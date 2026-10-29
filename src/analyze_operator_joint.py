"""Compare joint operator/candidate learning to the retained mapper and frozen pool."""
import argparse
import json
from pathlib import Path
from analyze_online_rapid import summarize


def analyze(root, seed):
    root = Path(root)
    methods = ['pool_context', 'operator_joint', 'operator_frozen']
    result = summarize(root, seed, methods)
    reports = {m: json.loads((root / f'seed{seed}-{m}.json').read_text()) for m in methods}
    a, b = (reports[m]['config'] for m in methods[1:])
    changes = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if changes - {'ppo_asa_disable_learning', 'report', 'asa_diagnostics'}:
        raise ValueError('Joint and frozen differ beyond learning toggle')
    if a['ppo_asa_disable_learning'] or not b['ppo_asa_disable_learning']:
        raise ValueError('Incorrect learning toggles')
    c = reports['pool_context']['config']
    changes = {k for k in a.keys() | c.keys() if a.get(k) != c.get(k)}
    if changes - {'ppo_asa_bandit_feature_set', 'report', 'asa_diagnostics'}:
        raise ValueError('Baseline differs beyond operator-pool feature mode')
    for m in methods[1:]:
        meta = reports[m]['algorithm_metadata']
        if sum(meta['operator_selected'].values()) != meta['policy_steps']:
            raise ValueError('Operator accounting mismatch')
    if reports['operator_frozen']['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] != 0:
        raise ValueError('Frozen policy trained')
    if reports['operator_joint']['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] != reports['pool_context']['algorithm_metadata']['bandit_optimizer_steps_this_invocation']:
        raise ValueError('Training budgets differ')
    costs = {m: reports[m]['best_cost'] for m in methods}
    result['gain_over_matched_frozen_percent'] = 100 * (1 - costs['operator_joint'] / costs['operator_frozen'])
    result['comparison_outcome'] = ('joint_beats_both' if costs['operator_joint'] < min(costs['pool_context'], costs['operator_frozen']) else 'no_win_over_both_controls')
    result['operator_selected'] = {m: reports[m]['algorithm_metadata'].get('operator_selected') for m in methods}
    result['note'] = ('One-seed development screen of joint operator/candidate selection. '
                      'Frozen control randomizes candidates, not just operator choice. '
                      'No separate operator-only attribution, new SA-relative estimate or published-DDPG comparison.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, default=35)
    a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory / 'operator-joint-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
