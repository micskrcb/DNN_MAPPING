"""Compare established, context-only and context-plus-training policies."""
import argparse
import json
from pathlib import Path
from analyze_online_rapid import summarize

METHODS = ['train_more', 'pool_context', 'pool_context32']

def analyze(root, seed):
    root = Path(root)
    result = summarize(root, seed, METHODS)
    reports = {m: json.loads((root / f'seed{seed}-{m}.json').read_text()) for m in METHODS}
    for left, right, allowed in [
        ('train_more', 'pool_context', {'ppo_asa_bandit_feature_set'}),
        ('pool_context', 'pool_context32', {'ppo_update_epochs'}),
    ]:
        a, b = reports[left], reports[right]
        changed = {k for k in a['config'].keys() | b['config'].keys()
                   if a['config'].get(k) != b['config'].get(k)}
        if changed - allowed - {'report', 'asa_diagnostics'}:
            raise ValueError('Unexpected configuration difference')
        am, bm = a['algorithm_metadata'], b['algorithm_metadata']
        if am['update_count'] != bm['update_count']:
            raise ValueError('Update-event counts differ')
        ratio = 2 if right == 'pool_context32' else 1
        if bm['bandit_optimizer_steps_this_invocation'] != ratio * am['bandit_optimizer_steps_this_invocation']:
            raise ValueError('Wrong optimizer-step ratio')
    cost = {m: reports[m]['best_cost'] for m in METHODS}
    result['combined_gain_percent'] = {m: 100*(1-cost['pool_context32']/cost[m])
                                       for m in METHODS[:2]}
    result['beats_both'] = all(cost['pool_context32'] < cost[m] for m in METHODS[:2])
    result['note'] = ('One-seed development screen; no fixed-SA or DDPG arm. '
                      'Cannot measure factorial interaction without a legacy9/32 arm. '
                      'Do not add percentages from separate seeds.')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, default=30); a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory/'pool-combined-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
