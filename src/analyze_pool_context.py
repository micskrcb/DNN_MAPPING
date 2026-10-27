"""Validate the descriptor-only pool-context ablation."""
import argparse
import json
from pathlib import Path
from analyze_online_rapid import summarize


def analyze(root, seed):
    root = Path(root)
    result = summarize(root, seed, ['train_more', 'pool_context', 'fixed_sa'])
    reports = [json.loads((root / f'seed{seed}-{m}.json').read_text())
               for m in ('train_more', 'pool_context')]
    a, b = (r['config'] for r in reports)
    changes = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if changes - {'ppo_asa_bandit_feature_set', 'report', 'asa_diagnostics'}:
        raise ValueError('Arms differ beyond pool-context features')
    a, b = (r['algorithm_metadata'] for r in reports)
    for key in ('update_count', 'bandit_optimizer_steps_this_invocation'):
        if a[key] != b[key]:
            raise ValueError('Training budgets differ: ' + key)
    result['note'] = ('Single-seed feature screen. Extra descriptors contain no exact '
                      'candidate costs. More inputs also add encoder parameters; '
                      'this is not a capacity-matched representation test.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, default=29)
    a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory / 'pool-context-summary.json').write_text(
        json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
