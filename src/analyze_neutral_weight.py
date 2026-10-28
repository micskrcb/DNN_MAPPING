"""Validate the neutral-example regression-weight ablation."""
import argparse
import json
from pathlib import Path
from analyze_online_rapid import summarize


def analyze(root, seed):
    root = Path(root)
    result = summarize(root, seed, ['pool_context', 'neutral_weight', 'fixed_sa'])
    reports = [json.loads((root / f'seed{seed}-{m}.json').read_text())
               for m in ('pool_context', 'neutral_weight')]
    a, b = (r['config'] for r in reports)
    changes = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if changes - {'ppo_asa_bandit_neutral_weight', 'report', 'asa_diagnostics'}:
        raise ValueError('Arms differ beyond neutral regression weight')
    a, b = (r['algorithm_metadata'] for r in reports)
    for key in ('update_count', 'bandit_optimizer_steps_this_invocation'):
        if a[key] != b[key]:
            raise ValueError('Training budgets differ: ' + key)
    result['note'] = ('Single-seed neutral-loss-weight screen with unchanged signed targets. '
                      'Loss magnitudes across arms are not directly comparable; '
                      'no published-DDPG comparison.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, default=32)
    a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory / 'neutral-weight-summary.json').write_text(
        json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
