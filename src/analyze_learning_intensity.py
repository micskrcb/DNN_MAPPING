"""Per-workload learning-intensity comparison; no pooled cross-model claim."""
import argparse
import json
import math
from pathlib import Path
from analyze_online_rapid import summarize

METHODS = ['exploit', 'train_more', 'fixed_sa']

def analyze(root, seed):
    result = {'seed': seed, 'workloads': {},
              'interpretation': 'Directional evidence on this seed only; not statistical significance or a published-method comparison.'}
    for model in ('vgg16', 'alexnet'):
        folder = Path(root) / f'{model}-fc'
        s = summarize(folder, seed, METHODS)
        if s['workload'] != {'model': model, 'workload_region': 'fc'}:
            raise ValueError('Workload does not match directory')
        reports = {m: json.loads((folder / f'seed{seed}-{m}.json').read_text()) for m in METHODS}
        configs = [reports[m]['config'] for m in METHODS[:2]]
        allowed = {'ppo_update_epochs', 'report', 'asa_diagnostics'}
        changed = {k for k in configs[0].keys() | configs[1].keys() if configs[0].get(k) != configs[1].get(k)}
        if changed - allowed:
            raise ValueError('Learning arms differ beyond training intensity: ' + str(sorted(changed - allowed)))
        base, more = (reports[m]['algorithm_metadata'] for m in METHODS[:2])
        if base['update_count'] != more['update_count']:
            raise ValueError('Update-event counts differ')
        if more['bandit_optimizer_steps_this_invocation'] != 4 * base['bandit_optimizer_steps_this_invocation']:
            raise ValueError('Expected fourfold optimizer-step ratio')
        times = s['runtime_seconds']
        if any(not math.isfinite(t) or t <= 0 for t in times.values()):
            raise ValueError('Runtime must be finite and positive')
        s['training_runtime_change_percent'] = 100 * (times['train_more'] / times['exploit'] - 1)
        s['candidate_beats_both_controls'] = all(
            reports['train_more']['best_cost'] < reports[m]['best_cost'] for m in ('exploit', 'fixed_sa'))
        s['placement_evaluations_per_arm'] = reports['train_more']['complete_placement_evaluations']
        result['workloads'][model] = s
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    result = analyze(a.directory, a.seed)
    (a.directory / 'learning-intensity-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
