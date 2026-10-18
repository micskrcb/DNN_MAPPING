"""Offline paired ranking diagnostic; no optimizer-win claim or deployable checkpoint.

Pools from seeds 30--33 train; seeds 34--35 are untouched test chains.
Compare the existing bounded regressor with same-architecture pairwise models.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def metrics(scores, gains, features):
    idx = np.arange(len(gains))
    selected = gains[idx, scores.argmax(1)]
    proxy = gains[idx, features[:, :, 7].argmax(1)]
    oracle = gains.max(1)
    denom = float((oracle - proxy).sum())
    return {"pools": len(gains),
            "advantage_over_proxy_seconds": float((selected - proxy).mean()),
            "advantage_over_uniform_seconds": float((selected - gains.mean(1)).mean()),
            "oracle_advantage_over_proxy_seconds": float((oracle - proxy).mean()),
            "oracle_capture_over_proxy": float((selected - proxy).sum() / denom) if denom > 1e-15 else None,
            "harm_fraction": float(np.mean(selected < proxy - 1e-15))}


def main():
    import torch
    import torch.nn.functional as F
    from run_multi_chip import ProposalBanditRegressor
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(2)
    pools = []
    for seed in range(30, 36):
        with np.load(args.directory / f'seed{seed}.npz', allow_pickle=False) as d:
            row = {k: d[k].copy() for k in ('global_states', 'candidate_features', 'true_gains')}
        report = json.loads((args.directory / f'seed{seed}.json').read_text())
        assert report['config']['seed'] == seed
        row['scale'] = max(abs(report['algorithm_metadata']['initial_cost']) * 1e-3, np.finfo(float).eps)
        row['seed'] = seed
        assert len(row['true_gains']) > 10
        assert all(np.isfinite(row[k]).all() for k in ('global_states','candidate_features','true_gains'))
        pools.append(row)
    train = pools[:4]
    gs = torch.tensor(np.concatenate([r['global_states'] for r in train]), dtype=torch.float32)
    xs = torch.tensor(np.concatenate([r['candidate_features'] for r in train]), dtype=torch.float32)
    raw = np.concatenate([r['true_gains'] for r in train])
    targets = np.concatenate([np.tanh(np.maximum(np.clip(r['true_gains']/r['scale'], -10, 10), 0)) for r in train])
    proxy = xs.numpy()[:, :, 7]
    result = {'protocol': 'offline chain-disjoint diagnostic, not an optimizer benchmark',
              'train_seeds': [30,31,32,33], 'test_seeds': [34,35],
              'epochs': args.epochs, 'models': {},
              'unreachable_positive_target_fraction_train': float(np.mean(np.abs(targets-proxy) > .25)),
              'training_pools': len(raw)}
    for mode in ('bounded_positive_regression', 'pairwise_signed', 'pairwise_positive'):
        torch.manual_seed(20261018)
        rng = np.random.default_rng(20261018)
        model = ProposalBanditRegressor(gs.shape[1], xs.shape[2], hidden_dim=128,
            scoring='proxy_residual' if mode.startswith('bounded') else 'direct', residual_limit=.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4)
        a, b = torch.triu_indices(xs.shape[1], xs.shape[1], offset=1)
        target = torch.tensor(targets, dtype=torch.float32)
        # Float64 labels preserve tiny physical gain differences when selecting informative pairs.
        label = torch.tensor(np.maximum(raw,0) if mode.endswith('positive') else raw, dtype=torch.float64)
        updates = 0
        for _ in range(args.epochs):
            for start in range(0, len(raw), 64):
                # Fixed-epoch, fixed-size random minibatches, independent of test outcomes.
                ids = torch.tensor(rng.choice(len(raw), min(64,len(raw)), replace=False))
                scores = model(gs[ids], xs[ids])
                if mode.startswith('bounded'):
                    loss = F.smooth_l1_loss(scores, target[ids])
                else:
                    delta = label[ids][:, a] - label[ids][:, b]
                    mask = delta.abs() > 1e-15
                    if not mask.any():
                        continue
                    loss = F.softplus(-delta.sign().float()[mask] * (scores[:,a]-scores[:,b])[mask]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                updates += 1
        rows=[]
        model.eval()
        for r in pools[4:]:
            with torch.no_grad():
                scores = model(torch.tensor(r['global_states']),torch.tensor(r['candidate_features'])).numpy()
            rows.append({'seed':r['seed'], 'all':metrics(scores,r['true_gains'],r['candidate_features'])})
            late = r['global_states'][:,0] >= .5
            rows[-1]['late'] = metrics(scores[late],r['true_gains'][late],r['candidate_features'][late]) if late.any() else None
        result['models'][mode]={'updates':updates,'test_chains':rows}
    result['interpretation']='Two test chains are exploratory. Positive offline ranking requires a later frozen rollout comparison; do not tune on these test seeds.'
    (args.directory/'ranking-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
