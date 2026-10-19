"""Offline paired ranking diagnostic; no optimizer-win claim; optional signed-ranker inference export.

Pools from seeds 30--33 train; seeds 34--35 are held-out diagnostic/model-selection chains.
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


def inference_checkpoint(model, candidate_count, updates, feature_set="topology16", focus=.5, slack=.05):
    """Inference-only payload compatible with run_ppo_asa's existing loader."""
    return {"algorithm": "bandit_sa_proposal_v1", "resumable": False,
            "candidate_count": candidate_count, "state_encoding": "relative_cost",
            "bandit_scoring": "direct", "bandit_pool_supervision_stride": 0,
            "annealing_schedule": "fixed", "focus_fraction": focus,
            "focus_stage_slack": slack, "bandit_feature_set": feature_set,
            "bandit": model.state_dict(), "update_count": updates,
            "training_method": "offline_signed_pairwise", "training_seeds": [30,31,32,33]}


def productive_loss(scores, gains):
    """Balance improving/neutral/worsening groups; rank gains among improvements.

    A scalar score is shared with the existing inference model. This is not a
    calibrated probability. No ordering reward is given for neutral > worsening.
    """
    import torch
    import torch.nn.functional as F
    positive = gains > 1e-15
    neutral = gains.abs() <= 1e-15
    negative = gains < -1e-15
    groups = [F.softplus(-scores[positive]).mean()] if positive.any() else []
    for mask in (neutral, negative):
        if mask.any():
            groups.append(F.softplus(scores[mask]).mean())
    classification = torch.stack(groups).mean()
    a, b = torch.triu_indices(scores.shape[1], scores.shape[1], offset=1, device=scores.device)
    delta = gains[:,a] - gains[:,b]
    mask = positive[:,a] & positive[:,b] & (delta.abs() > 1e-15)
    ranking = (F.softplus(-delta.sign().to(scores.dtype)[mask] *
               (scores[:,a]-scores[:,b])[mask]).mean() if mask.any() else scores.sum()*0)
    return classification + ranking


def main():
    import torch
    import torch.nn.functional as F
    from run_multi_chip import ProposalBanditRegressor
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--export-signed', type=Path, help='Train only on seeds 30--33 and export; do not read test chains')
    parser.add_argument('--export-productive', type=Path)
    args = parser.parse_args()
    if args.export_signed and args.export_productive:
        parser.error('choose only one export mode')
    export_path = args.export_signed or args.export_productive
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(2)
    pools = []
    for seed in range(30, 34 if export_path else 36):
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
    for mode in (('productive',) if args.export_productive else ('pairwise_signed',) if args.export_signed else ('bounded_positive_regression', 'pairwise_signed', 'pairwise_positive')):
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
                if mode == 'productive':
                    loss = productive_loss(scores, label[ids])
                elif mode.startswith('bounded'):
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
        if export_path:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = inference_checkpoint(model, xs.shape[1], updates)
            checkpoint['training_method'] = 'offline_' + mode
            torch.save(checkpoint, export_path)
            print(f"Exported {mode}: {updates} updates, seeds 30--33 only")
            return
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
