# Frozen signed-pairwise scorer inside SA

This follows the positive offline diagnostic at revision 952be72. It tests
whether supervised candidate ranking improves actual search trajectories.
It is not a new online RL algorithm or a claimed optimizer improvement.

## Fixed protocol

- Reuse archived training pools from seeds 30--33; never train on 34--35.
- Reconstruct the signed ranker with the original 100 epochs, 1,600 updates,
  model and sampler seeds, 128-unit architecture and learning rate 0.0003.
- Export inference-only weights through the existing checked checkpoint loader.
- Compare checkpoint, proxy-greedy and uniform on fresh seeds 40--42.
- VGG16-CONV, execution/partition VVA=1/1, 16 candidates, topology16 features,
  50% focus, 5% slack, fixed SA and matching top-m exploration schedules.
- 4,101 exact objective calls per method/seed, no oracle evaluations or updates
  during deployment. This matches training-chain length before extrapolating
  to longer searches. Additional sensitivity cost: 16 calls/run.
- Advance to a longer screen only with >=2/3 wins against each control and
  >=1% mean paired gain over proxy. Record losses and runtimes regardless.
- The short gate is exploratory; a pass does not establish broad generalization.

The original label collection cost 47,556 exact calls across six chains plus
96 sensitivity calls; four training chains account for 31,704 exact calls plus
64 sensitivity calls. Offline training and data collection must be disclosed
separately from deployment budgets. Seeds 34--35 served model selection, not
untouched final evaluation. Frozen policies retain exploration, so offline
argmax gains do not predict deployment gains directly.

The repository includes the four training datasets and reports with SHA-256
checks, plus the original diagnostic summary and test log. No test data are
needed by the export path. Retraining may have small floating-point differences
across library versions; it is not a byte-identical replay guarantee.

Run:

```bash
bash scripts/run_kaggle_pairwise_rollout.sh runs/kaggle-pairwise-rollout
```

The script checks data integrity, tests checkpoint round-trip, trains once, and
archives results on ordinary exit/failure. CPU execution; no GPU is required.
Runtime is unmeasured; the preceding oracle-collection runs took about nine
minutes per chain. The new run has nine deployment chains plus offline training.
