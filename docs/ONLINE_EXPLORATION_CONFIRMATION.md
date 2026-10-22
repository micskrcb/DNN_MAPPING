# Confirmation of lower-exploration online RL+SA

Seed12 is a positive exploratory screen:4.33616µs,0.83428% better than existing online RL+SA,2.02451% better than same-pool uniform,2.63697% better than fixed SA,2.87076% better than same-pool proxy. These are reconstructed-evaluator results, not a published SOTA comparison.

## Frozen protocol

Use all settings in [ONLINE_EXPLORATION.md](ONLINE_EXPLORATION.md) unchanged: VGG16-FC,100000 proposals, five arms, epsilon endpoint0.1 candidate versus0.5 reference, uniform replay,16 unfocused candidates, signed rewards, fixed SA. Seeds13 and14 were selected before seeing their outcomes. Run them separately to avoid a roughly12-hour combined session. Do not change parameters after seed13. Preserve both outcomes even if unfavorable.

Primary: mean paired final-best-cost improvement versus existing online mapper. Practical advancement rule: positive mean improvement versus each of the four comparators and wins versus existing online on both seeds. This is not a significance test or a5% claim. Report the seed12 exploratory result separately; the analyzer rejects pooling it with confirmation. Gains are100*(1-candidate_cost/control_cost). Lower is better. Report per-seed results and runtime.

## Run

Use a fresh output path each time, from a configured checkout (README setup). CPU2 threads, no GPU needed. Approximately5.5–7 hours per seed; not guaranteed.

```bash
SEEDS_OVERRIDE="13" PYTHONUNBUFFERED=1 bash scripts/run_kaggle_online_exploration_confirmation.sh runs/online-exploration-confirm-seed13
```

Then in a separate session, same pinned commit:

```bash
SEEDS_OVERRIDE="14" PYTHONUNBUFFERED=1 bash scripts/run_kaggle_online_exploration_confirmation.sh runs/online-exploration-confirm-seed14
```

Both use the same five-arm runner and preflight tests. ZIPs refresh after every completed arm; a failed run retains available logs. Download both ZIPs. To aggregate after placing both seeds' JSON reports in one directory:

```bash
python src/analyze_online_exploration.py path/to/combined-reports --seeds 13 14
```

## Engineering changes

`src/analyze_online_exploration.py` now verifies control selection, learner type, feature/state encoding, exploration, replay capacity, warmup, candidate count and annealing settings. `src/test_online_exploration.py` checks real archived reports and rejects a mislabeled control. `scripts/run_kaggle_online_exploration_confirmation.sh` preserves protocol defaults. Learning code/defaults are unchanged intentionally. No external code copied.

## Published DDPG comparison still required

Before claiming5% over TODAES2020 DDPG, audit the repository's DDPG reconstruction against the method, check legal actions/repair, rewards, training updates and checkpoint/evaluation behavior, and document missing simulator assumptions. Compare identical task graphs, partitioning, hardware, timing/routing objective, initializations and evaluation accounting. Match complete-placement objective calls rather than equating DDPG epochs with SA iterations; include calibration/training calls and runtime explicitly. Separate training exploration from evaluation and preserve comparable best-so-far reporting. A weak or broken reconstruction is not evidence against the published method. Numerical paper ratios alone are not a matched comparison. The confirmation run here contains no DDPG and cannot answer that question.
