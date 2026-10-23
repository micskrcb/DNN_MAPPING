## Confirmation update

Seeds13–14 have now completed and passed the directional rule. Mean paired gains:0.49% over previous RL+SA,1.92% over fixed SA,2.87% over proxy. The low-exploration reference is therefore supported by two confirmation seeds. No repeat is needed.

# Faster development of online bandit + fixed SA

## Why this run

The user requested less conservative iteration. Preserve the working low-exploration mapper, screen several useful variants together, and automatically test the best challenger at full budget. Do not spend the next sessions only repeating unchanged configurations. This is developmental selection, not evidence of statistical significance.

Latest actual result remains seed12:4.33616µs versus4.37264 previous online,4.45360 fixed SA. New variants have no measured performance yet. Existing seeds13–14 confirmation commands remain usable but are deferred.

## Screen: seed15,25000 proposals each

| Arm | Change relative to reference |
|---|---|
| reference | Working legacy9 signed regression, epsilon endpoint0.1 |
| exploit | Epsilon endpoint0.02 |
| topology | Existing topology16 candidate features |
| pool_regression | Evaluate all16 candidates every64 proposal steps; train signed regression on them |
| ranking | Same pool supervision plus within-pool pairwise ranking loss, weight0.25 |

All use VGG16-FC and the existing hardware/timing settings, unfocused candidates, fixed SA, uniform replay and the same initialization seed. Each has25001 reported complete-placement evaluations. Pool labels are charged, not free; pool methods can make fewer SA steps and a different number of updates. The unselected evaluated candidates are labels, not automatically accepted placements.

For scores s and clipped signed rewards y from a common-state pool, ranking loss is mean softplus(-(s_i-s_j)) over pairs with y_i>y_j+1e-8. Ties are excluded; all-tie pools give zero gradient. Add0.25 times this loss to selected-label smooth-L1 regression. Keep the newest256 evaluated pools, sampling up to8 per update. No reward-positive oversampling and no additional evaluator calls in the optimizer. Pool-regression control distinguishes richer labels from the ranking objective, though trajectories/update counts may diverge.

Implementation is original project code using standard PyTorch primitives; no external implementation copied. This objective alone is not a novelty claim.

## Automatic full run: seed16,100000 proposals each

Pick the lowest-cost challenger (reference excluded) and rerun from scratch with reference and fixed SA. If ranking is selected, include pool regression too. This is3 or4 arms,100001 reported evaluations each. Even if the short winner loses to reference, test it fully: short-budget order need not predict long-budget order. Do not call it a confirmed improvement automatically. Compare per-seed costs and runtime; do not merge screen and full stages into one mean. Seed16 selection is based only on seed15.

No further architecture changes between stages. No published-DDPG arm is included; that matched comparison remains outstanding. More experimental speed does not lower the standard for publication claims.

## Run and recover results

From a configured checkout (README setup):
```bash
PYTHONUNBUFFERED=1 bash scripts/run_kaggle_online_rapid.sh runs/kaggle-online-rapid
```
Use a fresh output directory. CPU2 threads, no GPU needed. Rough estimate6–9 hours, not benchmarked: topology features and ranking can add overhead. Internet needed to fetch the pinned commit on Kaggle. Save Version → Save & Run All.

The outer ZIP contains `screen/`, `full/` and `selected-challenger.txt`. Stage ZIPs update after each completed arm; outer ZIP updates after screen and on exit. Failure preserves available outputs/logs. Abrupt platform termination may leave only the last stage ZIP, which is still useful. Ranking buffers are training-session state; checkpoints are not resumable continuations. Do not assume stopping/restarting continues this experiment.

## Files

- `src/run_multi_chip.py`: optional ranking objective, bounded pool replay, diagnostics and CLI; defaults unchanged.
- `src/test_online_ranking.py`: order/tie-gradient and budget-accounted online integration checks.
- `src/analyze_online_rapid.py`: per-stage config/budget checks and challenger selection.
- `scripts/run_kaggle_online_rapid_stage.sh`: individual arm runner.
- `scripts/run_kaggle_online_rapid.sh`: two-stage automatic workflow and result packaging.

72 local tests passed (51 standard,21 focused). These validate mechanics, not real-workload improvement.

End-to-end VGG16 CLI smoke also passed for reference, topology and ranking at768 proposals each (769 evaluations). Ranking performed real updates with finite loss; these tiny runs are mechanics checks, not performance evidence.
