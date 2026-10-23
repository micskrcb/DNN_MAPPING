# Matched-budget DDPG reconstruction comparison

## What this can establish

A first direct comparison of our best current contextual bandit+fixed SA against our local paper-style DDPG reconstruction under the same evaluator. It cannot establish superiority over the published TODAES result: simulator assumptions, reconstructed partitions and unvalidated convergence remain. A numerical5% win is reported separately from a publication claim; the latter stays false in the analyzer.

## Settings and accounting

A separate10-placement/32-baseline timing pilot chooses up to100 epochs under a conservative4-hour DDPG target, reserving25% headroom. It starts independently and is excluded from comparator costs, but its actual runtime/evaluation evidence is retained. This is a runtime estimate, not a timeout guarantee.

Seed17, VGG16-FC, paper_targets, output_group_interleaved, paper_pipeline, paper_xy,4×4 chips with16×16 cores, executionVVA16/partitionVVA1, sensitivity256 for all arms. Same workload/objective, but different construction versus local-search trajectories; not identical starting placements or equal runtime.

DDPG: paper_cnn, sparse reward, existing objective-unit reward scaling, batch_z3, train_every1,up to100 epochs×30 complete placements=3000.1000 random-baseline evaluations, deterministic evaluation every100 training placements with candidates retained. Inclusive count4030 at that cap; lower selected epochs reduce all comparator budgets together. The existing `total_candidate_evaluations` excludes baseline and remains unchanged for compatibility; new metadata explicitly counts baseline-inclusive search and actual gradient updates. Old checkpoint baseline count can be unknown; fresh runs only in this protocol. Replay is not saved, so checkpoint continuation is not identical to uninterrupted training.

RL+SA: strongest current epsilon endpoint0.02, legacy9 features, uniform replay, unfocused16-candidate pools, signed regression, fixed SA.4029 proposals plus initialization. Fixed SA same4030 count; random4030 trials. SA temperature calibration is inside its proposal budget. Common sensitivity probes and post-run breakdown evaluation are outside all search budgets and not candidate selection.

No full-grid search, frozen-DDPG control or statistical convergence claim here. At3000 training placements, a DDPG loss may be undertraining; increasing budget or validating learning remains necessary before a paper claim. Report deterministic-policy diagnostics, collision repairs, baseline source, actor/critic loss, costs and runtimes rather than only best-so-far cost.

## Run

From the configured repository:
```bash
PYTHONUNBUFFERED=1 bash scripts/run_kaggle_ddpg_comparison.sh runs/kaggle-ddpg-comparison
```
Fresh directory required. Defaults seed17 and at most100 epochs after timing. `DDPG_EPOCHS_OVERRIDE` changes all comparator budgets together. Do not compare unmatched old100000-evaluation RL+SA results against this DDPG run.

Enable a Kaggle GPU if available: DDPG uses CUDA automatically; other methods use CPU2 threads. Save Version → Save & Run All. Enable GPU. Timing pilot determines the actual budget; complete-run runtime is not guaranteed and CPU may be much slower. The log prints seconds per training placement every10 placements, allowing an estimate. ZIP refreshes after each completed method and on ordinary failure; checkpoints saved every100 placements. Abrupt termination may require downloading the output folder/checkpoint directly. No automatic resume is implemented.

Files: `scripts/run_kaggle_ddpg_comparison.sh`, `src/analyze_ddpg_comparison.py`, and added DDPG accounting in `src/run_multi_chip.py`. Existing reconciliation test now verifies baseline-inclusive counts. No external code copied. The paper_cnn label denotes the existing reconstruction, not an independently validated author implementation.

Validation:74 local tests passed (51 standard,21 focused,2 accounting-analyzer tests). A full-size three-placement CPU smoke was stopped because it was too slow; it did not produce a completed result. CUDA timing and the full comparison have not been tested locally. The runner records a separate timing pilot and adapts the budget instead of assuming a GPU speedup.
