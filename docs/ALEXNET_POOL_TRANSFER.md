# AlexNet-FC: retained pool-context learner versus matched controls

Neutral-weight .25 failed to show a repeatable improvement (one win, one loss; mean -0.02781%). Keep the original pool18 learner with16 update steps and weight1. No new reward tuning in this experiment.

## Run

From the repository root after the existing README setup:

```bash
bash scripts/run_kaggle_alexnet_pool_transfer.sh runs/kaggle-alexnet-pool-transfer 34
```

Kaggle: Internet enabled; CPU is sufficient. No dataset or checkpoint upload needed. Three sequential arms, each200000 proposals plus initialization. Plan approximately6–10 hours; this is an estimate from VGG16 runs, not measured AlexNet timing. Completed arms resume with preserved outputs on the same commit; interrupted arms restart. The shared runner archives partial outputs on failure.

## Protocol

AlexNet-FC, seed34, paper_targets, paper_pipeline/paper_xy, output_group_interleaved,4x4 chips of16x16 cores, executionVVA16/partitionVVA1. Methods: pool_context, pool_frozen, fixed_sa. Learned and frozen policies have identical pool18 features and16-candidate pools; only learning is disabled in the frozen arm. Both use signed proposed_delta, fixed annealing, uniform replay, and epsilon endpoint.02. Weight1 remains default. Frozen score head starts uniform; it is a matched candidate-selection SA control, not independent random search.

The model starts from scratch on AlexNet. This measures recipe portability, not pretrained weight generalization. Retained-best cost and percent gain against BOTH frozen control and SA are the primary outcomes; runtime and exact evaluations are reported. A win over both supports follow-up; no win means report it and investigate before claiming portability. One seed cannot establish statistical significance or5% superiority to published DDPG.

## Files

- src/run_multi_chip.py: unchanged learner/loss and mapper.
- scripts/run_kaggle_online_rapid_stage.sh: new pool_frozen arm matching pool18.
- src/analyze_online_rapid.py: validates the new frozen arm.
- src/analyze_pool_transfer.py: strict trained/frozen configuration matching and explicit outcome.
- src/test_pool_transfer.py: real frozen learning-disable check and analyzer configuration-mismatch test.
- scripts/run_kaggle_alexnet_pool_transfer.sh: locked, resumable entry point, seed default34.

No external code copied. Previous methods remain available for reproducibility.
