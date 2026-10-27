# Pool-context descriptors for online RL+SA

## Evidence

VGG16-FC seed 28, revision ea00c34, 200001 evaluations per arm: 16-step learner 4.27584 us, 32-step learner 4.27184 us, fixed SA 4.36672 us. The 32-step gain is 0.09355% over the learner and 2.17280% over SA. It was behind the learner at 50k, 100k, 150k and 180k before finishing ahead. Total measured runtime was 5.88689 hours. Keep 16 steps as the established default; a one-seed 4 ns win does not establish a useful training-intensity improvement. Original reports and provenance are archived in artifacts/train32-horizon-20261027.

## Hypothesis and implementation

The existing scorer receives nine absolute descriptors for each candidate. The opt-in pool18 input retains those nine and appends nine relative descriptors: (x - pool mean) / max(pool standard deviation, 1e-6), clipped to [-3,3] and divided by 3. Tied columns become zero. Candidate order does not affect these statistics. Existing proxy feature index 7 is preserved.

This gives the scorer explicit information about alternatives at the current decision. It may help compare otherwise similar moves. The relative inputs are deterministic transformations of existing descriptors, not additional physical information, labels, or oracle queries. It cannot recover information missing from those descriptors and may amplify small feature differences. It adds encoder parameters, so any win is not a capacity-matched representation claim.

Replay stores the selected row with its context from the original pool; normalization is never recomputed over replay minibatches. Checkpoints reject a different feature set. The existing reward, 16-step fitting, candidate generation, epsilon and annealing schedules are retained. Historical topology/contention and pool-label/ranking experiments had negative results; this alternative avoids their exact-label cost and expensive topology descriptors, but remains an unproven experiment.

## Run and interpretation

`bash scripts/run_kaggle_pool_context.sh runs/kaggle-pool-context`

Fresh seed 29, VGG16-FC, 200000 proposals plus initialization, methods train_more, pool_context and fixed_sa. This is a one-seed development screen, consistent with the prior 32-step screen; it is not a confirmation experiment. The analyzer rejects hidden setting changes and mismatched training counts. A favorable screen requires a lower final cost than both controls. Always report the magnitude and runtime, even for a tiny win. No 5% published-DDPG claim follows.

Allow approximately 6–9 hours CPU, based on recent runs; actual runtime may vary. Internet enabled, no input ZIP required. Completed arms are reused at the same commit/output path; unfinished arms restart. Output is OUT.zip. All relevant tests should pass before the full run. Original project implementation; no external code copied for this feature.

Files: src/run_multi_chip.py (descriptor transformation), src/test_pool_context.py (feature and training contracts), src/analyze_pool_context.py (comparison checks), scripts/run_kaggle_pool_context.sh (experiment).
