# Unchanged online mapper: confirmation on seeds 9–10

Seed8 passed the directional screen against all four controls. Freeze that
configuration before running two additional seeds; do not tune between them.

```bash
SEEDS_OVERRIDE=9 bash scripts/run_kaggle_online_pool_confirmation.sh runs/pool-seed9
SEEDS_OVERRIDE=10 bash scripts/run_kaggle_online_pool_confirmation.sh runs/pool-seed10
```

Use separate Kaggle sessions: approximately6–7 hours each, based on the prior
5.7-hour full experiment. Both in one session could approach the time limit.
Each uses VGG16-FC, VVA16/1, 100000 proposals+initialization, and five arms:
original mapper, unfocused-pool online mapper, matched uniform, matched proxy,
and plain fixed SA. Same signed online learning, features, exploration and
calibration. No pretrained model or offline target change.

Keep seed8 separate. Report every seed, paired cost improvements, wins and
runtime. If both new seeds beat all controls, extend to more seeds/workloads;
do not claim statistical certainty from two. Mixed or negative results remain
part of the record. Do not promote the variant based on the screening seed alone.

Runner improvements are operational, not a new optimization algorithm:
- Capture Python, PyTorch and NumPy versions for reproducibility.
- Save ZIP after every completed arm and a validated summary after each seed.
- Aggregate paired differences only across the seeds requested.
- Reject duplicate seeds and mixed workload configurations/budgets.
- Keep run-status.txt to distinguish partial, failed and complete output.

These archives preserve completed work, not resumable mid-run optimizer state.
Ordinary failures print log tails. Abrupt termination can lose the current arm.

To aggregate two completed output directories, copy their seed9/seed10 report
JSON files into one folder, then run:
```bash
python src/analyze_online_pool.py path/to/combined --seeds 9 10
```
The analyzer requires all five reports per seed. It never implicitly pools
seed8 or historic random-search results into confirmation statistics.

Validation: the new tests recompute the actual seed8 percentages, exercise
multi-seed aggregation, and reject duplicates and workload mismatches. Mapper
source and its previously validated behavior are unchanged in this update.
