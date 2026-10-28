# Neutral-loss-weight verification

Nine focused tests passed, including gradient weighting, default-loss equality, all-neutral/all-nonneutral batches, actual training counts, invalid weights and checkpoint mismatch. The smoke runner's standard/focused preflight suites passed. Three real CLI arms completed 600 proposals each on a synthetic 64-task, 1x1-chip 8x8-core proxy workload and passed the strict analyzer. Generic microsecond fields in that synthetic summary are not physical latency results. No full-budget VGG16 improvement is claimed.

An analyzer mutation check rejected a changed learning rate. The full Kaggle experiment remains pending.
