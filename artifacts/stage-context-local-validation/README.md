# Stage-context validation

Seven focused tests passed, plus 51 standard and 45 existing preflight tests. Synthetic CLI smoke completed three arms at 600 proposals each and passed the strict feature-ablation analyzer. The synthetic summary's generic microsecond fields are not physical latency measurements.

A separate real VGG16-FC mechanics check ran 64 proposals (65 inclusive evaluations), seed31, with stage24. The mean within-pool standard deviations of raw minimum slack, mean slack and stage coverage were approximately 0.000993, 0.121141 and 0.174003. The new inputs are nonconstant on that short trace. This check does not establish a performance benefit or late-training feature diversity. It used 8 sensitivity trials and default training/exploration options, rather than the full run's settings.

Tests cover slack math, ties/zero-cost handling, input dimensions and equal training/objective budgets versus pool18. Existing feature-mismatch checkpoint guard applies to stage24. Full-budget Kaggle validation remains pending.
