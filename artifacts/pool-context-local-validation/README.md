# Local pool-context verification, 27 October

11 focused tests passed; the smoke runner's 51 standard and 45 focused preflight tests passed. The real CLI completed three arms on a synthetic 64-task, 1x1-chip 8x8-core proxy workload with 600 proposals each, followed by the strict analyzer. This is workflow evidence, not VGG16 performance evidence; the synthetic summary's generic microsecond labels do not represent physical latency. The full experiment remains pending.

Tests cover tied/constant columns, permutation equivariance, raw feature preservation, finite bounded outputs, actual training with equal update and objective budgets, checkpoint feature mismatch, and rejection of a changed learning rate by the analyzer.

Documentation correction: seed26's learned gain over frozen is 100*(1-4.28736/4.37920)=2.09719%, not 2.14211% (which used the learned cost as denominator). Together with seed25, mean paired gain is 1.77406%, not 1.79652%. The underlying costs and positive directions are unchanged.
