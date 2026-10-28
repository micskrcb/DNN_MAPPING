# Combined candidate local verification

Three real CLI arms completed 600 proposals each on a synthetic 64-task, 1x1 chip/8x8 core proxy workload. The strict combined analyzer passed: equal update events, 16/16/32 optimizer steps, equal objective budgets and initialization. The generic microsecond summary labels do not represent physical latency for this synthetic proxy test. No full-budget performance claim.

The runner's standard and focused preflight suites passed. Seven focused feature/training tests passed separately. A new analyzer regression test checks percentage arithmetic, ties, rejection of learning-rate changes, and incorrect optimizer counts using a clearly synthetic combined report derived from archived input reports. This validates reporting only.
