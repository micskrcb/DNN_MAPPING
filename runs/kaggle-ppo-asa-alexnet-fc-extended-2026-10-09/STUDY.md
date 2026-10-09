# AlexNet-FC PPO-guided ASA extended gate

This audit covers the one-seed extended Kaggle experiment produced by
`scripts/run_kaggle_ppo_asa_gate.sh extended` at source revision `1c5c633`.
The supplied archive passed ZIP integrity checking and has SHA-256
`525b111e84ccd389fc1a1fa268cceaead7f70eb41a1c3cd7873da9d0d8d47a03`.

Each condition started at 24.8184 microseconds and used exactly 4,101 true
objective evaluations: one initialization plus 4,100 proposals. The learned
and frozen PPO conditions used the same fixed Metropolis acceptance rule and
adaptive-SA temperature controller. The frozen score head remained exactly
uniform and performed no optimizer updates. All mechanics checks passed.

| Condition | Best latency (microseconds) | Runtime (seconds) |
| --- | ---: | ---: |
| PPO-guided ASA | 23.56016 | 82.41 |
| Frozen uniform PPO-ASA control | 23.42128 | 79.17 |
| Ordinary ASA | 23.43184 | 52.61 |

The learning and optimizer gates both failed. PPO-guided ASA was 0.593% worse
than its matched uniform control and 0.548% worse than ordinary ASA. It beat
the control at only 10 of 41 retained diagnostic checkpoints and beat ASA at
2 of 41.

The learned distribution stayed nearly uniform. Its final candidate entropy
was 2.76893 versus `ln(16) = 2.77259`, a gap of only 0.132%. Approximate KL was
`6.71e-6` and clip fraction was zero at the final update. Of 4,100 selected
proposals, 3,145 (76.7%) were objective-neutral and only 107 (2.61%) improved
the current placement. The frozen control found 114 improving moves. The
trained curve briefly led early in the run, but the advantage did not persist.

This is evidence that the implementation runs and updates safely, but it is
not evidence that PPO improves ASA. More seeds or the paper-scale budget must
not be spent on this unchanged candidate-ranking policy. The next bounded
ablation should first increase the density of informative moves, for example
by constructing matched proposal pools around tasks in the current bottleneck
stage, while retaining both the uniform-policy control and ordinary ASA under
the same true-objective budget. Any exact candidate deltas used as labels must
count against that budget.
