# Bottleneck-focused PPO-ASA extended gate

The supplied Kaggle archive passed ZIP integrity checking. Its SHA-256 is
`98dcef8a0e26a9f679d2e5f5f1ab8f097e30c7c672b138ec5a165b966eb17daa`.
All reports use source revision `91fd7dc` and each condition has exactly 4,101
true-objective evaluations.

| Condition | Best latency (microseconds) | Runtime (seconds) |
| --- | ---: | ---: |
| Focused PPO-ASA | 23.77920 | 84.15 |
| Focused frozen control | 23.75952 | 84.84 |
| Ordinary ASA | 23.43184 | 50.70 |

All mechanics checks passed. Trained PPO finished 0.083% worse than its frozen
focused control and 1.482% worse than ordinary ASA. It beat control at 29 of
41 retained checkpoints, but its final best was first recorded near evaluation
1,232 and did not improve afterward. Control found its final best at the end;
ASA also continued improving through evaluation 4,100.

Focusing did solve part of the diagnostic problem: the trained neutral-move
rate fell from 76.7% in the first extended gate to 37.2%. It did not create a
learning advantage. Trained PPO found 137 improving moves versus 151 for the
frozen control. Final entropy was 2.76463 versus the uniform maximum 2.77259,
with approximate KL `2.55e-6` and zero clip fraction. The policy therefore
remained close to uniform.

The focused candidate heuristic also produced worse placements than the first
unfocused gate. It must not be scaled to five seeds. The next RL test requires
training-scale, multi-instance experience rather than another candidate or
hyperparameter tweak on one short chain.
