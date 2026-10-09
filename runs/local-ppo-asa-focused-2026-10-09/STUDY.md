# Local bottleneck-focused PPO-ASA smoke

This seed-0 CPU smoke tested the `--ppo_asa_focus_bottleneck` proposal
neighborhood on AlexNet-FC. Each condition used one initial placement and
1,000 evaluated proposals. The learned and frozen conditions shared the same
focused candidate generation, Metropolis acceptance, adaptive temperature
controller, initialization, and objective budget.

| Condition | Best latency (microseconds) | Runtime (seconds) |
| --- | ---: | ---: |
| Focused PPO-ASA | 24.06272 | 116.11 |
| Focused frozen uniform control | 24.15984 | 88.80 |
| Ordinary ASA | 23.96288 | 12.77 |

Mechanics passed. Learned PPO was 0.402% better than its focused control but
0.417% worse than ordinary ASA. It beat control at seven of ten diagnostic
checkpoints. The trained run found 75 improving moves (7.5%) and 728 neutral
proposals (72.8%); the focused control found 67 improving moves and 721 neutral
proposals. The final policy entropy was 2.77109 versus the uniform maximum
2.77259, so the learned ranking was still weak.

This passes a one-seed learning-only gate. It does not establish an optimizer
advantage or justify five seeds. The next bounded run is one 4,101-evaluation
`focused-extended` Kaggle gate against the focused frozen control and ordinary
ASA.
