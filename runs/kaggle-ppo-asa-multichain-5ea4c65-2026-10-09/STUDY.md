# Training-scale multi-chain PPO-ASA result

Run date: 2026-10-09. Source revision: `5ea4c65ce2ed09d9b8baa6ff194ec6dd8234af2a`
(clean). Hardware: Kaggle Tesla T4 x2. Original archive SHA-256:
`24bb9405a4a5833a8ea0d4ea59b84d12d6230cad2f30fb1f0ba67035906ad235`.

## Protocol

Trained and frozen-control PPO-ASA each used 250,000 proposals across 1,954
independently initialized 128-proposal chains. Including every initialization,
each condition made 251,954 true objective calls. Ordinary ASA received one
initialization and 251,953 uninterrupted proposals, also 251,954 objective
calls. PPO used 16 legal candidates per proposal, a half global/half
bottleneck-focused pool, 4,096-transition rollouts, and 62 updates.

## Result

All mechanics checks passed. Best AlexNet-FC latency was:

| Method | Best latency (microseconds) | Difference from trained PPO-ASA |
| --- | ---: | ---: |
| Trained multi-chain PPO-ASA | 23.96800 | — |
| Frozen uniform PPO-ASA control | 24.08032 | trained is 0.466% better |
| Ordinary uninterrupted ASA | 23.14624 | trained is 3.550% worse |

This is a valid one-seed learning-only result. The learned policy beat the
frozen control at 3,413 of 3,906 matched checkpoints. Its improving-proposal
rate was 6.544% versus 6.228% for control, and its final entropy was 2.48776
versus the uniform maximum `ln(16)=2.77259`. The trained run first separated
from control after policy updates began and found its final best at proposal
58,624; control found its final best at 161,536. These observations show that
the PPO proposal distribution changed usefully rather than remaining random.

The optimizer gate failed because the training conditions restart after every
128 proposals, while ASA refines one placement for the entire budget. ASA kept
improving until proposal 230,432 and finished 0.82176 microseconds below the
trained method. The result therefore does not support claiming that PPO-ASA is
a better placement optimizer.

The trained/control pair took about 95.5 minutes concurrently, followed by
about 52.0 minutes for ASA. Total notebook time was roughly 2 hours 28 minutes.

## Next decision test

Training and optimizer evaluation are now separated. The preserved trained
checkpoint is evaluated without updates on five unseen, uninterrupted chains.
Each seed compares the loaded trained proposal policy, a freshly initialized
uniform policy, and ordinary ASA under exactly 4,101 objective evaluations.
This tests generalization and optimizer quality without charging the training
restarts against the inference search horizon. Use:

```bash
bash scripts/run_kaggle_ppo_asa_holdout.sh
```

The stored checkpoint is [`ppo-asa-trained.pt`](ppo-asa-trained.pt), SHA-256
`65187611b74498a2381ec5f2b2c648e88ab1b25f24f4ef973eb29ef096eb4a65`.
It was generated solely by this repository. The design-level external citation
remains [Qiu and Liang's RL-Based-SA repository](https://github.com/nathanqiu07/RL-Based-SA-Public);
no external source code is included.
