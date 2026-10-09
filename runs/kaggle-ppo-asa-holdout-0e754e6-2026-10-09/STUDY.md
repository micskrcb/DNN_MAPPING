# Frozen PPO-ASA five-seed holdout

Run date: 2026-10-09. Source revision: `0e754e6e28a2fedc2acf053d3d737e52be1d5b92`.
Original archive SHA-256:
`ffbfd7c8fe26ef5cf36a0e9c8a223d139fbb5190a204900621f799722509140f`.

## Protocol

The proposal policy trained for 250,000 proposals at revision `5ea4c65` was
loaded without further updates. Holdout seeds 1–5 each ran one uninterrupted
4,100-proposal chain. A freshly initialized uniform proposal policy and
ordinary ASA used the same initial cost and exactly 4,101 true objective calls
per seed. All mechanics checks passed.

## Result

| Method | Mean best latency (microseconds) | Sample standard deviation |
| --- | ---: | ---: |
| Frozen trained PPO-ASA | 23.63050 | 0.04462 |
| Fresh uniform PPO-ASA | 23.51504 | 0.03376 |
| Ordinary ASA | 23.50518 | 0.05942 |

The trained checkpoint lost all five paired comparisons to both controls. It
was 0.491% worse than uniform on average; the exploratory paired 95% interval
for improvement was `[-0.16406, -0.06685]` microseconds. It was 0.533% worse
than ASA, with interval `[-0.16564, -0.08498]` microseconds. The verdict is
`NO_HOLDOUT_POLICY_SIGNAL`.

The checkpoint's one-seed advantage during online 128-step training chains did
not transfer to 4,100-step uninterrupted optimization. This closes that
checkpoint and rejects simply running it longer. The next bounded experiment
must match the training episode horizon and progress state to deployment.

The machine-readable result is [`summary.json`](summary.json), SHA-256
`ce98817eb209a14fc8898cb19430e7955683ad17704696c7669cf80f128af841`.

