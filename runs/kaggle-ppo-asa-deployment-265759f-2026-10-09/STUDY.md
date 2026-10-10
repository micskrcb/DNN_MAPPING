# Deployment-matched PPO-ASA result

Source revision: `265759f495705687242aa4db6cc43434ebb2f2db`  
Uploaded archive SHA-256: `fec50450760eccb885ff3a08ab68d6f9e9d78b833f5eaa3c9cb48435584407b2`

The training run used 250,000 evaluated proposals, 61 independent
4,100-proposal chains, and 61 PPO updates. It finished in 4,974.8 seconds on a
Kaggle Tesla T4. The best placement observed while training was 23.41696
microseconds.

The frozen checkpoint was then evaluated on five unseen seeds. Each trained
policy, fresh exactly-uniform policy, and ordinary ASA condition received one
initial placement plus 4,100 proposals. All mechanics checks passed.

| Condition | Mean best latency (microseconds) | Sample SD |
| --- | ---: | ---: |
| Trained PPO-ASA | 23.516352 | 0.015554 |
| Uniform frozen proposal policy | 23.515040 | 0.033761 |
| Ordinary ASA | 23.505184 | 0.059418 |

The trained checkpoint beat uniform in three of five pairs, but its mean was
0.001312 microseconds worse (`-0.00558%` improvement). It beat ASA in one of
five pairs and was 0.011168 microseconds worse on average (`-0.04751%`). Both
exploratory paired 95% intervals include zero.

This rejects the checkpoint but does not yet reject the PPO-ASA architecture.
The policy remained almost uniform: final entropy was `2.763968`, compared
with the uniform maximum `ln(16) = 2.772589`; approximate KL was
`3.37e-6`, and clip fraction was zero. The run made only 61 PPO update cycles,
whereas the closest public RL-Based-SA experiment trains for 1,000 epochs.
See `PAPER_SCALE_DECISION.md` for the resulting sufficient-scale protocol.
