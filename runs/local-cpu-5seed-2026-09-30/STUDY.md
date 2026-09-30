# Local CPU five-seed diagnostic

Completed 2026-09-30 on the CPU branch, implementation commit `6bee27f`. Lower cost is better.

| Method | Mean latency (µs) | Sample SD (µs) |
|---|---:|---:|
| Trained DDPG | 38.24062 | 0.35591 |
| Untrained control | 38.46750 | 0.15533 |
| Random search | 41.13328 | 0.50133 |
| Fixed SA | 42.99555 | 0.93018 |
| Adaptive SA | 42.32438 | 1.19732 |
| Sequential | 42.12586 | 0.00000 |

| Seed | Trained best (µs) | Untrained best (µs) |
|---|---:|---:|
| 0 | 38.58094 | 38.58094 |
| 1 | 38.58094 | 38.29734 |
| 2 | 37.87195 | 38.58094 |
| 3 | 38.29734 | 38.29734 |
| 4 | 37.87195 | 38.58094 |

The paired mean benefit of training was 0.22687 µs. The approximate paired 95% t interval was [-0.33818, 0.79193] µs; five seeds and a discrete objective make this interval exploratory.

## Protocol and limits

- AlexNet CONV, 183 tasks, paper-target reconstruction, paper CNN, potential shaping and deterministic retention. These optional extensions are not the frozen sparse-reward paper condition.
- Per DDPG/control seed: 12 complete training rollouts plus six deterministic evaluations, with 64 random baseline samples also eligible as the saved best: 82 candidates. The control uses `train_every=100000000` and performs zero gradient updates. Trained runs use `train_every=10`.
- RS uses 82 trials. SA/ASA use 82 proposals plus one initial placement: 83 candidates. ASA calibration is included in the proposal budget. This is approximately matched candidate count, not equal compute or runtime.
- Seed 0 DDPG and sequential runs were preserved from the earlier ten-thread run using two preflight samples and an explicit override. Other completed runs use two threads and 64 preflight samples. Preflight preserves RNG state and does not contribute candidates to optimization. Thread count can affect floating-point reproducibility; timings are not directly comparable.
- The interrupted seed-1 trace is historical only and excluded from aggregates. Fresh seed-1 training completed without checkpoint resumption. Large checkpoints remain local and are excluded from Git.
- The short fixed-SA schedule is not temperature-calibrated to seconds. ASA barely exceeds its calibration/adaptation window. These are weak short-budget comparisons, not evidence against well-tuned annealing.
- The untrained control already obtains strong layouts. Beating RS/SA here does not establish that DDPG learned. Test longer runs and policy quality versus the untrained control before a paper-scale launch.

`run_remaining.py` recreates missing completed runs from the repository root, skipping existing reports. `run_untrained_controls.py` reproduces the zero-update control. Use the project Python environment. Raw reports and JSONL traces preserve actual settings and runtimes.
