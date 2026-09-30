# BatchNorm target-network diagnostic

## Purpose

This bounded CPU experiment checks the DDPG correction that keeps target
networks in evaluation mode and synchronizes their BatchNorm buffers. It is
not a convergence study and does not reproduce a paper benchmark.

## Configuration

- Workload: real torchvision AlexNet, `paper_targets`, CONV region.
- Objective: `paper_pipeline` with `paper_xy` routing on 4 x 4 chips, each
  with a 16 x 16 core grid.
- Agent: `paper_cnn`, baseline-normalized potential shaping, three actions per
  placement, 12 training placements, and a training update every 10 actions.
- Candidate accounting: 64 reward-baseline samples, 12 noisy training
  placements, and six retained deterministic diagnostic placements.
- Seeds: 0 through 4. Each run used two CPU intra-op threads and one
  inter-op thread.

Run the included script from the repository root with the project Python
environment. Checkpoint files are deliberately ignored because they are large
and machine-resume artifacts.

## Results

| Seed | Best latency (µs) | Earlier diagnostic (µs) | Difference (µs) |
| --- | ---: | ---: | ---: |
| 0 | 38.29734 | 38.58094 | -0.28359 |
| 1 | 38.10828 | 38.58094 | -0.47266 |
| 2 | 38.58094 | 37.87195 | +0.70898 |
| 3 | 37.87195 | 38.29734 | -0.42539 |
| 4 | 37.63563 | 37.87195 | -0.23633 |
| Mean | **38.09883** | **38.24062** | **-0.14180** |

Negative difference means lower latency after the repair. The five-seed mean
is 0.37% lower. The earlier seed 0 run used ten CPU threads, while all other
runs in both studies used two; excluding that pair gives 38.04920 µs after
the repair and 38.15555 µs before it, a 0.28% reduction. Three of those four
strictly thread-matched seeds improve and one regresses.

The effect is modest and this sample has only 12 placements per seed. Its value
is that the implementation no longer has minibatch-dependent target-network
BatchNorm behavior. A longer, predeclared multi-seed trained-versus-untrained
control is still required before claiming learning.

## Files

- `ddpg-seed*.json`: final machine-readable reports.
- `ddpg-seed*-ddpg.jsonl`: placement-level diagnostics.
- `ddpg-seed*.log`: complete console logs.
- `run_study.py`: the exact experiment launcher.
