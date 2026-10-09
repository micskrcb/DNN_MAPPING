# Multi-chain PPO-ASA local mechanics smoke

Run date: 2026-10-09.

## Purpose

This bounded AlexNet-FC run verifies the new independent-chain training path,
shared-policy updates, restart accounting, frozen-policy control, and matched
ordinary-ASA budget before spending Kaggle GPU time. It is not a learning or
optimizer-quality experiment.

## Configuration

- trained PPO-ASA and frozen PPO-ASA control: 512 proposals, restart every 64
  proposals, eight independently initialized chains, and 520 total true
  objective evaluations per condition;
- proposal pool: 16 candidates, half anchored on the current bottleneck stage;
- PPO: rollout 128, four updates, two update epochs, minibatch 64;
- ordinary ASA: 519 proposals plus one initialization, for the same 520 total
  objective evaluations;
- workload: AlexNet-FC, paper targets, paper pipeline timing, paper XY routing;
- device: local CPU.

## Result

Every mechanics check passed. Learned and frozen PPO-ASA both reached
24.31712 microseconds; ordinary ASA reached 24.05376 microseconds. PPO made
four finite updates, but the learned policy remained effectively uniform and
there was no learning signal at this deliberately tiny budget. The result only
authorizes the training-scale GPU gate.

The machine-readable result is [`summary.json`](summary.json), SHA-256
`c3147fac01ad1ab8d6ead5f8ae4fc9550d3104d85847ca7cc1a76722c4fed7c8`.

## Provenance

The independent-chain design follows the multi-instance training principle in
[Qiu and Liang's official RL-Based-SA repository](https://github.com/nathanqiu07/RL-Based-SA-Public).
The placement adapter, candidate representation, objective accounting, and
restart implementation in this repository are original; no external source
code was copied. See [`GITHUB_IMPLEMENTATION_AUDIT.md`](../../GITHUB_IMPLEMENTATION_AUDIT.md)
for the complete public-code review.
