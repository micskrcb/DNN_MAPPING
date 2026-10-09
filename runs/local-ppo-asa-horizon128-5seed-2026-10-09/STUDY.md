# Frozen checkpoint matched-horizon diagnostic

Run date: 2026-10-09. This local CPU diagnostic evaluates the preserved
checkpoint on five unseen 128-proposal chains, matching the training chain
length. Each trained, uniform, and ASA condition used 129 objective calls.

The trained policy averaged 24.28454 microseconds versus 24.37834 for its
uniform control, a 0.385% mean advantage. It won three of five pairs and its
exploratory interval `[-0.10792, 0.29551]` microseconds includes zero, so it
does not pass the prespecified four-of-five policy gate. It beat 128-proposal
ASA in all five pairs by 1.143% on average, but this tiny ASA budget is not a
competitive optimizer comparison.

Together with the 4,100-step holdout failure, this supports a horizon mismatch:
the checkpoint has a weak short-chain signal but does not support deep
refinement. The next training run uses 4,100-step episodes and within-chain
progress. This diagnostic is not a publication result.

The machine-readable result is [`summary.json`](summary.json), SHA-256
`779c753c47b89548e0b520fbe6bb2a10569ea4dae28a508f6f784bf88d14e39c`.
