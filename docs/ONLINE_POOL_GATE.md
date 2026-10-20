# Online mapper improvement: candidate-pool focus ablation

Run `bash scripts/run_kaggle_online_pool_gate.sh runs/online-pool-gate`.
Default VGG16-FC, seed8, VVA16/1, 100000 proposals+1 initialization.
Five arms: original online bandit, unfocused-pool online bandit, same-pool
uniform, same-pool proxy and plain fixed SA. Estimated6–7 hours on Kaggle CPU
from the latest measured per-arm times. Archives save after each arm.

## Actual algorithm change

The old pool already has eight bottleneck-anchored and eight unanchored moves.
Both use `_placement_neighbor`, also used by plain SA: randomly permute about
1% of tasks and optionally relocate to a free core. It would be incorrect to
claim this introduces a previously absent SA move operator.

`--ppo_asa_pool_focus_fraction 0` makes all16 candidates unanchored. Calibration
still uses the original focus fraction .5 and the same random stream, so this
ablation does not alter initial temperature. Candidate generation after
calibration is the only intended change. The bandit still learns online from
one selected move per step; signed targets, legacy9/clipped features, replay,
updates, exploration and acceptance remain fixed. No supervised pretraining.

Default omitted override preserves prior behavior. Checkpoints record the
effective pool fraction and reject a mismatched pool on loading. Old checkpoints
fall back to their recorded original focus fraction. Metadata distinguishes
pool focus from calibration focus. The baseline pool remains the default.

## Interpretation

If the unfocused learner beats same-pool uniform/proxy, that supports learning
within the new pool. If it also beats the original mapper and plain SA, extend
to additional seeds before adopting it. If only uniform improves, attribute
that to the neighborhood change rather than the learner. One seed is a
screen, not statistical confirmation. Keep all losses and timing results.
All arms use100001 exact optimization calls plus256 sensitivity evaluations.
No oracle labels or extra hidden objective calls are introduced.

## Validation

Tests cover default-versus-explicit-.5 equality, unchanged calibration
temperature, no anchored pool moves at zero focus, retained online updates,
exact toy budgets, and checkpoint mismatch rejection. The existing regression
suite remains required before experiments. No full new-seed result is claimed
from local tests.

Local CPU verification: all 59 unit tests and functional smoke tests passed,
with PyTorch enabled and no skipped PyTorch tests.
