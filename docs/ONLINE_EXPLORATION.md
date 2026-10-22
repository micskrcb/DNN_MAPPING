# Online RL+SA exploration experiment

## Evidence and decision

Seed11 recent replay finished successfully but cost4.38912µs versus reference4.37776µs (0.25949% worse). Reference remains the unfocused online contextual bandit with uniform replay, signed proposed_delta rewards and fixed SA. It beat same-pool uniform/proxy and fixed SA on this seed. This does not establish paper reproduction or general superiority.

## One-change hypothesis

The current learner still samples uniformly about50% of the time at the end. Test whether greater use of learned scores improves final placement. Existing `--ppo_asa_bandit_epsilon_final` supports this; no new learner or default change is needed. Candidate endpoint0.1; reference0.5. Both decay linearly from1 after512 warmup observations over the same duration. Thus exploration is lower across post-warmup training, not just its final phase. Actual final values are slightly above the endpoints. Uniform exploration can still select the greedy candidate; epsilon is mixture mass, not the observed fraction of nongreedy choices.

Lower exploration can reinforce prediction errors or reduce coverage. Treat this as an exploratory configuration, not an improvement until tested. Recent replay and mixed-size proposals stay off.

## Protocol

Seed12; VGG16-FC, paper_targets, paper_pipeline, paper_xy, output_group_interleaved,4×4 chips,16×16 cores, executionVVA16/partitionVVA1.100000 proposals plus initialization per arm;256 sensitivity/calibration trials recorded separately. Same16 unfocused candidates and focused calibration(.5), fixed annealing, signed rewards, legacy9 features, legacy_clipped state,50000 uniform replay,512 warmup,256 rollout and minibatch,4 update epochs,128 hidden units, learning rate.0003. CPU2 threads.

Arms: existing online; low-exploration online; same-pool uniform selection; same-pool proxy selection; fixed SA. SA retains its separate selection/calibration mechanics. The analyzer checks budgets, initial costs, candidate calibration, learning status, replay settings and exploration endpoints. It reports final epsilon, costs, gains, diagnostics and runtime.

Primary comparison: candidate versus existing online final best cost. Advance only if it also beats the controls on this exploratory seed; then confirm independently. A single seed cannot establish significance. Do not automatically extend failed configurations.

## Code and execution

- `src/run_multi_chip.py`: existing epsilon schedule and bandit implementation; unchanged this experiment.
- `scripts/run_kaggle_online_exploration.sh`: new controlled configuration and recoverable ZIP runner.
- `src/analyze_online_exploration.py`: validation and comparisons.
- `src/test_online_pool.py`: verifies lower exploration preserves objective budget, updates and calibration.

After setup from README, run from repository root:
```bash
PYTHONUNBUFFERED=1 SEEDS_OVERRIDE="12" BUDGET_OVERRIDE=100000 bash scripts/run_kaggle_online_exploration.sh runs/kaggle-online-exploration
```
Use a fresh output directory. Estimated5.5–7 hours from the preceding five-arm run; hardware contention varies. No GPU required. Kaggle Internet enabled; Save Version → Save & Run All. ZIP is refreshed after each completed arm, including logs on failure.68 tests passed locally (51 standard plus17 focused). No external code copied.
