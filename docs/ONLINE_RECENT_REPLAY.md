# Online bandit + fixed SA: recent replay experiment

The working unfocused online mapper is preserved. This experiment changes how its contextual-bandit regressor samples training data, not its placement proposals or objective. It remains online contextual-bandit learning, not PPO despite the shared `ppo_asa` entrypoint.

## Why

Seed9–10 traces show current signed proposal rewards more negative than replay targets, especially mid-run. Temperature, placement and exploration evolve while replay retains older data. This motivates (but does not prove) a stale-data hypothesis. More recent replay can also overfit or forget; retain some full-buffer sampling and compare directly.

## Prespecified exploratory protocol

VGG16 FC, execution VVA16 / partition VVA1, paper pipeline and XY routing, 4×4 chips with16×16 cores, output-group-interleaved logic order. Seed11;100000 proposals plus initialization per arm,256 calibration trials separately. CPU2 threads.16 unfocused candidates, ordinary move sizes, signed proposed_delta targets, legacy9 features, legacy_clipped states, fixed annealing, epsilon1→.5.389 updates expected at the full budget.

Five arms: unfocused online with uniform replay; identical online with recent replay; same-pool uniform selection; same-pool proxy selection; fixed SA. Candidate-arm temperature calibration remains focused at.5. Plain SA retains its own selection/calibration mechanics.

Recent replay takes192 of256 samples from the newest10000 observations, then64 from the full buffer excluding already selected entries. Thus actual recent fraction can exceed75%. At startup the window covers the whole buffer. Each minibatch is duplicate-free. Defaults (`fraction=0`) preserve the old sample sequence/RNG. New diagnostics record mean replay age and sampled recent fraction. The reward distribution is never balanced by oversampling positive moves.

Primary comparison is final best cost against the existing unfocused online mapper at equal objective budgets; also require directional wins against uniform, proxy and fixed SA before advancing. One exploratory seed is not confirmation. Record runtimes and learner diagnostics even if it loses. Do not automatically increase the budget or replace the reference after a failure.

## Files and commands

- `src/run_multi_chip.py`: sampler, CLI and diagnostic integration.
- `src/test_recent_replay.py`, `src/test_online_pool.py`: sampler and online integration checks.
- `src/analyze_online_recent_replay.py`: matched-budget, configuration and cost checks.
- `scripts/run_kaggle_online_recent_replay.sh`: sequential five-arm runner, preflight tests, recoverable ZIPs.

From a configured repository (dependencies/setup in README):
```bash
PYTHONUNBUFFERED=1 SEEDS_OVERRIDE="11" BUDGET_OVERRIDE=100000 bash scripts/run_kaggle_online_recent_replay.sh runs/kaggle-online-recent-replay
```
Use a fresh output path. Rough estimate6–7 hours based on prior five-arm CPU runs; machine contention can change this. GPU is unnecessary. Every completed arm refreshes the ZIP; failed runs preserve logs. Kaggle: Internet on, paste the pinned checkout block supplied with this commit and use Save Version → Save & Run All.

67 local tests passed (51 standard,16 focused). This establishes mechanics, not improvement. No external implementation was copied for this change.
