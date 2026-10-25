# Longer-horizon learning attribution

The priority target did not improve the established learner across three seeds. Keep proposed_delta, 16 optimizer steps/update, the original pool and the exact evaluator. Changing another algorithm component now would confound whether the established policy benefits from longer training.

Run `bash scripts/run_kaggle_learning_horizon.sh runs/learning-horizon`.

VGG16-FC seeds 25, 26, 27; 200000 proposals plus initialization per arm. Arms: train_more, frozen (same initialization and configuration, only learning disabled), fixed_sa. The analyzer verifies the two policy configurations differ only in disable-learning and output paths, equal objective budgets and initial costs, positive learned update count and zero frozen updates. Frozen is not random search: it uses the same candidate pool and annealing procedure with untrained scores. Its trajectories can diverge after updates consume RNG, so this is paired initialization, not identical proposal trajectories.

Reports include paired gains against both controls, each seed, standard deviation, wins/losses and logged costs near 50k/100k/150k/200k. Fixed-SA intermediate diagnostics are unavailable in this runner and remain null. Final costs are measured for all arms. Epsilon and temperature schedules are horizon-relative: the first 100k of this new run is not an exact extension/reproduction of the old 100k run. This experiment asks about a larger total-budget configuration, not isolated continuation. Reused seeds are development evidence. Three seeds do not prove significance or convergence; no published-DDPG superiority follows.

## Runtime and recovery

Allow roughly 12–16 hours on CPU (estimate, not timed); this may exceed a single Kaggle session. All three seeds are included by default. Completed arms are zipped after completion and marked done. Rerunning with the same directory and commit skips completed arms, checks identical saved commands, and reruns unfinished arms from scratch. This is not training-state resume. Each arm takes approximately 1.5–2 hours; download seed ZIPs if the overall ZIP is not refreshed before a forced stop. Restore the original folder layout under /kaggle/working before restarting in a new session. Hard termination cannot guarantee latest archive refresh. Never mark an unfinished run done manually.

## Code

- scripts/run_kaggle_learning_horizon.sh: three-seed orchestration and archiving.
- scripts/run_kaggle_online_rapid_stage.sh: frozen arm, command manifests, completed-arm recovery.
- src/analyze_learning_horizon.py: explicit learned/frozen comparison.
- src/analyze_priority_seeds.py: shared paired aggregation.
- src/analyze_online_rapid.py: arm mechanics validation.

No external code copied. The retained learning algorithm is unchanged; implementation improvements concern attribution, longer-budget testing, and recovery.
