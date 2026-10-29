# Joint operator and candidate scoring

## Why this experiment

The retained pool18 learner beats local fixed SA by about2.05% averaged over the latest VGG16-FC seeds32/33. Neutral weighting did not improve it reliably (mean paired change -0.02781%). AlexNet portability is a separate runnable experiment; its full result has not been received. The user now prioritizes algorithm improvement.

Earlier operator methods failed: the four-operator immediate/full-information/macro family did not beat Proxy-SA consistently, mixed-size candidates lost2.0054% to the earlier unfocused learner, and structured-only mixing lost1.02135% to the16-step learner. See historical NEXT_STEPS entries and STRUCTURED_BANDIT.md. These failures remain evidence against assuming mixed operators will help.

This experiment differs by retaining the current pool-relative neural candidate scorer and explicitly identifying operator types in each candidate. It is joint operator/candidate selection, NOT a new hierarchical RL controller or a revival of the old operator policy.

## Implementation

Opt-in feature mode `operator22` generates16 candidates:8 original scheduled-size moves,3 pure single-task relocations,3 pure pair swaps and2 existing communication-group axis reorderings. All moves are legal; group moves permute already occupied slots. If relocation or group generation is impossible, fall back to the original move and label it original. Append four one-hot actual operator indicators to the existing18 raw/pool-relative descriptors. The neural contextual bandit scores the resulting22-dimensional candidate representations. Epsilon-greedy selection chooses one candidate and its operator together.

The signed proposed-delta label, uniform replay,16 optimizer steps per update, fixed SA acceptance and objective are unchanged. Only the selected candidate receives an exact cost evaluation; unselected operators do not consume hidden objective labels. Pre-move descriptors and operator IDs contain no true candidate gain. The metadata reports actual selected operator counts. Feature-mode checkpoint checks prevent mixing pool18 and operator22 weights. No extra supervised oracle or new external code.

The default mapper stays pool18. operator22 also changes candidate distribution and neural input size, so a gain cannot be attributed exclusively to the operator identity feature. The frozen matched control samples candidates uniformly, not operators equally: its nominal allocation is8/3/3/2. It disables both learned operator choice and within-operator candidate ranking. A win supports the combined design; separate attribution would need a later ablation.

## Run and interpretation

```bash
bash scripts/run_kaggle_operator_joint.sh runs/kaggle-operator-joint 35
```

VGG16-FC, seed35,200000 proposals plus initialization per arm, CPU2 threads. Three arms: retained pool_context, operator_joint, identical operator_frozen with learning disabled. Same paper_pipeline/paper_xy,4x4 chips with16x16 cores, executionVVA16/partitionVVA1 and output_group_interleaved order. Same learner update budget. No pretrained checkpoint required. Plan about7–10 hours based on earlier runs, not a measured full-run runtime for this variant.

Success for this development screen means final retained-best cost lower than BOTH controls; also report wall time, training steps and selected operators. Ties or failures must be reported. One seed does not establish reliability. There is no fresh fixed-SA/random-search arm and no comparison to published DDPG. Do not interpret a win as reaching the5% publication target.

The wrapper locks output, runs tests, saves source commit/commands, and archives progress. Use a new output directory; same-commit restarts skip completed arms and restart unfinished arms.

## Files and provenance

- src/operator_candidates.py: legal operators and explicit type indicators.
- src/run_multi_chip.py: optional pool generation, scoring features, selected counts and compatibility checks.
- src/structured_moves.py: existing group move reused, originally documented as DOS-inspired structured search; not a DOS reproduction. See docs/DOS_PAPER_ASSESSMENT.md.
- src/analyze_operator_joint.py: budget/configuration checks, paired outcomes and operator counts.
- src/test_operator_candidates.py: legality, free-core fallbacks, real learning/frozen behavior, accounting and checkpoint incompatibility.
- scripts/run_kaggle_operator_joint.sh: reproducible experiment entry point.

No new external code was downloaded/copied; existing third-party notices remain applicable. New additions are project code. Local short runs validate mechanics only, not research performance.
