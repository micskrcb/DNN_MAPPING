# Learning intensity across two workloads

Run `bash scripts/run_kaggle_learning_intensity.sh runs/learning-intensity-UNIQUE` from the repository root with Python3, NumPy and PyTorch installed (see README setup; the root requirements.txt is historical). Uses CPU with two PyTorch threads and native BLAS/OpenMP thread caps; a GPU is not required.

Two stages: VGG16-FC and AlexNet-FC. Each uses seed21,100000 proposals+initialization=100001 evaluations, with three arms:

| Arm | Minibatch steps per update | Purpose |
|---|---:|---|
| exploit |4|Existing strongest configuration|
| train_more |16|Fit the same stream of move outcomes more thoroughly|
| fixed_sa |0|Unlearned search comparator|

Both learners use endpoint epsilon.02 over the full run, uniform50000 replay, warmup512, rollout256, minibatch256, legacy9 features, signed proposed_delta targets,16 unfocused candidates, fixed annealing. No pool supervision or ranking auxiliary loss. Workloads use paper_targets, output_group_interleaved, paper_pipeline/paper_xy,4×4 chips with16×16 cores, executionVVA16/partitionVVA1. Sensitivity256 remains external to the shared optimization budget.

More optimizer steps consume more wall time, but not more objective evaluations. Trajectories diverge; this is not training on an identical fixed dataset. Each learner trains from scratch on each workload. Report workload-specific cost, improvement over fixedSA and existing learner, actual optimizer steps, update events and runtime. New optimizer-step telemetry counts this invocation only; it must not be confused with cumulative checkpoint history. This runner does not resume checkpoints.

This is a two-workload development experiment, not confirmation across seeds or evidence against publishedDDPG. Do not average raw costs across workloads or promote a variant solely because it wins one case. A failed candidate does not replace the current implementation.

Estimated6–9h total, extrapolated from the latest VGG16 arms (~2.62h combined at4-step training) plus the second workload and increased training. This is not a measured duration; CPU allocation may change it. Both stages run sequentially. ZIP snapshots after each completed arm/stage; EXIT traps archive recoverable output after ordinary errors. Forced notebook termination can prevent the final outer snapshot; completed per-stage ZIPs remain in the repository runs folder.

Kaggle: enable Internet, paste the supplied pinned-revision block in one code cell, use Save Version → Save & Run All. Download the final ZIP. Existing output directories are refused to avoid overwriting results.

## Local validation

78 tests passed (51 standard+27 focused). VGG16-FC seed21 smoke completed all three arms at1025 evaluations each; actual optimizer steps8 versus32, SA0. Summary validation and ZIP generation passed. Smoke costs5.31568µs for both learners,5.84µs forSA. This validates mechanics, not a benefit from more training. The planned full run uses the same seed, so it is developmental rather than an independent held-out confirmation. AlexNet full workflow has not been run locally. Evidence: `artifacts/learning-followup-20261024/local-validation/`.
