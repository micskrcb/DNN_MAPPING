# Improvement-priority target: three-seed experiment

## Evidence and hypothesis

Attraction seed24 failed:4.56144µs versus working learner4.38496µs, uniform attraction4.47360µs and fixedSA4.43072µs. The learned attraction arm selected attraction moves about73% of the time and had91% acceptance in the last logged window. It may favor low-impact moves, but acceptance rate alone does not prove neutrality. It was still improving at100k, so convergence was not established. Preserve the negative result; do not promote attraction. Evidence: `artifacts/attraction-bandit-seed24-20261025/`.

The next experiment returns to the working random candidate pool and16-step learner. Only its local training target changes. Earlier positive_delta and other target/ranking variants were unsuccessful under their documented configurations; this is a distinct bounded asymmetric target, not a claim that the earlier results were ignored or solved.

## Target

Let g be the existing normalized proposed improvement, clipped to[-10,10]. The opt-in `improvement_priority` target is:

- abs(g)<=1e-9:0 (numerically neutral).
- g>1e-9:1+min(g,1), providing a distinct improvement bonus.
- g<-1e-9:-0.1*min(-g,1), retaining a bounded harm penalty.

The working reference still uses g directly. The new target asks the regressor to prioritize finding an improvement rather than precisely predict large negative magnitudes. It changes the surrogate objective and may overvalue small gains or underpenalize harmful moves; this must be tested, not assumed beneficial. No positive-label oversampling, extra labels, auxiliary oracle evaluations or altered exact objective are introduced. Loss and predicted target magnitudes across these arms are not directly comparable.

SA acceptance and retained-best cost use the original exact cost. Candidate generation,16 candidates,16 optimizer steps/update, epsilon endpoint.02, uniform replay50000, minibatch256, learning rate and annealing schedule remain unchanged. Structured and attraction fractions remain0. Pool supervision is not supported with this new target. Checkpoint loading cannot mix priority and other targets.

New diagnostics count selected improving/neutral/worsening proposals cumulatively at each log checkpoint and in the final report. Counts exclude calibration and use the normalized-gain tolerance above. They measure selected proposal outcomes, not accepted moves or proof of causal learning.

## Three-seed workflow

```bash
bash scripts/run_kaggle_priority_three_seed.sh runs/priority-three-seed
```

Seeds25,26,27, VGG16-FC,100000 proposals+initialization per arm. Three methods per seed: working16-step learner (`train_more`), new target (`priority`), fixedSA. All matched initialization, evaluator and budgets. This is a target ablation with an established learned reference and unlearned baseline, not a new candidate-pool experiment.

The outer ZIP and `three-seed-summary.json` are refreshed after each seed. The summary reports each seed, mean paired percentage improvement versus both controls, sample standard deviation, wins/ties/losses and best costs near25%,50%,75%,100% of budget. Actual logged evaluation indices are shown; missing diagnostics are null, not interpolated. Incomplete aggregates are marked provisional. Three seeds improve development reliability but do not guarantee significance, convergence or5% over publishedDDPG.

Allow approximately9–12h on Kaggle CPU, extrapolated from recent measured runtimes; it has not been timed end-to-end. New experiments default to three seeds per the user's instruction. We use three methods, not four, to keep the full-budget run practical. Internet on; Save Version → Save & Run All. No GPU needed.

## Recovery

Rerunning with the same output directory and commit validates and skips completed seeds. Incomplete seed directories are preserved under an interrupted-seed name, then that seed is restarted from scratch. This is **completed-seed recovery, not training-state checkpoint resume**. Different commits require different output directories. Completed arm ZIPs live beside each seed directory; the outer ZIP is refreshed after completed seeds and ordinary failure/exit. Forced termination may prevent the latest outer ZIP update. Download the available files before a Kaggle session reset; restoring them under the same path is required for recovery in a new session. There is no automatic result delivery to Codex yet.

## Files

- `src/run_multi_chip.py`: opt-in target, outcome counters and target-checkpoint guard.
- `src/analyze_priority_seeds.py`: three-seed aggregation and progress checkpoints.
- `scripts/run_kaggle_priority_three_seed.sh`: sequential run and completed-seed recovery.
- `src/test_improvement_priority.py`: target semantics, budget/calibration, checkpoint and summary tests.

Original project code; no external implementation copied. No full-budget improvement claim before results arrive.
