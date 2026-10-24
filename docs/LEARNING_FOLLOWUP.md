# Learning follow-up: research method first, short comparator diagnosis second

## Latest evidence

Horizon seed18: fixed SA5.43232µs, bandit5.55344µs, all DDPG/frozen/random5.81808µs at1186 candidates. Higher gamma improved final deterministic output, but not best cost; critic loss became extremely large. These short-budget results do not supersede full-budget bandit gains or establish published-method superiority.

Separately, historical AlexNet-FC300k run finished22.84912µs,7.93% below initial placement, in57.84minutes. Improvement continued late. It used executionVVA1, older focused pools/epsilon.5 and a dirty checkout. It is not comparable to VGG16/VVA16 costs. Negative target mean despite positive_delta metadata means exact modified source is needed before trusting the reward-mode label. No matched SA/random/frozen controls; no isolated RL advantage. The22.4µs compute floor bounds remaining reduction at1.97% from its final cost, conditional on that evaluator.

## Stage1: full-budget online RL+SA development

Seed19,100000 proposals+initialization per arm. Three arms: strongest existing epsilon.02 with decay fraction1; candidate epsilon.02 with decay fraction.5; fixed SA. Candidate reaches low exploration earlier after warmup. Same legacy9 features, unfocused16-candidate pools, uniform replay, signed regression, fixed annealing. No new rewards or move sizes. VGG16-FC,4×4 chips16×16 cores, paper_targets/paper_pipeline/paper_xy, output_group_interleaved, executionVVA16/partitionVVA1.

This is an aggressive full-budget developmental test, not multi-seed confirmation. Compare against the strongest existing method and SA, recording all outcomes even if candidate loses. Earlier exploitation can also worsen coverage.

## Stage2: bounded DDPG stability diagnosis

Seed20,30 training placements each; paper_cnn/sparse/batch_z3/train_every1/gamma.999. Two arms: original unclipped updates versus actor and critic global gradient-norm clipping at1. No Huber loss, reward rescaling, action projection or repair change.128 baseline samples+30 training+3 retained deterministic evaluations=161 search candidates per arm. This low budget tests numerical behavior, not placement quality/convergence. Do not compare its best costs with Stage1's100001 budget.

New diagnostics: mean absolute Q, Bellman target, TD error and replay reward, plus pre-clipping actor/critic gradient norms when clipping is enabled. Unclipped gradient-norm fields remain null. Logs average diagnostics over the logged episode's updates, not every preceding episode. Clipping may limit parameter updates; it does not guarantee bounded Q values or solve action repair.

`--ddpg_grad_clip 0` is the unchanged default. Positive settings are DDPG-only and alter checkpoint fingerprints. CUDA stability stage has not run locally. Tiny-agent tests verify finite diagnostics and actual clipped gradient norms.

## Execution

Enable GPU and Internet on Kaggle. From a configured checkout:
```bash
PYTHONUNBUFFERED=1 bash scripts/run_kaggle_learning_followup.sh runs/kaggle-learning-followup
```
Expected roughly4–6 hours on comparable hardware: three full-budget CPU bandit/SA arms then about30–45minutes of CUDA diagnosis. Estimate only. Save Version → Save & Run All. Fresh directory required. Outer ZIP includes `bandit/` and `ddpg-stability/`; nested stage ZIPs refresh per arm, outer ZIP after Stage1 and on normal exit/failure. Abrupt termination may require retrieving nested ZIPs.

Files: optional clipping/value diagnostics in `src/run_multi_chip.py`; faster-decay arm in existing rapid stage/analyzer; new `scripts/run_kaggle_ddpg_stability.sh`, `scripts/run_kaggle_learning_followup.sh`, and `src/test_ddpg_stability.py`.77 local tests passed. Defaults and historical runner configurations remain available. No external code copied; no novel-method or published-superiority claim.
