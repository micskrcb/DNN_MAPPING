# DDPG sparse-reward horizon diagnosis

## Evidence

Previous inclusive-budget CUDA run:1334 evaluations each. RL+SA5.27056µs, SA5.31840µs, DDPG/random5.83376µs. RL+SA beat local reconstructed DDPG9.65415%, but DDPG best remained its initial random baseline after330 placements and211797 updates. Repairs1878–1911/1924 tasks and42–49% action saturation indicate restricted effective action behavior. These observations do not isolate a cause or prove published superiority.

There are ceil(1924/3)=642 actions per placement. Default gamma.98 discounts terminal reward at the first action by .98^641≈0.000002376. Candidate gamma.999 gives≈0.5266. This mathematical attenuation is not evidence that changing gamma alone will fix training. We preserve collision repair to isolate the reward-horizon change. Repair is part of the transition function, so storing requested actions is not automatically a replay bug.

## Fixed experiment

Seed18; three DDPG arms: learned gamma.98, learned gamma.999, frozen gamma.98. Same random initialization seed, baseline search, paper_cnn architecture, sparse reward scaling, batch_z3, noise schedule, repair and diagnostics. Train_every1 for learned arms; frozen records transitions but skips optimizer updates. Frozen is a diagnostic, not a published baseline.

Each DDPG arm uses6 epochs×30=180 training placements,1000 baseline samples and6 retained deterministic evaluations(every30):1186 inclusive search candidates. RL+SA uses the existing best epsilon.02/legacy9/unfocused/uniform-replay configuration and1185 proposals+initialization. SA1185+initialization; RS1186 samples. Sensitivity probes and reporting evaluations remain outside search accounting equally as before.

VGG16-FC, paper_targets, output_group_interleaved, paper_pipeline, paper_xy,4×4 chips with16×16 cores, executionVVA16/partitionVVA1, seed18. No pool ranking, new features or repair change.

Assess final deterministic and noisy costs, best-source provenance, collision repairs, saturation, and actual gradient updates. A lower best cost alone is insufficient if learned policies still fail to beat the frozen control. Longer-horizon gamma is explicitly an extension; neither a win nor a loss here establishes published algorithm superiority or convergence.

## Run

Enable Kaggle GPU and Internet. From a configured repository:
```bash
PYTHONUNBUFFERED=1 bash scripts/run_kaggle_ddpg_horizon.sh runs/kaggle-ddpg-horizon
```
Use a fresh directory. Expected roughly3–5 hours on comparable CUDA hardware, based on two180-placement trained arms at the previous≈30.7s/placement, plus a cheaper frozen arm and CPU controls. Timing can vary. No automatic resume. Checkpoints every30 placements, but replay is not saved; continuation would not be identical training. ZIP refreshed after each completed arm and on normal failure; download it even if the run fails.

`DDPG_EPOCHS_OVERRIDE` changes all budgets consistently, but keep the default for this comparison. `SEED_OVERRIDE` defaults18.

## Changes and validation

`src/run_multi_chip.py`: adds DDPG-only gamma/frozen CLI switches and metadata; defaults unchanged. Nondefault gamma/frozen checkpoint fingerprints differ. These switches apply to `--algo ddpg`, not guided/hybrid learners.

`src/test_ddpg_horizon.py`: real tiny-DAG training and frozen-control checks. `scripts/run_kaggle_ddpg_horizon.sh`: six-arm runner. `src/analyze_ddpg_horizon.py`: config/budget/control checks and diagnostics. No external code copied. CUDA experiment remains unrun locally; no outcome is promised.

Validation:76 local tests passed (51 standard plus25 focused), including actual gradient updates in a tiny DAG and zero updates in its frozen counterpart. No full-size CUDA outcome is available yet.
