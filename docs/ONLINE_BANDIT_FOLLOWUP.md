# Online bandit: additional-seed follow-up

The restoration produced a positive single-seed result. Keep the method
unchanged rather than tuning away from that result.

Run:
```bash
bash scripts/run_kaggle_online_followup.sh runs/online-followup
```

Defaults: VGG16-FC, seeds6 and7, four arms (online learner, focused-uniform,
proxy-scoring, plain fixed SA), 100000 proposals +1 initialization per arm.
VVA16/1, legacy9 features, clipped state, signed proposed reward, epsilon_all
ending at .5, replay50000, warmup512, fixed SA. All optimizer and proposal
parameters match the corrected restoration. No pretrained model is loaded.
Plain SA has different proposal mechanics and is a practical baseline, while
uniform/proxy are matched proposal-selection controls. Sensitivity adds256
calls per run, recorded separately. No online target or evaluator changes.

Seed3 is kept separate from the follow-up aggregate. Report every result,
paired percentages, wins and runtime; do not infer significance from two seeds.
A useful next-stage signal requires both new seeds to beat uniform and proxy;
if not, report mixed/negative evidence rather than tuning on these seeds and
calling them held out again. Plain-SA comparison must also be disclosed.
No random-search comparison is made in this follow-up.

Runtime: approximately10–11 hours for both seeds, based on ~5 hours per
four-arm seed in the latest archive; allow variability. For shorter independent
sessions use SEEDS_OVERRIDE=6 and then SEEDS_OVERRIDE=7 with fresh output paths.
Each takes approximately5–6 hours. No GPU required.

Output is archived after each completed arm and on ordinary exit/error.
run-status.txt distinguishes running partial archives, failures and completion.
These are saved completed-run results, not resumable mid-training checkpoints.
An abrupt session termination may leave only the latest completed-arm archive;
there is no guarantee of saving a currently running arm.

Tests recompute reported percentages from the archived restoration reports and
reject unequal budgets. Existing PyTorch tests still run before experiments.
No full new-seed optimization results are claimed from local smoke checks.
