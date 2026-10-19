# Productive-move target: frozen scorer + fixed SA

Run `bash scripts/run_kaggle_productive_gate.sh runs/productive-gate`.
CPU, estimated 2–3 hours, including two training fits and nine searches.

## Change and rationale

Signed pairwise loss rewards neutral-over-worsening orderings, potentially
allowing them to dominate productive-move learning. The new scalar-score loss
averages logistic loss separately over improving, neutral and worsening groups,
then averages present groups equally. Improving scores are pushed up; neutral
and worsening scores down. A second pairwise term orders unequal positive gains.
Loss = balanced classification + positive-pair ranking, equal coefficients.
The absolute gain tolerance is 1e-15 seconds. Scores are not calibrated
probabilities. All-neutral and all-negative batches remain finite.

This differs from the earlier positive-pairwise experiment: it includes an
explicit class-balanced improvement loss and retains negative examples without
rewarding neutral-versus-negative ordering. It is a testable hypothesis, not a
claim of solving the optimization problem. No new external code is copied.

## Fixed comparison

- Same four archived training chains 30–33, same features and 128-unit network.
- Same initialization/sampling seeds, 100 epochs, 1600 updates per scorer.
- Productive frozen scorer, original signed frozen scorer, proxy control.
- Fresh seeds 43–45, same initial placements and exploratory action schedule.
- Same VGG16-CONV, 4x4 chips, 16x16 cores, VVA 1/1, fixed SA acceptance.
- 4101 search calls + 480 diagnostic calls per run (32 pools ×15 extra calls),
  plus 16 sensitivity calls. No diagnostic labels train or select actions.
- Summary reports final cost and audited improving/neutral selection fractions.
- Advance only with >=2/3 wins over each control and >=1% mean paired gain
  over proxy; this is a developmental screen, not publication confirmation.

Training-data collection and offline training costs remain additional to
inference budgets (training labels: 31704 exact calls plus 64 sensitivity).
No changes to SA and no online RL updates. If ranking improves without final
cost improvement, investigate longer-horizon consequences instead of treating
candidate-level metrics as optimizer success. A pass needs fresh confirmation.
