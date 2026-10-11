# Response to the independent Claude audit

Audit received: 2026-10-11. Repository response: implemented after revision
`1fd75c4`.

## Findings accepted

The audit correctly identified two untested explanations for the null
candidate-set PPO result.

First, the historical PPO-ASA reward was the improvement actually realized
after Metropolis acceptance. A rejected uphill proposal and an objective-
neutral proposal both produced zero reward. Acceptance randomness could
therefore hide proposal quality from the policy.

Second, the existing diagnostics could not establish whether the nine
candidate features predicted the true maximum-stage latency change. The proxy
gain measures routed communication, but it is not the reconstructed pipeline
objective. More training cannot repair an action representation with no
within-pool predictive signal.

The audit also correctly recommended measuring policy quality within the same
candidate pool. Final best-placement comparisons are noisy because a small
early action difference changes every later pool. A paired selected-candidate
gain versus the uniform expected gain has substantially greater diagnostic
power.

## Corrections and qualifications

The legacy 10,000-epoch `main`-branch log cannot be converted to 300,000
placements using the current runner's 30-placements-per-declared-epoch rule.
That convention belongs to the reconciled runner. The legacy log appears to
count one complete placement per printed epoch and uses a different graph and
objective, so no paper-scale conclusion follows from it.

The audit's claim that the reported KL is understated by roughly twofold is
not established. Including the first unchanged optimization pass lowers the
average, but the factor depends on the trajectory of all ten passes. The new
diagnostics retain KL while adding direct logit and probability-distance
measurements.

The proposed entropy deficit is treated as supporting evidence rather than a
hard success threshold. A policy can obtain a useful paired proposal advantage
while remaining close to uniform. The direct true-gain comparison is the
primary gate.

## Implemented diagnostic stages

### E0: candidate-pool oracle

For sampled pools, the runner evaluates the true objective for all 16 legal
candidates without changing the selected optimization trajectory. It reports:

- best-of-16 gain and uniform expected gain;
- selected-policy gain;
- paired confidence intervals;
- candidate-neutral and no-improvement pool fractions;
- agreement between proxy-best and true-best candidates;
- pooled Pearson/Spearman and within-pool centered correlations between every
  feature and true gain;
- the additional diagnostic objective-call count separately from the
  optimization budget.

### E1: planted-signal mechanics test

`--ppo_asa_reward_mode planted_proxy` uses the already-visible proxy-gain
feature as the reward. Shared and split actor/critic PPO are each compared
with an exactly uniform frozen control for 200 update cycles. This is a
mechanics test, not a placement result. The real-reward pilot is skipped if
the split model cannot learn this visible signal.

### E2: bounded real-reward split pilot

Only after E1 passes, the split model receives 300 update cycles with the
historical post-acceptance reward. Two thousand candidate pools are sampled
throughout training for oracle measurements. The resulting checkpoint is
frozen and evaluated on five unseen seeds against uniform and ordinary ASA
with 4,101 optimization objective calls per method.

The code also supports `--ppo_asa_reward_mode proposed_delta`, which removes
Metropolis acceptance from the training target. It is reserved for the next
one-variable ablation if E0 shows headroom and E1 passes but E2 fails.

## Added diagnostics

- policy score-head weight norm;
- policy-head and value-head gradient norms;
- separate actor and critic gradient norms in split mode;
- within-pool feature standard deviations;
- logit standard deviation and range;
- probability L1 distance from uniform;
- action agreement with a uniform policy using the same random draw;
- selected proxy advantage over the pool's uniform expectation;
- proposed, realized, and actual training rewards;
- selected, uniform, and oracle true gains in audited pools.

## Decision logic

- If E0 shows negligible best-of-16 headroom, stop using this neighborhood.
- If E0 shows headroom but features have no within-pool relationship to true
  gain, redesign features before further PPO training.
- If split PPO fails E1, fix the update implementation; do not run E2 or the
  1,250-update experiment.
- If E1 passes but E2 has no per-pool proposal advantage, compare realized and
  proposed-delta rewards while keeping all other settings fixed.
- If PPO beats uniform but not ASA, report genuine policy learning and reject
  this hybrid as the better optimizer.
- Run the full 1,250-update split experiment only after the bounded pilot shows
  a positive proposal-quality signal.

The reproducible command is:

```bash
bash scripts/run_kaggle_ppo_asa_diagnostic_gate.sh
```

