# Contextual operator-bandit simulated annealing

`--algo operator_sa` keeps fixed simulated annealing as the optimizer and uses
an online LinUCB contextual bandit only to choose the next move operator:

- `proxy`: best communication-distance proxy from a cheap candidate pool;
- `bottleneck`: move a task from the current slowest pipeline stage;
- `swap`: legal task-to-task core permutation without relocation;
- `relocation`: move one task to a currently unused physical core.

The context contains search progress, temperature, recent acceptance,
stagnation, best/current cost ratio, previous reward, and bottleneck dominance.
The reward is the signed proposed reduction in the exact configured objective.
SA still applies the Metropolis rule, so the search continues when the bandit
chooses a poor move. This is online unsupervised learning: it needs no dataset,
pretraining, or checkpoint.

After the first 20,000-call gate exposed unstable global operator credit, the
active v2 keeps separate early/middle/late models, discounts stale evidence by
0.995, guarantees 10% uniform exploration, and defaults to the proxy operator
unless another operator's lower confidence bound exceeds the proxy's upper
confidence bound. These settings are configurable with `--operator_sa_discount`,
`--operator_sa_minimum_exploration`, and `--operator_sa_phase_bins`.

`--operator_sa_disable_learning` selects operators uniformly and is the matched
control. Proxy-SA, fixed SA, and random search remain mandatory controls. Run
the three-seed 20,000-evaluation gate with:

```bash
bash scripts/run_kaggle_operator_sa_gate.sh runs/kaggle-operator-sa-gate
```

The prespecified per-workload gate requires Operator-SA to beat Proxy-SA and
fixed SA in at least two of three seeds, beat random in at least two seeds, and
improve on Proxy-SA by at least 1% on average. Passing this exploratory gate
triggers the exact embedded-proxy attribution test. Because v2 selected proxy
for roughly 92% of VGG16 steps, held-out seeds are permitted only if learned
selection also beats `--operator_sa_control proxy` in at least two of three
seeds and by at least 1% on average.

The completed attribution gate failed: learned selection was 2.306% worse than
the exact embedded-proxy control and won only one of three seeds. Consequently,
the selected-action-only operator-bandit configuration is closed. Its
strongest result is the training-free embedded proxy heuristic, which remains
the required control for any future supervised or RL gating method.

The follow-up full-information mode is enabled with
`--operator_sa_full_information_stride N`. Every `N`th decision evaluates all
four operators at the same state and trains from all four exact signed gains.
The proxy candidate advances the chain on those collection steps. This avoids
the biased feedback loop in which the policy only observes the move it chose.
The proxy control performs the same extra evaluations and discards their
labels, preserving equal objective-call budgets. The first gate uses stride 20,
zero forced random exploration, three seeds, and 20,001 calls per method:

```bash
bash scripts/run_kaggle_full_information_operator_gate.sh \
  runs/kaggle-full-information-operator
```

This is still online learning on one workload, without a pretrained dataset.
It becomes a positive RL+SA result only if it beats the matched proxy control
in at least two seeds and by at least 1% on mean cost.

This gate completed with valid accounting but failed: it won one seed, tied
two, and improved mean cost by 0.563%. The trained policy chose proxy on
52,030 of 52,061 decisions, consistent with the exact labels showing that the
other three operators were worse. Do not extend this mode unchanged.

## Proxy-gated discounted Linear Thompson Sampling

The selected-action follow-up avoids the full-information transition penalty.
Set `--operator_sa_selection_policy lints` and
`--operator_sa_proxy_gate_slate 2`. Cholesky Linear Thompson Sampling ranks
the four operators; the two highest-ranked operators each produce a proposal;
the cheap communication-distance proxy selects one; and only that proposal
uses the true objective and updates the model. Thus each post-calibration
decision still costs exactly one true evaluation.

`--operator_sa_reward signed` uses the historical signed proposed cost change.
`--operator_sa_reward outcome` gives bounded ALNS-style credit for a new best,
an improving move, an accepted worsening move, or a rejection, with recent
improvement magnitude as a secondary term. These are prespecified parallel
variants rather than post-result tuning. Run the paired gate with:

```bash
bash scripts/run_kaggle_proxy_gated_lints_gate.sh \
  runs/kaggle-proxy-gated-lints-gate
```

The decisive baseline is ordinary unthrottled embedded Proxy-SA, which also
uses one true objective per step. The outcome variant advances only with at
least two of three wins and at least 1% mean improvement over that baseline.
