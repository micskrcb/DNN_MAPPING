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
this online operator-bandit configuration is closed. Its strongest result is
the training-free embedded proxy heuristic, which remains the required control
for any future supervised or RL gating method.
