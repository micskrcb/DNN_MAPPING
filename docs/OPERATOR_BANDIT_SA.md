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

`--operator_sa_disable_learning` selects operators uniformly and is the matched
control. Proxy-SA, fixed SA, and random search remain mandatory controls. Run
the three-seed 20,000-evaluation gate with:

```bash
bash scripts/run_kaggle_operator_sa_gate.sh runs/kaggle-operator-sa-gate
```

The prespecified per-workload gate requires Operator-SA to beat Proxy-SA and
fixed SA in at least two of three seeds, beat random in at least two seeds, and
improve on Proxy-SA by at least 1% on average. Passing this exploratory gate
permits two held-out seeds; it is not publication evidence by itself.
