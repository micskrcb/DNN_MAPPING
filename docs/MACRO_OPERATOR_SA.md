# Delayed macro-action RL+SA

## Purpose

The one-step operator selectors learned valid policies but did not improve on
the strong embedded Proxy-SA control. `--operator_sa_macro_horizon N` changes
the credit problem: the contextual policy chooses a search strategy for `N`
successive SA proposals and is updated only after that segment.

The four macro-actions are proxy, bottleneck, swap, and relocation strategies.
For a non-proxy action, each step generates that action's proposal and a proxy
proposal; the cheap communication-distance proxy screens the pair before one
true objective evaluation. Fixed Metropolis acceptance and the measured fixed
SA temperature schedule remain unchanged.

The delayed reward is the bounded best-cost reduction achieved over the whole
segment. This lets a temporarily neutral or worsening move receive credit only
when its segment ultimately discovers a better placement. It is a contextual
bandit with delayed segment feedback, not yet a general long-horizon RL claim.

## Controls and accounting

- `--operator_sa_control uniform` holds a uniformly sampled macro-action for
  the same segment length and uses the same cheap proxy screen, without model
  updates.
- `--operator_sa_control proxy --operator_sa_macro_horizon 0` is ordinary
  unthrottled embedded Proxy-SA.
- Every post-calibration step uses exactly one true objective evaluation.
- Cheap proxy calculations are reported separately and never counted as true
  objective calls.

The first gate uses fresh seeds 9--11, a 64-step horizon, 20,001 true calls per
run, and VGG16-CONV. It advances only if learned macro selection beats both
controls in at least two seeds and improves mean cost over Proxy-SA by at least
1%.

```bash
bash scripts/run_kaggle_macro_operator_gate.sh \
  runs/kaggle-macro-operator-gate
```

Passing this short gate would justify shared training across multiple chain
restarts and workloads. Failure closes the current operator set and supports
keeping Proxy-SA as the practical optimizer.

## Completed result

The gate completed with correct accounting and active learning. Learned macro
selection beat the frozen macro controller in two of three seeds and by 0.816%
on average, but it beat Proxy-SA only once and averaged 2.297% worse. All
learned runs made 312 delayed updates and used every macro-action. The
prespecified gate therefore failed, and this operator family is closed rather
than retuned on the observed seeds.
