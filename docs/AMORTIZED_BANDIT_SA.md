# Amortized topology-aware bandit-guided SA

## Motivation

The immediate, full-information, Thompson, and delayed macro-action selectors
all learned measurable structure but failed to improve reliably on Proxy-SA
when trained from scratch on one placement instance. The remaining bounded
hypothesis is amortization: learn proposal corrections on multiple training
workloads, then evaluate the frozen policy on a different DNN without updating
it.

## Implementation

`topology16` extends the contention-aware candidate representation with four
portable environment features: relative off-chip latency, normalized chip
count, logical-task load of the allowed core region, and mesh/torus topology.
The checkpoint format records this feature contract and rejects incompatible
loads.

`--ppo_asa_bandit_control checkpoint` now performs genuine frozen inference
from a loaded contextual-bandit checkpoint. Earlier disabled-learning bandit
runs intentionally became uniform or proxy controls, so they could not test a
learned bandit checkpoint. Frozen checkpoint inference uses the same warm-up,
epsilon schedule, candidate pool, fixed-SA acceptance, and true-call budget as
the matched proxy and uniform controls. It makes no gradient updates.

When `--ppo_asa_finetune_loaded` is explicit, bandit optimizer state is also
restored. This permits sequential training without silently resetting Adam's
moments between workloads.

## Frozen gate

The first protocol trains sequentially on AlexNet-FC and AlexNet-CONV, then
freezes the resulting policy and evaluates ResNet50-CONV seeds 20--22. Each
training stage and every holdout method receives 20,001 true objective calls.
The holdout compares:

- the frozen amortized checkpoint;
- the matched proxy-greedy controller;
- the matched uniform candidate controller.

The policy advances only if it beats both controls in at least two of three
held-out seeds and improves mean cost over proxy by at least 1%.

```bash
bash scripts/run_kaggle_amortized_bandit_gate.sh \
  runs/kaggle-amortized-bandit-gate
```

This is an exploratory amortized-transfer gate. It is not evidence of broad
generalization unless it passes and is subsequently confirmed on additional
unseen workloads and hardware settings.

## Result: failed transfer gate

The protocol completed on Kaggle at revision `34c3101` with 51 passing tests.
The result archive SHA-256 is
`0b52220c8d9f81fab7a7c2e60e6d8f44d2c360469caa2be8d5d63f50f4466b20`.
Both AlexNet training stages performed 77 bandit updates. The held-out
checkpoint was loaded as `frozen_checkpoint_contextual_bandit` and performed
zero updates; all nine held-out jobs consumed exactly 20,001 true objective
calls.

On ResNet50-CONV seeds 20, 21, and 22, frozen checkpoint, proxy-greedy, and
uniform all reached the same best result: **588.784 microseconds**. The frozen
policy therefore had 0/3 wins and 0.000% mean improvement over proxy and over
uniform. The pre-registered advancement rule failed. This records a valid
null result, not a broken-checkpoint result; increasing only the evaluation
budget is not justified.
