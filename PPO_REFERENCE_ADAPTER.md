# PPO reference adapter audit

Audit date: 2026-10-11.

## Decision

The previous candidate-set PPO-ASA experiment did not justify more epochs. Its
policy and value function shared both feature encoders and were optimized by a
single combined loss. After 1,250 PPO updates, the policy was still effectively
uniform while the value loss was much larger than the policy loss. Critic
gradients could therefore overwrite the representation used to rank proposals.

The closest maintained open implementation is Qualcomm's
[`RL-Based-SA-Public`](https://github.com/nathanqiu07/RL-Based-SA-Public),
audited at commit
`6e64bec12b9a7c6db601a4a4ae254b942e4aed79`. Its
[`neuralsa/training/ppo.py`](https://github.com/nathanqiu07/RL-Based-SA-Public/blob/6e64bec12b9a7c6db601a4a4ae254b942e4aed79/neuralsa/training/ppo.py)
uses separate actor and critic modules, separate optimizers, discounted
reward-to-go critic targets, temporal-difference trace advantages, and the PPO
clipped surrogate loss. These are now available through
`--ppo_asa_update_style reference_separate`.

Stable-Baselines3 Contrib's maintained MaskablePPO independently supports the
same diagnosis: its policy API can disable feature-extractor sharing between
the policy and value networks. It is not used as a dependency because this
project's action is a dynamically generated set of legal placement candidates,
which requires a custom permutation-equivariant feature extractor and exact
objective-call accounting.

## What is reused

No external source file was copied. The adapter independently implements these
published/open-source PPO mechanics:

- disjoint actor and critic parameters;
- one Adam optimizer per network;
- discounted reward-to-go targets for the critic, bootstrapped only when a
  local rollout fragment ends before the placement chain;
- GAE-style temporal-difference advantages;
- clipped categorical policy ratios and normalized advantages;
- separate actor and critic gradient diagnostics.

The reference repository's BSD-style license and attribution are preserved in
`THIRD_PARTY_NOTICES.md`.

## What remains project-specific

The official repository cannot run this placement problem unchanged. It has
TSP, bin-packing, knapsack, and continuous-function adapters, while this
project requires legal physical-core assignments, a reconstructed multi-chip
latency objective, and graph-aware relocation/swap candidates. The placement
state, candidate encoder, reward scale, adaptive-SA controller, and exact
objective budget therefore remain local implementations.

## Controlled experiment

Run:

```bash
bash scripts/run_kaggle_ppo_asa_reference_split.sh
```

It first runs the reconciliation tests, then trains on 1,250,000 proposals for
exactly 1,250 PPO updates. It freezes that checkpoint and compares it on 10
unseen seeds against a freshly initialized uniform proposal policy and ordinary
ASA. Every method receives the same 4,101 true objective calls per seed.

The existing decision rule is retained: the checkpoint needs a positive paired
mean and at least eight wins out of 10 against uniform to establish a policy
signal. It must independently meet the same rule against ASA to establish an
optimizer win. If the split implementation still converges to uniform, the
next change must be the state/action representation or parallel environment
diversity rather than more PPO epochs.

