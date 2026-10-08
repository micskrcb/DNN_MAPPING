# DDPG Learning Failure: Research Findings and Actions

Updated: 8 October 2026

## Purpose

This document records the investigation prompted by the 3,000-placement
AlexNet-CONV Kaggle run, the independent Gemini research response, and a
follow-up search of papers, official implementations, GitHub discussions,
Stack Exchange, PyTorch discussions, and placement-specific research. It
separates observed evidence from hypotheses and records which recommendations
have been implemented.

The goal is to reproduce Wu et al.'s DDPG method honestly while developing a
separately labelled extension that can learn under the reconstructed simulator.
The extension must not be reported as the paper's original algorithm.

## Evidence from the failed run

The original 3,000-placement diagnostic produced the following observations:

- trained best cost: 37.6356 microseconds;
- matched untrained exploratory control best: 37.4466 microseconds;
- the trained best therefore did not demonstrate a learning advantage;
- mean collision repairs were about 176 of 183 assigned cores;
- deterministic collision repairs were about 180 of 183;
- the deterministic actor often intended only a handful of unique locations;
- actor outputs increasingly saturated near `-1` or `1`;
- critic loss reached approximately `1.48e12`;
- none of the retained best candidates came from the zero-noise deterministic
  policy;
- 86.3% of the best sampled latency was fixed computation, leaving about 13.7%
  as placement-sensitive communication headroom.

These measurements support policy collapse and critic divergence. A decreasing
best-so-far curve by itself is insufficient evidence of learning because it can
be produced by noisy exploration.

## Findings from the original paper

The source paper explicitly specifies a continuous DDPG actor, flooring to grid
coordinates, nearest-free Manhattan collision repair, terminal reward
`sqrt(B) - sqrt(L(P))`, discount factor 0.98, actor learning rate 0.0002,
critic learning rate 0.001, minibatch size 64, and 30 complete placements per
declared epoch. It reports convergence at roughly 300,000 to 400,000 evaluated
placements.

The paper leaves several material values unpublished, including the batch size
`z`, replay capacity, target update coefficient, OU-noise parameters, exact
CONV/FC masks, complete per-layer partitions, GRS timing, and the complete
simulator. It states that CONV and FC are placed in separate masked regions and
that different workloads can occupy different numbers of chips.

Source:

- Nan Wu et al., [Core Placement Optimization for Multi-chip Many-core Neural
  Network Systems with Reinforcement Learning](https://doi.org/10.1145/3418498)

## Gemini response: accepted findings

The following parts of the Gemini audit agree with the run evidence and the
primary literature:

1. The requested continuous action and the collision-repaired executed action
   were misaligned in replay.
2. DDPG is vulnerable to deterministic sparse-reward deadlock.
3. Sixty-one decisions for AlexNet-CONV make one-step terminal credit
   propagation difficult; the FC horizon is much longer at `z=3`.
4. Uniform replay under-samples the most informative transitions.
5. Zero-noise deterministic evaluation is required to distinguish policy
   learning from noisy search.
6. Demonstrations, prioritized replay, multi-step or complete-episode targets,
   legal-action selection, conservative critics, and delayed actor updates are
   reasonable experimental remedies.
7. A small controlled ablation should precede a paper-scale run.

## Gemini response: corrections and qualifications

Several statements in the response were too strong or technically inaccurate:

- DDPG does not backpropagate through the environment's `floor()` operation.
  The actor receives gradients through a learned differentiable critic. The
  actual problem is that flooring is discontinuous and many-to-one, while
  collision repair can make the stored action differ from the action that
  caused the transition.
- An untrained noisy policy is not mathematically guaranteed to beat a trained
  policy. That occurred in this run and is evidence against learning, but it is
  not a universal theorem.
- The 41.4168-microsecond figure was the minimum of a 64-sample sensitivity
  preflight. The 1,000-trial reward baseline was about 40.8497 microseconds.
- Removing BatchNorm is not universally required. The paper specifies
  BatchNorm, and careful treatment can work. The implementation now evaluates
  target networks with fixed running statistics and synchronizes target
  buffers.
- A straight-through estimator provides a biased surrogate gradient; it does
  not guarantee correct gradients. Standard Wolpertinger-style actor training
  does not require differentiating through the environment.
- The paper does not publish a numeric value for `z`, so no particular batch
  size can be described as the uniquely paper-correct value.
- Several citations in the Gemini response were generic listing pages,
  secondary summaries, or unrelated applications. The algorithmic direction
  is useful, but publication text should cite the primary works below.

## Internet and implementation research

### DDPG is a continuous-action algorithm

The maintained Stable Baselines discussion states that DDPG supports continuous
`Box` actions and points users with very large discrete action spaces toward the
Wolpertinger architecture. Discretizing DDPG output is possible, but it creates
problems the algorithm was not originally designed to solve.

- [Stable Baselines issue 37](https://github.com/hill-a/stable-baselines/issues/37)
- Lillicrap et al., [Continuous Control with Deep Reinforcement
  Learning](https://arxiv.org/abs/1509.02971)

### Sparse deterministic environments can deadlock DDPG

Matheron, Perrin, and Sigaud give a formal analysis of poor fixed points in
deterministic sparse-reward environments. This directly matches the observed
actor saturation and lack of deterministic-policy improvement.

- [The Problem with DDPG: Understanding Failures in Deterministic Environments
  with Sparse Rewards](https://arxiv.org/abs/1911.11679)

### Wolpertinger aligns continuous proposals with discrete execution

Wolpertinger lets an actor produce a continuous proto-action, retrieves nearby
discrete candidates, and lets the critic select which candidate is executed.
The replayed action can therefore describe the actual discrete transition.

- Dulac-Arnold et al., [Deep Reinforcement Learning in Large Discrete Action
  Spaces](https://arxiv.org/abs/1512.07679)

### Demonstrations and prioritized replay address exploration scarcity

DDPG from Demonstrations combines demonstration and agent transitions in
prioritized replay. Demonstrations are retained instead of being discarded by
the replay ring. Prioritized Experience Replay samples informative transitions
more frequently and uses importance weights to reduce sampling bias.

- Vecerik et al., [Leveraging Demonstrations for Deep Reinforcement Learning on
  Robotics Problems with Sparse Rewards](https://arxiv.org/abs/1707.08817)
- Schaul et al., [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

### Twin critics and delayed actor updates reduce value-estimation error

TD3 uses clipped double-Q targets, delayed policy updates, and target-policy
smoothing. Guided DDPG uses the applicable parts with complete-episode return
targets: two critics, conservative legal-candidate scoring, and delayed actor
updates. It should not be called a complete TD3 implementation because target
policy smoothing is not used.

- Fujimoto, van Hoof, and Meger, [Addressing Function Approximation Error in
  Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)

### Legal masking is well supported for policy-gradient methods

Invalid-action masking has a formal policy-gradient justification and becomes
more important as invalid actions dominate the nominal action space. This
supports a masked categorical PPO fallback if guided DDPG still fails.

- Huang and Ontanon, [A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms](https://arxiv.org/abs/2006.14171)

### Later placement systems favor discrete masked policies

Google's chip-placement formulation defines the action as a valid grid cell and
uses a feasibility mask. MaskPlace predicts a probability matrix over grid
cells, removes infeasible positions before sampling, and introduces denser
placement feedback. Both avoid repairing almost every actor output after the
decision.

- Mirhoseini et al., [Chip Placement with Deep Reinforcement
  Learning](https://arxiv.org/abs/2004.10746)
- Lai, Mu, and Luo, [MaskPlace](https://arxiv.org/abs/2211.13382)

A later core-placement paper uses PPO, graph convolution, and a one-step whole
placement policy. Its public code is useful as a reference but currently states
that multi-chip execution is unavailable, and its single-chip environment still
repairs overlaps. It should therefore not be copied as a validated replacement.

- Myung et al., [Policy Gradient-Based Core Placement Optimization for
  Multichip Many-Core Systems](https://doi.org/10.1109/TNNLS.2021.3117878)
- [Associated public repository](https://github.com/WOOSHIK-M/Core_Placement_with_Reinforcement_Learning)

## Implemented actions

The paper-aligned `--algo ddpg` remains available as the reproduction control.
The separately labelled `--algo ddpg_guided` now implements:

- compact state maps containing only the active masked region;
- legal collision-free candidate batches near the actor's proto-action;
- conservative twin-critic ranking of legal candidates;
- storage of the exact executed action;
- a best-available demonstration selected from ASA, the random baseline, or a
  resumed checkpoint;
- permanent demonstration transitions in replay;
- demonstration-aware proportional prioritized replay;
- discounted complete-episode return targets for every trajectory step;
- behavior-cloning pretraining and a decaying cloning weight;
- Huber critic losses, gradient clipping, and delayed actor updates;
- deterministic zero-noise evaluation and explicit evaluation accounting;
- initial uniformly random legal placements to broaden replay support before
  relying on actor proposals;
- independent seeded action and replay random streams, so learning does not
  change the random exploration sequence used by its paired control;
- `--guided_disable_learning`, which supplies a matched control with the same
  environment, demonstration generation, legal selection, exploration schedule,
  and candidate accounting but no pretraining or gradient updates;
- finite-Q, gradient, collision, action-saturation, replay, and policy-source
  diagnostics;
- resumable model/optimizer checkpoints and append-only main diagnostics.

## Staged experimental plan

### Gate 1: implementation sanity

Run bounded synthetic and AlexNet-CONV smoke tests. Required conditions:

- zero collision repairs;
- exactly one unique executed physical core per logic core;
- finite actor loss, critic loss, Q values, and gradient norms;
- exact cost agreement when replaying a demonstration;
- checkpoint resume succeeds without deleting earlier main diagnostics.

This gate passes locally.

### Gate 2: learning, not search luck

Run the trained and `--guided_disable_learning` conditions with identical
budgets for at least five seeds. Compare the zero-noise deterministic policy,
not only best-so-far cost. The trained condition passes only if it improves the
deterministic cost consistently while Q values remain bounded.

### Gate 3: stronger placement-sensitive workload

AlexNet-CONV fits inside one 16-by-16 chip and therefore has no off-chip traffic
under the minimal whole-chip mask reconstruction. It is useful for debugging but
is a weak final benchmark. After Gate 2, evaluate AlexNet-FC or VGG16-CONV,
which span multiple chips and have more communication headroom. Because `z` is
unpublished, treat any larger batch size used to control the horizon as an
explicit ablation.

### Gate 4: paper comparison

Freeze the simulator and compare BS, RS, fixed SA, paper-aligned DDPG, ASA,
guided DDPG, and the no-learning control with declared candidate accounting.
Use at least five seeds and report mean, sample standard deviation, minimum,
maximum, and ratios normalized to BS. Do not claim exact numerical reproduction
because the paper's simulator and several timing parameters are unpublished.

### Gate 5: fallback research algorithm

If guided DDPG fails Gate 2, stop tuning it blindly. Implement a separate masked
categorical PPO policy that samples only legal cores, optionally initializes
from ASA behavior cloning, and uses either potential-based dense feedback or
whole-placement normalized advantages. This would be a new experimental method,
not the paper's DDPG reproduction.

## Publication boundary

The current work is an honest reconstruction plus a motivated experimental
extension. A publishable claim requires multi-seed evidence, ablations isolating
each added mechanism, matched candidate budgets, stronger multi-chip workloads,
and comparison against masked PPO or another modern discrete placement policy.
Until those experiments exist, the correct statement is that the implementation
is internally consistent and ready for learning validation—not that it has
already reproduced the paper's reported improvements.

The detailed source-code comparison with maintained public DDPG
implementations and the remaining algorithmic differences are recorded in
[`DDPG_REFERENCE_AUDIT.md`](DDPG_REFERENCE_AUDIT.md).
