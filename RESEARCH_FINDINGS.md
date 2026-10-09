# DDPG Learning Failure: Research Findings and Actions

Updated: 8 October 2026

The direct public-code and training-scale comparison is recorded in
[`GITHUB_IMPLEMENTATION_AUDIT.md`](GITHUB_IMPLEMENTATION_AUDIT.md). Its main
finding is that the official RL-Based-SA configuration uses 10.24 million
transitions across 256 parallel problem instances, whereas the extended
PPO-ASA gates use 4,068 policy transitions from one fixed instance. Public
core-placement code does not include Wu et al.'s missing multi-chip simulator.
A maintained PPO library can improve reliability and parallel collection, but
no available repository is a drop-in replacement for this project's DNN
partition, routing, timing, and legal-placement environment.

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

## Five-seed guided-DDPG decision (8 October 2026)

The completed extensive Kaggle archive contains five trained/control pairs,
3,000 online placements per condition, all reports, all diagnostics, and valid
completion markers. Mechanics and numerical-stability checks passed, but the
trained policy lost every paired late deterministic comparison. Its aggregate
late deterministic mean was 39.0857 microseconds versus 38.7322 microseconds
for the frozen control, a regression of 0.3535 microseconds. The exploratory
paired 95% interval for improvement was -0.8766 to +0.1696 microseconds.

This closes Gate 2 for guided DDPG as a negative result. Occasional good
deterministic checkpoints do not reverse the conclusion because choosing the
best checkpoint after inspecting 100 diagnostics is selection bias, and the
trained best complete solution was also slightly worse in aggregate. Further
guided-DDPG tuning is paused.

The repository now contains `--algo ppo_masked`, a separate experimental
categorical PPO policy. It chooses one exact unused physical core for the next
logic core, applies a pre-softmax legality mask, and therefore has no repair
path. PPO uses clipped policy updates, generalized advantage estimation,
entropy regularization, a value baseline, gradient clipping, deterministic
diagnostics, and an exactly matched `--ppo_disable_learning` control. The
paper-aligned DDPG remains unchanged and available for reproduction tables.

## Five-seed masked-PPO decision (9 October 2026)

The extensive AlexNet-CONV archive contains all ten expected runs at commit
`e3576b0`: five trained and five frozen controls, each with 3,000 complete
placements, 1,000 random-baseline trials, 101 deterministic evaluations, and
valid completion markers. Initial deterministic costs match exactly within
each seed, every run has zero repairs, and all PPO diagnostics are finite.

Trained PPO beat its paired frozen deterministic policy in all five seeds. The
aggregate late deterministic mean was 42.6118 microseconds trained versus
45.3588 microseconds control, a 2.7471-microsecond or 6.06% improvement. The
exploratory paired 95% interval for improvement was 1.0972 to 4.3969
microseconds. A wider final-300-placement window gives the same conclusion.
This passes the learning-signal gate for this reconstructed workload.

The result does not yet establish a better optimizer or paper-level
performance. Best retained solutions averaged 39.9989 microseconds trained and
40.5094 microseconds control, only a 1.26% advantage, because random rollout
search is already strong. AlexNet-CONV occupies one chip and has zero off-chip
traffic. The next gate is therefore a matched-budget baseline table followed
by a paired masked-PPO run on AlexNet-FC or VGG16-CONV, where placement spans
multiple chips.

## AlexNet-FC multi-chip decision (9 October 2026)

The 3,000-placement one-seed extension used 4,101 complete-placement
evaluations for trained PPO, its frozen control, random search, and ASA. All
mechanics passed on the 932-core, four-chip FC region. The trained late
deterministic policy was 24.5048 microseconds versus 24.6424 for the frozen
control, a 0.558% directional improvement. Only 60 of 100 deterministic
checkpoints beat the control, the curve remained non-monotonic, and the trained
stochastic late-window mean was almost identical to control.

Training did not improve the retained optimizer result: trained PPO ended at
24.1080 microseconds and the frozen stochastic policy at 24.1072. Random search
was 24.1155, while ASA reached 23.4318 microseconds, 2.80% below trained PPO.
ASA improved steadily throughout its budget. Consequently, unchanged masked
PPO should not be scaled to five FC seeds. The useful next ablation is a
strictly budget-matched ASA/PPO hybrid or ASA-demonstration policy, always
compared with ASA alone; otherwise the extra RL machinery has no demonstrated
optimizer value on this fixed instance.

## Five-seed ASA baseline and hybrid research audit (9 October 2026)

The five-seed AlexNet-FC archive is complete and internally consistent at
revision `b147b14`. ASA best costs were 23.43184, 23.52528, 23.49808,
23.44928, and 23.59600 microseconds. Matched random-search costs were 24.11552,
24.12768, 24.16976, 24.07152, and 24.16800. ASA won every paired seed; its
mean was 23.50010 microseconds versus 24.13050, a 2.61% advantage. The paired
absolute advantage averaged 0.63040 microseconds, with an exploratory 95%
interval of 0.57216 to 0.68864. The sequential baseline was 24.92032
microseconds.

The official [RL-Based-SA implementation](https://github.com/nathanqiu07/RL-Based-SA-Public)
supports the proposal-learning direction but corrects a material error in the
Gemini survey. Qiu and Liang train the neighbor proposal with PPO, include
energy change in the state, and provide an LSTM variant; the Metropolis rule
and exponential cooling schedule remain fixed. Their implementation targets
knapsack, bin packing, TSP, and continuous test functions, so its training loop
and factorized proposal patterns are references rather than drop-in placement
code.

The survey's sequence-pair, B*-tree, Cartesian-overlap, and HPWL discussion
describes physical macro floorplanning. This project assigns fixed-size DNN
tasks to distinct positions in a predefined core grid and optimizes routed
communication latency. Replacing the placement representation with a B*-tree
would therefore change the problem rather than repair the policy. Claims about
combining BOPO, Wolpertinger, MCTS, and learned surrogates are plausible future
ideas but are not demonstrated for this evaluator by the cited evidence.

The next controlled implementation will let PPO choose a legal local proposal
while keeping the existing ASA acceptance and temperature logic fixed. It will
start from the same placements as a uniform-proposal ASA control and share the
same total objective-evaluation budget. This isolates whether learning improves
the neighborhood distribution. Temperature control, recurrent history, and
offline preference learning should be separate later ablations, not combined
in the first test.

That first implementation is now present as `--algo ppo_asa`. It uses the
reference method's core separation without copying its problem-specific code:
the policy selects a proposal, the existing Metropolis rule decides acceptance,
and the existing ASA controller changes temperature. The action is a
permutation-invariant choice among 16 legal candidates rather than a
932-by-1,024 flat action. A frozen zero-logit policy is exactly uniform, so the
learning comparison does not depend on a weak random neural initialization.

The 1,001-evaluation local smoke is intentionally reported as a provisional
gate. Learned PPO-ASA beat its uniform proposal control by 0.47% (23.9986
versus 24.1123 microseconds), but ordinary ASA reached 23.9629 and remained
0.15% better. PPO updates were finite and the learned condition recorded 41
improving moves versus 39 for its control. Roughly 89% of selected proposals
were objective-neutral, confirming that bottleneck-stage credit remains the
central difficulty. The full 4,101-evaluation one-seed gate must establish
whether additional online updates create a real optimizer advantage.

The 4,101-evaluation gate rejected this first learned neighborhood. The trained
run ended at 23.5602 microseconds, behind the uniform control at 23.4213 and
ordinary ASA at 23.4318. The policy stayed close to uniform: final entropy was
2.76893 against a 2.77259 maximum, final approximate KL was `6.71e-6`, and the
clip fraction was zero. It found 107 improving moves, fewer than the control's
114, while 3,145 of its 4,100 proposals were objective-neutral. This pattern
points to insufficiently informative proposal sets and credit, not evidence
that a well-separated learned proposal distribution needs only a longer run.

The next defensible ablation should change one factor: construct candidate
pools around tasks participating in the current bottleneck stage for both the
trained and uniform conditions. PPO may then rank destinations or swap partners
within that matched pool. Exact delta labels can also be tested, but every
candidate evaluated by the true objective must count against the common budget.
An LSTM or a larger training budget comes later because memory and duration do
not address a neighborhood in which most actions have no measured effect.

That single-factor ablation is implemented as
`--ppo_asa_focus_bottleneck`. A local 1,001-evaluation smoke reduced the neutral
rate from roughly 89% in the earlier short run to 72.8%, and the improving-move
rate reached 7.5%. The learned condition beat the identically focused uniform
control by 0.40% (24.0627 versus 24.1598 microseconds), but ordinary ASA still
won at 23.9629. The result is a learning-only signal. One focused 4,101-
evaluation gate is warranted to see whether the advantage persists; five seeds
remain conditional on beating ASA as well.

### Gate 1: implementation sanity

Run bounded synthetic and AlexNet-CONV smoke tests. Required conditions:

- zero collision repairs;
- exactly one unique executed physical core per logic core;
- finite actor loss, critic loss, Q values, and gradient norms;
- exact cost agreement when replaying a demonstration;
- checkpoint resume succeeds without deleting earlier main diagnostics.

This gate passes locally.

### Gate 2: learning, not search luck — passed for masked PPO on AlexNet-CONV

Masked PPO passed its trained-versus-frozen comparison at 3,000 placements over
five seeds. Guided DDPG failed the same gate. This conclusion concerns policy
learning on the reconstructed AlexNet-CONV objective, not paper reproduction.

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

## Paired Kaggle short result (8 October 2026)

The corrected one-seed trained/control run used commit `aa75b2d`, 300 online
placements per condition, identical seed 0 action randomness, 60 legal-random
warm-up placements, and separate replay randomness. Mechanics and numerical
stability passed: warm-up costs matched exactly, collision repairs were zero,
Q magnitude stayed below 0.127, losses were finite, and late proto-action
saturation was below 1%.

It did not show learning. The trained policy's late deterministic mean was
41.4736 microseconds versus 38.3919 microseconds for the no-learning control,
so training was 8.03% worse. The trained best candidate was 38.1083
microseconds versus 37.6356 microseconds for the control. This is one seed and
does not establish a population result, but it is direct evidence against the
current update at the 300-placement budget.

An older 3,000-placement guided run from commit `107899a` reached a 37.1630
microsecond best search trajectory and lower late noisy costs. It lacked a
matched no-learning control, permanent demonstration retention, legal-random
warm-up, and separated random streams. Its deterministic diagnostics did not
show a clean sustained improvement. It is useful for choosing a 3,000-placement
follow-up horizon but cannot be used as proof that policy training helped.
