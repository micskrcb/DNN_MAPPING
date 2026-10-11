# Local DDPG normalization experiments, 2026-09-30

Status: exploratory runs in progress; no successful learning claim.

All new runs start fresh with seed 4, AlexNet CONV, paper-target partitioning,
paper-pipeline latency, XY routing, Figure 9 CNN, batch_z=3, 100 complete
placements, 64 reward-normalizer samples, 64 sensitivity samples, one optimizer
update per ten actions, two CPU threads, gamma=0.98, tau=0.005 and an absolute
1,000-placement OU decay horizon. Deterministic diagnostics run every two
placements and may update the retained best. Retained best and current policy
quality must be assessed separately.

| Directory | Reward | Target running statistics | Critic mode for actor update |
| --- | --- | --- | --- |
| local-cpu-batchnorm-long-2026-09-30 | potential | hard copy | evaluation |
| local-cpu-polyak-2026-09-30 | potential | Polyak average | evaluation |
| local-cpu-polyak-sparse-2026-09-30 | sparse | Polyak average | evaluation |
| local-cpu-actor-bn-2026-09-30 | sparse | Polyak average | training; restore buffers after backward |
| local-cpu-untrained-sparse-2026-09-30 | sparse | unused | no optimizer updates |
| local-cpu-consistent-bn-2026-09-30 | sparse | Polyak average | evaluation; critic fitting also uses evaluation mode |

The consistent variant uses `src/run_consistent_bn_experiment.py` instead of
the normal entrypoint. It calibrates online running statistics once per sampled
batch, then computes both gradient objectives with evaluation normalization.
The actor uses that same normalization when choosing actions. It carries a
separate checkpoint marker; cross-mode resume is rejected. This normalization
choice is an experimental assumption, not specified in the paper.

The untrained control uses `train_every=100000000`, so its 6,100 environment
steps never trigger an update. It shares the same initialization, seed,
exploration schedule, reward baseline and diagnostic candidate budget with
the trained sparse runs. Its fixed deterministic policy is the direct
reference for learned-policy improvement. It does not save a checkpoint.

The first row resumed from placement 12 without replay restoration. The other
rows are uninterrupted fresh runs, so a comparison to the first row confounds
normalization and replay restoration. The two sparse runs form the direct
comparison for the actor-update change. Runtime comparisons are invalid because
these jobs overlap on the same CPU.

The paper specifies BatchNorm and sparse terminal rewards, but does not specify
this target-buffer policy or normalization modes during actor optimization.
These are implementation choices, not a claim to have recovered author code.
The reduced optimizer frequency, tiny baseline budget and short training budget
also make these diagnostics different from a full reproduction experiment.

## Mechanism probe

`src/diagnose_checkpoint_normalization.py` evaluates fixed prefixes of a saved
best layout with zero actions. At placement 10 of the Polyak/potential run, the
online critic's mean output was 26.91976 using running statistics and 0.15318
using batch statistics. The target critic's corresponding means were 0.15564
and 0.36232. This localizes a large discrepancy to the online critic even after
changing target updates. The probe distribution is constructed, not drawn from
replay; it does not prove a causal explanation of performance.

The actor-update variant uses batch statistics when differentiating the critic
with respect to actor actions. Critic weights are frozen, and running buffers
are restored after backward. Tests verify finite updates, actual actor parameter
changes, restored critic buffers, and target inference independence from batch
companions. Further diagnostics measure replay Q, inference-mode replay Q,
Bellman targets and sampled rewards. These changes do not guarantee learning.

## Decision gate

The seed-4 untrained control completed all 100 placements with zero optimizer
updates. Its deterministic cost remained 38.5809375 µs; its best retained
exploration candidate reached 37.871953125 µs. These are two different
comparators: final policy versus fixed policy, and best-of-search versus
best-of-search. A trained policy merely beating the random-search baseline
does not establish a training benefit.

Evaluate the full deterministic-cost trajectory, final policy cost, action
saturation, repair counts, and critic-value drift. An early retained minimum
alone is insufficient. Any promising variant must then be tested on additional
seeds against untrained controls with matched candidate and evaluation budgets.
Do not select a winning seed as evidence of general learning. Full paper
comparisons still require all workload regions and the declared larger budgets.

## Completed 100-placement diagnostics

All conditions below use seed 4 and are single-seed diagnostics, not a basis
for selecting an algorithm or claiming reproduction.

| Condition | Final deterministic latency | Best retained latency | Finding |
| --- | ---: | ---: | --- |
| Untrained sparse control | 38.58094 us | 37.87195 us | Reference; zero updates |
| Potential + Polyak targets | 47.65594 us | 37.63563 us | Policy worsened |
| Sparse + Polyak targets | 47.65594 us | 38.58094 us | Policy worsened |
| Sparse + actor batch statistics | 47.65594 us | 37.63563 us | Policy worsened |
| Consistent inference normalization | 43.30750 us | 39.14813 us | Less severe, still worse than control |
| Compact active-region input | 39.99891 us | 37.87195 us | Faster and more stable, but no learned-policy improvement |

The compact-state trial retained the physical placement simulator and objective;
it only cropped the constant masked part of the Figure-9 CNN input. It completed
100 placements in about two minutes locally, versus much longer full-frame
diagnostics. Its final policy was 3.7% worse than the untrained policy, so it
remains an ablation rather than a candidate fix.
