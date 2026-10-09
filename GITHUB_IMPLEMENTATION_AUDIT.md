# Public implementation and training-scale audit

Audit date: 2026-10-09.

## Why the current RL gates do not reproduce the paper's advantage

The completed PPO-ASA gates use 4,101 true-objective evaluations from one
fixed AlexNet-FC instance and perform 32 policy updates. This is a useful
mechanics and learning-signal test, but it is far below the training scale used
by the related published methods.

Wu et al. report convergence after roughly 300,000 to 400,000 complete DDPG
placements. Their random-search and simulated-annealing baselines each search
around one million placements. The exact simulator and source used for those
results are not public, so this project reconstructs unpublished workload
partition, routing, timing, and training details.

The official RL-Based-SA default PPO experiment trains for 1,000 epochs. Its
TSP configuration runs 256 independent problem instances for 40 annealing
steps per epoch: 10.24 million transitions before evaluation, with new random
instances generated each epoch. Our 4,068 post-calibration policy transitions
from one fixed instance are about 2,500 times fewer and contain far less state
diversity. Replacing PPO while retaining one short chain will not close this
difference.

The reconstructed maximum-stage pipeline objective also makes many local moves
uninformative. In the first 4,101-evaluation PPO-ASA gate, 76.7% of selected
proposals did not change the objective. Bottleneck focusing reduced this to
37.2%, but trained PPO still finished 0.083% behind its frozen focused control
and 1.482% behind ordinary ASA. The learned policy stopped improving near
evaluation 1,232; ASA continued improving through evaluation 4,100.

## Repositories checked

### RL-Based-SA-Public

Repository: <https://github.com/nathanqiu07/RL-Based-SA-Public>

This is the closest reusable implementation for a learned-neighbor hybrid. It
has a permissive three-clause-style license and contains PPO, Metropolis
acceptance, fixed cooling, energy-change state, and an LSTM branch. Its problem
adapters cover knapsack, bin packing, TSP, and continuous test functions. It
does not contain DNN core placement, the paper-pipeline objective, legal
physical-core mapping, or multi-chip routing. It can reproduce its own
benchmarks directly; using it here still requires a new vectorized placement
problem and policy/action adapter.

### Core_Placement_with_Reinforcement_Learning

Repository:
<https://github.com/WOOSHIK-M/Core_Placement_with_Reinforcement_Learning>

This is the closest public core-placement repository and the historical source
of this project. It contains PPO placement components, but its README states
that only single-chip mapping is available and multi-chip execution has not
been uploaded. The checked repository has no license file. It does not supply
the missing Wu et al. multi-chip simulator and is not a drop-in implementation
of the 2020 DDPG paper.

### Stable-Baselines3 Contrib MaskablePPO

Repository: <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>

MaskablePPO is a maintained implementation with discrete invalid-action
masking and multiprocessing vector environments. It can replace local PPO
bookkeeping after placement is exposed as a Gymnasium environment. It does not
provide the placement state, objective, action design, or candidate
neighborhood. Its practical value is reliable PPO and parallel independent
placement chains.

### AlphaChip / circuit_training

Repository: <https://github.com/google-research/circuit_training>

AlphaChip is a mature distributed PPO/GNN system for macro floorplanning with
wirelength, congestion, density, orientations, and DREAMPlace. It supports
distributed collection across many actors and relies on pretraining for its
strongest behavior. Its environment and checkpoints are incompatible with
fixed-grid DNN logic-core placement.

### MacroPlacement

Repository: <https://github.com/ABKGroup/MacroPlacement>

This independent assessment is useful context: its maintained simulated
annealing baseline can outperform evaluated RL macro placers. A strong ASA
result here is not evidence that the implementation is broken.

## Decision

Preserve and present the validated positive results immediately:

- five-seed masked PPO improves its matched frozen policy by 6.06% in late
  deterministic AlexNet-CONV latency, with all five pairs positive;
- five-seed ASA beats matched random search by 2.61% on AlexNet-FC and beats
  the sequential baseline by 5.70%;
- guided DDPG and both PPO-ASA neighborhoods are documented negative
  ablations.

For the next learning implementation, retain the current objective and legal
placement logic but train a shared proposal policy across many independent
placement chains. Use the official RL-Based-SA PPO structure or MaskablePPO,
and batch or parallelize environment collection. The first training-scale
target should be at least 250,000 true-objective evaluations, with a frozen
policy control and ordinary ASA receiving matched evaluation budgets. Every
true objective call remains part of the budget.

## Implemented response to the audit

The repository now supports repeated independently initialized PPO-ASA chains
through `--ppo_asa_restart_interval`. One proposal network and optimizer learn
from rollouts collected across all chains, while generalized-advantage
estimation treats each restart as terminal. A configurable mixture of global
and bottleneck-anchored proposals is available through
`--ppo_asa_focus_fraction`. New-chain initialization evaluations are included
in the true-objective budget, and the paired Kaggle runner increases ASA's
proposal count so all three conditions have identical totals.

The `multichain-train` protocol uses 250,000 proposals, 1,954 independent
chains, 4,096-transition PPO rollouts, and 251,954 objective evaluations per
condition. Collection inside each condition is sequential because the
placement evaluator is stateful; trained and frozen conditions run concurrently
on Kaggle's two GPUs. This obtains multi-chain data diversity without claiming
that the evaluator itself is vectorized.

The implementation follows the public RL-Based-SA experiment structure and
cites it in the source and documentation. No external source code was copied.
Stable-Baselines3 Contrib remains a licensed, maintained fallback if the
training-scale result identifies an optimizer problem; it was not added as an
unnecessary dependency before that evidence exists.

The completed training-scale gate showed that this independent implementation
can learn: trained PPO-ASA beat its frozen control by 0.466% and at 3,413 of
3,906 matched checkpoints. It did not beat ordinary ASA, which was 3.55%
better after one uninterrupted long chain. The follow-up protocol therefore
freezes the saved policy and evaluates unseen uninterrupted chains. This
addresses the different purposes of short diverse training episodes and deep
single-placement optimization before considering an external PPO dependency.
