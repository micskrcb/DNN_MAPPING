# Technical Report: RL-Guided Core Placement for Multi-Chip DNN Accelerators

**Repository:** `micskrcb/DNN_MAPPING`
**Technical-report branch:** `technical-report-draft`
**Implementation baseline:** cleaned-up branch, following commit `1106478`
**Status:** implementation and experimental report, not a submission-ready paper

## Abstract

This repository studies the placement of neural-network logic cores onto the
physical cores of a multi-chip, many-core accelerator. The intended reference
is Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural
Network Systems with Reinforcement Learning*, ACM TODAES 2020,
DOI [10.1145/3418498](https://doi.org/10.1145/3418498). The original paper
does not release its simulator or all reconstruction parameters. Accordingly,
this project is an independent, documented reconstruction rather than an exact
replication claim.

The current system models DNN layers as directed graphs of VMM
(vector-matrix-multiply) and VVA (vector-vector-accumulation) logic cores. A
placement assigns each logic core to a unique physical core in a hierarchical
multi-chip mesh. The primary objective is reconstructed bottleneck pipeline
latency under compute time, XY routing and shared-link contention.

Several learning formulations were implemented and tested: continuous DDPG,
guided DDPG, constructive masked PPO and PPO-guided simulated annealing. The
earlier DDPG and PPO formulations did not provide reliable optimizer gains.
The active method is a contextual-bandit proposal ranker inside **fixed
simulated annealing (SA)**. It scores a pool of legal relocation/swap proposals
while SA retains temperature control and Metropolis acceptance. The method has
positive matched-control evidence on AlexNet-FC and VGG16-FC, but it is not yet
established as superior to the strongest non-learned proposal heuristic or to
fixed SA under every workload.

This report explains the theory, implementation, evidence, file layout and
limitations needed to understand and extend the project safely.

## 1. Research question

Given a directed workload graph of logic cores, assign each logic core to a
distinct physical core such that pipeline latency is minimized. The central
question is whether a learning component can improve the *search process* over
strong classical optimization under equal objective-evaluation budgets.

The project deliberately separates three claims that are often conflated:

1. **The program runs:** it produces legal placements and finite costs.
2. **The learner changes decisions:** its parameters, scores and selected
   actions diverge from a frozen control.
3. **The learner improves optimization:** under matched budget and matched
   mechanics, it yields lower final placement cost than an appropriate control.

Only the third statement is an optimizer result. A low best-so-far cost by
itself is not evidence that RL learned; random exploration, initialisation or a
repair heuristic may have found it.

## Paper-aligned benchmark comparison

The reference paper reports latency and throughput normalized to its sequential
baseline (BS), but a random-search (RS) reference is less sensitive to the
unpublished sequential logical-core ordering. Its Figure 10 contains bars
rather than a numerical table, so the per-workload ratios below are
**two-significant-figure readings of those bars**, included to make the
comparison concrete rather than to claim precision that the chart does not
provide. The reported aggregate DDPG reductions are exact statements from the
paper: 50.5% versus BS, **38.4% versus RS**, and 18.6% versus SA. Its protocol
uses one million RS and SA placement evaluations; DDPG is described as passing
the best RS/SA result around 300k--400k evaluated placements.

| Paper Figure 10 latency, normalized to RS | SA / RS | DDPG / RS | DDPG improvement vs RS | Supporting RS / BS ratio |
| --- | ---: | ---: | ---: | ---: |
| AlexNet CONV | ~0.91 | ~0.82 | ~18% | ~0.88 |
| AlexNet FC | ~0.86 | ~0.74 | ~26% | ~0.58 |
| AlexNet overall | ~0.92 | ~0.82 | ~18% | ~0.84 |
| VGG16 CONV | ~0.60 | ~0.48 | ~52% | ~0.67 |
| VGG16 FC | ~0.70 | ~0.53 | ~48% | ~0.40 |
| VGG16 overall | ~0.60 | ~0.48 | ~52% | ~0.67 |
| ResNet50 CONV | ~0.74 | ~0.62 | ~38% | ~0.86 |
| ResNet50 FC | ~0.98 | ~0.94 | ~6% | ~0.88 |
| ResNet50 overall | ~0.74 | ~0.60 | ~40% | ~0.86 |
| **Paper geometric-average claim** | **~0.76 implied** | **0.616** | **38.4%** | — |

The reconstructed system implements the paper-facing topology and workload
counts, but cannot yet be called a numeric reproduction.  The missing VVA
throughput, logic-core numbering, per-layer grids and simulator details change
the amount of placement headroom.  The following table therefore compares
**matched project baselines** in the same format while keeping the operating
condition explicit.  Percentages are reductions in latency, so larger is
better.  They are valid comparisons *within each row*, not against a paper
bar in another row.

| Project protocol: AlexNet FC, 4x4 chips, 16x16 cores/chip | RS latency (µs) | Fixed SA / improvement vs RS | ASA / improvement vs RS | Best result compared with paper DDPG-vs-RS | Supporting BS context |
| --- | ---: | ---: | ---: | --- | --- |
| 1M evaluations, VVA = 1 op/cycle, fixed grids, seed 0 | 23.93728 | 22.70400 / **5.15%** | 23.09328 / **3.53%** | Fixed SA is below the paper's ~26% AlexNet-FC DDPG-vs-RS reading. | BS = 24.92032 µs; the compute floor is 89.9% of BS. |
| 1M evaluations, VVA = 16 ops/cycle, rate-one grids held fixed, seed 0 | 3.07200 | 2.05125 / **33.23%** | 2.39488 / **22.04%** | Fixed SA exceeds the paper chart's ~26% AlexNet-FC DDPG-vs-RS reading, but this is one exploratory assumed-rate seed and is not an RL result. | BS = 3.92032 µs; fixed SA is 0.5232× BS. |

The current learned bandit is **not** entered in the preceding paper table:
its 100,001-call experiments use a focused candidate pool and a different
control question.  It would be misleading to compare its 1.6--2.3% advantage
over a matched frozen control with the paper's one-million-placement DDPG/RS/SA
bars.  Its proper, paper-style improvement table is instead:

| Current learned-SA protocol | Budget | Comparator | Learned improvement | RS comparison |
| --- | ---: | --- | ---: | --- |
| AlexNet FC, held-out seeds 3--5 | 100,001 true-objective evaluations | Focused uniform frozen control | **1.965%** mean; 3/3 seeds | No same-protocol RS arm: do not calculate a paper-style RS percentage. |
| VGG16 FC, seeds 0--2 | 100,001 true-objective evaluations | Focused uniform frozen control | **2.255%** mean; 3/3 seeds | No same-protocol RS arm: do not calculate a paper-style RS percentage. |
| VGG16 FC, held-out seeds 3--5 | 100,001 true-objective evaluations | Focused uniform frozen control | **1.637%** mean; 3/3 seeds | No same-protocol RS arm: do not calculate a paper-style RS percentage. |
| VGG16 FC, all six seeds | 100,001 true-objective evaluations | Focused uniform frozen control | **1.946%** mean; 6/6 seeds | No same-protocol RS arm: do not calculate a paper-style RS percentage. |

The next defensible paper-aligned RL experiment is to freeze an execution-rate
and logical-order condition, run fixed SA, ASA and learned SA at the same
one-million complete-placement budget over multiple predeclared seeds, and
then report this exact table with confidence intervals.  Only then can a
learned-SA value be compared to the classical paper baselines without mixing
different budgets or controls.

## 2. System and workload model

### 2.1 Physical topology

The paper-mode configuration uses a `4 x 4` array of chips. Every chip contains
a `16 x 16` mesh of processing cores, giving

\[
N_{physical} = 4 \times 4 \times 16 \times 16 = 4096
\]

physical cores. The topology is implemented by
[`src/multi_chip_topology.py`](../src/multi_chip_topology.py).

A global core ID is chip-major. The topology can return its chip and local-core
coordinates, compute a legacy distance cost, or construct an explicit routed
path. The active paper-oriented routing model is `paper_xy`:

- traffic inside a chip follows dimension-ordered XY routing;
- inter-chip traffic first moves to a lower-left chip-periphery gateway;
- it travels through the chip grid in X then Y order;
- it routes from the destination gateway to the final core.

The gateway and exact global-routing-switch behavior are reconstruction
assumptions because the reference simulator is unavailable. `mesh` is the
paper-oriented topology; `torus` is supported only for exploratory legacy
distance experiments.

### 2.2 Logic cores and task graph

`src/run_multi_chip.py` traces built-in torchvision models and constructs a
directed task graph. A DNN layer is split into a two-dimensional partition grid:

- each `(input-group, output-group)` tile is a VMM logic core;
- each output group has a VVA logic core that accumulates its VMM partial sums.

For a layer with `M` output groups and `N` input groups, the layer contains
`M*N` VMM tasks plus `M` VVA tasks. Directed edges represent tensors produced by
one logic core and consumed by another. VMM outputs are treated as 32-bit
partial sums; VVA outputs are treated as 8-bit activations.

`src/compute_model.py` provides deterministic partition construction. In
`paper_targets` mode, `paper_partition_grids()` chooses each layer's `(M, N)`
grid such that the total number of logic cores matches the aggregate counts
published in Figure 6, while enforcing the 64 KB 8-bit weight-tile limit and
balancing estimated VMM and VVA work. Published target totals are:

| Model | CONV logic cores | FC logic cores |
| --- | ---: | ---: |
| AlexNet | 183 | 932 |
| VGG16 | 1,024 | 1,924 |
| ResNet50 | 512 | 37 |

The paper does not publish individual layer grids, VVA rate or logical task
ordering. These choices are therefore parameters and reconstruction assumptions,
not recovered ground truth.

### 2.3 Compute timing

`compute_seconds()` maps task work into idealized execution time. VMM tasks use
the documented 128 MACs per cycle at 400 MHz. VVA tasks use the explicit
parameter `--vva_ops_per_cycle`, because the paper does not specify the rate.
For task `i`, the simplified compute model is

\[
t_{comp}(i)=\frac{\lceil operations_i /(rate_i \cdot utilization) \rceil}
{400\text{ MHz}}.
\]

This excludes activation-buffer stalls, router startup, exact packet scheduling,
some transformation operations and detailed compute/communication overlap.

## 3. Objective function and evaluator

### 3.1 Placement representation

Let `P(i)` be the physical core chosen for logic task `i`. A valid placement is
injective: two logic tasks cannot occupy the same physical core. A workload can
be restricted to an `allowed_cores` mask, which lets CONV and FC regions use
separate contiguous whole-chip regions.

`MultiChipEnvironment` in
[`src/multi_chip_environment.py`](../src/multi_chip_environment.py) owns the
task graph, the current placement and legality checks.

### 3.2 Pipeline latency

The primary cost is a bottleneck-stage latency, intended to reflect the
paper's Equation 4 rather than total wirelength. Kahn topological levels form
dependency stages `S_k`. In the basic reconstructed model,

\[
L(P)=\max_k T_k(P),
\]

where a stage cost is the largest task service time in that stage. A task's
service time includes its compute time plus routed communication to placed
successors:

\[
t_i(P) = t_{comp}(i) +
\sum_{j \in succ(i)} bytes(i,j)\,c(P(i),P(j)).
\]

For `paper_pipeline`, the evaluator uses `paper_xy` routes and evaluates each
topological phase using the larger of:

- the largest routed byte-hop cost emitted by one source task; and
- the largest serialized load on any shared directed link.

This is implemented by `pipeline_stages()`, `pipeline_breakdown()` and
`_xy_contention_components()`.

The repository still exposes `total_communication_cost()` and
`chip_breakdown()` for diagnostics. They are **not** the optimization objective;
they sum pairwise traffic costs and do not represent the bottleneck pipeline.

### 3.3 Why objective sensitivity matters

Optimization cannot improve a placement-insensitive cost meaningfully. An
earlier AlexNet-CONV partition created one dominant VVA task whose compute time
was effectively fixed across placements. The apparent DDPG gain in that regime
was not meaningful. The present runner therefore has a random-placement
sensitivity preflight. It samples placements and can stop a long paper-mode
run when the observed relative objective span is below the configured minimum.

`--allow_flat_objective` bypasses this protection and should be used only for
diagnostics.

### 3.4 Incremental exact evaluation

SA and learned-SA moves change only a small number of task locations.
`IncrementalPipelineEvaluator` in `src/run_multi_chip.py` caches unaffected
stage costs and reevaluates only stages touched by moved tasks and their local
dependencies. It is intended to produce the same objective as full evaluation
while reducing the cost of search. The flags `--sa_full_evaluation` and
`--asa_full_evaluation` are reference/debug alternatives.

## 4. Search baselines

All methods optimize the same placement objective but must be compared under
the same number of **complete placement objective evaluations**.

| Method | CLI value | Role |
| --- | --- | --- |
| Sequential baseline | `bs` | Deterministic chip-major task placement reference. |
| Random search | `random` | Samples independent valid permutations; best sampled cost is retained. |
| Fixed SA | `sa` | Primary classical baseline with calibrated geometric cooling. |
| Adaptive SA | `asa` | Temperature and move-scale feedback baseline. |
| DDPG | `ddpg` | Historical paper-oriented continuous actor-critic. |
| Guided DDPG | `ddpg_guided` | Experimental legal-action DDPG with demonstrations. |
| Masked PPO | `ppo_masked` | Experimental constructive legal placement policy. |
| PPO/SA or bandit/SA | `ppo_asa` | Learned proposal selection inside SA. |

### 4.1 Fixed simulated annealing

Starting from a legal random placement, a fixed-SA step proposes a legal
relocation or swap. If the change in objective is

\[
\Delta=L(P')-L(P),
\]

the move is accepted whenever `Δ <= 0`; otherwise it is accepted with

\[
p_{accept}=\exp(-\Delta/T).
\]

The initial temperature is measured from budgeted uphill samples to target an
initial acceptance rate, normally 0.8. It then cools geometrically with the
paper-oriented factor `0.99`. This calibration matters because fixed absolute
temperatures are invalid when the objective is expressed in seconds rather than
old proxy units.

### 4.2 Adaptive simulated annealing

`run_adaptive_sa()` adapts temperature over windows, targets an acceptance
fraction near 0.30 and changes perturbation size when the chain stalls. It is
implemented and useful as a comparison, but under current tests it has usually
been weaker than calibrated fixed SA. This does not imply that adaptive
metaheuristics fail generally; it only describes this controller and evaluator.

## 5. Retained reinforcement-learning formulations

### 5.1 Continuous DDPG

The original paper-oriented implementation uses an actor that emits continuous
coordinates and a critic that estimates `Q(state, action)`. It includes MLP,
spatial CNN and paper-inspired CNN variants:

- `Actor` and `Critic`: two-hidden-layer MLPs;
- `CNNActor` and `CNNCritic`: grid encoder plus placement features;
- `PaperCNNActor` and `PaperCNNCritic`: Figure-9-inspired convolutional stack
  and fully connected layers.

DDPG actions are projected from continuous coordinates to discrete core IDs.
When several tasks target the same physical core, a collision repair heuristic
finds available locations. This created a serious credit-assignment problem:
the action in replay did not faithfully represent the executed placement after
many repairs. The DDPG path is retained for audit and ablation, not as the
active method.

`GuidedReplayBuffer` and `run_ddpg_guided()` are an attempted repair. They use
legal candidate actions, ASA demonstration trajectories, prioritized replay,
twin critics and behaviour cloning. These experiments did not yet establish a
robust learned advantage.

### 5.2 Constructive masked PPO

`run_masked_ppo()` places tasks sequentially. Its categorical action mask gives
zero probability to occupied or disallowed physical cores, so it eliminates
collision repairs by construction. PPO uses a clipped policy objective:

\[
L^{CLIP}(\theta)=\mathbb{E}\left[
\min\{r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\}
\right].
\]

The masked implementation proves legal-action mechanics but its tests did not
show a reliable final-quality advantage over strong SA baselines.

### 5.3 PPO-guided SA

The first hybrid uses PPO to rank a dynamically generated pool of legal SA
neighbors. It keeps SA's candidate generation and Metropolis acceptance. The
reference-compatible option, `--ppo_asa_update_style reference_separate`, uses
separate actor and critic networks, optimizers, reward-to-go targets and
GAE-style advantages. Its design and third-party attribution are documented in
[`docs/PPO_REFERENCE_ADAPTER.md`](PPO_REFERENCE_ADAPTER.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Long PPO tests showed weak or uniform proposal policies. The project therefore
does not interpret the lack of PPO gain simply as insufficient epochs: it ran
mechanics gates, planted-signal checks and larger update schedules before
changing formulation.

## 6. Active method: contextual-bandit-guided fixed SA

### 6.1 Motivation

In this problem a proposal has an observable one-step outcome after exactly one
objective evaluation. A contextual bandit can directly regress that immediate
outcome instead of learning long-horizon value estimates. The design therefore
uses RL only where it has a clear task: choose a promising proposal from a pool
of otherwise legal SA moves.

SA still owns global search behavior:

- initial placement;
- candidate legality;
- temperature schedule;
- acceptance of uphill moves;
- best-so-far retention.

The bandit changes proposal selection only.

### 6.2 Candidate pool and features

At every step, `run_ppo_asa()` constructs `K=16` legal neighbor proposals. In
the active configuration, half are anchored to tasks from the current
bottleneck stage and half are ordinary proposals. Every proposal receives nine
features:

1. whether its task belongs to the bottleneck stage;
2. whether the task is VVA;
3. normalized task arithmetic work;
4. task work as a fraction of its stage;
5. predecessor hop reduction;
6. successor hop reduction;
7. traffic-weighted hop reduction;
8. same-chip connectivity change;
9. `proxy_gain`, an inexpensive routed-communication heuristic.

The global state includes chain progress, previous objective delta, previous
accept/reject status and recent acceptance fraction.

### 6.3 Bandit model and update

`ProposalBanditRegressor` has a global-state encoder and a candidate encoder,
both one-layer `tanh` MLPs with 128 hidden units. The encodings are added and
passed through a scalar score head. The score head is initialized to zero, so
before training every candidate ties and selection is exactly uniform.

For an evaluated candidate, the signed normalized proposed reward is

\[
r = \operatorname{clip}\left(\frac{L(P)-L(P')}{s},-10,10\right),
\]

where `s` is a reward scale. Positive reward means that the candidate improved
the objective, whether or not it is ultimately accepted by SA. The selected
candidate's `(global state, feature row, proposed reward)` is stored in a
uniform replay buffer. AdamW and smooth-L1 loss train the score:

\[
\mathcal{L}_{bandit}=\operatorname{Huber}(\hat r(s,x),r).
\]

Uniform replay is intentional. Earlier positive-balanced replay overrepresented
rare improvements, causing predicted gains to become unrealistically positive
and harmful exploitation.

### 6.4 Action selection and control

The bandit uses epsilon-greedy selection. It begins uniformly, remains uniform
through warm-up, then decays its random-selection probability to 0.50 over the
whole run. A high final epsilon preserves exploration because the candidate
score is noisy and only selected candidates receive labels.

The primary causal control uses the same candidate generator, warm-up,
temperature schedule, objective-call budget and randomness, but disables
learning. Its zero-initialized scorer is exactly uniform. A second,
more demanding comparison uses `proxy_greedy`, which ranks candidates directly
by the hand-engineered `proxy_gain` feature without training.

## 7. What the experiments currently support

The project has evidence of learning and of optimization benefit in narrow,
well-defined settings. The most defensible statement is:

> Under equal true-objective budgets and the same focused candidate pool, the
> contextual-bandit selector improved over a focused-uniform frozen control on
> the tested AlexNet-FC and VGG16-FC workloads. It is not yet proven to be
> superior to the strongest non-learned proposal rule, to fixed SA on every
> workload, or to transfer broadly beyond these workloads.

Selected evidence recorded in `docs/PROJECT_STATE.md`:

| Workload and protocol | Main observation | Interpretation |
| --- | --- | --- |
| AlexNet-FC, held-out seeds 3–5, 100,001 evaluations | Learned method beat frozen control 3/3; mean +1.965%, reported paired interval [1.008%, 2.922%]. | Positive held-out-seed evidence on a graph used during development; not a new problem-instance test. |
| AlexNet-CONV, three seeds, 100,000 evaluations | Learned, frozen and fixed-SA methods reached the same final minimum. | A valid final-quality tie. Any earlier advantage is at most an unresolved early-convergence signal. |
| VGG16-FC, seeds 0–2, 100,001 evaluations | Learned beat frozen control 3/3, mean +2.255%; reported paired interval [1.107%, 3.403%]. | Positive transfer signal without retuning. |
| VGG16-FC, held-out seeds 3–5 | Learned beat focused-uniform 3/3, mean +1.637%; it was effectively tied with fixed SA. | Strengthens the matched-control result; does not prove fixed-SA superiority. |
| VGG16-FC, all six seeds | Learned beat focused-uniform 6/6, mean +1.946%; fixed-SA difference remains uncertain. | Strongest present evidence, still exploratory due to development sequence and limited independent units. |

The recent short proxy-greedy comparison is especially important because it
tests whether learning merely rediscovered the hand-crafted `proxy_gain`
feature. It must be completed at a longer matched budget before becoming a
publication-level conclusion.

## 8. Limits and threats to validity

### 8.1 Simulator fidelity

The evaluator is a documented approximation, not the authors' unpublished
simulator. Important missing or uncertain factors include exact global routing,
activation-buffer stalls, detailed GRS behavior, block scheduling, packet
scheduling, router startup, overlap of compute and communication, VVA rate and
logical ordering. Paper-normalized ratios are therefore diagnostic comparisons,
not proof of exact reproduction.

### 8.2 Attribution

The focused-uniform control isolates the trained selector package. It does not
fully isolate learned representation from the hand-crafted candidate generator,
bottleneck focus or `proxy_gain` feature. The proxy-greedy control, feature
ablation and shuffled-label tests are the right next causal checks.

### 8.3 Statistical scope

Seeds provide repeated stochastic runs but do not create new workload instances.
Three-seed intervals are descriptive and unstable. The repository should report
all individual seed results, the exact call budget, wall-clock time, uncertainty
intervals and predeclared decision rules. It should not report only the best
seed or treat a best-so-far improvement as policy learning.

### 8.4 Objective-call versus wall-clock cost

The learned method scores 16 candidate proposals and performs neural updates.
Equal complete-objective-call budgets make final-quality comparisons fair, but
they do not make wall-clock cost equal. Both should be reported.

## 9. Code map

| File or directory | Responsibility | Read when you need to… |
| --- | --- | --- |
| `src/run_multi_chip.py` | Workload extraction, RL models, all algorithms, CLI and reports. | Change an algorithm, reward, candidate feature or CLI setting. |
| `src/multi_chip_environment.py` | Legal placement state, objective, stage construction and diagnostics. | Change latency semantics or validate an objective. |
| `src/multi_chip_topology.py` | Core IDs, physical coordinates, distance and XY paths. | Change hardware topology or route construction. |
| `src/compute_model.py` | Logic-core grids, operation counts and timing conversion. | Change partition assumptions or VMM/VVA timing. |
| `src/run_paper_experiment.py` | Paper-style CONV/FC orchestration. | Reproduce a documented suite. |
| `src/run_multiseed_experiment.py` | Independent multi-seed subprocess scheduling. | Run matched repeated experiments. |
| `src/analysis/` | Summary and gate-analysis tools. | Analyze an existing archive without changing optimization. |
| `src/experiments/` | Historical or exploratory prototypes. | Inspect earlier approaches; do not treat them as supported runners. |
| `scripts/` | Kaggle and batch experiment gates. | Run preserved, parameterized protocols. |
| `tests/` | Smoke and reconciliation tests. | Verify a code change before a large run. |
| `docs/PROJECT_STATE.md` | Current quantitative record. | Make claims about results. |
| `docs/RECONCILIATION.md` | Paper assumptions and evaluator corrections. | Compare implementation with the paper. |
| `docs/RESEARCH_FINDINGS.md` | Literature, failed approaches and causal concerns. | Choose the next experiment. |
| `docs/PROJECT_JOURNEY.md` | Chronological history. | Understand how and why the project changed. |
| `docs/archive/` | Superseded but preserved evidence. | Audit older claims and historical paths. |

## 10. Reproducibility protocol

For any claimed result:

1. Record the Git commit, model, workload region, topology, timing model,
   VVA rate, logical task order, seed and device.
2. Run the objective-sensitivity preflight before long paper-mode search.
3. Use matched total complete-placement objective evaluations.
4. Keep trained and control conditions identical except for learning.
5. Log diagnostics: costs, acceptance, objective counts, updates, epsilon,
   score variation and finite losses/gradients.
6. Save report JSON, JSONL diagnostics, manifest and archive checksum.
7. Report all seeds, not only the best run.
8. Distinguish exploratory tuned seeds, held-out seeds and new workloads.

The standard active configuration is exposed by `--algo ppo_asa` together with
`--ppo_asa_learner bandit` and `--ppo_asa_annealing fixed`. Refer to the
repository README and `docs/IMP_README.md` for exact supported commands; do not
reconstruct a long experiment command from memory.

## 11. Publication path

The current work is appropriate for a carefully bounded workshop or
reproducibility-oriented paper only after the claims are matched to the
evidence. A positive-method paper needs at least:

1. A longer proxy-greedy matched-control study.
2. Feature and focus ablations establishing what the learned scorer contributes.
3. More held-out seeds and at least one additional workload family.
4. A frozen evaluator configuration and fully reproducible result archives.
5. A clear statement that the system is a reconstructed simulator where the
   original paper omits parameters.

The scientifically defensible proposed contribution is not “RL always beats
SA.” It is a contextual-bandit candidate-selection mechanism for SA in a
reconstructed multi-chip DNN placement simulator, together with controlled
evidence, explicit negative results and a reproducible methodology.

## 12. Related documentation and attribution

- The project history is in [`PROJECT_JOURNEY.md`](PROJECT_JOURNEY.md).
- The contributor-oriented architecture guide is in
  [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md).
- Paper fidelity assumptions are in [`RECONCILIATION.md`](RECONCILIATION.md).
- Current results and unresolved work are in
  [`PROJECT_STATE.md`](PROJECT_STATE.md) and [`NEXT_STEPS.md`](NEXT_STEPS.md).
- The PPO adapter's source relationship is documented in
  [`PPO_REFERENCE_ADAPTER.md`](PPO_REFERENCE_ADAPTER.md) and
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

This report should be updated whenever the evaluator, active method, control
ladder or publication claim changes.
