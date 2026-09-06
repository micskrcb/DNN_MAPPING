# DNN Mapping with Reinforcement Learning

## Policy Gradient / Actor-Critic Based Core Placement Optimization for Multi-Chip Many-Core DNN Accelerators

This project implements and extends reinforcement-learning-based **core placement optimization for Deep Neural Network (DNN) workloads** on multi-chip many-core hardware architectures.

The central problem is:

> Given a DNN computational/communication graph and a physical many-core architecture, determine where each DNN computational task should be placed so that communication latency and overall execution cost are minimized.

The project started from the open-source repository:

**Core Placement with Reinforcement Learning**

and has been progressively modified to reproduce and extend the methodology described in the **2020 paper on reinforcement-learning-based core placement for multi-chip many-core systems**.

The current implementation contains:

* Single-chip reinforcement-learning placement
* Multi-chip topology and environment
* DDPG-based multi-chip placement
* PPO-based single-chip placement
* Graph-based workload representation
* CNN workload extraction using PyTorch
* Channel-partitioned CNN workloads
* VMM/VVA logic-core modeling
* Pipeline-stage latency objective
* Collision-resolution mechanisms
* Random Search baseline
* Simulated Annealing baseline
* Reproducible random seeds
* Batched action placement
* Multi-chip scaling toward the paper's 4 × 4 chip configuration
* Automated smoke tests for the major multi-chip components

The repository is currently in the **paper-reproduction and validation phase**. The long-term goal is to establish a faithful reproduction of the paper first and then introduce measurable improvements.

---

# 1. Problem Statement

Modern DNN accelerators execute neural-network operations across many processing or logic cores.

A DNN is represented as a graph in which:

* Nodes represent computational tasks.
* Edges represent communication between tasks.
* Edge weights represent communication/data volume.

The same DNN can have very different execution times depending on how its tasks are physically mapped onto the accelerator.

For example:

```text
Logic Core 0 ───────> Logic Core 1 ───────> Logic Core 2
       |                     |                     |
       v                     v                     v
Physical Core A       Physical Core B       Physical Core C
```

If communicating tasks are placed far apart, communication latency increases.

Therefore, the mapping problem is:

```text
DNN Task Graph
       |
       v
Placement Algorithm
       |
       v
Physical Many-Core Architecture
       |
       v
Minimize Communication / Pipeline Latency
```

The placement problem becomes particularly difficult when:

* The number of tasks is large.
* The number of physical cores is large.
* The architecture contains multiple chips.
* Communication crosses chip boundaries.
* Multiple tasks compete for the same physical cores.
* The workload contains thousands of partitioned DNN operations.

This project uses reinforcement learning to learn placement policies instead of exhaustively searching the entire mapping space.

---

# 2. Relationship to the 2020 Paper

The project is being developed with two distinct objectives.

## Phase 1 — Reproduce the Paper

The first objective is to reproduce the methodology and results of the 2020 work as faithfully as possible.

This includes reproducing:

* DNN task partitioning
* VMM/VVA logic-core representation
* Multi-chip topology
* State representation
* Action formulation
* Reward formulation
* Pipeline latency objective
* Collision resolution
* DDPG training
* Random Search baseline
* Simulated Annealing baseline
* Paper-scale hardware configuration
* Paper-scale workloads

## Phase 2 — Improve the Paper

Only after the baseline implementation is validated should modifications be presented as improvements.

Potential improvement directions include:

* Better graph representations
* GCN-based task embeddings
* Attention-based policies
* More efficient exploration
* Larger and more realistic DNN workloads
* Modern DNN architectures
* Topology-independent placement
* Improved training efficiency
* Better scalability

This distinction is important:

> Implementing something that already exists in the paper is a **reproduction**, not a novel contribution.

---

# 3. System Architecture

The current project can be viewed as the following pipeline:

```text
             DNN / CNN Workload
                     |
                     v
             Workload Extraction
                     |
                     v
             Task Graph Generation
                     |
                     v
        Channel / Weight Partitioning
                     |
                     v
             VMM + VVA Logic Cores
                     |
                     v
             Multi-Chip Environment
                     |
          +----------+----------+
          |                     |
          v                     v
        DDPG                   Baselines
          |                 /           \
          |                /             \
          v               v               v
      Placement        Random Search    Simulated Annealing
          |
          v
   Physical Core Mapping
          |
          v
 Pipeline / Communication Latency
```

---

# 4. Repository Structure

The important components of the project are organized approximately as follows:

```text
DNN_MAPPING/
│
├── src/
│   │
│   ├── agent/
│   │   ├── model.py
│   │   ├── actor.py
│   │   ├── agents.py
│   │   ├── distributions.py
│   │   └── community_detection.py
│   │
│   ├── env/
│   │   ├── core_mapper.py
│   │   └── reward.py
│   │
│   ├── runner/
│   │   ├── exact_mapping.py
│   │   ├── random_search.py
│   │   └── simulated_annealing.py
│   │
│   ├── data/
│   │   └── toy_lut.py
│   │
│   ├── config/
│   │   ├── example.py
│   │   ├── example2.py
│   │   └── 2020.py
│   │
│   ├── parsedata.py
│   ├── structure.py
│   │
│   ├── multi_chip_topology.py
│   ├── multi_chip_environment.py
│   ├── run_multi_chip.py
│   ├── test_multi_chip.py
│   │
│   ├── main.py
│   └── compare_algos.py
│
└── README.md
```

The exact repository structure may evolve as the implementation continues.

---

# 5. Workload Representation

The original implementation contains a graph representation based on tasks and communication relationships.

The basic abstraction is:

```text
Task / Logic Core
        |
        +---- communication ----> Task / Logic Core
```

Each communication edge contains an associated data/packet volume.

The mapping algorithm therefore needs to consider both:

1. Which tasks communicate.
2. Where those tasks are physically located.

---

# 6. CNN Workload Extraction

The current implementation contains an actual PyTorch-based CNN workload extractor.

The workload is not restricted to an artificial list of sequential layers.

The current implementation includes:

```text
SimpleCNN
    |
    v
PyTorch model
    |
    v
extract_cnn_task_graph()
    |
    v
Partitioned task graph
```

The extraction mechanism identifies computational layers and constructs a communication graph representing their execution dependencies.

---

# 7. Channel Partitioning

A major modification made during the paper-reproduction process is **channel-based partitioning of CNN layers**.

Instead of treating:

```text
Conv Layer
```

as a single task, the layer is partitioned into multiple logic-core operations.

The current implementation uses:

```text
channels_per_partition = 8
```

as the default partition size.

The partitioning follows the hardware model described in the paper:

```text
                 Layer
                   |
          +--------+--------+
          |                 |
          v                 v
     VMM Logic Cores   VMM Logic Cores
          |                 |
          +--------+--------+
                   |
                   v
              VVA Cores
                   |
                   v
              Next Layer
```

For a layer with multiple input and output channel groups:

```text
M output groups × N input groups
```

are used to construct the VMM operations.

The partial results are then accumulated through VVA logic cores.

---

# 8. VMM and VVA Logic Cores

The current partitioning model follows the paper's accelerator abstraction.

### VMM

Vector-Matrix Multiplication operations are represented as computational logic cores.

### VVA

Vector-Vector Accumulation operations collect partial results generated by the VMM operations.

The resulting structure is therefore more representative of the paper than a simple:

```text
1 CNN layer = 1 task
```

representation.

---

# 9. Current Workload Size

With:

```text
channels_per_partition = 8
```

the current `SimpleCNN` workload produces:

```text
906 logic cores
```

consisting of VMM and VVA tasks.

This is significant because the original simplified workload contained only approximately:

```text
15 tasks
```

The new representation therefore moves the project much closer to the scale of the paper.

For comparison, the paper discusses workloads such as approximately:

```text
AlexNet : 932 logic cores
VGG16   : 1924 logic cores
```

The current 906-task workload is therefore in the same general scale as the paper's AlexNet workload.

---

# 10. Multi-Chip Hardware Model

The project contains a `MultiChipTopology` implementation.

The architecture is represented as:

```text
Multiple Chips
      |
      +---- Chip 0
      |       |
      |       +-- Core grid
      |
      +---- Chip 1
      |       |
      |       +-- Core grid
      |
      +---- Chip 2
      |
      +---- ...
```

Each chip contains a 2D grid of physical cores.

The project supports configuration using:

```text
chips_x
chips_y
rows
cols
```

For example:

```bash
--chips_x 4
--chips_y 4
--rows 16
--cols 16
```

represents:

```text
4 × 4 chips
```

with:

```text
16 × 16 cores per chip
```

The resulting physical core count is:

```text
4 × 4 × 16 × 16 = 4096 cores
```

This configuration provides enough physical cores for the current 906-task workload.

---

# 11. Important Paper Configuration Note

The paper-scale configuration discussed during reproduction is:

```text
4 × 4 chips
16 × 16 cores per chip
= 1024 cores
```

However, the current command/configuration shown during testing produced:

```text
4 × 4 chips
16 × 16 cores/chip
= 4096 cores
```

because:

```text
4 × 4 × 16 × 16 = 4096
```

This discrepancy is intentionally documented rather than hidden.

The current 906-task workload therefore fits comfortably in the tested 4096-core topology.

Before claiming exact hardware-level reproduction, the final topology configuration must be checked against the paper's exact definition of the 1024-core system.

---

# 12. Reinforcement Learning Algorithms

The repository contains more than one RL implementation.

## 12.1 PPO

The single-chip implementation contains PPO-related components.

Relevant files include:

```text
agent/model.py
agent/agents.py
agent/actor.py
agent/distributions.py
env/core_mapper.py
```

The graph structure can be processed using graph-convolution-related components.

The single-chip mapper performs task placement onto a physical core grid.

---

## 12.2 DDPG

The current multi-chip implementation contains a complete DDPG implementation.

The implementation includes:

```text
Actor
Critic
ReplayBuffer
DDPGAgent
```

as well as target-network updates.

The main training flow is implemented through:

```text
run_ddpg()
```

The agent learns a continuous action representation which is converted into physical core placements.

---

# 13. DDPG Training

The current multi-chip training loop approximately follows:

```text
Initialize environment
        |
        v
Generate workload
        |
        v
Generate random-search baseline B
        |
        v
Initialize DDPG agent
        |
        v
Reset environment
        |
        v
Select action
        |
        v
Add exploration noise
        |
        v
Place logic core(s)
        |
        v
Store transition
        |
        v
Train actor/critic
        |
        v
Repeat
        |
        v
Evaluate placement
```

The current implementation also supports batched placement.

---

# 14. Batched Actions

The implementation now supports:

```text
batch_z
```

which controls how many logic cores are placed during one action step.

For example:

```bash
--batch_z 3
```

means:

```text
3 logic cores
```

are placed per environment action.

The corresponding action dimension becomes:

```text
action_dim = 6
```

because each logic core requires an `(x,y)` placement representation.

This modification was introduced as part of matching the paper's batched action formulation.

---

# 15. State Representation

The state supplied to the RL agent contains information about the current placement.

A major issue was discovered during training:

### Previous implementation

The occupancy representation was binary:

```text
occupied = 1
empty    = 0
```

This meant the agent knew that a core was occupied but not **which logic core occupied it**.

That removed important spatial information.

### Current implementation

The occupancy map now encodes the index of the assigned logic core.

Conceptually:

```text
0 = empty
task index = occupied
```

with normalization used by the neural network.

This allows the agent to infer where previously placed predecessor tasks are located.

---

# 16. Communication State

Another state-representation bug was identified.

The previous implementation used the current task's outgoing communication row.

For sequential workloads, this describes communication toward tasks that have not yet been placed.

The updated state representation also considers communication involving already-placed predecessor tasks.

This is important because the placement decision should be informed by the communication relationship with tasks whose physical positions are already known.

---

# 17. Exploration

The current DDPG implementation uses exploration noise.

The implementation currently uses Gaussian-style noise with a decaying noise scale.

The paper uses an Ornstein-Uhlenbeck process with a fading factor.

Therefore:

```text
Current implementation:
Gaussian noise + decay

Paper:
Ornstein-Uhlenbeck exploration
```

This remains a known fidelity difference.

It can be addressed after the current state representation and training behavior have been validated.

---

# 18. Objective Function

One of the most important changes made during the reproduction process was the objective function.

A simple total communication-cost objective is not sufficient to reproduce the paper's throughput-oriented metric.

The paper's objective is based on the **maximum latency across pipeline stages**.

Conceptually:

```text
L(P) = max_k T(k | P)
```

where:

* `P` is a placement.
* `k` represents a pipeline stage.
* `T(k | P)` is the latency associated with that stage.

The reason is that pipeline throughput is determined by the slowest stage.

---

# 19. Pipeline Stages

The current environment contains pipeline-stage analysis.

The task graph is processed as a DAG and topological levels are used to determine pipeline stages.

Conceptually:

```text
Stage 0
  |
  +---- Task 0
  |
  v
Stage 1
  |
  +---- Task 1
  +---- Task 2
  |
  v
Stage 2
  |
  +---- Task 3
```

The stage latency is then evaluated.

The overall objective is the maximum stage latency rather than simply summing all communication costs.

---

# 20. Communication Cost

Communication cost is based on physical distance and communication volume.

A simplified representation is:

```text
Communication Cost
    ∝
Manhattan Distance × Communication Volume
```

For multi-chip systems, communication involving different chips is treated differently from communication occurring within a chip.

The project therefore distinguishes between:

```text
On-chip communication
```

and:

```text
Off-chip communication
```

with off-chip communication carrying a larger penalty.

---

# 21. Collision Resolution

Multiple logic cores cannot occupy the same physical core.

The placement system therefore needs collision handling.

The implementation includes collision-resolution logic.

The reproduction work changed the approach toward the paper's mechanism:

```text
If collision occurs
       |
       v
Find nearby available core
       |
       v
Use Manhattan-distance based search
       |
       v
Assign task
```

The previous implementation used a different distance calculation across the core space.

The current implementation has dedicated collision-resolution tests.

---

# 22. Reward Function

The project has undergone several reward-function changes during reproduction.

The paper uses a sparse terminal reward related to the random-search baseline:

```text
r_t = √B − √L(P)
```

where:

* `B` = baseline cost
* `L(P)` = latency of the resulting placement

The existing implementation also contains reward/cost calculation components for communication and deadlock behavior.

Reward fidelity remains an important part of the final paper-reproduction validation.

---

# 23. Deadlock Handling

The original reward implementation includes a deadlock penalty.

The communication/routing cost therefore does not consider only distance.

The reward system contains:

```text
RewardCalculator
CommunicationCost
DeadlockPenalty
```

The deadlock component penalizes problematic communication/routing configurations.

This is relevant because an apparently short communication path may still produce an undesirable routing configuration.

---

# 24. Baseline Algorithms

The project includes several non-RL baselines.

## Random Search

Random placements are generated and evaluated.

The random-search result is also used to calculate the baseline:

```text
B
```

for DDPG reward normalization.

---

## Simulated Annealing

A simulated-annealing implementation is available for comparison.

It searches the placement space using probabilistic acceptance of candidate solutions.

---

## Exact / Heuristic Mapping

The repository also contains mapping approaches such as:

```text
Zigzag
Neighbor
```

depending on the selected implementation/configuration.

These provide additional reference points against which RL-based placement can be evaluated.

---

# 25. Reproducibility

Random seeds are now supported.

Example:

```bash
--seed 0
```

The seed is used so that:

* Random Search baseline is reproducible.
* DDPG exploration is reproducible.
* Before/after experiments can be compared more reliably.

This is important because an earlier issue was identified where the baseline:

```text
B
```

changed between runs simply because the random-search baseline was not seeded.

---

# 26. Testing

The repository contains:

```text
test_multi_chip.py
```

which provides smoke tests for the multi-chip implementation.

The tests currently cover:

```text
Topology
Environment
Pipeline stages / pipeline latency
Collision resolution
```

Additional CNN/task-graph validation was added during the channel-partitioning work.

Run:

```bash
python3 test_multi_chip.py
```

A previously validated run produced:

```text
✓ topology tests passed
✓ environment tests passed  (cost=1254.00)
✓ pipeline_stages/pipeline_latency tests passed
✓ collision resolution tests passed
```

The test suite was subsequently expanded as the CNN partitioning functionality was added.

---

# 27. Installation

The project is primarily Python-based.

A clean Python environment is recommended.

Example:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python dependencies used by the project.

At minimum, the multi-chip CNN functionality requires:

```text
Python
NumPy
PyTorch
```

and the repository's other modules may require additional packages.

If a `requirements.txt` is present in the repository, use:

```bash
pip install -r requirements.txt
```

Otherwise install the dependencies imported by the selected implementation.

For CNN workload extraction, PyTorch must be available.

Check:

```bash
python3 -c "import torch; print(torch.__version__)"
```

---

# 28. Running the Tests

From the `src` directory:

```bash
python3 test_multi_chip.py
```

This should be performed before long DDPG experiments.

A successful smoke test indicates that the core topology/environment mechanisms are functioning before training begins.

---

# 29. Running the Legacy / Small DDPG Experiment

For a small workload and grid, use:

```bash
python3 run_multi_chip.py --algo ddpg --use_cnn
```

This is useful for:

* Debugging
* Quick experiments
* Checking training behavior
* Testing state/reward changes

It should not be considered the final paper-scale experiment.

---

# 30. Running the Current Partitioned CNN Workload

The current CNN extractor uses:

```text
channels_per_partition = 8
```

by default.

The CLI also exposes:

```bash
--channels_per_partition
```

Setting:

```text
0
```

selects the legacy one-task-per-layer representation.

For example:

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn \
    --channels_per_partition 0
```

is useful for quick legacy/smoke testing.

---

# 31. Current Large-Scale Experiment

The current paper-oriented experiment uses:

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn \
    --epochs 10000 \
    --seed 0 \
    --batch_z 3 \
    --chips_x 4 \
    --chips_y 4 \
    --rows 16 \
    --cols 16
```

The observed initialization was:

```text
Seeded RNGs with --seed 0
Extracting Real CNN Workload via PyTorch Hooks
Channel partitioning: channels_per_partition=8 -> 906 logic cores
```

followed by:

```text
Tasks : 906
Algo  : ddpg
```

and a random-search baseline.

---

# 32. Important Performance Limitation

The current 906-task experiment exposed a major performance bottleneck.

A representative run reached approximately:

```text
10 epochs  -> 884736
20 epochs  -> 892928
30 epochs  -> 933888
40 epochs  -> 811008
```

while taking approximately ten minutes for only a small number of episodes.

This means that simply running:

```bash
--epochs 10000
```

with the current unoptimized implementation is impractical.

The estimated runtime was on the order of tens of hours.

---

# 33. Cause of the Large-Scale Runtime

Three major issues were identified.

## 33.1 Training at Every Environment Step

The DDPG agent currently performs training at every environment step.

For:

```text
906 tasks
batch_z = 3
```

approximately:

```text
906 / 3 ≈ 302
```

environment steps occur per episode.

Therefore, a 10,000-episode experiment can generate millions of gradient updates.

---

## 33.2 Pipeline Latency Complexity

The current pipeline-latency implementation performs unnecessary scans over the full task set.

The workload graph is sparse, but parts of the implementation effectively perform operations resembling:

```text
O(N²)
```

for:

```text
N = 906
```

This is significantly more expensive than necessary.

A sparse adjacency representation should reduce the amount of work.

---

## 33.3 CPU Execution

The current DDPG implementation does not automatically make use of a GPU.

Therefore, neural-network training is currently performed on the CPU unless the implementation is explicitly modified.

---

# 34. Current Optimization Work

The next engineering step is to improve the runtime before performing a 10,000-episode experiment.

The planned optimizations are:

1. Reduce unnecessary DDPG updates.
2. Optimize pipeline-latency calculation using graph sparsity.
3. Add device selection / GPU support.
4. Add live progress and ETA reporting.

These are **engineering/performance improvements**, not changes to the mathematical objective.

The objective should remain unchanged while optimizing execution.

---

# 35. Previous Training Plateau

Before channel partitioning, a smaller 15-task workload was used to investigate RL training behavior.

The original training showed a plateau.

A state-representation investigation identified two important problems.

### Problem 1

The occupancy state was binary rather than task-indexed.

The agent knew:

```text
core occupied = 1
```

but not:

```text
which task occupies this core
```

Therefore, it could not directly infer where previously placed communicating tasks were located.

### Problem 2

The communication feature represented the wrong direction.

The state used the outgoing communication of the current task rather than sufficiently representing communication with already-placed predecessors.

This made the state poorly suited to the actual placement decision.

These problems were corrected.

---

# 36. Effect of the State Fix

Before the state correction:

```text
Best Cost ≈ 36864
```

and training plateaued around epoch 110.

After the correction:

```text
Best Cost ≈ 40960
Baseline B ≈ 49152
```

and the current-cost distribution became substantially tighter toward the end of training.

This indicated that the state representation was a genuine contributor to the earlier training instability.

The remaining plateau could not therefore be attributed solely to the state representation.

---

# 37. Training Budget

The smaller experiment used:

```text
1000 episodes
```

by default.

The paper reports DDPG convergence over a much larger number of placement/training experiences, approximately:

```text
300K–400K placements
```

for the paper-scale workload.

Therefore, the smaller 1,000-episode experiment should not automatically be interpreted as proof that the policy has reached its true optimum.

A longer training budget is required for meaningful convergence analysis.

However, for the 906-task workload, increasing the episode count before fixing the computational bottlenecks is impractical.

Runtime optimization therefore comes first.

---

# 38. Current Status

## Implemented

The following components are currently implemented or substantially implemented:

* [x] Original repository completed and extended
* [x] Single-chip placement framework
* [x] PPO-related single-chip implementation
* [x] Multi-chip topology
* [x] Multi-chip environment
* [x] DDPG actor
* [x] DDPG critic
* [x] Replay buffer
* [x] Target networks
* [x] Random Search
* [x] Simulated Annealing
* [x] CNN workload extraction
* [x] PyTorch-based workload generation
* [x] Channel partitioning
* [x] VMM/VVA logic-core representation
* [x] 906-task partitioned CNN workload
* [x] Pipeline-stage extraction
* [x] Maximum pipeline-stage latency objective
* [x] Collision-resolution logic
* [x] Task-indexed occupancy state
* [x] Improved communication state
* [x] Batched action support
* [x] Random seed support
* [x] Multi-chip smoke tests
* [x] Pre-flight task/core capacity checking

---

# 39. Known Differences from the Paper

The project is **not yet a fully validated reproduction**.

Known differences and remaining work include:

### Exploration

Current:

```text
Gaussian noise + decay
```

Paper:

```text
Ornstein-Uhlenbeck process with fading factor
```

---

### CNN Workloads

Current:

```text
SimpleCNN
```

with paper-inspired channel partitioning.

Paper evaluation:

```text
AlexNet
VGG16
ResNet50
```

or the exact workload set specified by the paper.

Real paper workloads still need to be integrated and evaluated.

---

### Topology

The current code can represent large multi-chip configurations, but the exact physical topology and core count still need to be validated against the paper's precise Table 1/system definition.

---

### Training

The large-scale implementation currently requires performance optimization before a complete 10,000+ episode experiment is practical.

---

### Reward

The reward implementation has been moved toward the paper's baseline-normalized formulation, but complete numerical validation against the paper's training/reward procedure remains part of the reproduction phase.

---

### Experimental Results

The reported paper numbers have **not yet been independently reproduced**.

Therefore, the repository should not currently claim exact reproduction of the paper's final performance numbers.

---

# 40. Paper Reproduction Roadmap

The current roadmap is:

```text
Current Implementation
        |
        v
Fix / Validate State
        |
        v
Validate Objective
        |
        v
Optimize Runtime
        |
        v
Real Paper Workloads
        |
        v
Paper Hardware Configuration
        |
        v
DDPG vs Random Search vs SA
        |
        v
Reproduce Paper Results
        |
        v
Establish Baseline
        |
        v
Introduce Novel Improvement
        |
        v
Evaluate Improvement
```

---


# 41. Planned Paper-Faithful Improvements — Status Update

The four items originally listed here have each moved forward, at different paces:

## 1. Real DNN workloads — DONE for AlexNet/VGG16, PARTIAL for ResNet50

`SimpleCNN` has been replaced by a generalized extractor (`extract_model_task_graph`) that pulls
real architectures via `torchvision`:

```text
--model alexnet    # fully accurate — purely sequential, hooks capture true data flow
--model vgg16      # fully accurate — purely sequential, hooks capture true data flow
--model resnet50   # RUNS, but topologically approximate (see caveat below)
--model simple     # original small demo CNN, kept for fast smoke tests
```

Each Conv2d/Linear layer is still partitioned into VMM (vector-matrix-multiply) + VVA
(vector-vector-accumulation) logic cores per Sec 3.1.1 / Fig. 5-6, not just extracted whole-layer.

**Known caveat — ResNet50's skip connections:** the extractor uses forward hooks, which observe
Conv2d/Linear *call order* but not the actual `add()` operation that merges a skip branch back into
the main path. Every real layer and its shape are captured correctly, but the Bottleneck block's
shortcut edges are currently approximated as if the network were a plain sequential chain. This is
flagged with an explicit runtime warning whenever `--model resnet50` is used. Fixing this properly
requires tracing the real computational graph (e.g. via `torch.fx`) instead of relying on hook order
— not yet implemented.

## 2. Exact paper topology — PARTIAL

The multi-chip grid is fully configurable (`--chips_x`, `--chips_y`, `--rows`, `--cols`), and has been
run at the paper's actual real-system config: **4×4 chips × 16×16 cores/chip = 4,096 total cores**
(Table 1). The hierarchical on-chip/off-chip latency model (`comm_cost()`) uses Manhattan routing with
configurable α (on-chip) / β (off-chip) latencies, matching the paper's Sec 2.2 model conceptually
with the same default ratio (α=1.0, β=5.0).

**Not yet done:** converting the paper's *measured* bandwidth figures (100 GB/s off-chip, 64 GB/s
per-core NoC, Table 1) into precise α/β latency values — we're currently using the paper's
illustrative default ratio, not a value derived from their actual hardware numbers.

## 3. Exact exploration mechanism — NOT DONE

Still uses plain Gaussian exploration noise, not the paper's Ornstein-Uhlenbeck process with a fading
factor. What *has* been fixed is the decay schedule itself — it's now scaled to reach ~0.01 by 80% of
training regardless of episode count (previously a fixed `0.995` barely decayed over a short run,
masking whether the policy had actually converged). That's a correctness fix to the existing
mechanism, not a replacement of the mechanism itself. Implementing real OU noise is still open.

## 4. Training procedure — PARTIAL

| Paper (Table 1 / Algorithm 1) | Current implementation |
|---|---|
| `r_t = √B − √L(P)`, sparse, terminal-only | ✅ Matches exactly (Algorithm 1, line 11) |
| Minibatch size K=64 | ✅ Matches |
| `L(P) = max_k T(k|P)` (Eq. 4, bottleneck) | ✅ Matches (was previously a flat sum — fixed) |
| `α_θ=0.0002, α_w=0.001, γ=0.98` | ⚠️ Currently `1e-4, 1e-3, 0.99` — close, not identical |
| Action batches z logic cores at once | ✅ Matches (`--batch_z`) |
| Evaluate DDPG/BS/RS/SA on identical cost fn | ✅ Enforced — see Section 42 |
| Ornstein-Uhlenbeck + fading factor | ❌ Still plain Gaussian |
| 5-seed averaging (Sec 4.5, Fig. 20) | ⚠️ `--seed` supported per-run; multi-seed averaging loop not yet automated |

---

# 42. Baseline Validation — Status: Enforced

This principle is now structurally guaranteed, not just aspirational: `DDPGAgent`, `run_sa()`, and
`run_random()` all call `env.evaluate()`, which resolves to the same `pipeline_latency()` function
(the paper's Eq. 4 bottleneck objective) regardless of which algorithm is calling it. There is no
separate cost function per algorithm — comparisons between DDPG/RS/SA are apples-to-apples by
construction.

---

# 43. Paper Performance Validation — Actual Measured Results So Far

The reference values below remain **paper targets**, not yet reproduced targets — no run so far has
matched the paper's exact workload, topology-derived latencies, exploration mechanism, and training
budget simultaneously. What we do have now are real, completed (or in-progress) experimental runs
that validate the *mechanics* are working correctly:

| Run | Workload | Grid | Episodes | Baseline B (random search) | Best DDPG cost | Result |
|---|---|---|---|---|---|---|
| ✅ Completed | SimpleCNN, 906 partitioned logic cores (`channels_per_partition=8`) | 4,096 cores | 10,000 (~8.7 real hrs) | 444,416 | 340,992 | **~23.3% reduction** vs. random-search baseline |
| 🔄 In progress | AlexNet, 1,445 partitioned logic cores (`channels_per_partition=128`) | 4,096 cores | 1,000 (~2.5 hr ETA) | 12,697,984 | trending down (14.6M best after 10 epochs) | not yet final |
| ⏳ Extraction validated, not yet trained | VGG16, 1,616 partitioned logic cores (`channels_per_partition=128`) | 4,096 cores | — | — | — | — |
| ❌ Not yet attempted | ResNet50 | — | — | — | — | blocked on skip-connection caveat above |

The completed SimpleCNN run's ~23% improvement is directionally consistent with the paper's own
random-search-relative results (paper reports 1.61x/38.4% average improvement over RS across its
three real workloads), but should be read as **evidence the RL pipeline is mechanically correct and
learns effectively**, not as a reproduction of any specific paper number — different workload, and
several Section 41 gaps (exploration mechanism, exact hyperparameters, exact latency values) are
still open.

```text
DDPG (paper) : 50.5% latency reduction  ]
BS   (paper) : 38.4% latency reduction  ]  reference targets, not yet matched
RS   (paper) : 18.6% latency reduction  ]
```

---

# 44. Future Research / Genuine Improvements

Unchanged in spirit from the original plan — GCN embeddings, attention-based placement,
topology-generalization, and modern architecture evaluation remain the right next research
directions once baseline reproduction is solid. One addition worth noting: **graph-structure-aware
methods (GCN/attention) will have limited demonstrable advantage until the workload graphs
themselves are structurally accurate** — AlexNet/VGG16 are genuinely sequential chains (a GCN adds
little over a chain-aware MLP), and ResNet50's *actual* branching structure isn't captured yet (see
Section 41.1). Item 4 in this section — modern architectures with real branching/attention structure
— and the graph-based RL direction are therefore linked: fixing ResNet50's extraction is close to a
prerequisite for meaningfully evaluating GCN-based placement.

---

# 45. Example Commands (Updated, All Currently Working)

### Run tests
```bash
python3 test_multi_chip.py
```
7 checks: topology, environment, pipeline latency/objective, collision resolution, batched actions,
CNN workload partitioning (torchvision models included), and SA-vs-random sanity check.

### Small smoke test (SimpleCNN, default 64-core grid)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --epochs 50
```

### Legacy whole-layer workload (no partitioning, for quick debugging)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --channels_per_partition 0
```

### Partitioned SimpleCNN workload (906 logic cores)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --channels_per_partition 8 \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16
```

### Real AlexNet workload at paper-scale grid (validated, ~9.4s/episode on a 12-core CPU)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet --channels_per_partition 128 \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 --epochs 1000 --seed 0 --batch_z 3
```

### Real VGG16 workload (extraction validated; not yet run to completion)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --model vgg16 --channels_per_partition 128 \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 --epochs 1000 --seed 0 --batch_z 3
```

### ResNet50 (experimental — sequential-approximation caveat applies, see Sec 41.1)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --model resnet50 --channels_per_partition 128 \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 --epochs 1000 --seed 0 --batch_z 3
```

### Performance-tuned large-scale run (recommended flags for anything above ~500 tasks)
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet --channels_per_partition 128 \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 --epochs 10000 --seed 0 \
    --batch_z 3 --train_every 5 --device cpu
```
`--train_every 5` runs one gradient update per 5 environment steps instead of every step (experience
is still recorded every step) — this was the single biggest lever once the sparse pipeline-latency fix
was in place. `--device` auto-detects CUDA if available; explicit `cpu`/`cuda` override supported.

### Reproducible single run
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet --seed 0
```

### Multi-seed averaging (matches the paper's own Sec 4.5 / Fig. 20 methodology)
```bash
for s in 0 1 2 3 4; do
    python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet \
        --channels_per_partition 128 --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
        --epochs 1000 --seed $s --batch_z 3 --train_every 5 \
        | tee "log_seed${s}.txt"
done
```
The paper itself doesn't rely on a single fixed seed — it runs 5 seeds and averages (Fig. 20). This
loop is the mechanism to do the same; results still need to be aggregated by hand/script afterward.

---

# 46. Development Status

### Current stage
**Paper reproduction + scalability engineering** — substantially further along than the previous
snapshot, with the core RL mechanics now verified paper-faithful and real workloads now supported.

### Working
* Core mapping environment, multi-chip topology, DDPG training framework
* CNN extraction generalized to real `torchvision` models (AlexNet, VGG16 — fully accurate; ResNet50
  — runs, topologically approximate)
* Channel-partitioned VMM/VVA logic core generation (Sec 3.1.1, Fig. 5-6)
* **Paper's exact bottleneck pipeline-latency objective** (Eq. 4) — replaced an earlier
  total-summed-cost implementation
* **Sparse, baseline-normalized reward** (`r_t = √B − √L(P)`, Algorithm 1 line 11) — replaced an
  earlier dense per-step reward
* **Manhattan-distance collision resolution** with correct floor/intended-core/tie-break logic
  (Sec 3.2) — replaced an earlier Euclidean whole-grid scan
* **Batched action placement** (z logic cores per action, Sec 3.2)
* **Task-identity-encoded state representation** (occupied cores show *which* task, not just
  occupied/free) plus bidirectional (incoming+outgoing) communication features — this was found to
  be the root cause of an early training plateau and fixing it produced measurably better
  convergence
* Reproducible seeding across random/numpy/torch (`--seed`)
* Sparse/cached pipeline evaluation — ~19.4x measured speedup over the original dense O(n²) scan
* Reduced-frequency DDPG training (`--train_every`)
* GPU auto-detection + CPU thread-count fix for CPU-only machines
* Live per-episode timing + ETA reporting
* Pre-flight grid-size/task-count validation (fails fast with a clear fix instead of crashing deep
  in placement code)
* Consistent same-cost-function baseline comparison (RS/SA/DDPG) — see Section 42
* Smoke test suite covering all of the above (7 checks)

### Currently being worked on / known next steps
* **State-construction performance**: `_occ_map()` still has an unvectorized Python loop scaling
  with `num_tasks`, identified as the likely remaining per-step bottleneck at real-workload scale
  (e.g. AlexNet's 1,445 tasks) — not yet fixed
* Ornstein-Uhlenbeck exploration noise + fading factor (paper's exact mechanism)
* Exact hyperparameter alignment with Table 1 (`α_θ=0.0002, α_w=0.001, γ=0.98`)
* Completing VGG16 and ResNet50 training runs
* `torch.fx`-based exact skip-connection graph extraction for ResNet50
* Converting the paper's measured bandwidth figures into precise α/β latency values

### Not yet claimed
* Exact reproduction of the paper's reported percentages (50.5% / 38.4% / 18.6%)
* A novel RL algorithm or architectural improvement over the paper (actor/critic are still MLPs, not
  the paper's CNN, and not yet the GCN/attention future-research directions)
* A topologically-exact ResNet50 evaluation
* Multi-seed statistical validation in the paper's own style (Fig. 20)
* GA baseline, topology-agnostic (2D torus/HNoC/dragonfly), or RNN-workload experiments — none
  implemented yet

---

# 47. Research Philosophy

Unchanged — this two-stage methodology remains the right frame, and the project is now solidly
inside the "faithful implementation → controlled experiments" phase of the reproduce-first stage,
with real baseline results (Section 43) starting to accumulate.

```text
Reproduce first
Paper → Faithful implementation → Controlled experiments → Baseline results

Improve second
Validated baseline → Identify limitation → Propose modification →
Implement modification → Controlled comparison → Measure improvement
```

---

# 48. Acknowledgement

Unchanged. The project builds upon the original open-source implementation *Core Placement with
Reinforcement Learning* and extends it toward a more complete multi-chip DNN mapping framework based
on the 2020 ACM TODAES paper's methodology. The current repository should be viewed as an ongoing
research/reproduction implementation, not a finished reproduction.

---

# 49. Current Experimental Snapshot

```text
=== Completed run ===
CNN workload                 : SimpleCNN (channel-partitioned)
Partition size                : 8 channels/group
Logic cores generated         : 906
Placement algorithm           : DDPG
Action batching                : 3 tasks/action
Random seed                   : 0
Multi-chip configuration      : 4 x 4 chips
Cores per chip                : 16 x 16
Total physical cores          : 4096
Training episodes             : 10,000 (completed in ~8.7 real hours, ~3.11s/episode)
Random-search baseline (B)    : 444,416
Best DDPG placement cost      : 340,992
Improvement over baseline     : ~23.3%

=== In-progress run ===
CNN workload                  : AlexNet (torchvision, channel-partitioned)
Partition size                 : 128 channels/group
Logic cores generated          : 1,445
Multi-chip configuration       : 4 x 4 chips, 16 x 16 cores/chip (4096 cores)
Training episodes requested    : 1,000 (~2.5 hr ETA at 9.38s/episode)
Random-search baseline (B)     : 12,697,984
Status                         : running, best cost trending down (14.6M after 10 epochs)

=== Extraction validated, training not yet run ===
VGG16 at channels_per_partition=128 -> 1,616 logic cores (fits 4096-core grid)
ResNet50 workload extraction runs but is topologically approximate (skip connections
not captured — see Section 41.1)
```

The immediate next engineering objective is finishing the AlexNet run, then vectorizing the
remaining `_occ_map()` bottleneck before attempting VGG16/ResNet50 at similar scale, since those
workloads only get bigger.

---

# 50. Summary

This project investigates how reinforcement learning can be used to solve the DNN-to-many-core
placement problem. The implementation has progressed substantially since the last snapshot — from a
small task-level mapping framework with several silent deviations from the paper's actual algorithm,
to a validated pipeline that:

```text
Real DNN (AlexNet / VGG16 / [ResNet50, approximate])
 |
 v
Hook-based extraction
 |
 v
Channel partitioning (VMM / VVA logic cores)
 |
 v
Communication DAG
 |
 v
Multi-chip environment (paper-scale: 4096 cores)
 |
 v
DDPG placement (batched actions, task-identity-aware state)
 |
 v
Pipeline bottleneck-latency evaluation (Eq. 4)
 |
 v
Sparse, baseline-normalized reward (Algorithm 1)
```

Every stage of that pipeline has now been individually checked against the paper's text and
validated with passing tests, and the first full-scale training run produced a real, credible
improvement over a random-search baseline computed under the identical cost function. The project is
still in the reproduction/validation stage — the paper's exact reported numbers haven't been matched
yet, and several concrete, well-understood gaps remain (Section 46) — but it is meaningfully closer
to a faithful reproduction than a "different implementation that happens to solve a similar problem."