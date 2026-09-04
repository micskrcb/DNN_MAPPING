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

# 41. Planned Paper-Faithful Improvements

The remaining reproduction work includes:

## 1. Real DNN workloads

Replace:

```text
SimpleCNN
```

with real architectures such as:

```text
AlexNet
VGG16
ResNet50
```

and generate the corresponding partitioned task graphs.

---

## 2. Exact paper topology

Validate and reproduce the exact:

* Number of chips
* Cores per chip
* Inter-chip communication model
* On-chip communication model
* Routing assumptions

specified in the paper.

---

## 3. Exact exploration mechanism

Implement the paper's:

```text
Ornstein-Uhlenbeck process
```

and fading factor.

---

## 4. Training procedure

Match the paper's:

* Training budget
* Batch configuration
* Learning rates
* Exploration schedule
* Reward formulation
* Evaluation methodology

as closely as possible.

---

# 42. Baseline Validation

The final reproduction experiment should compare:

```text
DDPG
Random Search
Simulated Annealing
```

using exactly the same:

* Workload
* Hardware
* Communication model
* Objective
* Evaluation procedure

The important principle is:

> Every algorithm must be evaluated using the same cost function.

Otherwise, improvements in the reported numbers cannot be attributed confidently to the placement algorithm.

---

# 43. Paper Performance Validation

The paper reports substantial latency improvements for its placement methods.

Previously discussed reference values include approximately:

```text
DDPG : 50.5% latency reduction
BS   : 38.4% latency reduction
RS   : 18.6% latency reduction
```

These numbers should be treated as **paper reference targets**, not as results reproduced by this repository yet.

The purpose of the reproduction stage is to determine how closely the current implementation can reproduce these trends under the same experimental assumptions.

---

# 44. Future Research / Genuine Improvements

Once the paper baseline is reproduced, the project can move beyond reproduction.

Potential contributions include:

## Graph-Based RL

Use the complete DNN DAG directly rather than relying primarily on a flattened state.

Possible architecture:

```text
DNN Task Graph
      |
      v
     GCN
      |
      v
Task Embeddings
      |
      v
 Actor / Critic
      |
      v
Core Placement
```

This direction is particularly relevant because DNN workloads are naturally graph-structured.

---

## Attention-Based Placement

A possible extension is to use attention mechanisms to determine which tasks and physical regions are most important for the current placement decision.

For example:

```text
Task Graph
    |
    v
Graph Encoder
    |
    v
Attention
    |
    v
Placement Policy
```

This could improve over a fixed-size MLP representation.

---

## Topology-Generalized Placement

The current project focuses on grid-based multi-chip architectures.

A future extension could investigate:

```text
2D Mesh
2D Torus
HNoC
Dragonfly
```

and other interconnect structures.

The objective would be to learn a placement strategy that is not tightly coupled to one topology.

---

## Modern DNN Architectures

The project can eventually be evaluated on:

```text
CNNs
ResNets
Transformers
Modern hybrid architectures
```

This would test whether the learned placement strategy generalizes beyond the sequential CNN workloads emphasized in the original work.

---

# 45. Example Commands

### Run tests

```bash
python3 test_multi_chip.py
```

### Small DDPG run

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn
```

### Legacy layer-level workload

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn \
    --channels_per_partition 0
```

### Partitioned CNN workload

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn \
    --channels_per_partition 8
```

### Seeded experiment

```bash
python3 run_multi_chip.py \
    --algo ddpg \
    --use_cnn \
    --seed 0
```

### Current large-scale experiment

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

**Note:** The large-scale command should currently be considered an experimental command rather than a recommended long-running command until the runtime optimizations are completed.

---

# 46. Development Status

### Current stage

**Paper reproduction + scalability engineering**

### Working

* Core mapping environment
* Multi-chip topology
* DDPG training framework
* CNN extraction
* Channel partitioning
* VMM/VVA task generation
* Pipeline objective
* Collision resolution
* State representation improvements
* Baseline algorithms
* Smoke tests

### Currently being worked on

* Large-scale runtime optimization
* Efficient pipeline-latency computation
* GPU/device support
* Efficient DDPG update scheduling
* Real DNN workloads
* Exact paper configuration
* Full paper-result reproduction

### Not yet claimed

* Exact reproduction of all paper results
* Novel RL algorithm
* State-of-the-art performance
* Full ResNet50/Transformer evaluation
* Definitive improvement over the 2020 paper

---

# 47. Research Philosophy

The project follows a strict two-stage methodology:

### Reproduce first

```text
Paper
  ↓
Faithful implementation
  ↓
Controlled experiments
  ↓
Baseline results
```

### Improve second

```text
Validated baseline
       ↓
Identify limitation
       ↓
Propose modification
       ↓
Implement modification
       ↓
Controlled comparison
       ↓
Measure improvement
```

This prevents changes that merely make the implementation different from the paper from being incorrectly presented as research contributions.

---

# 48. Acknowledgement

The project builds upon the original open-source implementation:

**Core Placement with Reinforcement Learning**

and extends it toward a more complete multi-chip DNN mapping framework based on the methodology described in the associated 2020 research work.

The current repository should therefore be viewed as an **ongoing research/reproduction implementation**, rather than a finished reproduction of the original paper.

---

# 49. Current Experimental Snapshot

The latest validated large-workload extraction produced:

```text
CNN workload                 : SimpleCNN
Partition size               : 8 channels
Logic cores generated        : 906
Placement algorithm          : DDPG
Action batching              : 3 tasks/action
Random seed                  : 0
Multi-chip configuration     : 4 × 4 chips
Cores per chip               : 16 × 16
Total physical cores         : 4096
```

The large-scale run successfully reached the training stage and produced a random-search baseline:

```text
Baseline B = 444416
```

but training performance was too slow for a practical 10,000-episode run.

Therefore, the **next immediate engineering objective is runtime optimization**, after which the large-scale experiment can be rerun under controlled conditions.

---

# 50. Summary

This project investigates how reinforcement learning can be used to solve the difficult **DNN-to-many-core placement problem**.

The implementation has progressed from a small task-level mapping framework to a much more realistic system containing:

```text
DNN
 ↓
CNN extraction
 ↓
Channel partitioning
 ↓
VMM/VVA logic cores
 ↓
Communication DAG
 ↓
Multi-chip environment
 ↓
DDPG placement
 ↓
Pipeline latency evaluation
```

The current implementation is substantially closer to the 2020 paper than the original starting repository, but the work is still in the **reproduction/validation stage**.

The immediate priority is to make the 906-task experiment computationally practical, then integrate the paper's real workloads and exact configuration, reproduce the reported baselines, and finally introduce and evaluate genuinely novel improvements.
