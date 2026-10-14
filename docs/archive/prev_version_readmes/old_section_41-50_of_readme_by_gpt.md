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