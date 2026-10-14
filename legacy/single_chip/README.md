# Legacy Single-Chip Placement Framework

This directory preserves the historical upstream single-chip core placement codebase originally forked from [`WOOSHIK-M/Core_Placement_with_Reinforcement_Learning`](https://github.com/WOOSHIK-M/Core_Placement_with_Reinforcement_Learning).

---

## Overview

The legacy system implements core placement for a single 2D mesh chip using:
- **Graph Convolutional Networks (GCN)**: Node embedding over static netlists (`agent/model.py`, `agent/actor.py`).
- **Proximal Policy Optimization (PPO)**: Discrete coordinate placement with Gaussian/Beta policy heads (`agent/agents.py`, `agent/distributions.py`).
- **Baseline Heuristics**: Exact mapping (Zigzag, Neighbor in `runner/exact_mapping.py`), Random Search (`runner/random_search.py`), and Simulated Annealing (`runner/simulated_annealing.py`).
- **Evaluation Metric**: Manhattan communication distance with deadlock collision penalties (`env/reward.py`, `env/core_mapper.py`).
- **Workloads**: Static Lookup Table (LUT) netlists (`data/toy_lut.txt`, `data/toy_lut.py`).

---

## Status in Current Research

As documented in `docs/RECONCILIATION.md` and `README.md`, **this single-chip netlist code is outside the validated multi-chip paper reproduction path**.

The current research focuses on:
- Multi-chip many-core accelerators (e.g. 4x4 chips, 16x16 cores/chip = 4,096 cores).
- Realistic DNN workload graphs (AlexNet, VGG16, ResNet50) partitioned into VMM and VVA logic cores per Section 3.1.1 of Wu et al. (ACM TODAES 2020).
- Kahn topological bottleneck pipeline latency objective ($L(P) = \max_k T(k|P)$).
- Contextual-bandit-guided Fixed Simulated Annealing (`--algo ppo_asa --ppo_asa_learner bandit --ppo_asa_annealing fixed`).

All active multi-chip algorithms, topologies, environments, and experiments reside in `src/`.

---

## Legacy Execution

To run the legacy single-chip mapper for historical comparison:

```bash
cd legacy/single_chip
python3 main.py --config config/example.py
```
