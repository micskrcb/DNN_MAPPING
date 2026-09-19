# Paper-reproduction execution plan

The primary objective is an honest reproduction of Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning*. Experimental improvements remain separate from paper-mode runs.

The paper does not publish its simulator or every implementation parameter. Exact numerical reproduction therefore cannot be promised. Reports and the experiment manifest record all reconstructed choices.

## Gate 1: optimization problem — partially complete

- [x] Match Figure 6 aggregate logic-core counts exactly for AlexNet, VGG16, and ResNet50.
- [x] Reconstruct deterministic per-layer input/output grids and enforce the 64 KB weight buffer.
- [x] Optimize CONV and FC independently in disjoint physical regions.
- [x] Use Table 1 compute rates, precisions, and link bandwidths.
- [x] Add block scaling, deterministic XY routes, and shared directed-link contention.
- [x] Validate routes/contention with hand-calculated tests and conservation tests.
- [ ] Model the 64 KB input/activation buffer, stalls, and compute/communication overlap.
- [ ] Establish a better validated GRS/multicast reconstruction if evidence becomes available.
- [ ] Validate or replace the assumed workload block counts and VVA rate.

Current paper targets:

| Workload | CONV cores | FC cores | Total |
| --- | ---: | ---: | ---: |
| AlexNet | 183 | 932 | 1,115 |
| VGG16 | 1,024 | 1,924 | 2,948 |
| ResNet50 | 512 | 37 | 549 |

## Gate 2: methods — implementation complete, long runs pending

- [x] BS in chip-major then core-major order.
- [x] Figure 9 `paper_cnn` actor and critic.
- [x] Grid-only paper state, `2z` continuous coordinates, floor conversion, and nearest-free Manhattan repair.
- [x] Actor/critic learning rates 0.0002/0.001, gamma 0.98, and minibatch 64.
- [x] Sparse terminal `sqrt(B) - sqrt(L(P))` reward.
- [x] Explicit 30 complete placements per declared epoch.
- [x] Independent DDPG, reward-baseline, RS, and SA accounting.
- [x] Default paper budgets: 300,000 DDPG placements and 1,000,000 RS/SA placements.
- [x] Record unpublished `z`, OU, replay, padding, LRN, and target-update choices as assumptions.
- [ ] Run the paper-scale budgets on the H100 and archive all manifests/logs.

## Gate 3: comparisons — pending GPU experiments

1. Run the bounded H100 validator and AlexNet CONV/FC smoke suite.
2. Measure per-placement and optimizer-update time before committing the allocation window.
3. Run AlexNet across at least five DDPG seeds plus BS, RS, and SA.
4. Report CONV/FC latency separately, normalized to BS, with seed mean, sample standard deviation, minimum, and maximum.
5. Inspect noisy/deterministic learning curves, losses, intended-core diversity, collision repairs, hop-distance reductions, and link-load distributions.
6. Proceed to VGG16 and ResNet50 only after the AlexNet evaluator and learning behavior are credible.
7. Add a validated large-batch fill/steady-state/drain model before presenting paper-style throughput.
8. Record Git commit, clean/dirty state, complete configuration, Torch/CUDA versions, visible GPU, memory, evaluation counts, checkpoint paths, and wall time.

## Gate 4: improvements after reproduction

Potential reward shaping, collision penalties, valid-action masks, discrete actions, graph encoders, alternative partitioning, and parallel environments are improvement experiments. Freeze and identify the paper-mode configuration first, then compare improvements under matched complete-placement budgets.
