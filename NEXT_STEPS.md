# Paper-reproduction execution plan

The primary objective is an honest reproduction of Wu et al., *Core Placement
Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement
Learning*. Experimental improvements must remain separate from paper-mode runs.

The paper does not publish its simulator or every implementation parameter.
Consequently, exact numerical reproduction cannot be promised. Every unspecified
choice must be recorded, and results should first be compared using the paper's
normalized metrics and trends.

## Gate 1: reproduce the optimization problem

Do not run long DDPG experiments until this gate passes.

1. **Match the logic-core allocation.** Reproduce the Figure 6 targets and the
   paper's compute-balancing/capacity rules instead of selecting one global
   `channels_per_partition` value.

   | Workload | CONV cores | FC cores | Total |
   | --- | ---: | ---: | ---: |
   | AlexNet | 183 | 932 | 1,115 |
   | VGG16 | 1,024 | 1,924 | 2,948 |
   | ResNet50 | 512 | 37 | 549 |

2. **Separate CONV and FC placement.** Allocate distinct masked physical-core
   regions and optimize them independently, as described in Section 3.1.2.
3. **Implement the paper timing objective.** Model block-by-block CONV streaming,
   layer-by-layer FC execution, pipeline stages/time phases, 8-bit activations
   and weights, 32-bit partial sums, minimal-path XY routing, link traffic,
   buffering, stalls and the maximum stage latency `L(P)`.
4. **Enforce Table 1 resources.** Use the 4x4-chip, 16x16-core system, 128 MACs at
   400 MHz, 64 KB weight buffer, 64 KB input/activation buffer, 64 GB/s/core NoC
   and 100 GB/s/chip off-chip bandwidth. Record the interpretation of GRS because
   its complete implementation is not specified in this paper.
5. **Validate the evaluator.** Add hand-calculated small cases and conservation
   checks before using the objective as an RL reward.

## Gate 2: reproduce the methods

1. Add the sequential-placement baseline (BS), ordered by chip index and then
   core index.
2. Retain the Figure 9 `paper_cnn` agent: CONV-32/64, pooling/LRN, FC-600/300,
   batch normalization, actor FC-`2z`, and critic action merge after FC-600.
3. Use the 2-D placement matrix alone as state, continuous `2z` coordinates,
   floor conversion, nearest-free Manhattan repair, Adam actor learning rate
   0.0002, critic learning rate 0.001, discount 0.98 and minibatch size 64.
4. Keep the paper reward exactly sparse: zero until completion, then
   `sqrt(B) - sqrt(L(P))`. Keep potential shaping out of paper-mode runs.
5. Resolve and document unspecified values: `z`, OU parameters/fading schedule,
   replay capacity, convolution padding, LRN parameters, target networks and
   soft-update coefficient.
6. Correct experiment accounting: the paper predicts 30 complete placements per
   epoch and reports convergence after roughly 300,000-400,000 evaluated
   placements. Record both epochs and complete-placement evaluations explicitly.
7. Run RS with 1,000,000 sampled placements and SA with approximately 1,000,000
   placements, cooldown 0.99 and 1% placement perturbations.

## Gate 3: reproduce reported comparisons

1. Run AlexNet, VGG16 and ResNet50 with batch size one for latency and a large
   batch for throughput.
2. Report CONV and FC results separately as well as the overall workload.
3. Normalize latency and throughput to BS, matching Figure 10; do not compare
   arbitrary proxy values with the paper's latency.
4. Report seed-level DDPG results and at least five-seed mean, sample standard
   deviation, minimum and maximum. The paper does not state uncertainty for its
   main Figure 10 results, so our additional statistics should be identified as
   such.
5. Record Git commit, clean/dirty state, complete configuration, Torch/CUDA
   versions, GPU, checkpoints, total placement evaluations and wall-clock time.
6. Verify hop-distance reductions and link-traffic distributions in addition to
   final cost. These intermediate measurements help distinguish a faithful trend
   from an accidentally similar final number.

## Gate 4: improvements after reproduction

Only after freezing a paper-mode configuration should we evaluate potential
reward shaping, collision penalties, valid-action masking, discrete actions,
graph encoders or parallel environments. Label all such results as improvements,
and compare them against the frozen paper-mode implementation under matched
placement-evaluation budgets.
