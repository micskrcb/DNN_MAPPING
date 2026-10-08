# Paper-reproduction execution plan

The primary objective is an honest reproduction of Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning*. Experimental improvements remain separate from paper-mode runs.

The paper does not publish its simulator or every implementation parameter. Exact numerical reproduction therefore cannot be promised. Reports and the experiment manifest record all reconstructed choices.

## Gate 1: optimization problem — partially complete

- [x] Match Figure 6 aggregate logic-core counts exactly for AlexNet, VGG16, and ResNet50.
- [x] Reconstruct deterministic per-layer input/output grids, balance estimated VMM/VVA cycles, and enforce the 64 KB weight buffer.
- [x] Optimize CONV and FC independently in disjoint physical regions.
- [x] Use Table 1 compute rates, precisions, and link bandwidths.
- [x] Add block scaling, deterministic XY routes, routed per-source byte-hop time, and shared directed-link contention.
- [x] Validate routes/contention with hand-calculated tests and conservation tests.
- [x] Add a placement-sensitivity preflight and compute/communication bottleneck report.
- [ ] Model the 64 KB input/activation buffer, stalls, and compute/communication overlap.
- [ ] Establish a better validated GRS/multicast reconstruction if evidence becomes available.
- [ ] Validate or replace the assumed workload block counts and VVA rate.

Current paper targets:

| Workload | CONV cores | FC cores | Total |
| --- | ---: | ---: | ---: |
| AlexNet | 183 | 932 | 1,115 |
| VGG16 | 1,024 | 1,924 | 2,948 |
| ResNet50 | 512 | 37 | 549 |

## Gate 2: methods — paper mode complete; guided DDPG rejected, masked PPO implemented

- [x] BS in chip-major then core-major order.
- [x] Figure 9 `paper_cnn` actor and critic.
- [x] Grid-only paper state, `2z` continuous coordinates, floor conversion, and nearest-free Manhattan repair.
- [x] Actor/critic learning rates 0.0002/0.001, gamma 0.98, and minibatch 64.
- [x] Sparse terminal `sqrt(B) - sqrt(L(P))` reward in 400-MHz cycle units.
- [x] Explicit 30 complete placements per declared epoch.
- [x] Independent DDPG, reward-baseline, RS, and SA accounting.
- [x] Default paper budgets: 300,000 DDPG placements and 1,000,000 RS/SA placements.
- [x] Record unpublished `z`, OU, replay, padding, LRN, and target-update choices as assumptions.
- [x] Add an experimental guided DDPG mode with ASA demonstrations, permanent demonstration replay, collision-free legal action projection, normalized complete-episode return targets, and deterministic candidate retention.
- [x] Add a matched `--guided_disable_learning` control and a uniform legal warm-up so policy learning can be separated from initialization and repair effects.
- [x] Complete the five-seed guided-DDPG ablation: trained lost all five late deterministic comparisons, so this path is closed as a negative result.
- [x] Add experimental masked categorical PPO with exact invalid-action masking and a matched frozen-policy control.
- [ ] Run the paper-scale budgets on an available GPU and archive all manifests/logs.

## Gate 3: comparisons — pending GPU experiments

1. Run `scripts/run_kaggle_masked_ppo_ablation.sh short`. It compares 300 trained placements with an exactly matched frozen-policy control at seed 0 and creates a downloadable ZIP.
2. Check zero repairs, matching initial-policy cost, finite losses, and late deterministic trained-versus-control cost. Do not scale a negative short result.
3. If the short gate is positive, run the `extensive` preset for five paired seeds and 3,000 placements per condition. Require consistent paired improvement rather than one lucky best candidate.
4. Inspect noisy/deterministic curves, critic/actor losses, unique intended positions, collision repairs, hop-distance reductions, and link loads. Require the trained policy to improve over the matched control consistently.
5. Repeat the matched experiment on AlexNet-FC or VGG16-CONV. AlexNet-CONV has zero off-chip traffic in the minimum whole-chip mask and only about 13.7% communication headroom, so it is a useful sanity test but a weak final benchmark.
6. Keep the completed guided-DDPG archive as a negative ablation; do not spend the paper-scale budget on it.
7. Only after the learning gate passes, run AlexNet across at least five seeds plus BS, RS, SA, and ASA under matched complete-placement budgets.
8. Proceed to VGG16 and ResNet50 after the evaluator and learning behavior are credible.
9. Add a validated large-batch fill/steady-state/drain model before presenting paper-style throughput.
10. Record Git commit, clean/dirty state, complete configuration, Torch/CUDA versions, visible GPU, memory, evaluation counts, checkpoint paths, and wall time.

## Gate 4: improvements after reproduction

Guided DDPG is now available as an explicitly experimental mode. It combines ASA demonstrations with permanent replay, collision-free legal projection, normalized complete-episode return targets, uniform legal replay warm-up, and retained deterministic candidates. Its matched no-learning control is required for every learning claim.

Masked categorical PPO, graph encoders, alternative partitioning, and parallel environments remain future improvement experiments. Freeze and identify the paper-mode configuration, then compare every improvement under matched complete-placement budgets. The evidence and design rationale are recorded in [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).
