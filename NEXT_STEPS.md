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
- [x] Complete the five-seed, 3,000-placement masked-PPO ablation: trained beat the frozen deterministic policy in all five seeds, with a 6.06% aggregate late-policy improvement and a positive exploratory paired interval.
- [ ] Run the paper-scale budgets on an available GPU and archive all manifests/logs.

## Gate 3: comparisons — pending GPU experiments

1. Freeze the successful AlexNet-CONV masked-PPO configuration and preserve its archive/hash with the experiment record.
2. Run BS, RS, fixed SA, ASA, paper DDPG, masked PPO, and the frozen PPO control with matched complete-placement accounting on the same evaluator.
3. The one-seed AlexNet-FC short gate completed: mechanics passed, but PPO's 0.163% directional policy improvement was weak and its best placement lost to matched-budget RS and ASA.
4. The 3,000-placement AlexNet-FC extension completed: the deterministic policy improved directionally by 0.558%, but trained PPO tied its frozen control's best result and ASA was 2.80% better. Do not scale unchanged PPO to five FC seeds.
5. The five-seed 4,101-evaluation AlexNet-FC baseline is complete: ASA beat random search in all five pairs, averaging 23.5001 versus 24.1305 microseconds (2.61%).
6. [x] Implement a fixed-budget PPO learned-proposal ASA. Metropolis acceptance and the cooling/adaptation controller remain fixed; PPO ranks legal relocation/swap candidates using graph, movement, bottleneck, cost-change, and search-history features.
7. [x] Complete the one-seed 4,101-evaluation PPO-ASA gate. Mechanics passed, but learned PPO-ASA reached 23.5602 microseconds versus 23.4213 for its uniform control and 23.4318 for ASA. It failed both gates.
8. [x] Add a matched bottleneck-focused neighborhood and test it locally. At 1,001 evaluations it cut neutral proposals from about 89% to 72.8% and raised improving moves to 7.5%. Trained PPO beat its focused control by 0.40% but remained 0.42% behind ASA.
9. [x] Complete the 4,101-evaluation focused gate. Neutral proposals fell to 37.2%, but trained PPO finished at 23.7792 microseconds versus 23.7595 for its frozen control and 23.4318 for ASA. Do not scale this neighborhood to five seeds.
10. [x] Replace single-chain online updates with a shared proposal policy trained across repeated independently initialized placement chains. The implementation retains one policy across restarts, terminates advantages at chain boundaries, and counts every initialization.
11. [x] Add a mixed global/bottleneck proposal pool and an exact paired runner for a 250,000-proposal trained policy, frozen control, and matched-budget ASA. The local 520-evaluation mechanics smoke passed; it was not a learning test.
12. [x] Complete `multichain-train`: mechanics passed and trained PPO-ASA beat its uniform control by 0.466%, but ordinary ASA was 3.55% better. Preserve this as a one-seed learning-only result.
13. [x] Preserve the trained proposal checkpoint and add frozen checkpoint loading for inference-only evaluation.
14. [x] Complete the five-seed uninterrupted holdout: the frozen checkpoint lost every pair, averaging 0.491% worse than uniform and 0.533% worse than ASA.
15. [x] Diagnose training horizon locally: at the original 128-proposal horizon, trained PPO had a weak 0.385% mean advantage over uniform but passed only three of five pairs. The 4,100-step failure is consistent with horizon specialization.
16. [x] Train once with 4,100-proposal episodes, within-chain progress, and PPO updates at every chain boundary; immediately run the frozen five-seed 4,101-call holdout.
17. [x] Reject that checkpoint: it received only 61 PPO updates, stayed nearly uniform, and had no positive holdout mean against uniform or ASA. This is enough to reject the checkpoint, not the architecture.
18. Run the paper-informed sufficient-scale protocol with 1.25 million proposals, exactly 1,250 PPO updates, no entropy bonus, and 10 frozen holdout seeds. This resolves the update-count shortfall relative to the closest public RL-Based-SA setup.
19. Close the current candidate-scoring architecture if it fails the positive-mean/eight-of-10 uniform gate. If it passes both uniform and ASA gates, repeat training with at least three independent training seeds.
20. Keep guided DDPG and both short-chain PPO-ASA variants as negative ablations.
21. Proceed to VGG16 and ResNet50 after the evaluator and learning behavior are credible.
22. Add a validated large-batch fill/steady-state/drain model before presenting paper-style throughput.
23. Record Git commit, clean/dirty state, complete configuration, Torch/CUDA versions, visible GPU, memory, evaluation counts, checkpoint paths, and wall time.

## Gate 4: improvements after reproduction

Guided DDPG is now available as an explicitly experimental mode. It combines ASA demonstrations with permanent replay, collision-free legal projection, normalized complete-episode return targets, uniform legal replay warm-up, and retained deterministic candidates. Its matched no-learning control is required for every learning claim.

Masked categorical PPO is implemented and has passed its first five-seed
learning gate. Graph encoders, alternative partitioning, and parallel
environments remain future experiments. Freeze and identify the paper-mode
configuration, then compare every improvement under matched complete-placement
budgets. The evidence and design rationale are recorded in
[`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).

The first PPO-guided ASA implementation now follows the verified RL-Based-SA
structure: PPO learns the neighbor proposal while simulated annealing retains
its accept/reject rule and schedule. Sequence-pair and B*-tree floorplanning
representations are not applicable to this fixed-grid core-mapping problem.
