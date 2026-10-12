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
18. [x] Run the paper-informed sufficient-scale protocol with 1.25 million proposals, exactly 1,250 PPO updates, no entropy bonus, and 10 frozen holdout seeds. The run completed in 6.47 hours with all mechanics checks passing.
19. [x] Close the current candidate-scoring architecture. It beat uniform in only three of 10 seeds, tied five, and failed both prespecified gates while converging to uniform entropy. Do not add more epochs to this configuration.
20. [x] Audit the closest official RL-Based-SA PPO source. It uses disjoint actor and critic modules/optimizers and reward-to-go critic targets; the rejected local scorer shared its encoders. Add a cited `reference_separate` adapter without changing the placement objective, legal neighborhood, or controls.
21. [x] Record the independent Claude audit, including its accepted reward/signal findings and corrections to its legacy-epoch conversion and unquantified KL claim.
22. [x] Implement E0 oracle-of-16 candidate headroom, E1 planted-signal shared/split controls, missing logit/gradient/feature diagnostics, and a gated 300-update E2 split pilot.
23. [x] Run `scripts/run_kaggle_ppo_asa_diagnostic_gate.sh`. Both planted-signal conditions passed, but the real split pilot stayed uniform and produced no holdout policy signal after 300,032 evaluations.
24. [x] Establish that oracle headroom exists: best-of-16 gained about 0.70% of current cost over uniform on sampled real pools. The existing proxy's 4.5% oracle match rate is below the 6.25% uniform expectation.
25. [x] Add exact oracle-dataset export and a trajectory-grouped supervised ranking gate. It collects ten independently restarted chains, holds out complete chains, reports selected advantage, confidence interval, oracle-headroom capture, and top-1 performance, and emits a PPO-compatible actor checkpoint.
26. [x] Complete the first oracle-supervised run at revision `8fe5da4`. The frozen supervised actor beat uniform and ASA in four of five fresh-seed pairs by 0.131% and 0.180% on average, but both five-seed intervals crossed zero. PPO fine-tuning then lost all five pairs to uniform and four of five to ASA; close PPO fine-tuning for this checkpoint.
27. Keep guided DDPG and failed candidate-set PPO-ASA variants as negative ablations.
28. Proceed to VGG16 and ResNet50 after the evaluator and learning behavior are credible.
29. Add a validated large-batch fill/steady-state/drain model before presenting paper-style throughput.
30. Record Git commit, clean/dirty state, complete configuration, Torch/CUDA versions, visible GPU, memory, evaluation counts, checkpoint paths, and wall time.
31. [x] Complete the trajectory-safe supervised gate at revision `164edfe`. It used ten chains with two held out, captured only 3.36% of oracle advantage on validation, and produced a selected-advantage interval crossing zero. The runner correctly stopped before optimizer holdouts and PPO fine-tuning.
32. [x] Close the current supervised candidate scorer. The earlier 0.18% five-seed ASA advantage remains a leakage-prone, statistically uncertain provisional result and is superseded by the failed chain-held-out representation gate.
33. Run fixed-workload RS, SA, and ASA calibration near the paper's one-million-placement budget and compare normalized ratios, evaluator sensitivity, and wall-clock cost before changing the learning architecture.
34. If RL+ASA remains the method goal after calibration, redesign what RL controls (for example, selecting among distinct ASA move operators or using a topology-aware representation), and require a chain-held-out supervised or planted-signal gate before any long PPO training.
35. [x] Reproduce the post-hoc leave-one-chain-out target analysis on the trajectory-safe dataset. Signed gain transferred negatively, while improvement-only utility showed 14.7% mean capture overall and 10.0% late-chain capture; treat this only as a hypothesis because the target was selected after seeing the data.
36. [x] Add an explicit improvement-only target, complete-chain bootstrap statistics, sampled-policy capture, and a strict gate while preserving the historical signed-gain behavior.
37. [x] Run `scripts/run_kaggle_improvement_ranker_gate.sh` on fresh data. Late argmax capture was 7.03% [4.28%, 9.81%], sampled capture was 0.73%, the improvement-rate ratio was 1.76x, and nine of ten chains were positive. It failed three of five frozen criteria.
38. [x] Stop before the frozen-optimizer holdout because the fresh gate failed. Preserve deterministic argmax ranking as a weak representation result, not an RL+ASA optimizer result.
39. [x] Repair fixed SA's objective-unit mismatch. The paper publishes cooldown 0.99 but not absolute temperatures; fixed SA now measures its initial temperature from budgeted uphill deltas and then follows a frozen geometric schedule. Historical absolute temperatures remain opt-in.
40. [x] Run `scripts/run_kaggle_paper_baseline_calibration.sh` at one million proposals per search method. Fixed SA beat RS, but RS/BS was 0.9606 and SA/BS was 0.9111, far above the paper figure's approximate 0.59 and 0.50. The reconstructed compute-only floor is already 0.8989 of BS, so this evaluator cannot reproduce the published reduction regardless of optimizer quality.
41. Run the exploratory `scripts/run_kaggle_evaluator_vva_sensitivity.sh` sweep. VVA throughput is unpublished and currently defaults to one addition/cycle; sweep 1, 8, 16, 32, 64, and 128 additions/cycle with matched short RS/SA searches to measure whether this assumption explains the missing placement headroom.
42. If a physically plausible VVA setting restores adequate headroom, freeze it before seeing new optimizer results, rerun the one-million-proposal calibration, and validate the choice on a held-out network or region. Do not restart RL until the non-learning baselines are credibly calibrated.

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
