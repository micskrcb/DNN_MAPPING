## 23 October: lower exploration passes seed12 screen; confirm unchanged

Evidence: `artifacts/online-exploration-seed12-20261023/` (source hash, reports, summary and test/completion logs). Commit d65d9e4;68 tests passed;100001 placement evaluations per arm;389 updates per learner; identical initial cost6.72µs. Total runtime5.84h.

| Method | Best cost µs | Lower-exploration improvement |
|---|---:|---:|
| Lower-exploration online RL+SA | 4.33616 | — |
| Existing online RL+SA | 4.37264 | 0.83428% |
| Same-pool uniform | 4.42576 | 2.02451% |
| Fixed SA | 4.45360 | 2.63697% |
| Same-pool proxy | 4.46432 | 2.87076% |

Actual final epsilon0.10462 versus0.50257. Lower exploration won all comparisons on this seed. This is exploratory evidence, not proof of general superiority. It is not a5% win over published DDPG/SOTA, and same-pool uniform is not standalone RS.

Next: lock this configuration and run confirmation seeds13 and14 in separate sessions. Do not tune between them. Keep seed12 separate. Advance if mean paired improvement is positive against all four comparators and candidate wins against the existing learner on both confirmation seeds; report mixed results honestly. This is a practical directional screen, not a statistical-significance claim. No mapper rewrite is justified by a successful single-seed result.

Code improvements: stricter analyzer validation of learner, features, replay, candidate count, annealing and actual control selection; reject mixing seed12 with confirmation; archived-report regression tests; dedicated confirmation runner. Protocol in [ONLINE_EXPLORATION_CONFIRMATION.md](ONLINE_EXPLORATION_CONFIRMATION.md). Published-method comparison remains a separate required experiment: matched simulator/workload and budget, independently validated DDPG reconstruction, training/evaluation costs reported. Do not present failure of an unvalidated DDPG implementation as a published-method win.

## 22 October: recent replay did not beat reference; exploration test next

Archive `artifacts/online-recent-replay-20261022/` records commit a9d8b6a, seed11, successful completion and67 passing tests. All five arms used100001 placement evaluations and the same initial cost8.016µs. Both learners made389 updates.

| Method | Best cost µs | Improvement over fixed SA |
|---|---:|---:|
| Existing unfocused online | 4.37776 | 1.09886% |
| Recent-replay online | 4.38912 | 0.84222% |
| Fixed SA | 4.42640 | 0% |
| Same-pool uniform | 4.42736 | -0.02169% |
| Same-pool proxy | 4.49296 | -1.50371% |

Recent replay lost to the existing learner by0.25949%. It beat uniform by0.86372% and proxy by2.31117%, but failed the primary advancement comparison. Existing learner beat uniform by1.12031% and proxy by2.56401%. These controls are not standalone random search. This single seed does not establish general inferiority of recent replay.

Sampling changed as intended: final mean replay age24806.69→9967.02 observations; recent fraction21.19%→79.88%. Better prediction/target mean agreement did not translate to a better final placement. Keep uniform replay as reference. Runtime summed across arms5.58h.

Next experiment uses the already-supported exploration endpoint0.1 versus reference0.5, with unchanged linear decay duration. This reduces exploration throughout post-warmup training, not only at the end. Actual terminal epsilon is slightly above its configured endpoint because warmup is excluded from learned steps. This is a hypothesis, not an established improvement. Seed12,100000 proposals, five matched arms. See [ONLINE_EXPLORATION.md](ONLINE_EXPLORATION.md). Prior entries below are historical.

## 22 October: multiscale rejected; improve online replay next

Both uploaded runs completed. The seed10 confirmation did run despite the previous plan to supersede it. Shared seed10 arms in the two ZIPs match exactly and count only once.

| Seed10 method | Best cost (µs) |
|---|---:|
| Original focused online | 4.41280 |
| Unfocused online | 4.37216 |
| Unfocused uniform | 4.44112 |
| Unfocused proxy | 4.45760 |
| Fixed SA | 4.44640 |
| Mixed-size online | 4.45984 |
| Mixed-size uniform | 4.46240 |
| Mixed-size proxy | 4.49168 |

Mixed-size online is 2.0054% worse than unfocused online and 0.3023% worse than fixed SA. Keep it opt-in for reproducibility; do not promote it. Unfocused online remains the working reference.

Across confirmation seeds9–10, unfocused online mean paired improvements are **0.4532% over original (1/2 wins), 0.9640% over uniform (2/2), 1.7076% over proxy (2/2), and 1.3688% over fixed SA (2/2)**. Screening seed8 is separate. These are small-sample VGG16-FC results, not broad superiority or paper reproduction.

Next: test recent-experience replay within the successful online mapper. In the middle 40k–60k proposals, current reward versus replay-target means were -5.348 versus -2.621 (seed9), and -4.555 versus -3.062 (seed10). This is consistent with lagging replay; it does not establish causation. Best placements still improved near proposals99732/99332, so the traces do not establish a plateau.

The new optional sampler draws at least75% of each minibatch from the latest10000 observations and the remainder from the full50000 buffer without duplicates. No reward-sign oversampling, new features, new moves, or SA changes. Default sampling remains exactly the old behavior. See [ONLINE_RECENT_REPLAY.md](ONLINE_RECENT_REPLAY.md) for the protocol and commands. No performance claim until its results arrive.

Evidence: `artifacts/online-multiscale-20261021/` and `artifacts/online-pool-seed10-20261021/`, including source hashes, reports, summaries, completion and test logs. Prior entries below are historical plans/results.

## 21 October: seed9 near-ties original; resume mapper development

56e0176 seed9 completed,62 tests passed,100001 calls/arm. Costs µs:
original4.41664, unfocused-online4.41728, same-pool uniform4.43392,
proxy4.48448, fixed-SA4.46496. Unfocused-online gains: -0.01449% original,
+0.37529% uniform, +1.49850% proxy, +1.06787% fixed-SA. This supports
learning versus controls, not superiority to the original mapper. Both
learners made389 updates. Total6.38h. Source summary is archived.

At the user's request, stop the planned unchanged seed10 confirmation and
resume algorithm development. Seed10 is now a new developmental experiment,
not confirmation of the previous pool-only hypothesis. New opt-in pool:
8 two-task moves +8 ordinary ~1%-task moves. Preserve online signed learning,
fixed calibration, features, exploration and objective budget. Original focused
and unfocused mappers remain reference arms. Same mixed-pool uniform/proxy and
plain SA remain controls. No promotion/performance claim before results.
See ONLINE_MULTISCALE.md.

## 21 October: unfocused online pool passes seed8 screen

Run e871e64 completed all59 tests and five100001-call arms. VGG16-FC seed8
best costs (µs): modified online4.36672, original4.40432, same-pool uniform
4.42544, same-pool proxy4.48320, plain-SA4.43520. Modified gain: +0.8537%
original, +1.3269% uniform, +2.5981% proxy, +1.5440% fixed-SA. Both online
learners made389 updates; initial placement and candidate-arm calibration
matched. Modified runtime71.58 minutes versus58.96 fixed-SA; total5.696h.
This is positive one-seed screening evidence, not a promoted replacement.
Source reports: artifacts/online-pool-seed8-20261020/.

Next: unchanged configuration, confirmation seeds9–10 in separate sessions.
Keep all five arms and100001-call budgets. No mapper/target/exploration changes.
Runner improvements: validated per-seed summaries, environment version logging,
aggregate paired gains and guards against duplicate seeds/mixed workloads.
Report screening seed8 separately. See ONLINE_POOL_CONFIRMATION.md.

## 20 October: additional seeds support learning, not superiority to plain SA

9f8db7b completed seeds6–7, 100001 calls/arm, all 57 tests passed.
Mean best costs µs: online4.42856, uniform4.50888, proxy4.55304,
fixed-SA4.41768. Mean paired gain: +1.7812% uniform (2/2 wins),
+2.7328% proxy (2/2), -0.2464% fixed-SA (0/2). Online made389 updates
per seed. Runtime10.394h total; online84.0 versus fixed-SA60.6 minutes/seed.
Seed3 stays separate. No random baseline was rerun.

Correction to proposed next step: the existing pool already contains 50%
unfocused plain-SA-style neighbors. New opt-in change tests an entirely
unfocused pool while retaining calibration focus=.5, the same online signed
reward, features, training schedule, exploration and SA acceptance. Default
behavior is unchanged. Compare original, unfocused online, same-pool uniform,
same-pool proxy and plain-SA, initially seed8 at100001 calls each. No new
performance improvement is claimed. See ONLINE_POOL_GATE.md.

## 20 October: positive online restoration; keep learner unchanged

Run 2348e08 completed, 55 tests passed. VGG16-FC seed3, VVA16/1,
100001 evaluations/arm and equal initial placements. Archived bacc334 and
current signed online learner both reached 4.41664 µs; uniform 4.50976,
proxy 4.55088, fixed-SA 4.44640, relative-state learner 4.43184 µs.
Current online learner gains: uniform 2.0649%, proxy 2.9498%, fixed-SA
0.6693%. Relative state lost 0.3442%, so retain legacy state. Online learners
made 389 updates and reached epsilon 0.502566. Total six-arm time 7.62 hours.

Same-run final-cost agreement is not exact historical replay: archived source
previously reached 4.45568 µs. Single reused seed does not establish a robust
multi-seed win. No new random-search run; do not relabel this as a new 23% gain.
Reports and test log are in artifacts/online-restoration-20261019/.

Next: unchanged online learner versus uniform/proxy/plain-SA on additional
seeds6–7. Remove duplicate archived arm and losing relative-state arm. Preserve
100001-call budgets and all learner/search settings. Runner archives after
every completed arm and checks budgets/initial costs/learning flags. Seed3
remains separate from follow-up aggregates. See ONLINE_BANDIT_FOLLOWUP.md.

## 19 October: return to the demonstrated online bandit

Priority is now restoration and controlled improvement of the original online
bandit+fixed-SA, not replacement by offline ranking. Archived VGG16-FC seeds
3–5 showed 22.405% over random, 1.637% over focused-uniform and 0.122% over
plain fixed SA. These baselines must not be conflated.

Found reward compatibility drift: old bandit replay stored signed proposed
reward despite positive_delta CLI metadata; current code honors the flag.
Historical bandit runner now explicitly selects proposed_delta. New restoration
runner includes archived bacc334 source, current signed online learner, a
single relative-state ablation, uniform, proxy and fixed-SA controls. Original
100001-call budget/VGG16-FC/VVA16/1 retained; default reused seed3. No new
performance claim. See ONLINE_BANDIT_RESTORATION.md for provenance and commands.

## 19 October: ranking-path diagnosis completed; productive target next

Revision dc5d536 completed eight runs, exit 0, all 54 tests passed.
VGG16-CONV seeds 40–41: mean best costs (µs) proxy exploratory 1508.632,
proxy greedy 1519.286, learned exploratory 1731.319, learned greedy 2591.430.
Mean paired learned disadvantage: 14.745% exploratory, 70.681% greedy.
Each run used 4101 search + 960 audit calls, plus 16 sensitivity calls.
Reused seeds make this diagnostic, not new held-out confirmation.

On learned greedy trajectories, scorer-selected moves in audited pools were
neutral in 63/64 and 64/64 pools. Strong raw-gain rankings can reflect avoiding
harm rather than generating progress. This supports target correction but does
not prove neutrality is the sole cause. Removing exploration worsened outcomes.
Summary and test log: artifacts/ranking-path-20261019/.

Next: balance improving/neutral/worsening example groups and rank gain among
improving candidates, retaining the same scalar architecture and data (30–33).
No new oracle data train the model. Compare productive/signed/proxy with
exploration on fresh seeds 43–45. This is supervised pretraining/proposal
learning, not demonstrated online RL. See PRODUCTIVE_RANKER.md.

## 19 October: greedy tie-breaking regression caught before training

Archive `kaggle-ranking-path-fixed-20261019T044048Z.zip` at `c9e08ba`
passed the 51-test suite but failed the added one-hot greedy assertion.
No training or new rollouts completed. With tied scores, PyTorch topk(1)
and argmax can choose different indices, splitting exploration/exploitation
mass between equal maxima. Top-m=1 now shares argmax tie-breaking. The test
explicitly zeros model weights to guarantee ties; its assertion is retained.
The audit runner now prints failure log tails in Kaggle output.
Local verification with temporary CPU PyTorch/torchvision passed all 51
reconciliation tests, functional smoke tests, and all 3 ranking tests
(including both frozen greedy controls); no PyTorch tests skipped.

## 19 October: ranking-path audit interrupted — implementation error

Run `0596e1a` passed 54 tests and reproduced both seed-40 exploratory costs
exactly (learned 1782.40832 µs, proxy 1517.47904 µs), then failed before the
first greedy rollout: validation rejected warmup=0. No exploration comparison
or complete gate conclusion is available. Fix permits zero warmup only with
learning disabled; tests now exercise both frozen greedy controls and verify
one-hot maximum-score selection. Re-run the unchanged diagnostic protocol.

## 19 October: frozen pairwise rollout fails proxy comparison

Archive `kaggle-pairwise-rollout-20261018T131223Z.zip`, revision `994a83d`,
exit 0; all 54 Kaggle tests passed. VGG16-CONV seeds 40–42 each had 4,101
objective calls, matched initial placements, and no deployment updates.
Mean best costs: proxy-SA 1,491.228 µs; learned-SA 1,694.697 µs;
uniform-proposal SA 1,885.773 µs. Mean paired learned improvement was
10.135% over uniform (3/3 wins), but -13.607% over proxy (0/3 wins).
Uniform-proposal SA is not random search. Nine rollouts took 91.68 minutes.
Offline ranking gains did not establish improved full-search optimization.
The planned longer run does not advance. Summary is archived in
`artifacts/pairwise-rollout-20261018/`.

Next: an eight-run diagnostic reuses seeds 40–41, comparing learned/proxy
scores with exploratory/greedy selection. This is developmental, not fresh
confirmation. Audit exact gains in up to 64 pools per run; export scores,
selection probabilities and chosen indices without changing selection or
training. All extra oracle calls are included in reported budgets.
See [ranking-path audit](RANKING_PATH_AUDIT.md).

# Paper-reproduction execution plan

## 18 October: ranking repair positive offline result

At `952be72`, archive SHA-256
`d1848d59939c2b2b0e11d0ebe8b66cfb26a4aefad6ea898e2da9c6d784fc02b8`,
all 53 tests passed. Six chains each consumed 7,926 exact calls, including
3,825 additional oracle calls (plus 16 sensitivity samples). Four chains
provided 1,020 training pools; seeds 34--35 were held out.

Signed pairwise ranking captured 79.94%/87.18% of available immediate oracle
gain over proxy, and 94.07%/97.63% in the late half. Late harm frequency was
0.78% in both chains. Positive ranking captured 88.46%/91.75% overall but
70.58%/68.23% late, with 38.28%/37.50% late harm. The bounded regressor
had negative total advantage and tied proxy late. 57.43% of its training
targets were outside its representable interval. These are offline ranking
results, not percentages of final placement improvement or a causal isolation
of the range restriction. Model selection used seeds 34--35.

Next: freeze signed ranking trained only on seeds 30--33, test complete
4,101-call SA rollouts on fresh seeds 40--42 against proxy and uniform.
The scorer is supervised from oracle labels, not an online RL update.
See `PAIRWISE_ROLLOUT.md`; no performance result exists yet.


## 18 October: VGG16 frozen-transfer result and ranking repair

Archive `kaggle-amortized-vgg16-transfer.zip`, SHA-256
`d2f75471b927bc0de83b45b08cb4b42716d5d341b6d6ba7324ada0a4a31a19f4`,
ran revision `7d64193`. All 51 tests passed, both training stages made 77
updates, and each held-out seed used 20,001 calls with frozen loaded weights.
Mean costs: proxy 1210.63451 us, checkpoint 1274.90944 us, uniform
1472.72832 us. The checkpoint beat uniform 3/3 by 13.316% mean paired gain
but lost to proxy 3/3 by 5.340%. The advancement gate failed. Since the scorer
contains a proxy prior, its uniform advantage cannot be attributed solely to
learning. No extension of this checkpoint is recommended.

The next diagnostic is `scripts/run_kaggle_ranking_repair.sh`, documented in
`RANKING_REPAIR.md`. It reuses chain-disjoint labels to compare bounded
positive regression with signed and positive pairwise ranking, preserving
VVA execution/partition rates 1/1. This is an offline representation test,
not a demonstrated improvement to the deployed optimizer.


## Immediate Proxy-SA portability gate

The 2,000-move portability gate passed mechanically on AlexNet-FC, ResNet18,
and a custom torus CNN. Proxy-SA was competitive but not uniformly superior.

1. The 20,000-evaluation gate is complete and failed advancement on both
   workloads; do not run unchanged seeds 3 and 4.
2. The exact attribution test is complete: learning lost by 2.306% on average
   and won only 1/3 seeds. Do not run unchanged held-out seeds.
3. Treat the exact eight-candidate proxy-only SA as the current strongest
   method. The budget-accounted full-information operator gate is now
   implemented: every twentieth decision labels all four operators while the
   matched proxy control pays for the identical labels. Run
   `scripts/run_kaggle_full_information_operator_gate.sh`; advance only if the
   learner wins at least two of three pairs and improves mean cost by at least
   1%.
4. Scale a confirmed method to 100,000 evaluations before any paper-scale
   500,000--1,000,000 evaluation experiment.
5. Treat the 17 October external review as a design audit, not an empirical
   result. Add ordinary unthrottled Proxy-SA to the decision comparison. If
   the full-information diagnostic fails, test selected-action proxy-gated
   discounted LinTS with one true call per step. See
   `EXTERNAL_RESEARCH_REVIEW_2026-10-17.md` for accepted findings, corrections,
   controls, and the separate long-horizon RL path.

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
41. [x] Run the exploratory `scripts/run_kaggle_evaluator_vva_sensitivity.sh` sweep. Eight VVA additions/cycle reduced the compute floor from 0.899 to 0.526 of BS while preserving the original AlexNet-FC partition grids; at 20,000 proposals, RS/BS improved from 0.966 to 0.842 and SA/BS from 0.928 to 0.661. The VVA assumption explains much of the missing headroom but does not yet reproduce Figure 10.
42. [x] Audit the VVA-rate literature. The corrected audit retracts the Noxim, full-overlap, multiplier-bypass, eight-wide reduction, SRAM-interface, and exact Figure-10 claims. Published evidence is insufficient to select one additions/cycle value; eight remains a sensitivity point, not a calibrated fact.
43. [x] Add `--partition_vva_ops_per_cycle` so execution-rate sweeps can hold the reconstructed logical graph fixed. This separates timing sensitivity from the grid changes that confounded rates 16 and above.
44. [x] Run `scripts/run_kaggle_evaluator_vva_sensitivity.sh <output> 20000 1` on fixed rate-one grids. SA/BS was 0.928, 0.661, 0.612, 0.671, 0.726, and 0.790 at execution rates 1, 8, 16, 32, 64, and 128. Objective sensitivity and sampled best cost saturated by rate 32, identifying compute-dominated, transition, and communication-dominated regimes without changing the graph.
45. [x] Run the conditional one-million-proposal calibration at execution rate 16 with rate-one grids (archive SHA-256 `f584151963f1e1a438c413ac427cf9835b624d292981d1ef0c85ff4312b0b294`). At seed zero, BS was 3.92032 microseconds, RS/BS was 0.7836, fixed-SA/BS was 0.5232, and ASA/BS was 0.6109. Fixed SA beat RS by 33.23%, and its 0.5232 ratio is close to the approximate Figure-10 SA ratio of 0.50. This is a one-seed, post-sweep, assumed-rate descriptive result: it does not identify the unpublished VVA rate or confirm reproduction. ASA was 16.75% worse than fixed SA under this matched budget.
46. [x] Run `scripts/run_kaggle_logic_order_ablation.sh` at execution rate 16 with rate-one grids (archive SHA-256 `116b43ec90f1e60c0e86f6dada8dbfec4b8a8d529aaeecfcbd197a6b3efed214`). Changing only logical numbering moved BS from 3.92032 to 9.92 microseconds, while 20,000-proposal RS changed from 3.14032 to 3.21536 and fixed SA from 2.40000 to 2.39328 microseconds. Consequently RS/BS moved from 0.8010 to 0.3241 and SA/BS from 0.6122 to 0.2413. Treat BS-normalized paper matching as unresolved and report absolute costs plus paired same-order controls; neither ordering is identified as the authors' unpublished convention.
47. Check AlexNet-CONV under the same declared timing assumption against its separate approximate Figure-10 targets (RS/BS 0.89, SA/BS 0.82). Also report hop-distance changes, noting that the paper's 4.5x AlexNet-FC value is DDPG versus BS, not an SA target.
48. [x] Add a fixed-annealing mode beneath the PPO proposal policy. It reuses fixed SA's measured initial temperature, 0.99 cooldown, constant 1% neighborhood, incremental evaluator, and Metropolis acceptance. PPO changes only which member of a legal candidate pool is evaluated. The immediate proposed objective delta is used as a contextual-bandit reward (`gamma=0`), and `--ppo_asa_disable_learning` is the matched uniform-proposal control.
49. [x] Run the first signed-delta fixed-SA+RL short gate (archive SHA-256 `1ce71f4631b5befe571acff6943c3f9df66d4bf67ab0f4433379ce702e62cfa2`). Learned PPO reached 2.45040 microseconds versus 2.44864 for the frozen control, a 0.0719% loss. Both used 20,001 objective evaluations, the same initial cost and fixed schedule. PPO made 78 finite updates, moved its score-head norm from 0.0028 to 0.348, and reached probability L1 distance 0.047 from uniform, so this is a functioning but negative optimizer result rather than failed learning mechanics.
50. [x] Rerun the short gate with improvement-only `positive_delta` reward, learning rate `3e-4`, and entropy coefficient `0.001` (archive SHA-256 `3abb540207ada402b7394013f47fca16e1641bec71279192daf8548410c85148`). Learned RL+SA reached 2.43328 microseconds versus 2.44864 for its frozen control, a 0.6273% improvement at the same 20,001 objective evaluations. It beat ASA by 1.298% and 20,000-sample random search by 22.51%, but remained 1.387% worse than plain fixed SA. PPO made 78 finite updates.
51. [x] Run the three-seed `extended` gate (archive SHA-256 `71a95ba004476beeb4a4a13cc0ad7a918a5c869327f203a05d1369bc98d740c8`). Learned RL+SA beat frozen control in two of three seeds and mean per-seed improvement was +0.105%, passing the prespecified directional gate. Mean costs were 2.23989 microseconds learned, 2.24213 control, 2.20576 fixed SA, and 2.41760 ASA. The paired three-seed 95% t interval for relative improvement is approximately [-2.41%, +2.62%], so this is a weak positive pilot rather than statistically reliable evidence.
52. [x] Run `scripts/run_kaggle_rl_fixed_sa_gate.sh adaptive-short` (archive SHA-256 `8d9db7e72154480f5b14dd291d56deb5cc52c769af5d351064b5f5e62ed0e349`). Learned RL+ASA reached 2.64784 microseconds versus 2.63840 for its frozen adaptive control, a 0.3578% loss at equal 20,001-evaluation budgets. It also lost to plain ASA at 2.46528 and fixed SA at 2.40000. PPO made 78 finite updates, so this is a negative optimizer result rather than a failed run. Do not run `adaptive-extended` with this controller.
53. Retain RL+fixed SA as the primary learned-annealing implementation. For a future RL+ASA method, first redesign the adaptive controller around the working geometric schedule (for example, bounded reheating or learned move-scale selection) and establish that its frozen adaptive control is competitive with fixed SA. Do not combine another weak ASA controller with PPO and attribute the resulting gap to proposal learning.
54. [x] Add a contextual-bandit proposal learner under the same fixed-SA mechanics. It predicts immediate gain from the selected move and preserves the exact uniform frozen-policy control and objective budget. The initial positive-target design is superseded by item 55.
55. [x] Diagnose the first bandit short gate. It lost to its frozen control by 8.04% despite 77 finite updates; positive-balanced replay overpredicted gains and 10% final exploration caused harmful exploitation. Change only the identified learning failure: signed gain targets, uniform replay, 50% final epsilon, and full-run decay. Repeat the one-seed short gate before any extended run.
56. [x] Complete the corrected signed-target short gate. Learned bandit+fixed-SA beat its frozen control by 0.1242% and ASA by 0.7983%, while remaining 1.90% worse than fixed SA. Prediction calibration and exploration diagnostics no longer show the first run's collapse. The directional gate passes, so run `bandit-extended`; require positive mean improvement and wins in at least two of three paired seeds.
57. [x] Complete the three-seed signed-bandit extended gate. It passed the directional rule with two wins and +0.491% mean improvement over control, beat ASA by 7.70%, and remained 1.16% worse than fixed SA; the paired interval crosses zero. Diagnose and fix the replay-cap exploration bug that stopped epsilon at 0.7525 instead of 0.5. Treat the archive as evidence for the actual 0.7525 configuration and rerun the intended schedule before final claims.
58. [x] Complete the epsilon-fixed paired rerun. Final epsilon reached 0.5026 in every seed; learned proposals won two of three pairs and averaged +0.664% over frozen control. Using archived baselines, learned RL+SA beat ASA by 7.87% but remained 0.98% worse than fixed SA. The paired interval still crosses zero. Freeze this configuration and next add genuinely held-out seeds and workloads rather than tune on seeds 0--2.
59. [x] Complete frozen held-out seeds 3--5. RL+SA beat the frozen control in all three pairs by 1.965% on average with a paired 95% interval [1.008%, 2.922%]. This is positive descriptive evidence on the tuned AlexNet-FC graph, not an unbiased confirmatory test. It beat ASA in all three by 9.288% and beat fixed SA in two of three by 0.669% on average, though the fixed-SA interval crosses zero. Freeze the method and move to cross-workload validation; do not tune further on AlexNet-FC.
60. [x] Add frozen workload overrides and complete the AlexNet-CONV short transfer gate without changing learner hyperparameters. RL+SA beat frozen control by 1.136%, fixed SA by 2.246%, and ASA by 4.394%, with 77 finite updates. Advance to the three-seed 100,000-evaluation CONV gate.
61. [x] Complete the three-seed AlexNet-CONV extension. Learned, frozen control, and fixed SA all reached the identical 37.02117-microsecond final minimum in all three seeds. Learned proposals had a small 0.357% mean best-so-far trajectory advantage over control, but its interval crossed zero. Record CONV as a final-quality tie and move to a new DNN workload without retuning.
62. [x] Complete the frozen VGG16-FC short transfer gate at seed zero. RL+SA reached 4.59088 microseconds, beating frozen control by 1.834%, fixed SA by 1.490%, and ASA by 2.739%, with 77 finite updates. This is a passed one-seed signal only.
63. [x] Complete the unchanged three-seed VGG16-FC `bandit-extended` gate at 100,000 evaluations per method. Learned RL+SA beat frozen control in all three pairs by 2.255% on average, with paired 95% interval [1.107%, 3.403%]. It beat fixed SA in all three by 0.908% on average, though that interval crosses zero, and beat ASA in all three by 5.418% with a positive interval. The cross-workload final-quality gate passes.
64. Add the minimum decisive attribution check: a non-learned proxy-greedy selector using the same focused candidate pools, epsilon schedule, and objective budget. Keep the current focused-uniform control. Add focus-zero or feature-removal ablations only if this comparison leaves the source of the gain unresolved; do not require a large control ladder before continuing useful experiments.
65. [x] Complete frozen VGG16-FC seeds 3--5 with matched random, fixed-SA, and ASA baselines. RL+SA beat focused-uniform control 3/3 by 1.637%, ASA 3/3 by 3.793%, and random 3/3 by 22.405%; it was effectively tied with fixed SA. Across all six VGG seeds it beat control 6/6 by 1.946%. This is strengthened exploratory transfer evidence, not a universal superiority claim.
66. Run the new `bandit-proxy-short` comparator on VGG16-FC. It uses identical focused candidate pools, warm-up, epsilon decay, SA mechanics, and true-objective budget, but ranks exploitation moves directly by `proxy_gain` with zero training updates. Use three seeds and skip already-completed context baselines. Escalate to 100,000 evaluations only if the short comparison is informative.
67. [x] Implement the external-review follow-up as opt-in modes without changing the frozen historical configuration: `contention12` adds cached peak-link, affected-stage slack, and source-serialization deltas; soft focus admits stages within 5% of the current bottleneck; `epsilon_top_m` retains greedy, top-four, and global-uniform probability; shuffled-label and bottleneck-random controls test attribution. Add exact cache-delta tests and `scripts/run_kaggle_contention_bandit_gate.sh`.
68. [x] Run the three-seed 20,000-evaluation VGG16-CONV contention-bandit gate. The learner beat shuffled labels in two seeds, bottleneck-random in all three, and random search in all three, but lost to proxy-greedy and fixed SA in all three. The prespecified extension gate failed, so do not run the unchanged 100,000-evaluation version. This rejects the `contention12` plus soft-focus plus top-four exploration package, not every RL+SA formulation.
69. Retain proxy-greedy as a required strong non-learned control. The next learned proposal experiment must add value beyond that proxy, preferably by learning a residual correction to the proxy score or by selecting among qualitatively different move operators. Require a short paired gate against proxy-greedy and fixed SA before scaling.
70. Do not tune further on VGG16-CONV seeds 0--2. After a new mechanism passes its mechanics and planted-signal tests, use fresh seeds or another workload for the decision gate.
71. [x] Implement opt-in bounded proxy correction and relative-cost state, selected-reward replay, checkpoint compatibility checks, and cached shared pool lookups. Add synthetic-learning and paired initial-proxy tests. See `RESIDUAL_BANDIT.md`; real performance is not yet established.
72. Run `scripts/run_kaggle_residual_bandit_gate.sh short` on VGG16-CONV seeds 3--5. Compare learned residual, proxy-greedy, fixed SA and random at 20,001 calls each. Advance only on the declared directional rule; do not retune using these held-out results and continue calling them held out.
73. [x] The seed 3--5 residual gate failed against proxy-greedy: one win, two losses, and -6.907% mean improvement. It beat fixed SA 3/3 by 23.788%. Do not extend this selected-only learner unchanged.
74. [x] Add opt-in whole-pool labels with exact objective accounting. A supervised pool charges all 16 candidates to the fixed budget, is selected before their costs are known, and adds every exact gain to replay. The 45-test suite and a four-method 65-call smoke pass.
75. Run the pool-supervised residual gate on fresh VGG16-CONV seeds 6--8 with stride 32 and 20,001 calls/method. Require positive mean and at least 2/3 wins against both proxy-greedy and fixed SA before any extension.
76. [x] Complete the stride-20 full-information operator gate on VGG16-CONV seeds 0--2. Mechanics and exact accounting passed, but learning beat the matched transition-throttled proxy in only one seed and improved mean cost by 0.563%, so the two-win/1% gate failed. The policy chose proxy on 52,030 of 52,061 decisions; do not extend this configuration.
77. [x] Implement the prespecified selected-action follow-up: discounted Cholesky LinTS, top-two cheap proxy screening, one true evaluation per step, and separate signed/outcome-aware reward variants. Preserve ordinary unthrottled embedded Proxy-SA as the decisive control and add reproducibility/accounting tests.
78. [x] Run `scripts/run_kaggle_proxy_gated_lints_gate.sh` on VGG16-CONV seeds 0--2 at 20,001 calls per method. Outcome reward lost all pairs and averaged -6.587% versus unthrottled proxy. Signed reward won two pairs but averaged -0.165%. Both failed the required +1% mean gate.
79. [x] Close one-step contextual selection among the existing four operators. Do not tune discount, Thompson scale, proxy slate, or reward weights on these seeds. Preserve unthrottled embedded Proxy-SA as the strong training-free result and mandatory future control.
80. [x] Implement the first delayed-credit RL+SA gate. The policy holds an operator strategy for 64 proposals, receives segment-level best-cost improvement, and is compared with a matched frozen uniform-macro control and unthrottled Proxy-SA at equal true-call budgets. Exact-budget and reproducibility tests pass.
81. [x] Run `scripts/run_kaggle_macro_operator_gate.sh` on fresh VGG16-CONV seeds 9--11. Learned macro selection beat its frozen macro control in two seeds by 0.816% on average, but beat Proxy-SA once and averaged -2.297%. The gate failed with valid mechanics.
82. [x] Close the current four-operator learning family. Preserve immediate, full-information, Thompson, and delayed-credit methods as negative ablations; do not tune their horizon, reward, discount, or sampling scale on completed seeds.
83. Treat unthrottled embedded Proxy-SA as the current practical optimizer. Any further RL+SA proposal must introduce new information through a topology-aware representation or amortized training across distinct placement instances, with Proxy-SA retained as the advancement baseline.
84. [x] Add a portable `topology16` proposal representation, true frozen-checkpoint bandit inference, and optimizer-state continuation for explicit cross-workload training. Preserve matched proxy and uniform controls and exact true-call accounting.
85. [x] Run `scripts/run_kaggle_amortized_bandit_gate.sh`: train on AlexNet-FC and AlexNet-CONV, then freeze and test on ResNet50-CONV seeds 20--22. All training and held-out call accounting passed, and the checkpoint was genuinely frozen at hold-out; nevertheless every method reached 588.784 microseconds in all three seeds. Zero wins and 0.000% mean improvement failed the two-win/+1% rule. Close this exact transfer protocol; do not add epochs merely to pursue this null result.
86. [x] Add a matched Proxy-SA versus uniform headroom preflight to the amortized runner. It stops before training if the chosen target cannot distinguish its decisive controls at the preflight budget.
87. Run the VGG16-CONV transfer gate: train sequentially on AlexNet-FC and AlexNet-CONV, then freeze the checkpoint on VGG16-CONV seeds 20--22. Advance only on at least two wins over both controls and +1% mean improvement over Proxy-SA. This target is chosen because prior tests showed proposal-selection headroom, not because of a result from this gate.
88. [x] Add a preregistered Proxy-SA headroom-certification runner. It reports initial cost, sensitivity span, compute-floor gap, and short (20,001-call) versus long (100,001-call) Proxy-SA results on AlexNet-CONV, VGG16-CONV, and ResNet50-CONV. A workload is eligible for further RL only if the median short result remains at least 2% above the long reference.
89. Run `scripts/run_kaggle_headroom_certification.sh` before the VGG16 transfer gate. Do not run RL on workloads that fail its 2% unresolved-headroom criterion.
90. [x] Complete headroom certification at `305999c` (archive SHA-256 `3e504b35d3c53138e6267abe5153e48648433fa881d47fa6be3d1a0985cb191f`). AlexNet-CONV and ResNet50-CONV had 0.000% median unresolved headroom and are ineligible. VGG16-CONV passed with 6.709% median unresolved headroom and a 40.706% median reference gap above the compute floor.
91. Run the one eligible frozen transfer gate only: AlexNet-FC then AlexNet-CONV training, frozen VGG16-CONV hold-out seeds 20--22, with Proxy-SA and uniform controls. Retain the prespecified two-win and +1% mean-versus-Proxy-SA advancement rule.

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
