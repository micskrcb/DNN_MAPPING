## 24 October: horizon result and independently supplied historical300k run

Horizon seed18 completed;76 tests passed. Matched1186 candidates: SA5.43232µs, RL+SA5.55344µs, both trained DDPG variants/frozen/random5.81808µs. Bandit beats reconstructed DDPG4.54858% but loses to SA2.23%. All DDPG bests remained random baseline. Gamma.999 final deterministic6.3616µs versus.98 at9.696 and frozen7.856; this improves the final actor output but not retained best. Long-gamma final logged critic loss≈106million,97.2% repairs; default98.8% repairs. Do not promote gamma as a solved fix. Evidence in `artifacts/ddpg-horizon-20261024/`.

New historical archive `dnn-mapping-300k.zip`: AlexNet-FC, seed0, executionVVA1/partitionVVA1,932 tasks,300001 candidates,1170 updates, RTX3060 Laptop GPU,3470.12s. Initial24.8184µs→22.84912µs (7.93476% lower). Logs still improve from23.50096 at102k to22.84912 at300k. Compute-only floor22.4µs leaves at most1.96559% reduction from the final cost within this evaluator. This is not the VGG16/VVA16 protocol. No matched controls, so it does not isolate learning. Reported revision1e9dee2 is dirty; `positive_delta` metadata conflicts with a negative final replay-target mean, consistent with historical reward-label issues but unverified for this modified checkout. Preserve as historical evidence, not a clean reproducible benchmark. Source/diff needed for exact reproduction. No checkpoint was loaded.

Next workflow prioritizes RL+SA: seed19,100000 proposals, strongest epsilon.02 reference versus faster decay to.02 (decay fraction.5 instead of1), plus fixed SA. Same other settings. Then a short seed20 DDPG numerical test:30 placements each, gamma.999, no clipping versus global actor/critic gradient clipping1,128 baseline+3 diagnostics=161 search candidates each. Added Q/target/TD/reward magnitude and pre-clipping gradient diagnostics. These are experimental opt-in changes, not promised improvements. Historical300k run supports continued optimization at larger budgets but does not justify transferring its outcome to this workload.

See [LEARNING_FOLLOWUP.md](LEARNING_FOLLOWUP.md).77 tests passed locally. No external code copied. Published-method5% target remains unmet as a verified claim.

## 24 October: preliminary DDPG win; diagnose reward horizon

Evidence: `artifacts/ddpg-comparison-20261024/`, source f423eaf;74 tests passed. Timing pilot selected11 epochs×30=330 training placements. Inclusive budget1334=1000 random baseline+330 training+4 deterministic diagnostics, matched across all methods. Costsµs: RL+SA5.27056; fixed SA5.31840; DDPG5.83376; random5.83376. RL+SA gains9.65415% versus our DDPG reconstruction and0.89952% versus SA. DDPG's best source remained random_baseline. CUDA DDPG10132.3s versus bandit54.28s. This does not establish5% over published DDPG: one seed, limited training and unvalidated reconstruction.

DDPG made211797 updates, but logged1878–1911 collision repairs among1924 tasks; noisy actions had42–49% saturation. Existing gamma.98 with batch_z3 gives642 steps and first-step terminal multiplier .98^641=2.37637e-6. Sparse delayed reward is therefore heavily discounted. This motivates an experiment, not a proven diagnosis of failure; collision handling, saturation and function approximation may also matter.

New opt-in `--ddpg_gamma`(.98 default) and `--ddpg_disable_learning`; checkpoint fingerprints reject differing settings, while default legacy fingerprints remain compatible. Metadata reports gamma, placement horizon, terminal-discount multiplier and learning status. No repair change or action/reward rewrite. New short real-training tests check discount and zero-update frozen control.

Next: fresh seed18,180 training placements per DDPG arm,1000 baseline+6 diagnostics=1186 inclusive candidates. Compare default.98, long-horizon.999 and frozen.98. At.999 first-step multiplier≈0.5266. Identical repair and train_every1 for trained arms. RL+SA, fixed SA, random get1186 evaluations. Gamma.999 is an experimental extension, not asserted to be the paper's parameter. See [DDPG_HORIZON_DIAGNOSIS.md](DDPG_HORIZON_DIAGNOSIS.md). No new performance result yet.

## 23 October: stronger exploitation wins rapid screen and full follow-up

Source archive and reports: `artifacts/online-rapid-20261023/`. Commit2d21749. Both stages complete;72 tests passed per stage. Seed15 screen(25001 evaluations): stronger exploitation+0.88196% over reference; topology-0.50593%; pool regression-1.36738%; ranking-5.23023%. Topology also cost roughly2.35× reference runtime. Do not promote ranking or topology from this result; only the winning exploitation variant was run fully.

Fresh seed16 full run(100001 evaluations): exploitation4.33888µs, reference4.37376µs, fixed SA4.40832µs. Improvements0.79748% over reference and1.57520% over SA. Both learners389 updates. Final epsilon0.02503 versus0.10462. Preserve stronger exploitation as candidate. Do not compare this single-seed SA percentage directly with the earlier two-seed average. No published-method comparison yet.

Next: inclusive-budget local DDPG reconstruction comparison, seed17. Fix missing baseline accounting: existing DDPG candidate count excludes baseline search although baseline can supply the returned best placement. New `search_evaluations_including_baseline` includes baseline, training and deterministic diagnostics; historical count remains unchanged for compatibility. Track actual optimizer updates this invocation and persist baseline count in new checkpoints; old checkpoints with unknown count report null rather than fabricated totals.

A separate10-placement timing pilot automatically selects up to100 epochs against a conservative4-hour DDPG target; it is not reused as evidence. Runner uses DDPG paper_cnn, sparse rewards, batch_z3, train_every1,up to100 epochs×30 placements=3000,1000 baseline trials and periodic retained deterministic diagnostics. Inclusive search budget is computed from the actual selected epochs; at the cap it is4030. RL+SA/fixed SA/random receive that same actual budget. Identical VGG16-FC partition/hardware/timing/routing. DDPG auto-selects CUDA if available; bandit/SA CPU. This is a preliminary reconstruction comparison, not paper reproduction, convergence evidence or a5% published-SOTA claim. See [DDPG_MATCHED_COMPARISON.md](DDPG_MATCHED_COMPARISON.md).

## 23 October: both exploration confirmation seeds pass

Seed13 lower-exploration cost4.38688µs; seed14 cost4.34928µs. Completed successfully with100001 placement evaluations per arm. Both beat previous online RL+SA, same-pool uniform, proxy and fixed SA. Mean paired gains across confirmation seeds13–14: **0.49213% over previous online,2.24199% over uniform,2.86536% over proxy,1.92062% over fixed SA**;2/2 wins for every comparator. Seed12 remains a separate exploratory result.

The prespecified directional confirmation rule passes, with only two seeds and one workload. Keep epsilon0.1 as the development reference. This does not establish5% improvement over published DDPG/SOTA; neither is included in these runs. Evidence and combined summary: `artifacts/online-exploration-confirmation-20261023/`.

Continue the requested accelerated development workflow: seed15 screen and seed16 full-budget follow-up. Do not run seeds13–14 again. Previous entries describing confirmation as pending/deferred are historical and superseded by these received results.

## 23 October: accelerated development replaces immediate confirmation

User requested faster algorithm development. Seed12 remains the latest measured result: lower-exploration RL+SA4.33616µs,0.83428% over previous online mapper and2.63697% over fixed SA. No new Kaggle outcome has arrived; no new gain is claimed. Seeds13–14 confirmation runner remains available but is no longer the next requested run.

New automated development run: seed15,25000 objective proposals per arm, five arms: working epsilon0.1 reference; stronger exploitation endpoint0.02; topology16 features; pool regression; within-pool pairwise ranking. Pool-regression and ranking evaluate a full16-candidate pool every64 proposal steps, charging every label to the same objective budget. Ranking adds weight0.25 softplus ordering loss to the existing signed regression loss. Comparisons are within one common-state pool, never arbitrary cross-state move pairs. It retains at most256 pools and samples up to8 per update. No extra objective calls occur during training.

Automatically select the lowest-cost challenger and run it on fresh seed16 at100000 proposals with working reference and fixed SA; if ranking wins the screen, also retain pool regression in the full run. Run the best challenger even if the short screen is negative: this is an aggressive developmental test, not an advancement or publication gate. Keep screen and full-stage results separate. Both budgets include initialization in reported evaluations (+1). Extra pool evaluations reduce accepted search opportunities and may change update counts; report these tradeoffs.

New code/defaults are opt-in; existing learner and confirmation runners remain intact.72 local tests passed. Full run has not been measured. Details and execution: [ONLINE_RAPID_DEVELOPMENT.md](ONLINE_RAPID_DEVELOPMENT.md). Published DDPG comparison remains outstanding; this run cannot establish a5% win over a published method.

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

# DDPG Learning Failure: Research Findings and Actions

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


Updated: 17 October 2026

## External operator-selection review (17 October 2026)

The current internet review and supplied Gemini Deep Research report are
audited in
[`EXTERNAL_RESEARCH_REVIEW_2026-10-17.md`](EXTERNAL_RESEARCH_REVIEW_2026-10-17.md).
Accepted recommendations include an unthrottled Proxy-SA control,
selected-action discounted LinTS, cheap two-proposal proxy screening,
outcome-aware reward as a separate ablation, and anytime best-cost metrics.
The report incorrectly describes the active selector as stationary
infinite-memory LinUCB; the code already uses phase-specific models and 0.995
discounting. It also overstates the cost and theoretical rejection of the
stride-20 full-information diagnostic. The current gate remains useful, but a
positive result must also beat ordinary unthrottled Proxy-SA.

The direct public-code and training-scale comparison is recorded in
[`GITHUB_IMPLEMENTATION_AUDIT.md`](archive/GITHUB_IMPLEMENTATION_AUDIT.md). Its main
finding is that the official RL-Based-SA configuration uses 10.24 million
transitions across 256 parallel problem instances, whereas the extended
PPO-ASA gates use 4,068 policy transitions from one fixed instance. Public
core-placement code does not include Wu et al.'s missing multi-chip simulator.
A maintained PPO library can improve reliability and parallel collection, but
no available repository is a drop-in replacement for this project's DNN
partition, routing, timing, and legal-placement environment.

## Full-information operator result (17 October 2026)

The VGG16-CONV gate at revision `8a60616` passed all mechanics checks. Each
learned run used 20,001 true objective calls and obtained 3,472 exact labels
from 868 same-state four-operator pools. Relative to the matched proxy control,
the learned costs improved by 1.689%, 0%, and 0% across seeds 0--2. The mean
was 0.563% with one win, below the prespecified two wins and 1% mean rule.

The labels revealed a more specific limitation than "insufficient epochs."
The trained policy selected proxy for 52,030 of 52,061 total decisions. The
supervised signed-reward totals consistently ranked proxy above bottleneck,
swap, and relocation proposals. More iterations of the same mechanism would
mostly reinforce proxy selection while the four-call collection steps reduce
the number of sequential state transitions.

The next bounded hypothesis restores one true evaluation per step. Discounted
Cholesky LinTS ranks operators, the cheap communication proxy chooses between
the top two generated proposals, and only the executed operator is updated.
Signed immediate reward and an outcome-aware reward are prespecified parallel
variants. Ordinary unthrottled embedded Proxy-SA is the decisive control. The
method advances only with at least two of three paired wins and at least 1%
mean improvement over that control.

## Proxy-gated LinTS result (17 October 2026)

The one-evaluation-per-step follow-up passed its mechanics checks but failed
its performance gate. Against ordinary unthrottled embedded Proxy-SA, the
outcome-reward variant produced paired improvements of -8.079%, -10.035%, and
-1.647%, averaging -6.587% with no wins. The signed-reward variant produced
+0.338%, +0.237%, and -1.070%, averaging -0.165% with two wins. The required
mean improvement was +1%.

This is evidence against the current action space and credit mechanism rather
than evidence that no learned annealing method can work. All learned runs made
19,968 updates. The top-two proxy screen overrode the sampled primary operator
about 10,500--11,350 times, and the executed proposal mix was materially
different from pure proxy. Outcome-aware rewards selected non-proxy moves more
often and harmed the final result. Signed immediate rewards remained close to
the strong proxy baseline but did not improve it.

Together with the full-information result, this closes one-step selection
among the existing `proxy`, `bottleneck`, `swap`, and `relocation` operators.
Further tuning on the same workload and seeds would be post-hoc. A credible
next RL+SA experiment needs multi-step actions and delayed return, trained
across independent chains or workloads, while retaining unthrottled Proxy-SA
as the decisive non-learned control.

## Delayed macro-action result (17 October 2026)

The fresh-seed VGG16-CONV gate used 64-step macro-actions and segment-level
best-cost reward. Learned selection improved on the frozen uniform-macro
controller by 4.377%, -3.674%, and 1.746% across seeds 9--11, averaging
+0.816% with two wins. This is a small directional learning signal.

The decisive Proxy-SA comparison was negative: -0.757%, +2.097%, and -8.232%,
averaging -2.297% with one win. Every learner completed 312 delayed updates,
used all four actions, and accumulated finite nonzero rewards, so the failure
cannot be attributed to a frozen model or missing feedback. The macro policy
changed search behavior but did not reliably improve it.

The evidence now covers selected-action immediate credit, full-information
immediate credit, proxy-gated Thompson sampling, and delayed 64-step credit.
None beats unthrottled embedded Proxy-SA under the frozen gate. The honest
conclusion is that the present hand-designed operator family adds no reliable
learned advantage. Future RL work must change the representation or train an
amortized policy across genuinely different placement instances; it should not
retune this family on the completed seeds.

### Amortized topology-aware transfer result (2026-10-17)

The first explicit amortization test did change both representation and
training instances: `topology16` was trained on AlexNet-FC, fine-tuned on
AlexNet-CONV, then frozen for ResNet50-CONV. Its mechanics were verified:
both stages made 77 updates, the checkpoint loaded for hold-out inference, and
each of the nine held-out runs consumed exactly 20,001 true calls. Yet all
three controllers (frozen checkpoint, proxy-greedy, and uniform) ended at
588.784 microseconds in every one of the three held-out seeds. This supplies
no transfer signal. It closes this particular two-source/one-target protocol;
it does not establish that every possible amortized RL+SA design is impossible.

### Independent audit action (2026-10-17)

An external audit of the source and archived VGG16-FC runs agreed that the
remaining issue is likely objective headroom and candidate-label information,
not checkpoint plumbing. Its external repository and license claims were not
treated as verified evidence. The actionable, testable recommendation has been
adopted as a pre-registered protocol: compare 20,001-call Proxy-SA with a
100,001-call reference across fixed CONV workloads, record floor and
sensitivity diagnostics, and require at least 2% median unresolved headroom
before running further RL. This is a screening criterion, not a claim that a
workload failing it has a wrong simulator.

The completed certification supports the screen. AlexNet-CONV and
ResNet50-CONV each had 0.000% median short-to-reference gap despite broad
random-placement spans, showing that random sensitivity alone does not certify
late-search headroom. VGG16-CONV retained 6.709% median unresolved headroom,
with its 100,001-call reference 40.706% above the compute floor. Therefore,
only VGG16-CONV is eligible for the next controlled learned-transfer test.

## Purpose

This document records the investigation prompted by the 3,000-placement
AlexNet-CONV Kaggle run, the independent Gemini research response, and a
follow-up search of papers, official implementations, GitHub discussions,
Stack Exchange, PyTorch discussions, and placement-specific research. It
separates observed evidence from hypotheses and records which recommendations
have been implemented.

The goal is to reproduce Wu et al.'s DDPG method honestly while developing a
separately labelled extension that can learn under the reconstructed simulator.
The extension must not be reported as the paper's original algorithm.

## Evidence from the failed run

The original 3,000-placement diagnostic produced the following observations:

- trained best cost: 37.6356 microseconds;
- matched untrained exploratory control best: 37.4466 microseconds;
- the trained best therefore did not demonstrate a learning advantage;
- mean collision repairs were about 176 of 183 assigned cores;
- deterministic collision repairs were about 180 of 183;
- the deterministic actor often intended only a handful of unique locations;
- actor outputs increasingly saturated near `-1` or `1`;
- critic loss reached approximately `1.48e12`;
- none of the retained best candidates came from the zero-noise deterministic
  policy;
- 86.3% of the best sampled latency was fixed computation, leaving about 13.7%
  as placement-sensitive communication headroom.

These measurements support policy collapse and critic divergence. A decreasing
best-so-far curve by itself is insufficient evidence of learning because it can
be produced by noisy exploration.

## Findings from the original paper

The source paper explicitly specifies a continuous DDPG actor, flooring to grid
coordinates, nearest-free Manhattan collision repair, terminal reward
`sqrt(B) - sqrt(L(P))`, discount factor 0.98, actor learning rate 0.0002,
critic learning rate 0.001, minibatch size 64, and 30 complete placements per
declared epoch. It reports convergence at roughly 300,000 to 400,000 evaluated
placements.

The paper leaves several material values unpublished, including the batch size
`z`, replay capacity, target update coefficient, OU-noise parameters, exact
CONV/FC masks, complete per-layer partitions, GRS timing, and the complete
simulator. It states that CONV and FC are placed in separate masked regions and
that different workloads can occupy different numbers of chips.

Source:

- Nan Wu et al., [Core Placement Optimization for Multi-chip Many-core Neural
  Network Systems with Reinforcement Learning](https://doi.org/10.1145/3418498)

## Gemini response: accepted findings

The following parts of the Gemini audit agree with the run evidence and the
primary literature:

1. The requested continuous action and the collision-repaired executed action
   were misaligned in replay.
2. DDPG is vulnerable to deterministic sparse-reward deadlock.
3. Sixty-one decisions for AlexNet-CONV make one-step terminal credit
   propagation difficult; the FC horizon is much longer at `z=3`.
4. Uniform replay under-samples the most informative transitions.
5. Zero-noise deterministic evaluation is required to distinguish policy
   learning from noisy search.
6. Demonstrations, prioritized replay, multi-step or complete-episode targets,
   legal-action selection, conservative critics, and delayed actor updates are
   reasonable experimental remedies.
7. A small controlled ablation should precede a paper-scale run.

## Gemini response: corrections and qualifications

Several statements in the response were too strong or technically inaccurate:

- DDPG does not backpropagate through the environment's `floor()` operation.
  The actor receives gradients through a learned differentiable critic. The
  actual problem is that flooring is discontinuous and many-to-one, while
  collision repair can make the stored action differ from the action that
  caused the transition.
- An untrained noisy policy is not mathematically guaranteed to beat a trained
  policy. That occurred in this run and is evidence against learning, but it is
  not a universal theorem.
- The 41.4168-microsecond figure was the minimum of a 64-sample sensitivity
  preflight. The 1,000-trial reward baseline was about 40.8497 microseconds.
- Removing BatchNorm is not universally required. The paper specifies
  BatchNorm, and careful treatment can work. The implementation now evaluates
  target networks with fixed running statistics and synchronizes target
  buffers.
- A straight-through estimator provides a biased surrogate gradient; it does
  not guarantee correct gradients. Standard Wolpertinger-style actor training
  does not require differentiating through the environment.
- The paper does not publish a numeric value for `z`, so no particular batch
  size can be described as the uniquely paper-correct value.
- Several citations in the Gemini response were generic listing pages,
  secondary summaries, or unrelated applications. The algorithmic direction
  is useful, but publication text should cite the primary works below.

## Internet and implementation research

### DDPG is a continuous-action algorithm

The maintained Stable Baselines discussion states that DDPG supports continuous
`Box` actions and points users with very large discrete action spaces toward the
Wolpertinger architecture. Discretizing DDPG output is possible, but it creates
problems the algorithm was not originally designed to solve.

- [Stable Baselines issue 37](https://github.com/hill-a/stable-baselines/issues/37)
- Lillicrap et al., [Continuous Control with Deep Reinforcement
  Learning](https://arxiv.org/abs/1509.02971)

### Sparse deterministic environments can deadlock DDPG

Matheron, Perrin, and Sigaud give a formal analysis of poor fixed points in
deterministic sparse-reward environments. This directly matches the observed
actor saturation and lack of deterministic-policy improvement.

- [The Problem with DDPG: Understanding Failures in Deterministic Environments
  with Sparse Rewards](https://arxiv.org/abs/1911.11679)

### Wolpertinger aligns continuous proposals with discrete execution

Wolpertinger lets an actor produce a continuous proto-action, retrieves nearby
discrete candidates, and lets the critic select which candidate is executed.
The replayed action can therefore describe the actual discrete transition.

- Dulac-Arnold et al., [Deep Reinforcement Learning in Large Discrete Action
  Spaces](https://arxiv.org/abs/1512.07679)

### Demonstrations and prioritized replay address exploration scarcity

DDPG from Demonstrations combines demonstration and agent transitions in
prioritized replay. Demonstrations are retained instead of being discarded by
the replay ring. Prioritized Experience Replay samples informative transitions
more frequently and uses importance weights to reduce sampling bias.

- Vecerik et al., [Leveraging Demonstrations for Deep Reinforcement Learning on
  Robotics Problems with Sparse Rewards](https://arxiv.org/abs/1707.08817)
- Schaul et al., [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

### Twin critics and delayed actor updates reduce value-estimation error

TD3 uses clipped double-Q targets, delayed policy updates, and target-policy
smoothing. Guided DDPG uses the applicable parts with complete-episode return
targets: two critics, conservative legal-candidate scoring, and delayed actor
updates. It should not be called a complete TD3 implementation because target
policy smoothing is not used.

- Fujimoto, van Hoof, and Meger, [Addressing Function Approximation Error in
  Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)

### Legal masking is well supported for policy-gradient methods

Invalid-action masking has a formal policy-gradient justification and becomes
more important as invalid actions dominate the nominal action space. This
supports a masked categorical PPO fallback if guided DDPG still fails.

- Huang and Ontanon, [A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms](https://arxiv.org/abs/2006.14171)

### Later placement systems favor discrete masked policies

Google's chip-placement formulation defines the action as a valid grid cell and
uses a feasibility mask. MaskPlace predicts a probability matrix over grid
cells, removes infeasible positions before sampling, and introduces denser
placement feedback. Both avoid repairing almost every actor output after the
decision.

- Mirhoseini et al., [Chip Placement with Deep Reinforcement
  Learning](https://arxiv.org/abs/2004.10746)
- Lai, Mu, and Luo, [MaskPlace](https://arxiv.org/abs/2211.13382)

A later core-placement paper uses PPO, graph convolution, and a one-step whole
placement policy. Its public code is useful as a reference but currently states
that multi-chip execution is unavailable, and its single-chip environment still
repairs overlaps. It should therefore not be copied as a validated replacement.

- Myung et al., [Policy Gradient-Based Core Placement Optimization for
  Multichip Many-Core Systems](https://doi.org/10.1109/TNNLS.2021.3117878)
- [Associated public repository](https://github.com/WOOSHIK-M/Core_Placement_with_Reinforcement_Learning)

## Implemented actions

The paper-aligned `--algo ddpg` remains available as the reproduction control.
The separately labelled `--algo ddpg_guided` now implements:

- compact state maps containing only the active masked region;
- legal collision-free candidate batches near the actor's proto-action;
- conservative twin-critic ranking of legal candidates;
- storage of the exact executed action;
- a best-available demonstration selected from ASA, the random baseline, or a
  resumed checkpoint;
- permanent demonstration transitions in replay;
- demonstration-aware proportional prioritized replay;
- discounted complete-episode return targets for every trajectory step;
- behavior-cloning pretraining and a decaying cloning weight;
- Huber critic losses, gradient clipping, and delayed actor updates;
- deterministic zero-noise evaluation and explicit evaluation accounting;
- initial uniformly random legal placements to broaden replay support before
  relying on actor proposals;
- independent seeded action and replay random streams, so learning does not
  change the random exploration sequence used by its paired control;
- `--guided_disable_learning`, which supplies a matched control with the same
  environment, demonstration generation, legal selection, exploration schedule,
  and candidate accounting but no pretraining or gradient updates;
- finite-Q, gradient, collision, action-saturation, replay, and policy-source
  diagnostics;
- resumable model/optimizer checkpoints and append-only main diagnostics.

## Staged experimental plan

## Five-seed guided-DDPG decision (8 October 2026)

The completed extensive Kaggle archive contains five trained/control pairs,
3,000 online placements per condition, all reports, all diagnostics, and valid
completion markers. Mechanics and numerical-stability checks passed, but the
trained policy lost every paired late deterministic comparison. Its aggregate
late deterministic mean was 39.0857 microseconds versus 38.7322 microseconds
for the frozen control, a regression of 0.3535 microseconds. The exploratory
paired 95% interval for improvement was -0.8766 to +0.1696 microseconds.

This closes Gate 2 for guided DDPG as a negative result. Occasional good
deterministic checkpoints do not reverse the conclusion because choosing the
best checkpoint after inspecting 100 diagnostics is selection bias, and the
trained best complete solution was also slightly worse in aggregate. Further
guided-DDPG tuning is paused.

The repository now contains `--algo ppo_masked`, a separate experimental
categorical PPO policy. It chooses one exact unused physical core for the next
logic core, applies a pre-softmax legality mask, and therefore has no repair
path. PPO uses clipped policy updates, generalized advantage estimation,
entropy regularization, a value baseline, gradient clipping, deterministic
diagnostics, and an exactly matched `--ppo_disable_learning` control. The
paper-aligned DDPG remains unchanged and available for reproduction tables.

## Five-seed masked-PPO decision (9 October 2026)

The extensive AlexNet-CONV archive contains all ten expected runs at commit
`e3576b0`: five trained and five frozen controls, each with 3,000 complete
placements, 1,000 random-baseline trials, 101 deterministic evaluations, and
valid completion markers. Initial deterministic costs match exactly within
each seed, every run has zero repairs, and all PPO diagnostics are finite.

Trained PPO beat its paired frozen deterministic policy in all five seeds. The
aggregate late deterministic mean was 42.6118 microseconds trained versus
45.3588 microseconds control, a 2.7471-microsecond or 6.06% improvement. The
exploratory paired 95% interval for improvement was 1.0972 to 4.3969
microseconds. A wider final-300-placement window gives the same conclusion.
This passes the learning-signal gate for this reconstructed workload.

The result does not yet establish a better optimizer or paper-level
performance. Best retained solutions averaged 39.9989 microseconds trained and
40.5094 microseconds control, only a 1.26% advantage, because random rollout
search is already strong. AlexNet-CONV occupies one chip and has zero off-chip
traffic. The next gate is therefore a matched-budget baseline table followed
by a paired masked-PPO run on AlexNet-FC or VGG16-CONV, where placement spans
multiple chips.

## AlexNet-FC multi-chip decision (9 October 2026)

The 3,000-placement one-seed extension used 4,101 complete-placement
evaluations for trained PPO, its frozen control, random search, and ASA. All
mechanics passed on the 932-core, four-chip FC region. The trained late
deterministic policy was 24.5048 microseconds versus 24.6424 for the frozen
control, a 0.558% directional improvement. Only 60 of 100 deterministic
checkpoints beat the control, the curve remained non-monotonic, and the trained
stochastic late-window mean was almost identical to control.

Training did not improve the retained optimizer result: trained PPO ended at
24.1080 microseconds and the frozen stochastic policy at 24.1072. Random search
was 24.1155, while ASA reached 23.4318 microseconds, 2.80% below trained PPO.
ASA improved steadily throughout its budget. Consequently, unchanged masked
PPO should not be scaled to five FC seeds. The useful next ablation is a
strictly budget-matched ASA/PPO hybrid or ASA-demonstration policy, always
compared with ASA alone; otherwise the extra RL machinery has no demonstrated
optimizer value on this fixed instance.

## Five-seed ASA baseline and hybrid research audit (9 October 2026)

The five-seed AlexNet-FC archive is complete and internally consistent at
revision `b147b14`. ASA best costs were 23.43184, 23.52528, 23.49808,
23.44928, and 23.59600 microseconds. Matched random-search costs were 24.11552,
24.12768, 24.16976, 24.07152, and 24.16800. ASA won every paired seed; its
mean was 23.50010 microseconds versus 24.13050, a 2.61% advantage. The paired
absolute advantage averaged 0.63040 microseconds, with an exploratory 95%
interval of 0.57216 to 0.68864. The sequential baseline was 24.92032
microseconds.

The official [RL-Based-SA implementation](https://github.com/nathanqiu07/RL-Based-SA-Public)
supports the proposal-learning direction but corrects a material error in the
Gemini survey. Qiu and Liang train the neighbor proposal with PPO, include
energy change in the state, and provide an LSTM variant; the Metropolis rule
and exponential cooling schedule remain fixed. Their implementation targets
knapsack, bin packing, TSP, and continuous test functions, so its training loop
and factorized proposal patterns are references rather than drop-in placement
code.

The survey's sequence-pair, B*-tree, Cartesian-overlap, and HPWL discussion
describes physical macro floorplanning. This project assigns fixed-size DNN
tasks to distinct positions in a predefined core grid and optimizes routed
communication latency. Replacing the placement representation with a B*-tree
would therefore change the problem rather than repair the policy. Claims about
combining BOPO, Wolpertinger, MCTS, and learned surrogates are plausible future
ideas but are not demonstrated for this evaluator by the cited evidence.

The next controlled implementation will let PPO choose a legal local proposal
while keeping the existing ASA acceptance and temperature logic fixed. It will
start from the same placements as a uniform-proposal ASA control and share the
same total objective-evaluation budget. This isolates whether learning improves
the neighborhood distribution. Temperature control, recurrent history, and
offline preference learning should be separate later ablations, not combined
in the first test.

That first implementation is now present as `--algo ppo_asa`. It uses the
reference method's core separation without copying its problem-specific code:
the policy selects a proposal, the existing Metropolis rule decides acceptance,
and the existing ASA controller changes temperature. The action is a
permutation-invariant choice among 16 legal candidates rather than a
932-by-1,024 flat action. A frozen zero-logit policy is exactly uniform, so the
learning comparison does not depend on a weak random neural initialization.

The 1,001-evaluation local smoke is intentionally reported as a provisional
gate. Learned PPO-ASA beat its uniform proposal control by 0.47% (23.9986
versus 24.1123 microseconds), but ordinary ASA reached 23.9629 and remained
0.15% better. PPO updates were finite and the learned condition recorded 41
improving moves versus 39 for its control. Roughly 89% of selected proposals
were objective-neutral, confirming that bottleneck-stage credit remains the
central difficulty. The full 4,101-evaluation one-seed gate must establish
whether additional online updates create a real optimizer advantage.

The 4,101-evaluation gate rejected this first learned neighborhood. The trained
run ended at 23.5602 microseconds, behind the uniform control at 23.4213 and
ordinary ASA at 23.4318. The policy stayed close to uniform: final entropy was
2.76893 against a 2.77259 maximum, final approximate KL was `6.71e-6`, and the
clip fraction was zero. It found 107 improving moves, fewer than the control's
114, while 3,145 of its 4,100 proposals were objective-neutral. This pattern
points to insufficiently informative proposal sets and credit, not evidence
that a well-separated learned proposal distribution needs only a longer run.

The next defensible ablation should change one factor: construct candidate
pools around tasks participating in the current bottleneck stage for both the
trained and uniform conditions. PPO may then rank destinations or swap partners
within that matched pool. Exact delta labels can also be tested, but every
candidate evaluated by the true objective must count against the common budget.
An LSTM or a larger training budget comes later because memory and duration do
not address a neighborhood in which most actions have no measured effect.

That single-factor ablation is implemented as
`--ppo_asa_focus_bottleneck`. A local 1,001-evaluation smoke reduced the neutral
rate from roughly 89% in the earlier short run to 72.8%, and the improving-move
rate reached 7.5%. The learned condition beat the identically focused uniform
control by 0.40% (24.0627 versus 24.1598 microseconds), but ordinary ASA still
won at 23.9629. The result is a learning-only signal. One focused 4,101-
evaluation gate is warranted to see whether the advantage persists; five seeds
remain conditional on beating ASA as well.

## Multi-chain training implementation (9 October 2026)

The training-scale response to the public-code audit is implemented without
copying external source. One PPO proposal policy is retained across repeated
independently initialized placement chains. Restarts terminate generalized-
advantage estimates, reset the placement and annealing state, and retain the
shared model and optimizer. A half-focused candidate pool supplies both
bottleneck-relevant and global moves. Every restart initialization is counted
as a true objective evaluation, and ordinary ASA receives the exact same total.

The local mechanics smoke used eight 64-proposal chains. All mechanics checks
passed, both PPO conditions received 520 objective evaluations, and four PPO
updates were finite. Trained and frozen PPO-ASA tied at 24.31712 microseconds;
ASA reached 24.05376. This is expected to be inconclusive at 512 proposals and
does not establish learning. The decision experiment is the 250,000-proposal
Kaggle `multichain-train` gate, which produces 1,954 chains and 251,954 total
objective evaluations for each of trained PPO-ASA, frozen control, and ASA.

That training-scale run completed at clean revision `5ea4c65`. Trained
PPO-ASA reached 23.96800 microseconds versus 24.08032 for its uniform frozen
control, a 0.466% advantage. It beat control at 3,413 of 3,906 matched
checkpoints, selected 6.544% improving moves versus 6.228%, and reduced final
entropy to 2.48776 from the uniform maximum 2.77259. This establishes a useful
one-seed learning signal. Ordinary ASA reached 23.14624 microseconds and was
3.55% better, so the optimizer gate failed.

The comparison exposed a protocol issue rather than an unstable PPO update:
the learned method was intentionally restarted every 128 proposals to collect
diverse training experience, while ASA refined one chain for 251,953 proposals.
The next test therefore loads the preserved model without further updates and
evaluates it on five unseen uninterrupted chains. A fresh uniform scorer and
ASA receive the same 4,101 objective evaluations in each seed. Training cost
and inference optimization cost are reported separately.

The holdout rejected the checkpoint. Frozen trained PPO-ASA lost every seed,
averaging 23.63050 microseconds versus 23.51504 for uniform and 23.50518 for
ASA. Paired exploratory intervals excluded zero in the wrong direction. A
five-seed local check at the original 128-proposal horizon was directionally
better: trained PPO averaged 0.385% below uniform and won three pairs, but the
interval included zero. Thus the checkpoint contains a weak short-horizon
preference and no transferable 4,100-step optimizer policy.

The official RL-Based-SA code uses immediate realized gain by default, as this
implementation does, but its training and evaluation annealing horizons match.
Its TSP configuration also learns over 256 newly generated instances per
epoch. The final bounded correction here therefore matches the 4,100-proposal
training and deployment horizons, normalizes progress within each chain, and
updates PPO at chain boundaries. No external code is copied. Failure of the
resulting frozen holdout closes PPO-ASA rather than prompting more tuning.

### Gate 1: implementation sanity

Run bounded synthetic and AlexNet-CONV smoke tests. Required conditions:

- zero collision repairs;
- exactly one unique executed physical core per logic core;
- finite actor loss, critic loss, Q values, and gradient norms;
- exact cost agreement when replaying a demonstration;
- checkpoint resume succeeds without deleting earlier main diagnostics.

This gate passes locally.

### Gate 2: learning, not search luck — passed for masked PPO on AlexNet-CONV

Masked PPO passed its trained-versus-frozen comparison at 3,000 placements over
five seeds. Guided DDPG failed the same gate. This conclusion concerns policy
learning on the reconstructed AlexNet-CONV objective, not paper reproduction.

### Gate 3: stronger placement-sensitive workload

AlexNet-CONV fits inside one 16-by-16 chip and therefore has no off-chip traffic
under the minimal whole-chip mask reconstruction. It is useful for debugging but
is a weak final benchmark. After Gate 2, evaluate AlexNet-FC or VGG16-CONV,
which span multiple chips and have more communication headroom. Because `z` is
unpublished, treat any larger batch size used to control the horizon as an
explicit ablation.

### Gate 4: paper comparison

Freeze the simulator and compare BS, RS, fixed SA, paper-aligned DDPG, ASA,
guided DDPG, and the no-learning control with declared candidate accounting.
Use at least five seeds and report mean, sample standard deviation, minimum,
maximum, and ratios normalized to BS. Do not claim exact numerical reproduction
because the paper's simulator and several timing parameters are unpublished.

### Gate 5: fallback research algorithm

If guided DDPG fails Gate 2, stop tuning it blindly. Implement a separate masked
categorical PPO policy that samples only legal cores, optionally initializes
from ASA behavior cloning, and uses either potential-based dense feedback or
whole-placement normalized advantages. This would be a new experimental method,
not the paper's DDPG reproduction.

## Publication boundary

The current work is an honest reconstruction plus a motivated experimental
extension. A publishable claim requires multi-seed evidence, ablations isolating
each added mechanism, matched candidate budgets, stronger multi-chip workloads,
and comparison against masked PPO or another modern discrete placement policy.
Until those experiments exist, the correct statement is that the implementation
is internally consistent and ready for learning validation—not that it has
already reproduced the paper's reported improvements.

## VVA literature-audit status (12 October 2026)

An external research report reviewed Tianjic and Simba after the VVA
sensitivity sweep. Its usable conclusion is narrow: none of the cited primary
sources was shown to publish the VVA additions/cycle used by Wu et al.'s
simulator. The report did not provide the requested page-level evidence for a
rate of eight and therefore does not authorize freezing that value.

Several stronger assertions in that report are excluded from the project
evidence. Wu et al. do not mention Noxim in the paper, and Section 4.1 states
that each time phase is measured by adding computation and communication
latency; this does not support a fully hidden VVA cost. The report also
described RL as changing partition grids even though the VVA sweep ran only
BS, RS, and SA and this implementation reconstructs grids before placement
search. Its SRAM-bandwidth calculation assumes port widths, operand residence,
and read/write traffic that were not sourced. Simba's eight-wide vector MAC is
evidence about Simba, not direct evidence that the hybrid evaluator gives a
VVA core eight additions/cycle.

The rate-eight result remains an important sensitivity point because its
AlexNet-FC grids match the rate-one grids, so that comparison isolates the
execution-rate change. It is not a recovered hardware parameter. A follow-up
source audit must quote exact pages or explicitly conclude that the parameter
is unavailable. If it remains unavailable, report VVA throughput as a
sensitivity range and keep claims conditional on the assumed rate.

The adversarial follow-up made the required correction. It retracted the
Noxim, hidden/overlapped VVA, multiplier-bypass, 128-addition, eight-wide
post-processing, assumed SRAM-interface, fabricated reduction-equation, RL
partitioning, and exact Figure-10 assertions. Its surviving conclusion is:
the published evidence is insufficient to select a single VVA
additions-per-cycle value. The project therefore treats throughput as a
sensitivity parameter. The next baseline sweep fixes the rate used for grid
reconstruction while changing only the execution rate, preventing the two
effects from being conflated.

An independent review of the fixed-grid result identified logic-core indexing
as the next high-impact reconstruction assumption. The paper defines BS as
placing logic cores sequentially by physical chip index and then physical core
index, but it does not publish how VMM and VVA logic cores are numbered. This
implementation numbers each output group's VMM tasks followed by its VVA task,
which may give BS unusually strong locality once communication dominates.
Alternative logic-index orders are therefore a baseline-sensitivity ablation,
not a change to the optimizer.

The same review confirmed that Myung et al. do not resolve VVA timing: their
learning objective is a communication-cost proxy and their separate C++
cycle-level routing simulator is unavailable. Their large reported headroom
cannot calibrate Wu's computation-plus-communication objective. Wu's Equation
4 does explicitly define overall latency as the maximum stage latency, so the
max-versus-sum aggregation is known; the contents and timing of reconstructed
stages remain assumptions. Wu also reports 4.5x mean-hop reduction for
AlexNet-FC from BS to DDPG, which can serve as a later structural diagnostic
but is not a target for RS or SA.

The detailed source-code comparison with maintained public DDPG
implementations and the remaining algorithmic differences are recorded in
[`DDPG_REFERENCE_AUDIT.md`](archive/DDPG_REFERENCE_AUDIT.md).

## Learned-SA attribution audit (14 October 2026)

An external audit of the frozen contextual-bandit method raised a valid
causal-attribution issue. `--ppo_asa_disable_learning` gives a focused-uniform
control with the same candidate pool and objective-call budget, so learned
versus frozen measures the value of the trained non-uniform selector as a
package. It does not distinguish learned relationships from direct use of the
hand-built `proxy_gain` candidate feature. Plain fixed SA is also not an exact
control because it uses a different proposal neighborhood and temperature
calibration. Objective-call parity does not imply wall-clock parity because
the learned method constructs and featurizes 16 candidates and performs model
updates.

The later VGG16-FC transfer result remains useful and was not available to the
audit: the unchanged method beat focused-uniform control in all three seeds by
2.255% on average, with a three-seed paired t interval [1.107%, 3.403%]. This
answers the narrow concern that the effect appeared only on the tuned
AlexNet-FC graph. It does not answer whether proxy-greedy selection, fixed
random scoring, or focus changes explain the advantage. Three-seed t intervals
are descriptive because their variance estimates are unstable and exact
distribution-free tests have little power.

The minimum decisive next control is proxy-greedy selection with the same
focused pools, epsilon schedule, and objective-call budget. If the learner
beats that control, the proxy alone does not explain the gain. Focus-zero,
random-weight, shuffled-target, and feature-removal ablations remain useful if
the result is ambiguous, but they are not all mandatory before continuing the
planned held-out VGG runs. Larger 10--20-seed studies strengthen a formal
significance claim; they are recommendations rather than a retroactive reason
to discard the current exploratory evidence. Continue to report full runs as
the independent units, wall-clock time, and raw archive hashes.

## Independent literature review audit (14 October 2026)

A Gemini literature report supports a useful but narrow novelty hypothesis:
to the best of the reviewed evidence, applying an online contextual-bandit
candidate ranker inside fixed SA to reconstructed-latency multi-chip DNN core
placement has not been reported in exactly this form. This must be phrased as
"to our knowledge," not as proof that no such work exists. Broad claims of
being first to combine RL and SA, first to learn an SA proposal distribution,
or first to apply RL to multi-chip core placement are already excluded by
Vashisht et al., Correia et al., Qiu and Liang, and Myung et al.

Primary-source verification corrected several errors in the report:

- [Neural Simulated Annealing](https://proceedings.mlr.press/v206/correia23a.html)
  learns SA proposal distributions for general optimization and has an
  [official public implementation](https://github.com/Qualcomm-AI-research/neural-simulated-annealing).
- [Qiu and Liang's AAMAS 2025 paper](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1718.pdf)
  is not a multi-agent placement application. It learns proposal policies for
  Knapsack, Bin Packing, TSP, and continuous test functions, uses immediate
  energy gain, and is a close algorithmic reference. Its
  [public implementation](https://github.com/nathanqiu07/RL-Based-SA-Public)
  is already audited in `THIRD_PARTY_NOTICES.md`.
- [Vashisht et al.](https://arxiv.org/abs/2011.07577) use RL to produce a
  rough initialization followed by SA refinement, rather than ranking each SA
  proposal inside the loop.
- Myung et al.'s correct DOI is
  [10.1109/TNNLS.2021.3117878](https://doi.org/10.1109/TNNLS.2021.3117878),
  not `10.1109/TNNLS.2021.3115456`. Their
  [public repository](https://github.com/WOOSHIK-M/Core_Placement_with_Reinforcement_Learning)
  states that the uploaded executable path is single-chip and that multi-chip
  execution is not available.

The report's positive interpretation is also too strong. Learned versus
focused-uniform supports a causal statement about the trained selector package
under equal true-objective calls; it is not "irrefutable proof" that learned
relationships rather than direct exploitation of `proxy_gain` explain the
benefit. Three seeds do not warrant "highly significant" language, and the
tested ASA result does not establish a general failure of adaptive
metaheuristics. The VGG16-FC 3/3 transfer win remains positive evidence and the
running held-out VGG test remains appropriate.

The defensible paper angle is therefore a contextual-bandit-guided SA proposal
mechanism for latency-aware multi-chip DNN core placement, evaluated against a
matched focused-uniform control plus standard search baselines. VLSID or an
ML-for-systems/EDA workshop is a realistic first target. MICRO, DAC, ICCAD, or
TNNLS would require broader workload validation, stronger evaluator evidence,
and comparison with the best non-learned proposal heuristic; they should not
be described as immediate or guaranteed fits.

## Paired Kaggle short result (8 October 2026)

The corrected one-seed trained/control run used commit `aa75b2d`, 300 online
placements per condition, identical seed 0 action randomness, 60 legal-random
warm-up placements, and separate replay randomness. Mechanics and numerical
stability passed: warm-up costs matched exactly, collision repairs were zero,
Q magnitude stayed below 0.127, losses were finite, and late proto-action
saturation was below 1%.

It did not show learning. The trained policy's late deterministic mean was
41.4736 microseconds versus 38.3919 microseconds for the no-learning control,
so training was 8.03% worse. The trained best candidate was 38.1083
microseconds versus 37.6356 microseconds for the control. This is one seed and
does not establish a population result, but it is direct evidence against the
current update at the 300-placement budget.

An older 3,000-placement guided run from commit `107899a` reached a 37.1630
microsecond best search trajectory and lower late noisy costs. It lacked a
matched no-learning control, permanent demonstration retention, legal-random
warm-up, and separated random streams. Its deterministic diagnostics did not
show a clean sustained improvement. It is useful for choosing a 3,000-placement
follow-up horizon but cannot be used as proof that policy training helped.

## Contention-feature attribution result (15 October 2026)

The VGG16-CONV three-seed gate tested the opt-in `contention12` feature set,
5% soft bottleneck focus, and epsilon top-four exploration at 20,001 true
objective evaluations per method. Mean learned cost was 728.898 microseconds.
This was 66.218% better than random search and 57.172% better than the matched
bottleneck-random selector. Against shuffled-label training it won two of
three seeds, with a mean paired improvement of 19.835%, but the third pair
reversed direction.

The decisive controls were negative. Proxy-greedy reached 558.921
microseconds on average and beat learning in all three pairs; fixed SA reached
609.994 microseconds and also beat learning in all three. Proxy-greedy found
619, 880, and 800 improving moves, compared with 391, 489, and 471 for the
learner. The unchanged 100,000-evaluation extension is therefore not
justified.

Nonzero weights alone are insufficient evidence, but the real-versus-shuffled
diagnostics support the narrower conclusion that training extracted label
structure: learned score-head norms ended at 0.526--0.612 versus 0.052--0.095
for shuffled labels, and learned logit dispersion was two orders of magnitude
larger. The useful research question has shifted from whether the bandit can
update to whether it can add information beyond `proxy_gain`. A residual
ranker with proxy score as an explicit prior, or a bandit choosing among
distinct move operators, is a better next hypothesis than more epochs for the
rejected package. Any such method must be paired against proxy-greedy and
fixed SA under the same objective-call budget.
