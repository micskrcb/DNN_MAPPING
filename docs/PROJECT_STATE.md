## 24 October: learning follow-up received; test learning intensity across workloads

Archived results and SHA256: `artifacts/learning-followup-20261024/`. Existing RL+SA (epsilon endpoint.02, decay1) gives4.34080µs versus faster decay4.34352µs and fixedSA4.42912µs on VGG16-FC seed19,100001 evaluations each. Existing method wins over SA1.99408%; faster decay loses0.06266%. Retain existing schedule. Both learners completed389 update events. DDPG clipping improves final numerical diagnostics but both bests remain random baseline6.07168µs; no convergence or published-DDPG claim.

Next: AlexNet-FC and VGG16-FC seed21,100001 evaluations per arm, existing RL+SA versus16 minibatch optimizer steps per update instead of4, and fixedSA. Same feature set, reward, epsilon, replay, candidate pool, annealing and hardware. This targets learner fitting, not SA changes or extra objective queries. Record actual optimizer-step counts and runtime; the higher-training arm spends extra training compute. Exploratory workload-specific results, no guaranteed improvement or promotion from one seed. No additional DDPG run in this workflow.

Reviewed Wang etal DOS paper DOI10.1109/TCAD.2025.3640508. It combines dimensional ordering with time-sharing communication; its routing-time results are not directly our pipeline objective or mapping-program runtime. No author implementation located in targeted search. See `docs/DOS_PAPER_ASSESSMENT.md`. Our next experiment is not a DOS implementation; future structured moves require explicit partition coordinates and a matched unlearned structured control.

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

# Project state: paper-faithful DNN core placement

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


## 2026-10-17: amortized topology-aware RL+SA gate

Following an independent code-and-results audit, the next prerequisite is a
pre-registered headroom certification rather than another learned-policy run.
The new runner compares short and long Proxy-SA budgets on AlexNet-CONV,
VGG16-CONV, and ResNet50-CONV, records their random-placement sensitivity and
distance to the compute-only floor, and permits later RL only where at least
2% median short-to-reference headroom remains. This criterion prevents small,
plateau-level differences from being misrepresented as learning progress.

The certification completed at revision `305999c` with archive SHA-256
`3e504b35d3c53138e6267abe5153e48648433fa881d47fa6be3d1a0985cb191f`.
Only VGG16-CONV passed: its median 20,001-call Proxy-SA result was 6.709%
above its 100,001-call reference and the reference remained 40.706% above the
compute-only floor. AlexNet-CONV and ResNet50-CONV each had 0.000% median
unresolved headroom and are excluded from further RL gates under this
protocol. VGG16-CONV is the sole justified target for the next frozen transfer
test.

**Result: failed frozen-transfer gate.** The Kaggle run at revision `34c3101`
completed with a clean worktree, 51 passing tests, and archive SHA-256
`0b52220c8d9f81fab7a7c2e60e6d8f44d2c360469caa2be8d5d63f50f4466b20`.
Both AlexNet training stages made 77 bandit updates, and each held-out run
used the exact 20,001 complete-placement budget. The ResNet50-CONV frozen
checkpoint was genuinely loaded (`frozen_checkpoint_contextual_bandit`) and
made zero updates, as intended.

However, frozen checkpoint, proxy-greedy, and uniform each obtained exactly
588.784 microseconds for every hold-out seed (20--22). Thus there were zero
wins and 0.000% mean improvement over either control, far below the two-win
and +1% advancement rule. This is a valid null transfer result: do not claim
that the topology-aware checkpoint generalizes, and do not extend this exact
training/hold-out protocol by merely increasing its budget.

The next method changes the information available to learning rather than
retuning the closed operator controller. A `topology16` proposal representation
adds hardware context to the existing contention features. Bandit checkpoints
can now be evaluated as genuinely frozen policies, and explicit cross-workload
fine-tuning restores Adam state instead of silently restarting the optimizer.

The frozen protocol trains on AlexNet-FC and AlexNet-CONV, then evaluates on
unseen ResNet50-CONV seeds 20--22 against matched proxy-greedy and uniform
candidate controls. Every condition uses exact complete-placement accounting.
The implementation compiles, the 51-test local suite passes, and PyTorch
feature/checkpoint tests are included for the Kaggle environment. See
[`AMORTIZED_BANDIT_SA.md`](AMORTIZED_BANDIT_SA.md). Performance is not known.

## 2026-10-17: delayed macro-action gate result

The VGG16-CONV macro gate completed on fresh seeds 9--11 at revision
`ad056fc`, with a clean worktree and exact 20,001-call accounting for all nine
runs. Archive SHA-256:
`35f8a9a0e4ba6288d24b47f6e39591465d9c6b393f6eb441380e48ce213c52c6`.

Learned macro selection beat its frozen uniform-macro control in two of three
seeds and by 0.816% on mean paired cost. Against unthrottled Proxy-SA it won
only seed 10: paired improvements were -0.757%, +2.097%, and -8.232%, for a
-2.297% mean. It therefore failed the prespecified gate.

Mechanics were healthy. Every learned run made 312 delayed segment updates,
used all four macro-actions, and reported finite nonzero delayed rewards. This
is weak evidence that delayed credit can distinguish search strategies, but it
does not improve the strongest non-learned method. Together with the one-step
and full-information failures, this closes the current four-operator family.
Do not tune its horizon or Thompson parameters on these results. Proxy-SA
remains the practical best method in this line of experiments.

## 2026-10-17: delayed macro-action RL+SA implementation

The next bounded method is implemented as an opt-in extension of
`operator_sa`. Instead of updating from every immediate proposal, the policy
holds a search strategy for 64 steps and receives the best-cost reduction over
the complete segment. Non-proxy strategies are cheaply screened against a
proxy proposal while spending exactly one true evaluation per step. A matched
uniform-macro control has identical segment mechanics and no learning; ordinary
unthrottled embedded Proxy-SA remains the decisive baseline.

The implementation records macro choices, delayed reward, proxy overrides,
true-call accounting, and update counts. Reproducibility, exact-budget, learned
and frozen-control tests pass as part of the 50-test local suite. The frozen
gate uses fresh VGG16-CONV seeds 9--11 and is documented in
[`MACRO_OPERATOR_SA.md`](MACRO_OPERATOR_SA.md). Performance is not yet known.

## 2026-10-17: proxy-gated LinTS result

The selected-action VGG16-CONV gate completed at revision `006d2bc` with a
clean worktree and exact 20,001-call accounting in all nine runs. Archive
SHA-256: `6a8cfe987ae3fa0fae6465a4736e183137655bb0ff2b76fc471afeb969e728a7`.

Outcome-aware LinTS lost all three pairs to unthrottled embedded Proxy-SA and
was 6.587% worse on mean cost. Signed-reward LinTS won seeds 0 and 1 by 0.338%
and 0.237%, lost seed 2 by 1.070%, and was 0.165% worse on mean cost. It
therefore failed the prespecified two-win plus 1% mean advancement rule.

The method was active rather than frozen: every learned run made 19,968
updates, screened 39,936 cheap candidates, and frequently overrode its primary
sampled operator. Outcome credit caused substantially more non-proxy execution
and clearly degraded placement quality. Cheap proxy gating protected the
signed variant enough to approximately match Proxy-SA, but learning did not
add value beyond the proxy. Close this four-operator contextual-bandit family;
do not tune its discount, sampling scale, or reward weights on these seeds.
The next learned-SA hypothesis must change the action/credit problem, using
multi-step macro-actions and training across independent chains or workloads.

## 2026-10-17: full-information result and next gate

The full-information VGG16-CONV gate completed at revision `8a60616`. All
six runs used exactly 20,001 objective calls. Each learned seed collected
3,472 exact labels from 868 four-operator pools. Learning improved on its
matched, transition-throttled proxy control by 1.689% in seed 0 and tied it in
seeds 1 and 2, for a 0.563% mean and one win. It therefore failed the frozen
two-of-three and 1% advancement rule. Archive SHA-256:
`913cfec6305bdd58da71ff97da3161f061189e189416af2d0e649559cb87b7a4`.

The result is informative rather than a broken run. The learned selector used
proxy on 52,030 of 52,061 decisions; all three other operators had much worse
summed exact signed rewards. Full-information labels therefore taught the
model that the existing proxy dominated this operator set, while sacrificing
sequential search transitions. Do not extend this configuration.

The next prespecified short gate is now implemented: discounted Cholesky
Linear Thompson Sampling ranks the four operators, a cheap proxy screens its
top two proposals, and exactly one true objective is evaluated per step. It
compares signed and outcome-aware rewards with ordinary unthrottled embedded
Proxy-SA on three seeds. This is contextual-bandit-guided SA, not a claim of
long-horizon RL. Run `scripts/run_kaggle_proxy_gated_lints_gate.sh`.

## 2026-10-17: external bandit and SA research audit

The internet review and supplied Gemini report have been reconciled with the
current code in
[`EXTERNAL_RESEARCH_REVIEW_2026-10-17.md`](EXTERNAL_RESEARCH_REVIEW_2026-10-17.md).
The full-information gate remains a diagnostic rather than the final proposed
method. Its decisive missing baseline is ordinary unthrottled Proxy-SA. If it
fails, the next bounded implementation is selected-action, proxy-gated,
discounted Cholesky LinTS with one true evaluation per step; its reward design
must be ablated separately. No external source code has been imported.

## 2026-10-17: full-information operator-learning gate

This bounded RL+SA experiment evaluated every twentieth operator
decision evaluates `proxy`, `bottleneck`, `swap`, and `relocation` from the
same pre-move placement, charges all four exact calls to the declared budget,
and trains the phase-specific LinUCB models from all four signed gains. The
chain still advances with the proxy proposal at these collection steps, so the
learner receives no oracle trajectory advantage. On every other step it may
override proxy using the learned confidence gate. The matched proxy control
makes the same four calls and discards the labels. A regression test and a
reduced end-to-end archive smoke passed with exact accounting. The completed
negative performance result and replacement experiment are recorded above.

## 2026-10-16: training-free portable Proxy-SA mode

The tested proxy-greedy control is now exposed directly as `--algo proxy_sa`.
It requires no labels, checkpoint, or gradient updates. A bounded portability
runner compares it with matched-budget fixed SA and random search on AlexNet-FC
paper mode, ResNet18 full-frame mode, and a custom CNN on torus hardware. Local
validation passed all 46 tests and all three portability/accounting cases.
See `PROXY_GUIDED_SA.md`; Kaggle results remain required before making a
performance or generalization claim.

## 2026-10-16: operator-selection RL+SA gate

Implemented `--algo operator_sa`, an online LinUCB policy that selects among
four legal SA move operators while fixed Metropolis acceptance remains active.
It learns from exact proposed cost changes and includes a uniform-operator
control. The three-seed, 20,000-call gate compares Operator-SA, its control,
Proxy-SA, fixed SA, and random on AlexNet-FC and VGG16-CONV. All 47 tests and a
reduced end-to-end archive smoke passed; performance is not yet established.

### Completed 20,000-evaluation result

The three-seed gate completed at revision `b0a2c56` with exact 20,001-call
accounting throughout. On AlexNet-FC, Operator-SA beat its uniform-operator
control in 3/3 seeds by 0.521% on average and Proxy-SA in 2/3 by 0.174%, but
finished 0.046% behind fixed SA on average. On VGG16-CONV it beat random by
54.982% on average, but lost to Proxy-SA in 3/3 seeds by 2.589% and beat its
uniform control and fixed SA in only 1/3 seeds. Neither workload passed the
prespecified gate, so seeds 3--4 must not be run unchanged.

The learner was active: every learned run made 19,968 updates. AlexNet policies
collapsed mostly to swaps, while VGG16 operator proportions varied sharply by
seed. The next design should address this unstable/nonstationary credit signal,
preferably with phase-conditioned or discounted operator values and explicit
minimum exploration, while retaining Proxy-SA as the decisive control.

That v2 design is now implemented: separate early/middle/late LinUCB models,
0.995 evidence discount, 10% minimum exploration, and a confidence-bounded
Proxy-SA fallback. All 47 tests and a 129-call CLI accounting smoke pass. Its
three-seed 20,000-call result passed the formal VGG16-CONV gate: 9.455% over
uniform operator selection, 5.661% over the earlier Proxy-SA control, 9.307%
over fixed SA, and 58.686% over random. It failed on AlexNet-FC.

Attribution is not yet complete. VGG16 selected the proxy operator for roughly
92% of post-calibration steps; nearly all non-proxy choices were forced
exploration. The earlier Proxy-SA control also uses 16 candidates and a
different epsilon rule, whereas the embedded proxy operator uses eight and a
confidence fallback. An exact proxy-only operator control is now implemented
with separated proposal, selection, and acceptance random streams. Run its
three-seed attribution gate before held-out seeds or an RL improvement claim.

### Exact proxy attribution result

The attribution gate completed at revision `7132c9b` with exact 20,001-call
accounting. Learned Operator-SA costs were 1170.504, 1165.447, and 1121.771
microseconds; the exact embedded-proxy control reached 1105.252, 1112.386, and
1165.565 microseconds. Learning won only seed 2 and was 2.306% worse on average.
The control made zero updates and selected the identical proxy operator for all
19,968 post-calibration steps. Learned runs made 19,968 updates but selected
proxy for 91--92% of steps; most alternatives were forced exploration.

Therefore the earlier VGG16-CONV improvement is attributable to the improved
eight-candidate embedded proxy search, not to RL operator selection. The
unchanged learned method must not advance to held-out seeds. The exact proxy
heuristic is a valid positive optimization result and strong future baseline,
but it is training-free rather than an RL contribution.

Updated 2026-10-12.

## Goal and branch

The goal is to reproduce, then improve upon Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning* (ACM TODAES 2020, DOI 10.1145/3418498).

The maintained development branch is `cleaned-up` in `micskrcb/DNN_MAPPING` (continued from `cpu` and `codex/reconciled-paper-implementation`) and still supports `--device cuda`. `main` remains historical. Code and tests are the source of truth for current behavior; old run logs use different objectives and cannot be compared numerically with paper mode.

## What is implemented

- FX extraction for torchvision AlexNet, VGG16, and ResNet50.
- Exact Figure 6 aggregate task counts: AlexNet 183 CONV/932 FC, VGG16 1024/1924, and ResNet50 512/37.
- Deterministic per-layer `(M,N)` grid reconstruction proportional to MAC work, with exact aggregate counts, 64 KB 8-bit weight-tile capacity, and an explicit VMM/VVA cycle-balance term.
- Separate CONV and FC optimization over disjoint, contiguous whole-chip masks.
- Table 1 4×4-chip, 16×16-core hardware parameters and physical byte/time units.
- Reconstructed X-then-Y routes, lower-left periphery gateway for inter-chip traffic, routed byte-hop time, and shared directed-link contention per time phase.
- Configurable CONV block scaling; default four follows Figure 7's illustration.
- Figure 9 `paper_cnn`, sparse terminal reward, paper learning rates/gamma/batch size, batched actions, coordinate conversion, and Manhattan collision repair.
- Absolute-placement OU exploration scheduling that remains stable across cumulative checkpoint stages.
- Sequential BS, random search, fixed simulated annealing, adaptive simulated annealing, DDPG, and DDPG→ASA.
- Experimental guided DDPG with ASA demonstrations, permanent demonstration replay, collision-free legal projection, normalized complete-episode return targets, a uniform legal warm-up, deterministic candidate retention, and a matched no-learning control.
- Experimental masked categorical PPO with exact legal-action masking, clipped policy updates, deterministic diagnostics, checkpoints, and a matched frozen-policy control.
- Experimental PPO-guided ASA with legal candidate proposals, a permutation-equivariant candidate scorer, PPO updates, fixed Metropolis acceptance, the existing adaptive temperature controller, and an exactly uniform frozen-policy control.
- Opt-in contention-aware contextual-bandit proposals with cached shared-link and source-serialization deltas, soft near-bottleneck stage focus, structured top-m exploration, shuffled-label and bottleneck-random controls, and a staged VGG16-CONV Kaggle gate. Historical nine-feature checkpoints and defaults remain unchanged; the new method has not yet passed its empirical gate.
- A reference-compatible PPO-ASA update with disjoint actor/critic networks and optimizers, reward-to-go critic targets, GAE-style advantages, source attribution, and a matched 10-seed decision runner.
- A staged PPO-ASA diagnosis with true best-of-16 candidate evaluation, planted-signal trained/control tests, separate proposed/realized rewards, policy-head diagnostics, and a bounded split-policy pilot that must pass before the full run.
- Explicit placement accounting: DDPG epochs × placements/epoch, separate reward-normalizer trials, and independently configurable RS/SA budgets.
- Five-seed orchestration, periodic JSONL diagnostics, checkpoints, JSON reports, BS-normalized summaries, hop counts, link-load summaries, and a placement-sensitivity preflight.
- One command that orchestrates separate CONV and FC paper-mode suites.

## Documented reconstruction assumptions

The paper gives aggregate logic-core counts but not each layer's partition grid. The current dynamic program chooses exact-count grids close to MAC-proportional targets and enforces the weight buffer. The exact CONV/FC physical masks are not published; the code assigns the minimum rectangular group of whole chips, with CONV first and FC following it.

The complete GRS route implementation is unavailable. Inter-chip routing uses a lower-left gateway inferred from Figure 3 and deterministic XY routes. A stage's reconstructed communication time is the larger of its maximum per-source routed byte-hop time and maximum shared-link serialization load. The evaluated networks' block counts are unpublished; four CONV blocks is a configurable default inferred from the Figure 7 example. Residual branch traffic is traced, while the addition itself is assigned to the destination transformation/VVA path without adding a Figure 6 task.

The paper says computation is balanced per core but does not publish the grids or VVA throughput. The grid search therefore combines MAC-proportional core-count error with estimated VMM/VVA cycle imbalance. The weight of this term and VVA operations/cycle are recorded configuration parameters. Seconds-based latency is scaled to 400-MHz cycle units before the square-root DDPG reward by default; this changes reward magnitude, not placement ordering.

The paper also does not fully specify `z`, OU parameters and fade schedule, replay capacity, CNN padding, LRN parameters, target-network updates, or soft update coefficient. The implementation records its choices in reports and manifests.

## Remaining fidelity gaps

The simulator does not model 64 KB input/activation-buffer occupancy and stalls, exact GRS multicast, router startup, exact packet scheduling, transformation/bias/pooling costs beyond a minimum output pass, compute/communication overlap, or a cycle-accurate block pipeline. VVA throughput defaults to one addition/cycle because the paper does not state it. Consequently, `paper_pipeline` is a documented sensitivity model, not the authors' unpublished simulator.

Batch-one latency can be compared after validation. True large-batch throughput still needs an explicit fill/steady-state/drain calculation; the current inverse objective ratio is labeled as such and must not be presented as measured throughput.

## Evidence completed locally

- The corrected `cpu` head passes `src/test_multi_chip.py`, all 25 reconciliation tests with no skips, and the bounded CPU device validator under torch 2.14.0+cpu and torchvision 0.29.0+cpu. The validator completed 57 measured optimizer updates and a checkpoint round trip.
- Paper-target extraction returns all six exact CONV/FC counts.
- Hand-calculated XY gateway and shared-link contention tests pass.
- Masked-region tests confirm that baselines and the mapper cannot use other cores.
- Remainder work, byte units, torus IDs, SA budget/neighborhood, BS order, potential shaping, residual dependencies, real optimizer updates, and all agent architectures have regression coverage.
- AlexNet BS completed for both reconstructed regions on CPU: 183 tasks in a 256-core CONV region and 932 tasks in a 1024-core FC region.
- A one-placement AlexNet `paper_cnn` CPU smoke completed. It verifies execution only; replay had not reached minibatch size and no learning claim follows.
- The paper experiment runner completed a one-seed, one-placement CPU smoke for both regions and the configured baseline/optimizer matrix, producing manifests and BS-normalized summaries. This verifies orchestration, not learning.
- A 2,080-placement AlexNet-CONV CPU run exposed a flat objective: the old `(M=1,N=61)` reconstruction made a 5.2488-ms VVA task placement-independent, limiting visible headroom to roughly 0.001%. That checkpoint is incompatible with and must not resume into the corrected model.
- The corrected pure-model AlexNet-CONV audit preserves 183 cores with grids `(7,2), (12,4), (6,4), (6,6), (5,5)`. Across 256 seeded random placements the objective span was 16.34%; the best sample was 86.60% compute and 13.40% communication. This is meaningful placement sensitivity, not paper-result reproduction.
- The corrected pure-model AlexNet-FC audit preserves 932 cores with grids `(16,36), (16,16), (4,16)`. Across 64 seeded random placements the objective span was 4.05%; the best sample was 92.29% compute and 7.71% communication.
- In a bounded 2,000-candidate comparison on that reconstructed graph, normalized-to-BS costs were RS 0.9697, fixed SA 0.9899, and ASA 0.9697. These small-budget single-seed numbers validate execution only.
- Real torchvision extraction reproduces the same partitions. With 64 sampled placements, AlexNet CONV spans 13.89% and FC spans 4.05%. A three-placement CPU `paper_cnn` smoke completed 120 optimizer updates, checkpointing, deterministic diagnostics, and reporting at about 62 seconds per placement. Its noisy best was 7.29% below BS, but its deterministic policy worsened by placement three, so this is not evidence of learning.
- A 12-placement diagnostic found that deriving OU decay from each temporary checkpoint target exhausted exploration in the first stage. The fixed absolute-placement schedule retained noise and increased intended-core diversity; in a matched six-placement run, unique intended cores rose from 45 to 117 and occupied-core repairs fell from 177 to 101.
- An initial potential-shaping diagnostic produced critic losses in the hundreds because cycle scaling also inflated every intermediate potential. Baseline normalization reduced mean critic loss from `294` at placement six to `0.111`; the deterministic cost held at `3.9148e-05` instead of worsening to `4.3780e-05`. This is bounded stability evidence for an optional improvement condition, not a convergence claim.
- A 30-placement local CPU diagnostic confirmed that the policy still oscillates: its deterministic cost was best at placement six and later regressed. It also exposed that deterministic diagnostic placements were evaluated but discarded. The opt-in `--retain_deterministic_candidates` mode now preserves those placements and reports their evaluation count separately. A matched 12-training-placement run retained `3.829734375e-05`, 8.58% below its `4.188953125e-05` baseline, using 12 training plus six deterministic diagnostic candidate evaluations. This is a single-seed bounded result, not paper-level learning evidence.
- A 10,000-candidate ASA refinement of that retained placement completed locally in about 19 seconds. It accepted 9,585 moves and recorded 113 improving moves, but did not beat `3.829734375e-05`. Hybrid summaries now include deterministic diagnostic evaluations in their combined total when retention is enabled.

Earlier 1,445-task AlexNet logs, decimal-valued junior runs, and old proxy/full-frame best costs were produced by different extraction or objective versions. They remain useful historical diagnostics but are not evidence of paper result reproduction.

## Kaggle learning diagnosis (2026-10-08)

The 3,000-placement AlexNet-CONV Kaggle run executed correctly but did not demonstrate policy learning. The trained deterministic policy ended at 37.6356 microseconds while the matched untrained policy was 37.4466 microseconds. Approximately 176 of 183 actions required collision repair, the deterministic policy used only a few unique intended positions, and critic loss reached roughly `1.48e12`. The run's best candidate therefore came from search/exploration rather than a learned deterministic policy.

This diagnosis changed the immediate plan. The experimental `ddpg_guided` mode now seeds replay with ASA transitions that are never overwritten, projects every action to a legal unused core before evaluation, assigns normalized discounted complete-episode return targets to all trajectory steps, retains deterministic candidates, and starts with configurable uniform legal placements to diversify replay. `--guided_disable_learning` runs the identical data-generation path without gradient updates, providing the control needed to establish whether training contributes anything.

A comparison with OpenAI Spinning Up, Stable Baselines3, TD3, Wolpertinger, and DDPGfD found that the trained and no-learning conditions still shared one global NumPy stream. Replay sampling in the trained run therefore changed its later exploration samples. Guided replay and action selection now use independent seeded streams, and a paired synthetic integration test confirms identical random-warm-up costs, zero collision repairs, and finite training diagnostics. `DDPG_REFERENCE_AUDIT.md` records the full comparison and remaining differences.

The first corrected Kaggle pair used seed 0 and 300 online placements per condition. Mechanics and stability passed, but the trained late deterministic mean was 41.4736 microseconds versus 38.3919 microseconds for the no-learning control, 8.03% worse. The five-seed extensive run then completed 3,000 placements per condition. Training lost all five late deterministic comparisons; aggregate means were 39.0857 microseconds trained and 38.7322 microseconds control. Mean paired improvement was -0.3535 microseconds, with an exploratory 95% interval of -0.8766 to +0.1696 microseconds. Guided DDPG is therefore a valid negative ablation, not a successful learning method.

The original paper and later work do not make continuous DDPG an obviously suitable choice for this discrete placement problem. DDPG can deadlock under sparse deterministic rewards; Wolpertinger-style methods use a continuous proto-action only to retrieve discrete candidates; invalid-action masking and later masked placement policies enforce legality directly. If guided DDPG does not beat its matched control across seeds, the next agent should be masked categorical PPO rather than further tuning an unstable continuous actor. See [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md) for the evidence, Gemini review, sources, and decision gates.

## Masked-PPO learning result (2026-10-09)

The five-seed extensive masked-PPO archive is complete and valid at commit
`e3576b0`. Every trained/control pair used 3,000 placements, began with the
same deterministic policy, produced zero collision repairs, and retained
finite diagnostics. Training beat the frozen deterministic policy in all five
seeds. Mean late deterministic latency was 42.6118 microseconds trained versus
45.3588 microseconds control, an aggregate 6.06% improvement. The exploratory
paired 95% interval for absolute improvement was 1.0972 to 4.3969
microseconds, entirely above zero.

The best retained solution improved more modestly: 39.9989 microseconds trained
versus 40.5094 microseconds control on average, about 1.26%. This distinction
matters: PPO has demonstrated policy learning, while random exploration still
contributes much of the best-placement quality. The result is confined to the
one-chip AlexNet-CONV reconstruction and is not yet a paper-comparable
multi-chip benchmark.

## GPU state

DDPG networks and sampled tensors support CUDA; the environment, routing, collision handling, replay storage, and process orchestration remain CPU-side. GPU acceleration therefore speeds neural-network work but does not remove the CPU evaluator bottleneck.

The planned H100 12 GB slice was unavailable. Kaggle completed the original diagnostic, the five-seed guided-DDPG ablation, and the five-seed masked-PPO AlexNet-CONV learning gate on two T4 GPUs.

## Next execution sequence

1. Preserve guided DDPG and both 4,101-evaluation PPO-ASA variants as negative ablations; do not scale their unchanged policies.
2. [Implemented locally] Train one shared proposal policy across repeated independently initialized placement chains, using 250,000 proposals and matched frozen-policy and ASA controls. Chain initializations are part of the objective budget.
3. [Complete] The training-scale Kaggle gate used 1,954 chains and 251,954 objective evaluations per condition. Trained PPO-ASA beat its frozen control by 0.466% but lost to uninterrupted ASA by 3.55%. Mechanics passed and the policy changed usefully; this is a learning-only result.
4. [Complete] The five-seed uninterrupted holdout rejected the frozen checkpoint. It lost all five pairs, averaging 23.63050 microseconds versus 23.51504 for uniform and 23.50518 for ASA.
5. A matched 128-proposal local diagnostic showed a weak short-horizon signal: trained PPO averaged 0.385% below uniform, won three of five pairs, and beat severely budget-limited ASA. This identifies horizon specialization rather than useful deep refinement.
6. Run one final deployment-matched experiment: train on 4,100-proposal chains with within-chain progress and chain-boundary PPO updates, then automatically run the five-seed 4,101-call holdout.
7. [Complete] The deployment-matched checkpoint received 61 PPO updates and showed no holdout signal: 23.51635 microseconds trained versus 23.51504 uniform and 23.50518 ASA. Its policy remained nearly uniform, so the checkpoint is rejected but training was insufficient to reject the architecture against the closest 1,000-epoch paper.
8. [Complete] The sufficient-scale protocol ran 1.25 million proposals, exactly 1,250 PPO updates, and 10 frozen holdout seeds. Trained PPO averaged 23.50258 microseconds versus 23.50722 uniform and 23.48939 ASA. It won only three uniform pairs and four ASA pairs; final entropy was effectively uniform.
9. [Complete] Audit the official RL-Based-SA training source. It uses disjoint actor and critic networks and optimizers; the rejected scorer shared its feature encoders, allowing value gradients to alter the policy representation.
10. [Implemented] Record the independent audit and correct its legacy-epoch and KL qualifications. Add direct oracle headroom, feature-informativeness, planted-signal, and policy-gradient diagnostics.
11. [Complete] The staged diagnostic passed both planted-signal tests but the real split pilot stayed at uniform entropy after 300,032 evaluations. Its five-seed holdout beat uniform twice and tied three times; the mean interval included zero.
12. [Complete] The exact oracle audit found useful best-of-16 headroom, while the current proxy matched the best candidate only 4.5% of the time versus 6.25% expected from uniform choice.
13. [Implemented] Export exact-gain candidate pools with trajectory identifiers and train the same split actor with listwise supervised targets. The Kaggle gate collects ten independently restarted chains and reserves complete chains for validation so adjacent annealing states cannot cross the train/validation boundary.
14. Run `scripts/run_kaggle_oracle_supervised_ranker.sh`. Failed held-out ranking closes the current features; a pass automatically triggers frozen and PPO-fine-tuned five-seed optimizer holdouts.
15. Do not scale unchanged AlexNet-FC masked PPO; its 3,000-placement run tied the frozen control and lost to ASA.
16. Produce a matched-budget table for BS, RS, SA, ASA, paper DDPG, masked PPO, and the hybrid only after the hybrid gate passes.
17. Add activation-buffer/streaming behavior and router timing when defensible evidence is available.
18. Implement and validate true large-batch throughput before reproducing that panel of Figure 10.

The paper-faithful mode remains frozen separately from guided DDPG and future masked-policy experiments.

## AlexNet-FC multi-chip short gate (2026-10-09)

The matched one-seed gate used 932 FC logic cores across four chips and 567
complete-placement evaluations for each PPO condition, random search, and ASA.
Mechanics passed: trained and frozen policies started identically, repairs were
zero, all diagnostics were finite, and the final placement used all six
directed off-chip links available in the four-chip region.

The late deterministic PPO mean was 24.6022 microseconds versus 24.6424
microseconds for its frozen control, a directional improvement of only 0.163%.
The curve was non-monotonic, stochastic rollout cost worsened from 24.7541 to
25.0429 microseconds between the first and last five diagnostics, and only one
seed was tested. This is insufficient for a multi-chip learning claim.

The optimizer gate failed. Best costs were 24.2075 microseconds for trained
PPO, 24.1472 for matched-budget random search, and 24.1734 for matched-budget
ASA. PPO was therefore 0.25% worse than random search and 0.14% worse than ASA.
All three beat the 24.9203-microsecond sequential baseline. The next bounded
test should extend one paired seed to 3,000 placements with 4,101 matched
evaluations; scale to five seeds only if both the deterministic policy and best
placement improve materially.

That extension is now complete. The late deterministic advantage increased to
0.558%, but only 60 of 100 diagnostic checkpoints beat control and the learned
policy remained non-monotonic. Best retained costs were 24.1080 microseconds
trained, 24.1072 frozen control, 24.1155 random search, and 23.4318 ASA. Thus
training contributed no best-placement advantage, while ASA beat PPO by 2.80%
under the same 4,101-evaluation budget. Unchanged PPO should not be scaled to
five FC seeds. A future hybrid must reserve one fixed total budget across its
PPO and ASA phases and beat a full-budget ASA-only control.

## Five-seed AlexNet-FC ASA baseline (2026-10-09)

The validated archive at revision `b147b14` contains five paired ASA and random
search runs. Each ASA condition used one initial placement and 4,100 proposals;
each random condition evaluated 4,101 complete placements. ASA won all five
pairs. Its mean best latency was 23.5001 microseconds with sample standard
deviation 0.0653, compared with 24.1305 and 0.0408 for random search. The paired
mean advantage was 0.6304 microseconds (2.61%); an exploratory paired 95%
interval was 0.5722 to 0.6886 microseconds. ASA was 5.70% below the
24.9203-microsecond sequential baseline and averaged 53.2 seconds per seed,
versus 74.0 seconds for random search.

The report's generic `complete_placement_evaluations` field records 4,100 for
ASA because it counts proposals and excludes the separately evaluated initial
placement. The matched total is nevertheless 4,101 objective evaluations.
Generated run files made the repository provenance appear dirty, but every
report records the same source revision and the archive passed integrity
checking.

The accompanying Gemini survey is useful background for learned neighbor
proposals, but it discusses physical macro floorplanning, HPWL, sequence pairs,
and B*-trees that do not represent this fixed-grid DNN core-mapping problem. It
also incorrectly says the AAMAS 2025 RL-Based-SA implementation learns the
temperature schedule. The official implementation trains a PPO neighbor
proposal policy, augments state with energy change, optionally uses an LSTM,
and leaves Metropolis acceptance and exponential cooling fixed. The next method
will follow that narrower verified structure and must beat the five-seed ASA
baseline under matched evaluations.

The first implementation is now complete. It ranks 16 legal, unevaluated
swap/relocation candidates at each step and invokes the true objective only for
the selected candidate. The state includes progress, temperature, current and
best improvement, previous proposed energy change, acceptance history, and
neighborhood size. Candidate features include communication pressure,
movement, chip crossing, current-bottleneck-stage coverage, and a cheap routed
distance proxy. A zero-initialized score head makes the frozen control exactly
uniform. The implementation does not import code from the reference project;
it follows its verified separation between learned proposal, Metropolis
acceptance, and temperature scheduling.

A local one-seed AlexNet-FC smoke used 1,001 total objective evaluations per
condition. Learned PPO-ASA reached 23.9986 microseconds, its uniform control
24.1123, and ordinary ASA 23.9629. Learning therefore showed a 0.47%
directional advantage over its direct control, but the optimizer gate failed
by 0.15% against ASA. The next run is the 4,101-evaluation paired Kaggle gate;
five seeds remain conditional on beating both controls.

The 4,101-evaluation Kaggle gate is now complete at revision `1c5c633`. All
mechanics checks passed, but learned PPO-ASA reached 23.5602 microseconds,
versus 23.4213 for the uniform frozen policy and 23.4318 for ordinary ASA. It
was therefore 0.593% worse than its direct control and 0.548% worse than ASA.
The learned condition beat the control at only 10 of 41 recorded checkpoints.
Its final entropy was 2.76893 versus the uniform maximum `ln(16)=2.77259`, and
76.7% of selected proposals were objective-neutral. The unchanged method will
not be scaled to five seeds. See
`runs/kaggle-ppo-asa-alexnet-fc-extended-2026-10-09/STUDY.md`.

## Oracle-supervised proposal ranking result (2026-10-11)

The staged AlexNet-FC oracle-supervised run completed from revision `8fe5da4`.
Its 8,000 diagnostic pools contained 16 candidates each. The exact best
candidate had a mean 0.6639% advantage over a uniformly selected candidate;
58.56% of candidates were objective-neutral and 39.04% of pools contained an
improving candidate. The old handcrafted routed-distance proxy matched an
exact best candidate in only 4.95% of pools.

The supervised actor passed its original randomly held-out pool gate. On 1,600
validation pools it captured 86.02% of measured best-of-pool advantage and
matched an exact best candidate in 56.19% of informative pools. The relevant
random expectation was 35.13%, rather than 6.25%, because many candidates tied
for the best exact gain. Its mean selected advantage over uniform was 0.1348
microseconds per pool, with the reported normal-approximation interval fully
above zero. However, revision `8fe5da4` randomly split pools from continuous
annealing trajectories, so adjacent correlated states could cross the
train/validation boundary. This validation result is therefore provisional.
Revision `164edfe` fixes the protocol by exporting chain identifiers,
collecting ten independently restarted chains, and holding out complete
chains.

The more informative independent optimizer holdout used five fresh seeds and
kept the supervised actor frozen. It averaged 23.462912 microseconds versus
23.493728 for the uniform proposal control and 23.505184 for ordinary ASA.
The supervised actor won four of five pairs against each baseline, a mean
improvement of 0.1312% over uniform and 0.1798% over ASA. Both exploratory
five-seed 95% intervals crossed zero, so this is a positive directional signal
and not yet a statistically reliable superiority claim.

PPO fine-tuning from that checkpoint failed. After 200,032 proposals and 200
updates, the frozen fine-tuned actor averaged 23.536416 microseconds. It lost
all five pairs to uniform by 0.1817% on average; that paired interval was
entirely negative. It lost four of five pairs to ASA by 0.1329% on average.
Entropy fell from a moderately selective supervised policy while logit spread
increased, showing that PPO made the policy more confident while degrading its
generalization. More PPO fine-tuning of this objective is not justified.

At this point the apparent positive lead was **supervised candidate ranking
plus ASA**, not PPO plus ASA, and still required confirmation from revision
`164edfe`. The provisional archive SHA-256 is
`7f774ba7d79ac4b052e5e52c88551c54cedd06dad87e3981153d024fa39299f1`.

The trajectory-safe confirmation subsequently completed at revision
`164edfe` and failed its prespecified gate. It collected all 8,000 pools from
ten independently restarted chains, used eight complete chains for training,
and reserved chains 5 and 7 for validation. The validation actor captured only
3.36% of the oracle advantage. Its exact-best match was 35.55% versus a
tie-adjusted random expectation of 33.38%, while its selected advantage over
uniform was only 0.00223 microseconds per pool with a 95% interval from
-0.00147 to 0.00594 microseconds. Because the interval crossed zero, the runner
correctly stopped before optimizer holdouts and PPO fine-tuning.

The safe dataset also exposed stronger trajectory dependence than the original
random split suggested. Candidate neutrality rose to 84.61%, best-of-16
headroom fell to 0.2815% of current cost, and the previously strong manual
feature correlations fell to roughly 0.08 to 0.11 in magnitude. The actor
captured 10.31% of oracle advantage on its eight training chains but did not
transfer that ranking reliably to the two held-out chains. More epochs on the
same representation are therefore not justified.

This result supersedes the earlier five-seed directional result as the
authoritative representation test. The earlier frozen checkpoint remains a
documented provisional result, but its 0.18% ASA advantage was statistically
uncertain and was trained with a leakage-prone random pool split. There is now
no confirmed positive learned-ranking or PPO+ASA result. ASA remains the
strongest validated optimizer. Further RL+ASA work requires a new state/action
design and a fresh prespecified gate after simulator calibration, rather than
additional training of the current scorer. The trajectory-safe archive
SHA-256 is
`cb70a2b4361ff51f5483e5bccf8912d1d3bb05e32086ccbea3295d15910e60b4`.

An exploratory leave-one-chain-out analysis of that same ten-chain dataset
identified a narrower hypothesis that the signed-gain target had obscured.
The signed target transferred negatively overall: mean argmax oracle capture
was -1.6%, with a chain-bootstrap 95% interval from -5.6% to 2.0%. Training
the same linear listwise analysis on improvement-only utility
`max(exact_gain, 0)` instead produced 14.7% mean capture, with an interval
from 12.0% to 17.2%, and positive capture on all ten held-out chains. In the
late 40% of each chain it retained 10.0% mean capture, with an interval from
6.2% to 13.2% and nine of ten chains positive. The selected candidate improved
11.2% of the time versus a 6.0% candidate base rate.

This is a post-hoc target analysis on the existing data, not an optimizer win
or a confirmatory learning result. It also exposed a deployment gap: argmax
selection was useful, while softmax sampling captured only 3.4% of headroom
overall and 2.7% late. The dataset contained only 832-step chains, so it does
not establish transfer through a complete 4,100-proposal run. The next
revision therefore adds an explicit `improvement_only` training target,
chain-level bootstrap reporting, and a prespecified fresh-data gate using 20
complete 4,100-proposal chains. In this project, "improvement-only" means the
clipped utility `max(gain, 0)`; it is not called accepted gain because finite-
temperature simulated annealing can accept neutral and worsening proposals.
The historical signed-gain mode remains available for reproducibility.

The fresh improvement-only gate then completed at revision `91a70bc` using
16,394 audited pools from 20 complete 4,100-proposal chains. Ten chains were
used for gradient training and ten complete chains were held out. The
prespecified gate failed. Across all held-out pools, deterministic argmax
captured 9.22% of improvement-only oracle headroom, with a chain-bootstrap
95% interval from 7.54% to 10.87%, and all ten chains were positive. In the
prespecified late half, capture fell to 7.03%, with an interval from 4.28% to
9.81%. The selected candidate improved 3.63% of the time versus a 2.07%
candidate base rate (1.76x), and nine of ten chains remained positive.
However, softmax sampling captured only 0.73% of late headroom.

The run therefore passed the hit-rate-ratio and positive-chain criteria but
failed all three capture criteria: 10% late argmax capture, a 5% lower bound,
and 5% sampled-policy capture. The thresholds remain frozen; no optimizer
holdout is authorized from this checkpoint. This result confirms a weak,
transferable deterministic ranking signal while rejecting it as a deployed
stochastic RL+ASA proposal policy. The collection used 82,020 optimization
objective evaluations plus 245,910 additional oracle evaluations and took
5,744.8 seconds. The archive SHA-256 is
`b8266b249398967bf2e5c5f2dc61d02737ceab7b763016e52a3f37ec91a7ed2e`.

Reviewing the next baseline-calibration step exposed a separate paper-mode
problem in fixed SA. The paper fixes the cooldown at 0.99 but states that T0
and Tend are application-specific; it does not publish their values. The code
had used absolute temperatures 100 and 0.1 against a seconds-valued objective,
making almost every practical uphill move acceptable. Fixed SA now measures
T0 from budgeted uphill deltas in the current objective units, freezes a
geometric 0.99 schedule down to a prespecified ratio, and reports the measured
temperatures and exact evaluation counts. The old behavior remains available
only through `--sa_legacy_absolute_temperature` for historical reproduction.

The paper-scale AlexNet-FC baseline calibration then completed at revision
`80aec4f` with one million proposals per search method. BS was 24.92032
microseconds. Random search reached 23.93728 microseconds (0.96055 of BS),
corrected fixed SA reached 22.70400 microseconds (0.91106 of BS), and ASA
reached 23.09328 microseconds (0.92668 of BS). Fixed SA beat random search by
5.15%, so the corrected annealing mechanics and optimizer ordering are
credible. ASA was 1.71% worse than fixed SA and remains a project extension.

The calibration nevertheless rejects the current evaluator as a quantitative
reproduction of Figure 10. The published plot visually places AlexNet-FC
random search near 0.59 of BS and fixed SA near 0.50, whereas the reconstructed
ratios are 0.961 and 0.911. More decisively, the reconstructed compute-only
floor is 22.4 microseconds, already 0.89886 of BS. No placement algorithm can
reach the paper's approximate SA or DDPG ratio under this objective. The run
therefore diagnoses simulator headroom rather than an optimizer failure. It
took about 6.76 hours for RS and 5.51 hours for fixed SA; archive SHA-256 is
`455656bc06b78f3cb163f276060c54bbd6cd6d32dd03ac9d772951a560963977`.

The leading explicit assumption is VVA throughput. The paper publishes 128
MACs per VMM core but no VVA additions/cycle; the reconstruction currently
uses one. In AlexNet FC this makes the first layer's accumulation take 8,960
cycles, exactly the 22.4-microsecond compute floor. The next experiment is an
exploratory matched sweep over plausible VVA rates. It must precede more RL,
and any selected setting must be frozen and validated on fresh workloads
before it is treated as calibration rather than fitting to Figure 10.

That exploratory sweep completed at revision `d6493b5` with 20,000 matched RS
samples and fixed-SA proposals at 1, 8, 16, 32, 64, and 128 VVA
additions/cycle. At one addition/cycle, RS/BS was 0.966 and SA/BS was 0.928.
At eight additions/cycle, the original AlexNet-FC grids remained unchanged,
the compute floor fell from 0.899 to 0.526 of BS, RS/BS reached 0.842, and
SA/BS reached 0.661. This is strong evidence that the assumed VVA rate causes
much of the missing optimizer headroom, but it remains short of Figure 10's
approximate 0.59 RS and 0.50 SA ratios.

Rates 16 and above changed the reconstructed per-layer grids as well as the
execution rate. Their results are consequently non-monotonic and cannot be
interpreted as a pure throughput sweep: SA/BS was 0.781, 0.819, 0.892, and
0.763 at 16, 32, 64, and 128 additions/cycle. Eight additions/cycle is the
cleanest candidate because it changes the FC compute rate without changing
the FC grids, but selecting it from favorable AlexNet results would be
post-hoc calibration. The next gate is a literature audit of the accelerator
architectures cited by Wu et al., followed by a frozen full-budget run and
held-out workload validation. Archive SHA-256 is
`36fdd96e087703ad3641f455a269ff9a01f6e3fbf1e236a462928bdf590c7c21`.

The first external literature report did not resolve the missing parameter.
It acknowledged that no primary source states the simulator's VVA rate, then
recommended eight additions/cycle using assumed SRAM port widths and a
cross-architecture analogy to Simba. Those assumptions are not accepted as
calibration evidence. The report's claim that Wu et al. used Noxim could not
be found in the paper, and its recommendation to fully overlap VVA with NoC
traffic conflicts with the paper's statement that time-phase latency adds
computation and communication. It also incorrectly attributed partition-grid
changes to RL although the sweep contained only BS, RS, and SA. Rate eight
therefore remains a clean and useful sensitivity result, not a recovered
hardware fact.

The corrected external audit explicitly retracted the unsupported Noxim,
full-overlap, multiplier-bypass, eight-wide reduction, SRAM-interface, and
exact Figure-10 claims. Its defensible conclusion is that the published
evidence is insufficient to select a single VVA additions/cycle value. The
implementation now exposes `--partition_vva_ops_per_cycle` separately from
`--vva_ops_per_cycle`. This permits a fixed-grid sensitivity study in which
the logical graph is reconstructed once under a declared rate while execution
timing is swept independently. The historical coupled behavior remains the
default for reproducibility.

The fixed-grid sweep completed at revision `0341745`. With every condition
using the rate-one AlexNet-FC grids, RS/BS was 0.966, 0.842, 0.801, 0.904,
1.015, and 1.080 at execution rates 1, 8, 16, 32, 64, and 128; fixed SA/BS
was 0.928, 0.661, 0.612, 0.671, 0.726, and 0.790. At rates 32 and above the
sampled objective span was identical (74.90%) and the best sampled cost was
2.976 microseconds, showing that the relevant sampled bottleneck had saturated
in a communication-dominated regime. Rate one is compute-dominated, rate eight
is transitional, and rates 16-32 expose large communication-sensitive
headroom. The archive SHA-256 is
`9b8d4075d75f9e6bce75652b1249deec643144e5b76b07e939493820a8feea1b`.

The conditional full-budget baseline calibration completed at execution rate 16
with rate-one grids held fixed (archive SHA-256
`f584151963f1e1a438c413ac427cf9835b624d292981d1ef0c85ff4312b0b294`). At
seed zero and one million proposals per search method, BS was 3.92032
microseconds, RS was 3.07200 microseconds (0.7836 of BS), fixed SA was
2.05125 microseconds (0.5232 of BS), and ASA was 2.39488 microseconds
(0.6109 of BS). Fixed SA beat RS by 33.23% and is close to the approximate
Figure-10 SA ratio of 0.50, while the approximate Figure-10 RS ratio of 0.59
is still materially lower than 0.7836. Fixed SA accepted 336,113 moves and
made 37,105 improving moves; ASA was 16.75% worse than fixed SA under the
matched budget. This is an exploratory, one-seed, assumed-rate result, not a
recovery of an unpublished hardware value or a confirmatory reproduction.

The routing diagnostics show why latency ratios alone cannot validate the
reconstruction. Weighted mean hops were 9.68 for BS, 24.41 for RS, and 19.09
for fixed SA, yet fixed SA had the lowest latency. It lowered the maximum
serialization/link bottleneck enough to outweigh its longer routes. Wu's
Figure 11 reports a 4.5x *reduction* in AlexNet-FC mean hops for DDPG versus
BS, so this direction mismatch is a structural diagnostic, not support for a
paper-equivalent result.

An independent review identified a second baseline sensitivity: Wu defines BS
as sequential physical chip/core placement but does not publish the numbering
of VMM and VVA logic cores. The reconstruction currently numbers each output
group's VMM cores immediately before its VVA core, potentially giving BS
strong locality in communication-dominated regimes. The rate-16 calibration
remains descriptive because no numerical pass tolerance was frozen before
launch. Logic-index ordering must be ablated before treating BS-normalized
ratios as calibrated.

That ablation completed at revision `7ee8bc2` with 20,000 proposals per search
method (archive SHA-256
`116b43ec90f1e60c0e86f6dada8dbfec4b8a8d529aaeecfcbd197a6b3efed214`).
Changing only logical numbering from output-group-interleaved to
all-VMM-then-VVA increased BS from 3.92032 to 9.92 microseconds. In contrast,
RS changed only from 3.14032 to 3.21536 microseconds and fixed SA from 2.40000
to 2.39328 microseconds. The normalized ratios therefore changed from 0.8010
to 0.3241 for RS and from 0.6122 to 0.2413 for fixed SA. BS traffic-weighted
mean hops also rose from 9.68 to 27.40, while the search methods' hop metrics
were similar across orderings.

This confirms that BS normalization is dominated by an unpublished logical
index convention in the reconstruction. It cannot identify which convention
the authors used. Method-development experiments therefore keep one declared
ordering fixed and make their causal learning comparison against a paired
frozen policy under that same ordering. Absolute objective values, evaluation
budgets, and routing diagnostics must accompany BS-normalized values.

The review also corrected two scope points. Myung et al.'s communication-only
proxy and unavailable routing simulator do not inform Wu's VVA compute rate.
Conversely, Wu's Equation 4 explicitly fixes the objective as the maximum
pipeline-stage latency; max versus sum is not an unresolved parameter. Figure
11 reports a 4.5x AlexNet-FC mean-hop reduction for DDPG versus BS, useful as a
structural diagnostic but not as an RS/SA acceptance threshold.

A second, controlled proposal neighborhood is now available through
`--ppo_asa_focus_bottleneck`. Every candidate is anchored on a task in the
current maximum-latency pipeline stage, and the anchor is guaranteed to move.
Both trained and frozen conditions use the same focused pools; candidate
ranking, Metropolis acceptance, temperature adaptation, and true-objective
accounting otherwise remain unchanged.

## Fixed-SA + RL development direction

The primary method-development path now places learning inside the strongest
working search mechanics. `--algo ppo_asa --ppo_asa_annealing fixed` retains
fixed SA's measured initial temperature, 0.99 cooldown, constant 1% task
perturbation, Metropolis acceptance, and exact incremental objective. PPO only
ranks a set of legal unevaluated neighbors, and exactly one selected neighbor
consumes the optimization objective evaluation for that step.

The training target is the selected proposal's immediate pre-acceptance cost
improvement. With `--ppo_gamma 0 --ppo_gae_lambda 0`, this is a contextual
proposal-selection problem rather than a long-horizon return-estimation
problem. The causal control uses the same code, seed, candidate pools,
annealing schedule, and budget with `--ppo_asa_disable_learning`; its zero
score head remains uniform. Plain fixed SA and ASA remain context baselines,
but failure to beat fixed SA does not by itself reject learning. The first
gate asks whether learned proposals beat their matched frozen control.

The short gate is `scripts/run_kaggle_rl_fixed_sa_gate.sh short` at 20,000
proposals for seed zero. A positive finite-update result advances to
`extended`, which uses 100,000 proposals for seeds zero, one, and two. This
focuses development on RL-guided annealing without requiring ASA's reheating
controller to work first.

The first short-gate launch at revision `50c41ad` stopped during command-line
validation before either optimizer evaluated a proposal (archive SHA-256
`7fcd62da2f302e0db3c1429584242a1a502af2097fc3d0ed02e54d73db2b0d27`).
The function accepted the intended contextual-bandit value `gamma=0`, but a
shared CLI check still required `gamma>0` for every PPO mode. That check now
allows zero only for PPO-ASA and retains the stricter masked-PPO rule. The
failed archive contains no optimization result and must not enter result
tables.

The corrected signed-delta short gate then completed at revision `7f371fa`
(archive SHA-256
`1ce71f4631b5befe571acff6943c3f9df66d4bf67ab0f4433379ce702e62cfa2`).
At seed zero and 20,000 proposals, learned fixed-SA reached 2.45040
microseconds and the matched frozen control reached 2.44864 microseconds, so
learning was 0.0719% worse. Both used 20,001 total objective evaluations,
started at 3.81840 microseconds, and used identical measured temperature,
cooling, perturbation fraction, and candidate count. Plain fixed SA reached
2.40000 microseconds and ASA reached 2.46528 microseconds as context baselines.

This was not a frozen-policy failure. PPO performed 78 finite updates; score
head weight norm grew from 0.0028 to 0.348, late logit standard deviation was
0.102, and late probability L1 distance from uniform was 0.047. The learned
chain made 1,307 improving moves versus the control's 1,218, but did not find
a better minimum. Its signed proposed-delta reward was dominated by uphill
proposals (mean -4.77 normalized units). The next gate therefore uses
improvement-only positive delta, increases learning rate from `1e-4` to
`3e-4`, and reduces entropy regularization from `0.02` to `0.001`. Fixed-SA
cooling and acceptance remain frozen.

The improvement-only short gate completed at revision `168add0` (archive
SHA-256
`3abb540207ada402b7394013f47fca16e1641bec71279192daf8548410c85148`).
Learned fixed-SA reached 2.43328 microseconds versus 2.44864 for the frozen
control, a 0.6273% improvement with identical 20,001-evaluation budgets and
annealing schedules. It also beat ASA's 2.46528 microseconds by 1.298% and the
matched-configuration 20,000-sample random-search result by 22.51%, while
plain fixed SA remained better at 2.40000 microseconds by 1.387%.

This advances to the prespecified extended gate but is not yet evidence across
seeds. PPO made 78 finite updates, but late probability L1 distance from
uniform was only 0.00765 and learned proposals produced 1,142 improving moves
versus 1,218 for control. The better minimum may therefore be a useful policy
effect or a favorable single-chain path. The three-seed gate requires positive
mean improvement and wins in at least two seeds before supporting a learning
claim.

The 100,000-proposal, three-seed extended gate completed at revision `8e7c7e9`
(archive SHA-256
`71a95ba004476beeb4a4a13cc0ad7a918a5c869327f203a05d1369bc98d740c8`).
Learned RL+SA beat its frozen control in two of three seeds. Per-seed relative
improvements were -0.896%, +1.125%, and +0.086%, giving a mean of +0.105% and
passing the prespecified directional rule. Mean costs were 2.23989
microseconds learned, 2.24213 control, 2.20576 plain fixed SA, and 2.41760 ASA.
Thus learned fixed-SA proposals beat ASA by 7.35% on mean cost but remained
1.55% worse than plain fixed SA.

The effect is not statistically resolved. The mean paired absolute benefit was
0.00224 microseconds with a sample standard deviation of 0.02272; the paired
three-seed 95% t interval was [-0.05420, +0.05868] microseconds. Expressed as
relative improvement, the interval was approximately [-2.41%, +2.62%]. Every
learned run made 391 finite updates and every learned/control pair used 100,001
objective evaluations. This is therefore a weak positive pilot with correct
mechanics, not a publication-level learning claim.

The next method experiment follows the project goal directly: keep the same
improvement-only proposal policy and put it back inside adaptive SA. The runner
now supports `adaptive-short` and `adaptive-extended`. These preserve matched
learned/frozen schedules within each seed while allowing ASA's temperature
feedback, reheating, and neighborhood adaptation. The adaptive short gate must
beat its frozen adaptive control before any extended run.

The adaptive short gate completed at revision `d4d9e38` (archive SHA-256
`8d9db7e72154480f5b14dd291d56deb5cc52c769af5d351064b5f5e62ed0e349`).
Learned RL+ASA reached 2.64784 microseconds versus 2.63840 for its frozen
adaptive control, a 0.3578% loss with identical 20,001-evaluation budgets.
Plain ASA reached 2.46528 microseconds and fixed SA reached 2.40000
microseconds. The learned adaptive hybrid was therefore 7.40% worse than plain
ASA and 10.33% worse than fixed SA.

PPO mechanics were active: the learned run made 78 finite updates, its score
head norm reached 0.152, and late probability L1 distance from uniform reached
0.0144. It made 855 improving moves versus the frozen control's 824, but did
not find a better minimum. Both learned and frozen adaptive hybrids reheated
four times and ended with the same 0.5% perturbation fraction. This is a valid
negative result for the current adaptive controller, so `adaptive-extended`
is stopped.

The primary learned method remains RL+fixed SA, which at least passed the
three-seed directional gate. A future RL+ASA attempt should redesign adaptation
around the working geometric schedule and first show that its frozen adaptive
control is competitive. Adding PPO to an already weaker controller does not
isolate a useful learning contribution.

The next fixed-SA learning gate now uses a contextual bandit rather than PPO.
The bandit predicts signed immediate gain from the same global state
and legal candidate features, learns only from the selected move's budgeted
objective evaluation, and uses decaying epsilon-greedy exploration. This is a
one-step RL formulation aligned with the observed reward; it removes PPO's
unnecessary value estimation and trajectory credit assignment while leaving
candidate generation, fixed cooling, Metropolis acceptance, and evaluation
count unchanged. Its frozen control stays uniform. This implementation is
The first `bandit-short` run at revision `6a4c2fb` was negative (archive
SHA-256 `46808a06a40a5595a85145f066317ded0741096db891a96f8d063d9eeb22af0e`).
Learned cost was 2.64544 microseconds versus 2.44864 for the frozen control,
an 8.04% loss. It made 77 finite updates, but late policy distance from uniform
reached 0.84375 and its mean predicted gain was positive while selected moves
had mean clipped signed reward near -9.81. Positive-only targets, balanced
positive replay, and rapid decay to 10% exploration produced severe
overprediction and harmful exploitation. The corrected gate uses signed
targets, uniform replay, 50% final exploration, and decay over the full run.

The corrected signed-target `bandit-short` gate at revision `a52c073`
completed successfully (archive SHA-256
`e84ce397dd46399c88a75336a3599b94a5911fd957f8affb2236ae03d3e182f8`).
Learned bandit+fixed-SA reached 2.44560 microseconds versus 2.44864 for its
frozen control, a 0.1242% improvement at identical 20,001-evaluation budgets.
It beat ASA at 2.46528 microseconds by 0.7983%, but plain fixed SA remained
1.90% better at 2.40000 microseconds. The learner made 77 finite updates. Late
mean prediction was -3.95 versus a sampled signed target of -4.50, final
exploration was 0.513, and probability distance from uniform was 0.456. This
fixes the first run's severe positive overprediction and passes the
prespecified short directional gate, but the 0.124% one-seed effect is too
small for a claim. Proceed to the unchanged three-seed `bandit-extended` gate.

The three-seed signed-bandit extended gate at revision `d4b46ce` completed
(archive SHA-256
`f9a81276acc281f5f9e51aad95480db0b32da1e6284fb90b0bf7aa7d12be324a`).
Learned proposals beat the frozen control in two of three seeds. Per-seed
improvements were -1.129%, +2.072%, and +0.532%; mean per-seed improvement was
+0.491%. Mean costs were 2.23131 microseconds learned, 2.24213 control, 2.20576
fixed SA, and 2.41760 ASA. The learned hybrid beat ASA by 7.70% but remained
1.16% worse than fixed SA. The three-seed 95% t interval for improvement over
control was approximately [-3.49%, +4.47%], so the directional gate passes but
the effect is not statistically resolved.

The archive also exposed an implementation error in the declared exploration
schedule. Replay capacity was 50,000, and epsilon decay used current replay
length rather than total observations. Epsilon therefore stopped at 0.7525 in
the 100,000-proposal runs instead of reaching 0.5. The result remains a valid
test of the actual 0.7525-exploration configuration, but not of the declared
0.5 endpoint. The code now counts all bandit observations independently of
replay eviction and reports final epsilon in run metadata. Rerun before using
the extended result as final method evidence.

The epsilon-fixed paired rerun at revision `0608cf0` completed (archive
SHA-256 `8da3ae0c743abdec001ab6a2e743c5c5698e07a0857024a4ba8f89614f55499a`).
All learned runs reached final epsilon 0.5026, recorded 99,968 bandit
observations, made 389 finite updates, and used 100,001 total objective
evaluations. Learned costs were 2.26688, 2.17600, and 2.23904 microseconds;
matched frozen-control costs were 2.26704, 2.23216, and 2.22720. Relative
improvements were +0.007%, +2.516%, and -0.532%. The method therefore won two
of three seeds with +0.664% mean per-seed improvement, passing the directional
gate. Its approximate paired three-seed 95% t interval remains wide at
[-3.38%, +4.70%].

Combining the rerun with already archived same-seed baselines gives mean costs
of 2.22731 microseconds learned, 2.24213 frozen control, 2.20576 fixed SA, and
2.41760 ASA. Learned bandit+SA beats ASA by 7.87% but remains 0.98% worse than
fixed SA. This is repeated evidence of a small learned-proposal benefit over
the causal control, not yet evidence that the hybrid is the strongest search
method. Freeze this configuration before adding seeds or workloads.

The frozen-configuration held-out test on previously unused seeds 3--5 at
revision `073f003` completed (archive SHA-256
`e229999cba4d9ed89c0a8fafd1c3dcf7c01bfd339c39e7c87e7759440cf0387e`).
RL+SA beat its frozen control in all three held-out pairs. Per-seed improvements
were 2.284%, 1.537%, and 2.074%, for a mean of 1.965% and a paired 95% t
interval of [1.008%, 2.922%]. Mean held-out costs were 2.20656 microseconds
learned, 2.25077 control, 2.22160 fixed SA, and 2.43248 ASA. RL+SA beat fixed
SA in two of three seeds and by 0.669% on average, but that interval crosses
zero. It beat ASA in all three seeds by 9.288% on average, with interval
[7.833%, 10.742%]. All mechanics and budgets passed: 389 finite updates,
99,968 observations, final epsilon 0.5026, and 100,001 objective evaluations
per learned/control run.

Across all six epsilon-corrected seeds 0--5, RL+SA beat the frozen control in
five of six pairs and improved by 1.314% on average; the combined interval is
approximately [-0.024%, 2.653%], narrowly including zero because seeds 0--2
were more variable. RL+SA and fixed SA are effectively tied across the six
seeds (RL+SA 0.161% worse by mean paired relative difference, interval crossing
zero), while RL+SA beats ASA in all six by 8.572% on average. The held-out
control result is positive descriptive held-out-seed evidence. It is not an
unbiased confirmatory test because the method was developed on the same graph,
the seed unit does not create a new instance, and n=3 gives a fragile interval.
The next requirement is cross-workload validation, not more tuning on
AlexNet-FC.

The first frozen cross-workload transfer gate completed on AlexNet-CONV at
revision `cedf929` (archive SHA-256
`fa0fb4aff3b2e746f32a0b7126b49dbf4f9688611b65f8ded16521bf419b4c3d`).
At seed zero and 20,001 objective evaluations, learned bandit+SA reached
37.02117 microseconds versus 37.44656 for its frozen control, 37.87195 for
fixed SA, and 38.72273 for ASA. It therefore improved over control by 1.136%,
fixed SA by 2.246%, and ASA by 4.394%. The learner made 77 finite updates,
recorded 19,968 observations, and reached final epsilon 0.5128. No learner,
annealing, feature, or exploration setting was changed from the frozen
AlexNet-FC method. This passes the one-seed transfer gate and justifies a
three-seed 100,000-evaluation AlexNet-CONV extension.

The three-seed AlexNet-CONV extension at revision `ab8b352` completed (archive
SHA-256 `20ded5670df2e2a97eae86d4af4bd3e5608070ef0d1b4ca4549eeb8959095491`).
At 100,000 proposals, learned RL+SA, frozen control, and fixed SA all reached
the same 37.02117-microsecond best cost in every seed. RL made 389 finite
updates per seed and reached final epsilon 0.5026, so this is a valid optimizer
tie rather than a failed learning run. ASA means were worse, but varied by
seed.

The short-run CONV advantage was therefore an early-convergence signal rather
than a better final minimum. Learned RL+SA reached the shared floor at roughly
59,932, 36,432, and 33,432 proposals; frozen control reached it at roughly
61,132, 30,432, and 41,032. Mean best-so-far trajectory cost favored learned
proposals by 0.357%, but the three-seed interval crossed zero. Fixed SA reached
the floor no earlier than the 64,000, 92,000, and 62,000 logged checkpoints,
showing that candidate-pool proposal mechanics accelerate this workload, while
the extra contribution from learning is not resolved. Do not claim improved
final CONV quality or statistically supported sample efficiency.

The next frozen cross-workload short gate completed on VGG16-FC at revision
`40f60e9` (archive SHA-256
`9d29cbafcd59b3cb362cbe653129a3935b96f45de5e1fc06737be6d8001a3969`).
At seed zero and 20,001 objective evaluations, learned bandit+SA reached
4.59088 microseconds versus 4.67664 for its frozen control, 4.66032 for fixed
SA, and 4.72016 for ASA. It improved over control by 1.834%, fixed SA by
1.490%, and ASA by 2.739%. The learner made 77 finite updates and reached
epsilon 0.5128. No learner or search hyperparameter changed from the frozen
AlexNet-FC method. This passes the one-seed short gate, but it is not yet
multi-seed evidence. Run the unchanged three-seed, 100,000-evaluation
VGG16-FC extension before making a transfer claim.

The three-seed VGG16-FC extension at revision `6ab5a41` completed (archive
SHA-256 `3e9f5874ebc704fee430f2b7913a236432a72fafc1fa6395cbb4cffee2826def`).
At 100,001 evaluations per learned/control run, learned RL+SA beat its frozen
control in all three pairs. Per-seed improvements were 1.729%, 2.593%, and
2.444%; the mean was 2.255% with a paired 95% t interval of approximately
[1.107%, 3.403%]. Each learner made 389 finite updates and reached epsilon
0.5026, while each frozen control made zero updates and remained uniform.

Learned RL+SA also beat fixed SA in all three seeds by 0.908% on average,
although the three-seed interval [-0.489%, 2.305%] crosses zero. It beat ASA
in all three by 5.418% on average with interval [3.656%, 7.179%]. This passes
the prespecified extended gate and is the first positive cross-workload final
quality result for the frozen learned proposal method. Random search was not
included in this archive and must not be inferred from these comparisons.

The genuinely new VGG16-FC seeds 3--5 completed at revision `bacc334`
(archive SHA-256
`87e08605cc75b02e9de3d595e73f8781ca236d86f3068d4a7402aac3c3a6c7ba`).
Learned RL+SA beat focused-uniform control in all three pairs by 1.637% on
average, with paired three-seed t interval [0.336%, 2.938%]. It beat ASA in
all three by 3.793% and matched fixed SA: one win, two small losses, and 0.122%
mean advantage with an interval crossing zero. The newly included random
baseline was worse in all three seeds by 22.405% on average.

Across all six VGG16-FC seeds, learned RL+SA beat focused-uniform control 6/6
times by 1.946% on average, with paired t interval [1.362%, 2.530%]. The exact
two-sided sign test for six wins is 0.03125, reported descriptively because the
first three seeds followed the positive short screen. RL+SA beat ASA 6/6 by
4.605% and beat fixed SA 4/6 by 0.515%; the fixed-SA interval
[-0.200%, 1.230%] crosses zero. On held-out seeds, mean wall-clock time was
59.0 minutes for learned and frozen-control runs, 43.5 for fixed SA, 41.4 for
ASA, and 61.3 for random. Claims must distinguish objective-call quality from
wall-clock efficiency.

An external methodology audit correctly identified a remaining attribution
gap. The matched frozen control isolates the effect of selecting candidates
non-uniformly with the trained scorer, but it does not show that learned
scoring is better than directly exploiting the hand-built `proxy_gain`
feature. Both learned and frozen runs also use a pool with 50% bottleneck-
focused candidates, whereas plain fixed SA uses a different proposal
neighborhood and temperature calibration. Therefore the supported claim is a
2.255% VGG16-FC benefit over the focused-uniform control under equal true-
objective calls. It is not yet a claim of superiority over the best
non-learned heuristic, equal-wall-clock superiority, or broad RL
generalization. The immediate check is a matched proxy-greedy control, followed
by the already planned VGG16-FC seeds 3--5. A larger control ladder and 20-seed
study are possible publication-strength extensions, not conditions for treating
the current result as a valid positive exploratory result.

The local 1,001-evaluation focused smoke produced 24.0627 microseconds trained,
24.1598 for its focused uniform control, and 23.9629 for ordinary ASA. Trained
PPO beat its control by 0.40% but lost to ASA by 0.42%. Its improving-move rate
rose to 7.5%, while neutral proposals fell to 72.8%. This passes a bounded
learning-only gate and justifies one 4,101-evaluation `focused-extended` run;
it does not justify five seeds yet.

That extended gate is complete at revision `91fd7dc`. Trained focused PPO-ASA
reached 23.7792 microseconds, its frozen focused control reached 23.7595, and
ordinary ASA reached 23.4318. Training therefore finished 0.083% behind its
control and 1.482% behind ASA. Focusing reduced neutral proposals to 37.2%,
but trained PPO found fewer improving moves than control (137 versus 151) and
its best value stopped improving around evaluation 1,232. This closes the
short-chain PPO-ASA path. The next RL experiment requires multi-instance,
training-scale data rather than another neighborhood or hyperparameter tweak.

## Five-seed local control study (2026-09-30)

Completed five AlexNet-CONV seeds with 12 training placements and six retained deterministic evaluations per seed. DDPG also used 64 baseline samples (82 selectable candidates total). RS used 82 samples; SA/ASA used 82 proposals plus initialization. Mean best latency: trained DDPG 38.24062 µs, untrained control 38.46750 µs, RS 41.13328 µs, fixed SA 42.99555 µs, ASA 42.32438 µs, sequential 42.12586 µs. Training beat its untrained control in two seeds, tied two, and lost one. Its mean benefit was only 0.22687 µs (about 0.59%); the exploratory paired 95% interval [-0.33818, 0.79193] µs includes zero. The initial policy and collision repair already yield strong layouts, so gains over RS/SA do not establish learning. Short annealing budgets and temperature calibration limit that comparison. See `runs/local-cpu-5seed-2026-09-30/STUDY.md` and raw reports. Next priority: improve and validate learning against the untrained control before scaling the training budget.

## Target-network BatchNorm repair (2026-09-30)

The later seed-4 extension to 100 placements did not improve the retained
37.635625-microsecond layout. Its deterministic policy ended at 48.081328125
microseconds with only three intended cores and 181 collision repairs. The
archived normalization, compact-state, and untrained-control runs are therefore
negative exploratory ablations rather than validated repairs. Their settings,
reports, and small reproducibility runners are preserved under
`runs/NORMALIZATION_EXPERIMENTS.md`; generated model checkpoints remain
excluded because each is hundreds of megabytes.

The CNN target actor and critic previously calculated Bellman targets in training mode, so BatchNorm made target values depend on the composition of each replay minibatch. Target buffers also were not synchronized after the soft parameter update, and the critic accumulated unnecessary gradients while optimizing the actor. The code now evaluates targets in inference mode, copies BatchNorm buffers from the online networks, and freezes the critic for the actor-only step. A regression test checks fixed target outputs across minibatch companions, buffer synchronization, a single critic statistics update, and restored critic gradients.

The same bounded five-seed DDPG diagnostic after this repair had mean best latency 38.09883 µs versus 38.24062 µs before (0.14180 µs, 0.37%, lower). The four strictly two-thread comparable seeds averaged 38.04920 µs versus 38.15555 µs (0.28% lower); three improved and one worsened. This is numerical-stability evidence only, not evidence that DDPG has converged. The architecture/training fingerprint was changed deliberately, so earlier DDPG checkpoints cannot resume. See `runs/local-cpu-batchnorm-2026-09-30/STUDY.md`.

## VGG16-CONV contention-bandit gate (15 October 2026)

The three-seed short gate completed at revision `a00784a` with 20,001 true
objective evaluations for each learned and matched-control run. The archive
SHA-256 is
`187fa7c1b1628583318d9f7c6c477dd283c8989ff7a6b216838d3f6376238843`.
Mean best latency was 728.898 microseconds for the learned contention bandit,
966.531 for shuffled-label learning, 558.921 for proxy-greedy, 1,701.868 for
bottleneck-random, 609.994 for fixed SA, and 2,158.185 for random search.

The learned method beat shuffled labels in two of three seeds, bottleneck-
random in all three by 57.172% on average, and random search in all three by
66.218%. It lost to proxy-greedy in every seed by 31.208% on average and to
fixed SA in every seed by 19.668%. It therefore failed the prespecified gate
requiring wins over shuffled labels, proxy-greedy, and fixed SA in at least
two seeds. Do not run the unchanged 100,000-evaluation extension.

This was a valid learning run rather than a frozen or random policy: each
learner made 77 finite updates, final learned score-head norms were
0.526--0.612, and final logit standard deviations were 0.638--0.793. The
shuffled-label heads remained much smaller (0.052--0.095) with logit standard
deviations near 0.004--0.005. Those diagnostics show that the real labels
contain learnable structure, but the resulting ranking is inferior to direct
use of the hand-built proxy. The next method must learn a residual improvement
over proxy-greedy or control a different part of search, and proxy-greedy must
remain a mandatory matched control.

## Bounded proxy correction (15 October 2026)

Implemented opt-in bounded proxy correction, relative-cost state inputs,
selected-reward replay and shared candidate-pool caching. All 44 CPU tests
passed with no skips. A 36-second four-method CLI smoke passed matching
65-call budgets and archive checks. These validate mechanics, not real gains.
Next: VGG16-CONV seeds 3--5 at 20,001 calls/method against proxy-greedy,
fixed SA and random. See [RESIDUAL_BANDIT.md](RESIDUAL_BANDIT.md).
