# Project state: paper-faithful DNN core placement

Updated 2026-10-12.

## Goal and branch

The goal is to reproduce, then improve upon Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning* (ACM TODAES 2020, DOI 10.1145/3418498).

The maintained development branch is `cpu` in `micskrcb/DNN_MAPPING`; it is a descendant of `codex/reconciled-paper-implementation` and still supports `--device cuda`. `main` remains historical. Code and tests are the source of truth for current behavior; old run logs use different objectives and cannot be compared numerically with paper mode.

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
