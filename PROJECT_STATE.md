# Project state: paper-faithful DNN core placement

Updated 2026-10-08.

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
7. Close PPO-ASA if that frozen policy does not beat uniform in at least four of five seeds. Claim an optimizer improvement only if it also beats ASA.
8. Do not scale unchanged AlexNet-FC masked PPO; its 3,000-placement run tied the frozen control and lost to ASA.
9. Produce a matched-budget table for BS, RS, SA, ASA, paper DDPG, masked PPO, and the hybrid only after the hybrid gate passes.
10. Add activation-buffer/streaming behavior and router timing when defensible evidence is available.
11. Implement and validate true large-batch throughput before reproducing that panel of Figure 10.

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

A second, controlled proposal neighborhood is now available through
`--ppo_asa_focus_bottleneck`. Every candidate is anchored on a task in the
current maximum-latency pipeline stage, and the anchor is guaranteed to move.
Both trained and frozen conditions use the same focused pools; candidate
ranking, Metropolis acceptance, temperature adaptation, and true-objective
accounting otherwise remain unchanged.

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

The CNN target actor and critic previously calculated Bellman targets in training mode, so BatchNorm made target values depend on the composition of each replay minibatch. Target buffers also were not synchronized after the soft parameter update, and the critic accumulated unnecessary gradients while optimizing the actor. The code now evaluates targets in inference mode, copies BatchNorm buffers from the online networks, and freezes the critic for the actor-only step. A regression test checks fixed target outputs across minibatch companions, buffer synchronization, a single critic statistics update, and restored critic gradients.

The same bounded five-seed DDPG diagnostic after this repair had mean best latency 38.09883 µs versus 38.24062 µs before (0.14180 µs, 0.37%, lower). The four strictly two-thread comparable seeds averaged 38.04920 µs versus 38.15555 µs (0.28% lower); three improved and one worsened. This is numerical-stability evidence only, not evidence that DDPG has converged. The architecture/training fingerprint was changed deliberately, so earlier DDPG checkpoints cannot resume. See `runs/local-cpu-batchnorm-2026-09-30/STUDY.md`.
