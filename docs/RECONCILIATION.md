# Reconciliation and validation status

Status date: 2026-09-30

This branch began by reconciling GitHub commit `41f9d02` with the user-supplied Gemini archives `files (4).zip` and `files (5).zip`. Their three shared files were byte-identical; archive 4 also contained topology and environment files. Archive prose and old logs are historical evidence, not proof of current behavior.

## Corrections retained from the archive audit

- FX dependency tracing replaces sequential hook order.
- Overlapping channel ranges replace unconditional cross-layer all-to-all traffic.
- Pooling and flatten traffic use consumer shapes.
- Remainder tiles conserve unsplit layer work.
- Row-major policy coordinates convert to chip-major physical IDs.
- Proxy and seconds-based objectives remain separate.
- SA accepts relative to current cost, uses 0.99 cooling, and can move into free cores.
- Invalid capacity, cyclic graphs, unavailable devices, and incompatible checkpoints fail explicitly.
- Checkpoints store networks, optimizers, baseline, best placement, counters, and RNG state. Replay remains unpersisted, so resume is not bit-exact.
- OU noise can use one explicit absolute placement horizon, preventing early checkpoint stages from exhausting exploration for later resumed stages.

## Paper-mode additions

`--partition_mode paper_targets` reconstructs per-layer input/output partition grids while matching Figure 6 aggregate counts exactly and enforcing the 64 KB weight buffer. The reconstruction now balances estimated VMM and VVA cycles as well as each layer's target core count; this replaces grids such as AlexNet CONV2 `(M=1, N=61)`, whose single VVA task made the objective almost placement-insensitive. `--workload_region conv|fc` optimizes each class independently in a disjoint whole-chip mask.

`--timing_model paper_pipeline --routing_model paper_xy` uses Table 1 work/precision/bandwidth values, configurable CONV block scaling, deterministic X-then-Y routing through a lower-left chip-periphery gateway, routed per-source byte-hop time, and shared directed-link contention. Reports include edge and traffic-weighted hop counts plus on/off-chip link-load summaries. A preflight samples random valid placements, reports the objective range and bottleneck decomposition, and stops optimization if the sampled span is below 0.1% unless explicitly overridden.

`--agent_arch paper_cnn` implements the Figure 9 spatial actor and critic. Paper mode retains the sparse terminal reward, actor/critic learning rates 0.0002/0.001, gamma 0.98, and minibatch 64. The seconds-based latency is multiplied by 400 MHz before the square-root reward, matching the cycle units used by the paper while preserving the optimization ordering. BS, random search, SA, adaptive SA, DDPG, hybrid DDPG+ASA, multi-seed summaries, and explicit placement budgets are available. `src/run_paper_experiment.py` creates separate CONV and FC suites.

## Assumptions that remain

The paper does not publish per-layer partition grids, VVA throughput, exact physical masks, full GRS behavior, or workload-specific block counts. The implementation uses MAC-proportional exact-count grids plus a documented VMM/VVA balance score, one VVA operation per cycle, minimum contiguous whole-chip regions, a lower-left gateway inferred from Figure 3, and four CONV blocks by default from Figure 7's example.

Input/activation-buffer occupancy and stalls, exact multicast, router startup, transformation-unit costs, and compute/communication overlap are not modeled. An `N=1` VVA task receives one output-transformation pass so that it is not free, but this cost is a reconstruction assumption. Residual branch traffic is retained, while the add executes in the destination transformation/VVA path without a separate Figure 6 core. Concatenation and grouped convolution are unsupported.

The paper also leaves `z`, OU details, replay capacity, CNN padding, LRN parameters, and target-update details incomplete. Reports and the paper-run manifest identify these choices. Potential reward shaping and the `mlp`/`cnn` agents are improvement conditions, not paper mode.

Potential shaping uses a baseline-normalized potential. This preserves the fixed-horizon terminal objective through telescoping while avoiding the unstable, cycle-scaled intermediate rewards observed in the initial diagnostic.

## Validation

The original 2,080-placement AlexNet CONV run used the earlier objective. Its reported `0.00524885` latency was almost exactly the placement-independent CONV2 VVA compute time (`0.0052488` seconds), and random placements varied by only about `0.0015%`. That checkpoint is incompatible with the corrected objective and is retained only as historical evidence.

The corrected pure-model AlexNet CONV audit produces exact Figure 6 count `183` with grids `(7,2), (12,4), (6,4), (6,6), (5,5)`. Across 256 seeded random placements, latency spanned `16.34%`; the best sample's bottleneck was `86.60%` compute and `13.40%` communication. The corresponding FC audit preserves 932 cores with grids `(16,36), (16,16), (4,16)`; 64 random placements spanned `4.05%`, with the best sample `92.29%` compute and `7.71%` communication. A bounded 2,000-candidate CONV execution check gave BS-normalized values RS `0.9697`, fixed SA `0.9899`, and ASA `0.9697`. These figures prove the evaluator is placement-sensitive; they are not convergence or paper-result claims.

Current local validation uses torch 2.14.0+cpu and torchvision 0.29.0+cpu:

- `src/test_multi_chip.py`, all 19 reconciliation tests, and the bounded CPU device validator pass with no skipped tests. The validator completed 57 measured optimizer updates and a checkpoint round trip.
- Real torchvision AlexNet extraction returns the corrected 183 CONV and 932 FC partitions. Across 64 placements, the CONV objective spans `13.89%` and FC spans `4.05%`.
- A three-placement CPU `paper_cnn` smoke completed 120 optimizer updates, checkpointing, deterministic diagnostics, and reporting at about 62 seconds per placement. The noisy best was `7.29%` below BS, but the deterministic policy worsened by placement three; the run validates mechanics, not learning.
- At a 2,000-candidate CONV budget, BS-normalized results were RS `0.9697`, fixed SA `0.9899`, and ASA `0.9697`. These are single-seed bounded execution checks.
- A staged-training diagnostic showed that stage-relative noise decay collapsed exploration. With a fixed 1,000-placement horizon, intended-core diversity at placement six increased from 45 to 117 and occupied-core repairs decreased from 177 to 101. Baseline-normalized potential shaping reduced the placement-six mean critic loss from `294` to `0.111` and held deterministic cost at `3.9148e-05`; the unnormalized condition worsened to `4.3780e-05`. These short diagnostics establish numerical behavior only.
- A 30-placement follow-up found an early deterministic candidate of `3.829734375e-05` that the previous bookkeeping discarded while retaining a worse noisy candidate. The opt-in deterministic-retention mode now preserves such candidates and reports training, diagnostic, and total evaluation counts. Its matched 12-placement verification retained that result from 18 total evaluated candidates. It remains an improvement condition outside frozen paper mode.
- A subsequent 10,000-candidate ASA warm-start accepted 9,585 moves and found 113 improving transitions without beating the retained DDPG candidate. Hybrid accounting now uses DDPG's total evaluated candidates, including retained deterministic diagnostics, plus the ASA budget.

These are functionality checks. They do not demonstrate learning, convergence, H100 performance, or agreement with the paper's percentages. CUDA tensor placement is implemented, but the planned H100 12 GB slice has not been available locally.

Run:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/run_paper_experiment.py --model alexnet --device cuda --output_dir runs/paper-alexnet --dry_run
```

See `README.md` for setup and commands, `PROJECT_STATE.md` for the current handoff, and `NEXT_STEPS.md` for remaining reproduction gates. `run_multi_chip_fast.py` and the older single-chip programs are outside the validated path.

## Five-seed local control study (2026-09-30)

Completed five AlexNet-CONV seeds with 12 training placements and six retained deterministic evaluations per seed. DDPG also used 64 baseline samples (82 selectable candidates total). RS used 82 samples; SA/ASA used 82 proposals plus initialization. Mean best latency: trained DDPG 38.24062 µs, untrained control 38.46750 µs, RS 41.13328 µs, fixed SA 42.99555 µs, ASA 42.32438 µs, sequential 42.12586 µs. Training beat its untrained control in two seeds, tied two, and lost one. Its mean benefit was only 0.22687 µs (about 0.59%); the exploratory paired 95% interval [-0.33818, 0.79193] µs includes zero. The initial policy and collision repair already yield strong layouts, so gains over RS/SA do not establish learning. Short annealing budgets and temperature calibration limit that comparison. See `runs/local-cpu-5seed-2026-09-30/STUDY.md` and raw reports. Next priority: improve and validate learning against the untrained control before scaling the training budget.

## Target-network BatchNorm repair (2026-09-30)

The Figure 9 CNN target actor and critic now run in evaluation mode while computing Bellman targets. Their BatchNorm buffers are synchronized from the online networks after each update, and the critic is frozen during the actor-only update. This removes replay-minibatch-dependent target statistics and unintended critic BatchNorm updates. A five-seed bounded CPU repeat lowered mean best latency from 38.24062 µs to 38.09883 µs (0.37%); in the four exactly two-thread-comparable seeds the reduction was 0.28%, with three improvements and one regression. It is a short numerical-stability diagnostic, not a learning or paper-performance claim. The revised training fingerprint intentionally rejects earlier DDPG checkpoints. See `runs/local-cpu-batchnorm-2026-09-30/STUDY.md`.
