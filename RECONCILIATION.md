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
- OU noise uses an explicit absolute placement horizon so cumulative checkpoint stages share one exploration schedule.

## Paper-mode additions

`--partition_mode paper_targets` reconstructs per-layer input/output partition grids while matching Figure 6 aggregate counts exactly, enforcing the 64 KB weight buffer, and balancing estimated VMM/VVA cycles. This replaces grids such as AlexNet CONV2 `(M=1,N=61)`, whose single VVA task made the old objective almost placement-insensitive. `--workload_region conv|fc` optimizes each class independently in a disjoint whole-chip mask.

`--timing_model paper_pipeline --routing_model paper_xy` uses Table 1 work/precision/bandwidth values, configurable CONV block scaling, deterministic X-then-Y routing through a lower-left chip-periphery gateway, routed per-source byte-hop time, and shared directed-link contention. A preflight samples valid placements, reports the objective range and bottleneck decomposition, and stops optimization if the sampled span is below 0.1% unless explicitly overridden.

`--agent_arch paper_cnn` implements the Figure 9 spatial actor and critic. Paper mode retains the sparse terminal reward, actor/critic learning rates 0.0002/0.001, gamma 0.98, and minibatch 64. Seconds-based latency is multiplied by 400 MHz before the square-root reward, matching the cycle units used by the paper. BS, random search, SA, DDPG, multi-seed summaries, and explicit placement budgets are available.

## Assumptions that remain

The paper does not publish per-layer partition grids, VVA throughput, exact physical masks, full GRS behavior, or workload-specific block counts. The implementation uses MAC-proportional exact-count grids plus a documented VMM/VVA balance score, one VVA operation per cycle, minimum contiguous whole-chip regions, a lower-left gateway inferred from Figure 3, and four CONV blocks by default from Figure 7's example.

Input/activation-buffer occupancy and stalls, exact multicast, router startup and packet scheduling, exact transformation-unit costs, and compute/communication overlap are not modeled. An `N=1` VVA task receives one minimum output pass so it is not free; this is a reconstruction assumption. Residual branch traffic is retained, while the add executes in the destination transformation/VVA path without a separate Figure 6 core.

The paper also leaves `z`, OU details, replay capacity, CNN padding, LRN parameters, and target-update details incomplete. Reports and the paper-run manifest identify these choices. Potential reward shaping and the `mlp`/`cnn` agents are improvement conditions, not paper mode.

Potential shaping uses a baseline-normalized potential. Its fixed-horizon shaping terms still telescope to zero, while intermediate feedback remains near unit scale under the 400-MHz reward conversion.

## Validation

The earlier 2,080-placement AlexNet CONV run used a nearly flat objective: its `0.00524885` latency was almost exactly the placement-independent 5.2488-ms CONV2 VVA compute time. That checkpoint is incompatible with the corrected model.

The corrected pure-model AlexNet audits preserve Figure 6 counts. CONV uses grids `(7,2), (12,4), (6,4), (6,6), (5,5)` and spans `16.34%` across 256 random placements; the best sample is `86.60%` compute. FC uses `(16,36), (16,16), (4,16)` and spans `4.05%` across 64 placements; the best sample is `92.29%` compute. These prove placement sensitivity, not learning or paper-result reproduction.

Current local validation uses torch 2.14.0+cpu and torchvision 0.29.0+cpu:

- `src/test_multi_chip.py`, all 16 reconciliation tests, and the bounded CPU device validator pass with no skipped tests.
- The validator completed 57 measured optimizer updates and a checkpoint round trip.
- Balanced-grid selection, nonzero `N=1` VVA work, reward scaling, objective decomposition/sensitivity, topology, extraction, and optimizer updates are regression-tested.
- A short staged-training diagnostic showed that a fixed exploration horizon increased intended-core diversity from 45 to 117 and reduced occupied-core repairs from 177 to 101 at placement six. Normalized potential shaping reduced mean critic loss from `294` to `0.111` and avoided the deterministic-cost degradation seen in the unnormalized condition. These are numerical checks, not learning or convergence claims.

These are functionality checks. They do not demonstrate learning, convergence, H100 performance, or agreement with the paper's percentages. CUDA tensor placement is implemented, but the planned H100 12 GB slice has not been available locally.

Run:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/run_paper_experiment.py --model alexnet --device cuda --output_dir runs/paper-alexnet --dry_run
```

See `README.md` for setup and commands, `PROJECT_STATE.md` for the current handoff, and `NEXT_STEPS.md` for remaining reproduction gates. `run_multi_chip_fast.py` and the older single-chip programs are outside the validated path.
