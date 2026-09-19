# Reconciliation and validation status

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

## Paper-mode additions

`--partition_mode paper_targets` reconstructs per-layer input/output partition grids while matching Figure 6 aggregate counts exactly and enforcing the 64 KB weight buffer. `--workload_region conv|fc` optimizes each class independently in a disjoint whole-chip mask.

`--timing_model paper_pipeline --routing_model paper_xy` uses Table 1 work/precision/bandwidth values, configurable CONV block scaling, deterministic X-then-Y routing through a lower-left chip-periphery gateway, and shared directed-link contention. Reports include edge and traffic-weighted hop counts plus on/off-chip link-load summaries.

`--agent_arch paper_cnn` implements the Figure 9 spatial actor and critic. Paper mode retains the sparse terminal reward, actor/critic learning rates 0.0002/0.001, gamma 0.98, and minibatch 64. BS, random search, SA, DDPG, multi-seed summaries, and explicit placement budgets are available. `src/run_paper_experiment.py` creates separate CONV and FC suites.

## Assumptions that remain

The paper does not publish per-layer partition grids, exact physical masks, full GRS behavior, or workload-specific block counts. The implementation uses MAC-proportional exact-count grids, minimum contiguous whole-chip regions, a lower-left gateway inferred from Figure 3, and four CONV blocks by default from Figure 7's example.

Input/activation-buffer occupancy and stalls, exact multicast, router startup, transformation-unit costs, and compute/communication overlap are not modeled. VVA defaults to one addition/cycle. Residual branch traffic is retained, while the add executes in the destination transformation/VVA path without a separate Figure 6 core. Concatenation and grouped convolution are unsupported.

The paper also leaves `z`, OU details, replay capacity, CNN padding, LRN parameters, and target-update details incomplete. Reports and the paper-run manifest identify these choices. Potential reward shaping and the `mlp`/`cnn` agents are improvement conditions, not paper mode.

## Validation

Local validation uses torch 2.14.0+cpu and torchvision 0.29.0+cpu:

- Both maintained suites pass, including real extraction, optimizer updates, masked regions, hand-calculated contention, residual dependencies, and all agent architectures.
- Exact AlexNet counts are regression-tested; exact VGG16 and ResNet50 counts were checked during implementation.
- AlexNet paper-mode BS and one-placement DDPG smokes completed for CONV and FC.
- ResNet50 BS paper mode completed end to end for 512 CONV and 37 FC tasks.
- The paper wrapper completed both AlexNet regions and all four algorithms with bounded one-placement/two-search-trial settings, producing manifests and summaries.

These are functionality checks. They do not demonstrate learning, convergence, H100 performance, or agreement with the paper's percentages. CUDA tensor placement is implemented, but the planned H100 12 GB slice has not been available locally.

Run:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/run_paper_experiment.py --model alexnet --device cuda --output_dir runs/paper-alexnet --dry_run
```

See `README.md` for setup and commands, `PROJECT_STATE.md` for the current handoff, and `NEXT_STEPS.md` for remaining reproduction gates. `run_multi_chip_fast.py` and the older single-chip programs are outside the validated path.
