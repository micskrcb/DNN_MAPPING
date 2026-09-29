# Project state: paper-faithful DNN core placement

Updated 2026-09-29.

## Goal and branch

The goal is to reproduce, then improve upon Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning* (ACM TODAES 2020, DOI 10.1145/3418498).

The maintained remote branch is `codex/reconciled-paper-implementation` in `micskrcb/DNN_MAPPING`. `main` remains historical. Code and tests are the source of truth for current behavior; old run logs use different objectives and cannot be compared numerically with paper mode.

## What is implemented

- FX extraction for torchvision AlexNet, VGG16, and ResNet50.
- Exact Figure 6 aggregate task counts: AlexNet 183 CONV/932 FC, VGG16 1024/1924, and ResNet50 512/37.
- Deterministic per-layer `(M,N)` grid reconstruction proportional to MAC work, with exact aggregate counts, 64 KB 8-bit weight-tile capacity, and an explicit VMM/VVA cycle-balance term.
- Separate CONV and FC optimization over disjoint, contiguous whole-chip masks.
- Table 1 4×4-chip, 16×16-core hardware parameters and physical byte/time units.
- Reconstructed X-then-Y routes, lower-left periphery gateway for inter-chip traffic, routed byte-hop time, and shared directed-link contention per time phase.
- Configurable CONV block scaling; default four follows Figure 7's illustration.
- Figure 9 `paper_cnn`, sparse terminal reward, paper learning rates/gamma/batch size, batched actions, coordinate conversion, and Manhattan collision repair.
- Sequential BS, random search, simulated annealing, and DDPG.
- Explicit placement accounting: DDPG epochs × placements/epoch, separate reward-normalizer trials, and independently configurable RS/SA budgets.
- Five-seed orchestration, per-placement JSONL diagnostics, checkpoints, JSON reports, BS-normalized summaries, hop counts, link-load summaries, and a placement-sensitivity preflight.
- One command that orchestrates separate CONV and FC paper-mode suites.

## Documented reconstruction assumptions

The paper gives aggregate logic-core counts but not each layer's partition grid. The current dynamic program chooses exact-count grids close to MAC-proportional targets and enforces the weight buffer. The exact CONV/FC physical masks are not published; the code assigns the minimum rectangular group of whole chips, with CONV first and FC following it.

The complete GRS route implementation is unavailable. Inter-chip routing uses a lower-left gateway inferred from Figure 3 and deterministic XY routes. A stage's reconstructed communication time is the larger of its maximum per-source routed byte-hop time and maximum shared-link serialization load. The evaluated networks' block counts are unpublished; four CONV blocks is a configurable default inferred from the Figure 7 example. Residual branch traffic is traced, while the addition itself is assigned to the destination transformation/VVA path without adding a Figure 6 task.

The paper says computation is balanced per core but does not publish the grids or VVA throughput. The grid search therefore combines MAC-proportional core-count error with estimated VMM/VVA cycle imbalance. Seconds-based latency is scaled to 400-MHz cycle units before the square-root reward by default.

The paper also does not fully specify `z`, OU parameters and fade schedule, replay capacity, CNN padding, LRN parameters, target-network updates, or soft update coefficient. The implementation records its choices in reports and manifests.

## Remaining fidelity gaps

The simulator does not model 64 KB input/activation-buffer occupancy and stalls, exact GRS multicast, router startup, exact packet scheduling, exact transformation/bias/pooling costs beyond a minimum output pass, compute/communication overlap, or a cycle-accurate block pipeline. VVA throughput defaults to one operation/cycle because the paper does not state it. Consequently, `paper_pipeline` is a documented sensitivity model, not the authors' unpublished simulator.

Batch-one latency can be compared after validation. True large-batch throughput still needs an explicit fill/steady-state/drain calculation; the current inverse objective ratio is labeled as such and must not be presented as measured throughput.

## Evidence completed locally

- Paper-target extraction returns all six exact CONV/FC counts.
- Hand-calculated XY gateway and shared-link contention tests pass.
- Masked-region tests confirm that baselines and the mapper cannot use other cores.
- Remainder work, byte units, torus IDs, SA budget/neighborhood, BS order, potential shaping, residual dependencies, real optimizer updates, and all agent architectures have regression coverage.
- AlexNet BS completed for both reconstructed regions on CPU: 183 tasks in a 256-core CONV region and 932 tasks in a 1024-core FC region.
- A one-placement AlexNet `paper_cnn` CPU smoke completed. It verifies execution only; replay had not reached minibatch size and no learning claim follows.
- The paper experiment runner completed a one-seed, one-placement CPU smoke for both regions and all four algorithms, producing manifests and BS-normalized summaries. This verifies orchestration, not learning.
- A 2,080-placement AlexNet-CONV CPU run exposed a flat objective: the old `(M=1,N=61)` reconstruction made a 5.2488-ms VVA task placement-independent, limiting visible headroom to roughly 0.001%. That checkpoint is incompatible with the corrected model.
- The corrected pure-model AlexNet-CONV audit preserves 183 cores with grids `(7,2), (12,4), (6,4), (6,6), (5,5)`. Across 256 seeded random placements the objective span was 16.34%; the best sample was 86.60% compute and 13.40% communication.
- The corrected AlexNet-FC audit preserves 932 cores with grids `(16,36), (16,16), (4,16)`. Across 64 random placements the objective span was 4.05%; the best sample was 92.29% compute and 7.71% communication.

Earlier 1,445-task AlexNet logs, decimal-valued junior runs, and old proxy/full-frame best costs were produced by different extraction or objective versions. They remain useful historical diagnostics but are not evidence of paper result reproduction.

## GPU state

DDPG networks and sampled tensors support CUDA; the environment, routing, collision repair, replay storage, and process orchestration remain CPU-side. Previous junior measurements showed substantial CNN-update acceleration on a different GPU, but an H100 speedup cannot be claimed until measured with this commit and identical settings.

The planned device is an H100 12 GB slice accessed through SSH. Access was not available during local development. Once available, first run `src/validate_device.py --device cuda`, then the bounded paper smoke, inspect memory and CPU/GPU timing, and only then launch the default 300,000-placement five-seed runs.

## Next execution sequence

1. Rerun both suites in a Torch-enabled environment.
2. Run the AlexNet CONV/FC sensitivity preflight and bounded BS/RS/SA comparisons.
3. Validate the corrected commit on the H100 and archive the validator JSON.
4. Benchmark one placement and one optimizer update.
5. Require deterministic-policy improvement in a bounded DDPG diagnostic before a paper-scale launch.
6. Run AlexNet BS/RS/SA and five DDPG seeds with declared budgets.
7. Review learning diagnostics before spending the full VGG16/ResNet50 budget.
8. Add missing simulator behavior when defensible evidence is available.
9. Implement true large-batch throughput before reproducing that panel of Figure 10.

Potential reward shaping, different agents, collision penalties, discrete actions, graph encoders, and parallel environments remain improvement experiments and should be run only after freezing the paper-mode configuration.
