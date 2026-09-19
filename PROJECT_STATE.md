# Project state: paper-faithful DNN core placement

Updated 2026-09-19.

## Goal and branch

The goal is to reproduce, then improve upon Wu et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning* (ACM TODAES 2020, DOI 10.1145/3418498).

The maintained remote branch is `codex/reconciled-paper-implementation` in `micskrcb/DNN_MAPPING`. `main` remains historical. Code and tests are the source of truth for current behavior; old run logs use different objectives and cannot be compared numerically with paper mode.

## What is implemented

- FX extraction for torchvision AlexNet, VGG16, and ResNet50.
- Exact Figure 6 aggregate task counts: AlexNet 183 CONV/932 FC, VGG16 1024/1924, and ResNet50 512/37.
- Deterministic per-layer `(M,N)` grid reconstruction proportional to MAC work, with exact aggregate counts and 64 KB 8-bit weight-tile capacity.
- Separate CONV and FC optimization over disjoint, contiguous whole-chip masks.
- Table 1 4×4-chip, 16×16-core hardware parameters and physical byte/time units.
- Reconstructed X-then-Y routes, lower-left periphery gateway for inter-chip traffic, and shared directed-link contention per time phase.
- Configurable CONV block scaling; default four follows Figure 7's illustration.
- Figure 9 `paper_cnn`, sparse terminal reward, paper learning rates/gamma/batch size, batched actions, coordinate conversion, and Manhattan collision repair.
- Sequential BS, random search, simulated annealing, and DDPG.
- Explicit placement accounting: DDPG epochs × placements/epoch, separate reward-normalizer trials, and independently configurable RS/SA budgets.
- Five-seed orchestration, per-placement JSONL diagnostics, checkpoints, JSON reports, BS-normalized summaries, hop counts, and link-load summaries.
- One command that orchestrates separate CONV and FC paper-mode suites.

## Documented reconstruction assumptions

The paper gives aggregate logic-core counts but not each layer's partition grid. The current dynamic program chooses exact-count grids close to MAC-proportional targets and enforces the weight buffer. The exact CONV/FC physical masks are not published; the code assigns the minimum rectangular group of whole chips, with CONV first and FC following it.

The complete GRS route implementation is unavailable. Inter-chip routing uses a lower-left gateway inferred from Figure 3 and deterministic XY routes. The evaluated networks' block counts are unpublished; four CONV blocks is a configurable default inferred from the Figure 7 example. Residual branch traffic is traced, while the addition itself is assigned to the destination transformation/VVA path without adding a Figure 6 task.

The paper also does not fully specify `z`, OU parameters and fade schedule, replay capacity, CNN padding, LRN parameters, target-network updates, or soft update coefficient. The implementation records its choices in reports and manifests.

## Remaining fidelity gaps

The simulator does not model 64 KB input/activation-buffer occupancy and stalls, exact GRS multicast, router startup, transformation/bias/pooling costs, compute/communication overlap, or a cycle-accurate block pipeline. VVA throughput defaults to one addition/cycle because the paper does not state it. Consequently, `paper_pipeline` is the closest documented reconstruction, not the authors' unpublished simulator.

Batch-one latency can be compared after validation. True large-batch throughput still needs an explicit fill/steady-state/drain calculation; the current inverse objective ratio is labeled as such and must not be presented as measured throughput.

## Evidence completed locally

- Paper-target extraction returns all six exact CONV/FC counts.
- Hand-calculated XY gateway and shared-link contention tests pass.
- Masked-region tests confirm that baselines and the mapper cannot use other cores.
- Remainder work, byte units, torus IDs, SA budget/neighborhood, BS order, potential shaping, residual dependencies, real optimizer updates, and all agent architectures have regression coverage.
- AlexNet BS completed for both reconstructed regions on CPU: 183 tasks in a 256-core CONV region and 932 tasks in a 1024-core FC region.
- A one-placement AlexNet `paper_cnn` CPU smoke completed. It verifies execution only; replay had not reached minibatch size and no learning claim follows.
- The paper experiment runner completed a one-seed, one-placement CPU smoke for both regions and all four algorithms, producing manifests and BS-normalized summaries. This verifies orchestration, not learning.

Earlier 1,445-task AlexNet logs, decimal-valued junior runs, and old proxy/full-frame best costs were produced by different extraction or objective versions. They remain useful historical diagnostics but are not evidence of paper result reproduction.

## GPU state

DDPG networks and sampled tensors support CUDA; the environment, routing, collision repair, replay storage, and process orchestration remain CPU-side. Previous junior measurements showed substantial CNN-update acceleration on a different GPU, but an H100 speedup cannot be claimed until measured with this commit and identical settings.

The planned device is an H100 12 GB slice accessed through SSH. Access was not available during local development. Once available, first run `src/validate_device.py --device cuda`, then the bounded paper smoke, inspect memory and CPU/GPU timing, and only then launch the default 300,000-placement five-seed runs.

## Next execution sequence

1. Validate the pushed commit on the H100 and archive the validator JSON.
2. Run the bounded AlexNet paper smoke for both regions.
3. Benchmark one placement and one optimizer update to estimate wall time.
4. Run AlexNet BS/RS/SA and five DDPG seeds with the declared budgets.
5. Review learning diagnostics before spending the full VGG16/ResNet50 budget.
6. Add activation-buffer/streaming behavior if it materially changes the hand-checked evaluator or paper trends.
7. Implement and validate true large-batch throughput before reproducing that panel of Figure 10.

Potential reward shaping, different agents, collision penalties, discrete actions, graph encoders, and parallel environments remain improvement experiments and should be run only after freezing the paper-mode configuration.
