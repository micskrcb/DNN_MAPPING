# DNN Mapping with Reinforcement Learning

This repository maps DNN computation tasks onto a multi-chip many-core accelerator using DDPG, random search, fixed simulated annealing, adaptive simulated annealing (ASA), a DDPG→ASA hybrid, or the sequential baseline (BS). Its reproduction target is Wu et al., **Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning**, ACM TODAES 2020 ([DOI 10.1145/3418498](https://doi.org/10.1145/3418498)).

The `cpu` branch is the CPU-oriented continuation of the reconciled paper implementation. It is runnable and tested on CPU, explicitly controls PyTorch thread use, supports concurrent independent experiments, reduces diagnostic overhead, and uses exact affected-stage reevaluation for ASA. The paper does not publish its simulator or every parameter, so the code records reconstruction assumptions instead of claiming exact numerical reproduction.

## Present state

| Paper feature | Current status |
| --- | --- |
| 4×4 chips, 16×16 cores/chip | Implemented |
| Figure 6 logic-core totals | Exact aggregate counts for AlexNet, VGG16, and ResNet50 |
| Separate CONV and FC placement | Implemented with disjoint whole-chip masks |
| Figure 9 actor/critic | Implemented as `--agent_arch paper_cnn` |
| Sparse reward `sqrt(B) - sqrt(L(P))` | Implemented; zero before a complete placement |
| BS, RS, SA, and DDPG | Implemented |
| Adaptive SA and DDPG→ASA | Implemented as research extensions with matched-budget support |
| Objective sensitivity preflight | Implemented; flat paper-mode objectives stop before long optimization |
| 30 placements/epoch and paper search budgets | Explicitly accounted for by the paper runner |
| XY routing and link contention | Reconstructed and implemented |
| 64 KB weight-buffer constraint | Enforced during paper partition reconstruction |
| 64 KB activation-buffer stalls and exact GRS | Not published in enough detail; not implemented |
| Full paper-scale results | Not run yet |

Paper-mode reports include routed mean hop counts and on/off-chip link-load summaries. Multi-seed summaries report the objective normalized to BS and its inverse ratio. The inverse ratio is useful for comparison, but it is not labeled as measured accelerator throughput.

## Repository layout

- `src/run_multi_chip.py` runs one BS, DDPG, random-search, SA, ASA, or DDPG→ASA experiment.
- `src/run_multiseed_experiment.py` runs selected methods across seeds and can schedule independent CPU jobs concurrently.
- `src/run_paper_experiment.py` runs separate CONV and FC paper-mode suites.
- `src/compute_model.py` reconstructs partitions and converts work/traffic to physical units.
- `src/multi_chip_topology.py` implements physical IDs and mesh/torus routing.
- `src/multi_chip_environment.py` implements the placement objective and diagnostics.
- `src/validate_device.py` performs a bounded CPU/CUDA functionality profile.
- `PROJECT_STATE.md` gives the detailed handoff state and known limitations.
- `NEXT_STEPS.md` tracks the reproduction gates.
- `prev version readmes/` preserves earlier README snapshots.

The older single-chip PPO/GCN programs and `run_multi_chip_fast.py` are not part of the validated paper reproduction path.

## Complete CPU setup and DDPG→ASA workflow

All commands in this section run from a terminal. The hybrid order is **DDPG first, then Adaptive Simulated Annealing**: DDPG constructs the warm-start placement and ASA refines the best placement found by DDPG.

### 1. Clone the CPU branch

For a new checkout:

```bash
git clone --branch cpu https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
git branch --show-current
git log -1 --oneline
```

`git branch --show-current` must print `cpu`.

If the repository is already cloned and the local `cpu` branch exists:

```bash
cd DNN_MAPPING
git fetch origin
git switch cpu
git pull --ff-only origin cpu
```

If the repository is already cloned but the local `cpu` branch does not exist:

```bash
cd DNN_MAPPING
git fetch origin
git switch --track -c cpu origin/cpu
```

Before pulling later updates, preserve or commit any local edits. Then update with:

```bash
git switch cpu
git pull --ff-only origin cpu
```

### 2. Create the CPU Python environment

On Ubuntu/Debian, install the virtual-environment package if it is missing:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Create and activate a project-local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Activate the environment again after opening a new terminal:

```bash
cd DNN_MAPPING
source .venv/bin/activate
```

Verify that PyTorch is using its CPU build and inspect the available thread count:

```bash
python -c "import torch, torchvision, numpy; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch threads:', torch.get_num_threads())"
nproc
```

The historical root `requirements.txt` contains old pins and is not the environment specification for this branch.

### 3. Run validation before a long experiment

```bash
mkdir -p runs
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/validate_device.py --device cpu --output runs/cpu-validation.json
```

All tests should finish successfully before starting a paper-scale run.

Paper-mode runs now sample 64 valid placements before optimization and print
the minimum, median, maximum, relative objective span, and bottleneck
compute/communication split. The run stops when the sampled relative span is
below 0.1%, because a nearly placement-independent objective cannot validate a
placement algorithm. `--allow_flat_objective` is available only for deliberate
diagnostic work.

### 4. Run a small end-to-end DDPG→ASA smoke test

This validates model extraction, CPU DDPG training, checkpoint creation, ASA warm-starting, diagnostics, and the final JSON report. It is deliberately too small for scientific conclusions.

```bash
mkdir -p runs/smoke

python src/run_multi_chip.py \
  --algo ddpg_asa \
  --device cpu \
  --cpu_threads 4 \
  --use_cnn \
  --model simple \
  --channels_per_partition 128 \
  --timing_model full_frame \
  --agent_arch paper_cnn \
  --reward_mode sparse \
  --epochs 10 \
  --placements_per_epoch 1 \
  --iters 50 \
  --baseline_trials 10 \
  --batch_z 3 \
  --train_every 1 \
  --diagnostics_every 5 \
  --checkpoint_every 5 \
  --seed 0 \
  --save_checkpoint runs/smoke/ddpg.pt \
  --diagnostics runs/smoke/ddpg.jsonl \
  --asa_diagnostics runs/smoke/asa.jsonl \
  --report runs/smoke/report.json
```

Inspect the main result:

```bash
python -m json.tool runs/smoke/report.json | less
```

### 5. Run exactly 500,000 combined DDPG→ASA candidates

Do not start this command until the smoke test reports a credible objective
range and runtime. On the measured 20-thread CPU, the earlier implementation
required about 43 seconds per DDPG placement; a 400,020-placement DDPG phase
would therefore take roughly 200 days. Use a bounded run or CUDA after the
preflight gate passes.

This configuration retains the paper's **30 DDPG placements per declared epoch**. It assigns approximately 80% of the matched budget to DDPG and 20% to ASA:

```text
13,334 epochs × 30 DDPG placements = 400,020
ASA candidate evaluations             =  99,980
Combined optimization candidates      = 500,000
```

The following one-shot command runs the AlexNet CONV region using the paper-oriented topology, partitioning, objective, and Figure 9 network. Replace `$(nproc)` with a smaller number if the machine is shared.

```bash
mkdir -p runs/alexnet-conv-500k

python src/run_multi_chip.py \
  --algo ddpg_asa \
  --device cpu \
  --cpu_threads "$(nproc)" \
  --cpu_interop_threads 1 \
  --use_cnn \
  --model alexnet \
  --partition_mode paper_targets \
  --workload_region conv \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --agent_arch paper_cnn \
  --reward_mode sparse \
  --epochs 13334 \
  --placements_per_epoch 30 \
  --exploration_decay_placements 320016 \
  --iters 99980 \
  --baseline_trials 1000000 \
  --batch_z 3 \
  --train_every 1 \
  --diagnostics_every 1000 \
  --checkpoint_every 1000 \
  --seed 0 \
  --save_checkpoint runs/alexnet-conv-500k/ddpg-seed0.pt \
  --diagnostics runs/alexnet-conv-500k/ddpg-seed0.jsonl \
  --asa_diagnostics runs/alexnet-conv-500k/asa-seed0.jsonl \
  --report runs/alexnet-conv-500k/report-seed0.json
```

The run can take a long time on CPU. The safer method is to train DDPG in cumulative checkpointed targets and run ASA after the final DDPG target.

### 6. Run the 500,000-candidate experiment in resumable stages

The targets below are cumulative. They train DDPG to 99,990, 199,980, 300,000, and finally 400,020 placements. `--epochs` is a total target when loading a checkpoint, not an additional number of epochs. Every stage uses the same `--exploration_decay_placements 320016`, so OU noise reaches its minimum at 80% of the final DDPG budget instead of being exhausted during the first checkpoint stage.

```bash
mkdir -p runs/alexnet-conv-500k

for TARGET_EPOCHS in 3333 6666 10000 13334
do
  python src/run_multi_chip.py \
    --algo ddpg \
    --device cpu \
    --cpu_threads "$(nproc)" \
    --cpu_interop_threads 1 \
    --use_cnn \
    --model alexnet \
    --partition_mode paper_targets \
    --workload_region conv \
    --timing_model paper_pipeline \
    --routing_model paper_xy \
    --chips_x 4 --chips_y 4 \
    --rows 16 --cols 16 \
    --agent_arch paper_cnn \
    --reward_mode sparse \
    --epochs "$TARGET_EPOCHS" \
    --placements_per_epoch 30 \
    --exploration_decay_placements 320016 \
    --baseline_trials 1000000 \
    --batch_z 3 \
    --train_every 1 \
    --diagnostics_every 1000 \
    --checkpoint_every 1000 \
    --seed 0 \
    --load_checkpoint runs/alexnet-conv-500k/ddpg-seed0.pt \
    --save_checkpoint runs/alexnet-conv-500k/ddpg-seed0.pt \
    --diagnostics "runs/alexnet-conv-500k/ddpg-seed0-${TARGET_EPOCHS}.jsonl" \
    --report "runs/alexnet-conv-500k/ddpg-seed0-${TARGET_EPOCHS}.json"
done
```

If the checkpoint does not exist on the first invocation, training starts from scratch and creates it. Later invocations restore the actor, critic, target networks, optimizers, best DDPG placement, exploration state, RNG state, counters, and the random-search reward baseline. The replay buffer is not persisted, so it refills after each restart and a resumed run is not bit-exact.

Checkpoints created before the balanced-partition and cycle-scaled-reward fix
are intentionally incompatible. Start a new checkpoint after pulling this
revision; the old placement-2000 checkpoint used the flat objective and must
remain historical evidence rather than a training warm start.
Checkpoints made before the fixed absolute exploration schedule are also
incompatible because their noise history depends on the temporary stage target.

After the DDPG checkpoint reaches 400,020 placements, refine its saved best placement with exactly 99,980 ASA candidate evaluations:

```bash
python src/run_multi_chip.py \
  --algo ddpg_asa \
  --device cpu \
  --cpu_threads "$(nproc)" \
  --cpu_interop_threads 1 \
  --use_cnn \
  --model alexnet \
  --partition_mode paper_targets \
  --workload_region conv \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --agent_arch paper_cnn \
  --reward_mode sparse \
  --epochs 13334 \
  --placements_per_epoch 30 \
  --exploration_decay_placements 320016 \
  --iters 99980 \
  --baseline_trials 1000000 \
  --batch_z 3 \
  --train_every 1 \
  --seed 0 \
  --load_checkpoint runs/alexnet-conv-500k/ddpg-seed0.pt \
  --asa_diagnostics runs/alexnet-conv-500k/asa-seed0.jsonl \
  --report runs/alexnet-conv-500k/hybrid-seed0.json
```

Because the checkpoint has already reached the requested DDPG target, this final command restores the best DDPG placement without retraining and starts ASA from it. ASA always retains the warm start, so its reported best result cannot be worse than the saved DDPG best.

### 7. Run standalone ASA

This starts ASA from a random valid placement for the paper-oriented AlexNet CONV workload and performs 100,000 adaptive candidate evaluations:

```bash
mkdir -p runs/asa

python src/run_multi_chip.py \
  --algo asa \
  --device cpu \
  --cpu_threads "$(nproc)" \
  --use_cnn \
  --model alexnet \
  --partition_mode paper_targets \
  --workload_region conv \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --iters 100000 \
  --seed 0 \
  --asa_diagnostics runs/asa/seed0.jsonl \
  --report runs/asa/seed0.json
```

ASA calibrates its initial temperature from observed uphill cost changes. It then adapts temperature using the measured acceptance ratio, reheats after stagnant windows, and expands or contracts the moved-task fraction. Reports record accepted and improving moves, reheats, temperatures, neighborhood size, initialization, and exact candidate count.

### 8. Run multiple seeds and methods on CPU

For a machine with 20 logical CPU threads, this example runs two independent subprocesses at a time and gives ten PyTorch threads to each subprocess:

```bash
python src/run_multiseed_experiment.py \
  --device cpu \
  --jobs 2 \
  --cpu_threads 10 \
  --seeds 0,1,2,3,4 \
  --algorithms bs,ddpg,random,sa,asa,ddpg_asa \
  --model alexnet \
  --partition_mode paper_targets \
  --workload_region conv \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --epochs 1000 \
  --placements_per_epoch 30 \
  --search_budget 30000 \
  --baseline_trials 10000 \
  --output_dir runs/alexnet-conv-multiseed
```

Keep `jobs × cpu_threads` at or below the machine's logical CPU count to avoid oversubscription. Start with `--jobs 1` if memory is limited.

### 9. Understand the output files

- `*.pt` is a resumable DDPG checkpoint.
- DDPG `*.jsonl` contains periodic noisy-policy and deterministic-policy diagnostics, losses, exploration noise, and collision repairs.
- ASA `*.jsonl` contains temperature, acceptance ratio, perturbation fraction, current/best cost, and reheat count per adaptation window.
- The final `*.json` report contains configuration, best cost and placement, objective units, evaluation counts, runtime, routing diagnostics, and algorithm metadata.
- `summary.json` from the multi-seed runner contains per-method means, sample standard deviations, minima, maxima, and BS-normalized comparisons.

The CPU branch reduces diagnostic rollouts, controls PyTorch thread allocation, avoids repeated free-core scans, and uses an exact affected-stage evaluator for ASA. In a controlled 100-task test, incremental ASA returned the identical placement and cost as full reevaluation while running about 9.6 times faster; actual speedup depends on graph structure and neighborhood size.

## Paper-mode commands

Inspect the two generated CONV/FC commands and manifest without starting a long run:

```bash
python src/run_paper_experiment.py \
  --model alexnet --device cuda \
  --output_dir runs/paper-alexnet --dry_run
```

Run the full default AlexNet suite on the H100 allocation:

```bash
python src/run_paper_experiment.py \
  --model alexnet --device cuda \
  --output_dir runs/paper-alexnet
```

The defaults are intentionally large: five seeds; 10,000 declared DDPG epochs; 30 complete placements per epoch (300,000 per seed); one million random trials to form DDPG's fixed baseline; and one million RS/SA/ASA placements. The hybrid divides the same 300,000-placement optimization budget between DDPG and ASA (80/20 by default), while its reward-normalizer trials are reported separately. The runner executes CONV and FC separately and saves a manifest, reports, logs, checkpoints, diagnostics, and `summary.json` files.

For a 20-thread CPU, run two independent experiments at a time with ten threads each:

```bash
python src/run_paper_experiment.py \
  --model alexnet --device cpu --jobs 2 --cpu_threads 10 \
  --output_dir runs/paper-alexnet-cpu
```

Use `--algorithms bs,ddpg,asa,ddpg_asa` to select a smaller method set. Increasing `--jobs` can shorten a multi-seed campaign, but each job receives its own model and environment in memory.

For a bounded end-to-end paper-mode smoke test:

```bash
python src/run_paper_experiment.py \
  --model alexnet --device cpu --seeds 0 \
  --epochs 1 --placements_per_epoch 1 \
  --baseline_trials 2 --search_budget 2 \
  --output_dir runs/paper-smoke
```

One placement cannot train or establish convergence; it only verifies that both regions and all methods complete.

To run one region or method directly:

```bash
python src/run_multi_chip.py \
  --algo bs --use_cnn --model alexnet \
  --partition_mode paper_targets --workload_region conv \
  --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
  --seed 0 --report runs/alexnet-conv-bs.json
```

Valid paper-target workloads are `alexnet`, `vgg16`, and `resnet50`. Their exact aggregate logic-core counts are:

| Workload | CONV | FC | Total |
| --- | ---: | ---: | ---: |
| AlexNet | 183 | 932 | 1,115 |
| VGG16 | 1,024 | 1,924 | 2,948 |
| ResNet50 | 512 | 37 | 549 |

## What paper mode reconstructs

FX traces Conv2d/Linear dependencies. For each layer, deterministic integer `(M,N)` output/input partitions are selected so that total VMM plus VVA tasks match Figure 6 exactly, approximate MAC-balanced allocation, and keep every 8-bit weight tile within 64 KB. The paper publishes only aggregate counts, so these per-layer grids are assumptions.

CONV and FC are optimized independently. Each uses the minimum number of contiguous whole chips that can hold its tasks. CONV begins at chip row zero and FC begins at the next chip row. This preserves disjoint regions but is a reconstruction because the exact masks are not published.

Same-chip messages take deterministic X-then-Y routes. Inter-chip messages run from the source core to a lower-left chip-periphery gateway, traverse the chip grid X then Y, then travel from the destination gateway to its core. A time phase combines the maximum task compute time with the larger of the routed per-source byte-hop time and the bottleneck serialized load on any shared directed link. The gateway is inferred from Figure 3; the paper's complete GRS implementation is unavailable.

CONV work and traffic are divided across `--conv_blocks` (default 4, taken from the illustrative Figure 7). FC uses one layer work unit. Workload-specific block counts are not published. ResNet branch dependencies are retained; residual addition is assigned to the destination transformation/VVA path without adding a Figure 6 core.

## Objectives

| Mode | Meaning | Units |
| --- | --- | --- |
| `proxy` | Communication volume × hierarchical distance | Arbitrary score |
| `full_frame` | Ideal arithmetic plus per-task byte-hop serialization | Seconds |
| `paper_pipeline` | Reconstructed block/layer phases with XY shared-link contention | Seconds |

`paper_pipeline` requires `paper_targets`, `workload_region conv|fc`, and `paper_xy`. It uses Table 1's 128 MACs at 400 MHz, 64 GB/s/core on-chip links, 100 GB/s/chip off-chip links, 8-bit activations/weights, and 32-bit partial sums. VVA defaults to one addition/cycle because its throughput is unpublished. The reconstructed per-layer grids now balance estimated VMM and VVA cycles while preserving the exact Figure 6 aggregate counts and 64 KB weight constraint; `--partition_balance_weight` exposes the unpublished trade-off.

For routed communication, a phase uses the larger of the busiest shared-link serialization time and the maximum routed byte-hop time emitted by one source task. This retains both hop-distance and contention effects. Router startup, exact packet scheduling, GRS behavior, and cycle-accurate streaming remain unavailable.

The objective continues to be reported in seconds. Before applying the paper's square-root reward, seconds-based latency is multiplied by 400 MHz by default so the critic sees cycle-scaled rewards. Override this only for a documented experiment with `--reward_scale`.

The remaining simulator gaps are material: 64 KB input/activation-buffer stalls, exact multicast/GRS behavior, router startup, exact transformation-unit costs, compute/communication overlap, and workload-specific block schedules. Results must therefore be called a documented reconstruction, not an exact replay of the authors' simulator.

## DDPG and baselines

Paper-mode DDPG uses the Figure 9 spatial CNN, the 2-D placement grid, batched `2z` continuous coordinates, floor conversion, nearest-free Manhattan repair, actor learning rate 0.0002, critic learning rate 0.001, gamma 0.98, minibatch 64, and sparse terminal reward. `batch_z`, OU parameters, replay capacity, padding, LRN parameters, target networks, and soft-update coefficient are not fully specified by the paper and remain recorded assumptions.

`--reward_mode potential` and the `mlp`/`cnn` agents are improvement conditions. The shaping potential is normalized by the fixed random-search baseline so intermediate rewards remain near unit scale while the shaping terms still telescope to zero. Do not mix these results into the frozen paper-mode comparison. Collision repairs are expected because continuous coordinates may select the same or a masked core; diagnostics separate occupied-core repairs from mask repairs and also evaluate the deterministic policy.

`--retain_deterministic_candidates` is another explicit improvement condition. When enabled, a deterministic diagnostic rollout may update the saved best placement. Reports separate `training_candidate_evaluations`, `deterministic_candidate_evaluations`, and `total_candidate_evaluations`, preventing those additional objective evaluations from being hidden. Without the flag, diagnostics remain monitoring-only and cannot change the saved result.

BS fills allowed physical cores in chip-major order. RS samples complete valid placements. SA uses current-cost acceptance, cooling factor 0.99, and roughly 1% placement perturbations that may use free cores.

ASA and DDPG→ASA are project extensions, not features claimed by the source DNN-mapping paper. Keep fixed SA in result tables as the paper-aligned baseline. The runner matches the hybrid's combined candidate count to DDPG. For a direct hybrid-versus-ASA ablation, set `--search_budget` equal to `--epochs × --placements_per_epoch`; otherwise the paper-scale defaults deliberately give SA/ASA one million candidates and DDPG/hybrid 300,000.

## Interpreting results

A successful run proves that the program executed; it does not prove that DDPG learned. Use the JSONL diagnostics to compare noisy and deterministic policy costs, actor/critic losses, unique intended cores, and collision repair counts. Judge convergence across at least five seeds and compare all methods under the declared complete-placement budgets. Diagnostic policy rollouts are additional objective evaluations; use `total_candidate_evaluations` whenever deterministic retention is enabled.

Report CONV and FC separately. Normalize latency to BS as in the paper, include seed mean/sample standard deviation/minimum/maximum, and examine hop counts and link loads. Do not compare old proxy scores, full-frame scores, and paper-pipeline seconds as though they were the same metric.

See `PROJECT_STATE.md` for validated evidence and `NEXT_STEPS.md` for the work still required before claiming paper-comparable results.
