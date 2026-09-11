# DNN Mapping with Reinforcement Learning

Place DNN computation across a multi-chip many-core accelerator using DDPG, random search, or simulated annealing.

The project aims to reproduce **Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning** by Nan Wu, Lei Deng, Guoqi Li, and Yuan Xie, published in ACM TODAES (2020). [Read the paper's publication record](https://doi.org/10.1145/3418498).

**Status: a tested research implementation with explicit approximations. It is not yet a complete reproduction of the paper or its reported results.** CPU extraction and training have been tested. CUDA selection is implemented; H100 execution and memory use on a 12 GB slice remain unverified.

Development follows two stages: establish a defensible reproduction first, then evaluate improvements. Bug fixes and assumed simulator details are not claimed as research contributions.

## What the code does

1. Trace a PyTorch model with torch.fx to discover Conv2d/Linear dependencies.
2. Partition input and output channels into VMM tasks and VVA reduction tasks.
3. Construct communication edges between overlapping channel ranges.
4. Assign those tasks to distinct physical cores on a mesh or torus.
5. Evaluate a communication proxy or a full-frame compute/communication approximation.
6. Search for a low-cost placement and optionally save a checkpoint and JSON report.

VMM means vector–matrix multiplication; VVA means vector–vector accumulation. Both refer to **logic tasks** that are assigned to physical cores.

## Changes included in this version

The reconciled implementation restores features from the supplied earlier archives and fixes inconsistencies in both those archives and the previous checkout.

| Area | Current behavior |
| --- | --- |
| Dependency extraction | FX tracing replaces sequential forward-hook order; custom models are supported |
| Communication graph | Overlapping channel ranges replace unconditional all-to-all broadcasts; pooling/flattening use consumer shapes |
| Operation counts | Remainder tiles have exact sizes, preserving total layer MAC counts |
| Timing units | Physical compute time is never added to an arbitrary communication score |
| Core coordinates | Row-major policy coordinates convert to chip-major physical IDs before evaluation |
| Topology | Mesh and torus configurations are available |
| Placement overhead | Occupancy construction and collision searches use NumPy vectorization |
| Simulated annealing | Acceptance uses current cost, cooling uses 0.99, and proposals can reach unused cores |
| Exploration | DDPG uses fading Ornstein–Uhlenbeck noise |
| Checkpoints | Workload/configuration fingerprints reject incompatible checkpoints; RNG states are saved |
| Reporting | Optional JSON includes configuration, objective units, best placement, runtime, and limitations |
| Failure handling | Unsupported timing, cyclic graphs, over-capacity workloads, and unavailable requested devices fail explicitly |

See [RECONCILIATION.md](RECONCILIATION.md) for implementation provenance and detailed assumptions.

## Installation

Use the reconciled branch; all commands below run from the repository root.

~~~bash
git clone --branch codex/reconciled-paper-implementation https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

The multi-chip path needs **NumPy, PyTorch, and torchvision**. The root requirements.txt contains historical dependencies for the original project, including old Torch pins; it is not the installation specification for this reconciled path.

For a CPU environment:

~~~bash
python -m pip install numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~~~

For a GPU environment, use the GPU provider's configured environment or install compatible torch/torchvision packages using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Choose the CUDA build appropriate to the host. Do not install CPU-only wheels into the environment intended for the H100.

The local validation environment used Python 3.13, torch 2.14.0+cpu and torchvision 0.29.0+cpu. These describe the tested environment, not a CUDA compatibility guarantee.

## Start with tests and a small run

~~~bash
python src/test_multi_chip.py
python -m unittest discover -s src -p test_reconciliation.py -v
~~~

The first suite can skip extraction when PyTorch is missing. A run with skipped extraction does not validate the real-model path. The second suite tests actual training updates on CPU and also CUDA when available.

Run a small, complete training smoke test:

~~~bash
mkdir -p runs
OMP_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model simple \
  --channels_per_partition 128 --timing_model full_frame \
  --device cpu --epochs 10 --baseline_trials 10 \
  --train_every 1 --batch_z 3 --seed 0 \
  --save_checkpoint runs/smoke.pt \
  --checkpoint_every 5 --report runs/smoke.json
~~~

This configuration produces 29 logic tasks on the default 64-core grid. Its small search budget is for checking functionality, not measuring paper-level performance.

Resume to a **total** of 12 episodes:

~~~bash
OMP_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model simple \
  --channels_per_partition 128 --timing_model full_frame \
  --device cpu --epochs 12 --baseline_trials 10 \
  --train_every 1 --batch_z 3 --seed 0 \
  --load_checkpoint runs/smoke.pt --save_checkpoint runs/smoke.pt \
  --report runs/resume.json
~~~

Checkpoints restore networks, optimizers, best placement, baseline, counters, and RNG states. **Replay is not persisted**, and changing the total episode target changes the fading schedule; resumed training is not bit-exact. Old or incompatible checkpoints are rejected. A missing load path starts a new run; use a save path if the result should persist.

## Running on an H100 slice

First check the device from inside the allocated GPU environment:

~~~bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('Visible memory (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))"
~~~

Run the tests, then repeat the small smoke command with **--device cuda**. The actor, critic, and sampled training tensors use CUDA. Environment evaluation, collision resolution, replay storage, and search control remain on CPU. Multiple environments are not batched on the GPU.

For the planned H100 12 GB slice, inspect actual visible memory and runtime before scaling. No full-card H100 capacity or 12 GB fit is assumed. A GPU smoke test verifies execution, not convergence or paper fidelity.

The paper's physical grid is **4 × 4 chips, each with 16 × 16 cores: 4,096 physical cores**. An example larger-workload smoke configuration is:

~~~bash
python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model alexnet \
  --channels_per_partition 512 --timing_model full_frame \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
  --device cuda --epochs 10 --baseline_trials 10 \
  --batch_z 3 --train_every 1 --seed 0 \
  --save_checkpoint runs/alexnet-smoke.pt \
  --report runs/alexnet-smoke.json
~~~

AlexNet extraction at partition size 512 has been tested; this larger CUDA command has not. The partition size is a capacity-conscious example, not the paper's verified allocation.

## Objectives and their units

| Mode | Contents | Units |
| --- | --- | --- |
| proxy (default) | Communication volume multiplied by hierarchical hop costs | Arbitrary score |
| full_frame | Per-task arithmetic time plus outgoing byte-hop serialization | Seconds |

Both modes minimize the **maximum task service cost**. Computing nested maxima over DAG levels does not implement the paper's block-streaming schedule.

### Communication proxy

The defaults --on_lat 1 and --off_lat 5 weight on-chip and inter-chip hops. Proxy scores cannot be interpreted as seconds or compared numerically to full_frame results. Even proxy results can change after graph and coordinate fixes; record the code version for comparisons.

### Full-frame approximation

This mode requires --use_cnn and positive channel partitioning.

- VMM time: ceil(MACs / (128 × utilization)) / 400 MHz.
- VVA time: ceil(additions / (assumed additions per cycle × utilization)) / 400 MHz.
- VMM outputs are 32-bit partial sums; VVA outputs are 8-bit activations.
- Communication charges bytes × hops / bandwidth, summed over outgoing edges.
- Defaults: 64 GB/s on-chip and 100 GB/s off-chip, using decimal GB.
- Default VVA throughput is **1 addition/cycle**, an explicit assumption because the paper does not specify it.

The bandwidth values and arithmetic hardware parameters draw from the paper's Table 1. Treating bandwidth as byte-hop serialization is an approximation; bandwidth alone does not specify routing latency.

This objective is **not end-to-end inference latency**. It excludes shared-link contention, multicast, router startup, buffer stalls, overlap, input/output transfer, and bias/activation/pooling arithmetic. It does not reproduce the block-streaming schedule or separate CONV/FC placement regions.

In full_frame mode, bandwidth parameters determine communication coefficients; --on_lat and --off_lat are overridden. The old --compute_ops option is retained only to emit an error: an untyped MAC vector cannot describe VVA throughput or communication units safely.

## Workload support

| Workload | Current support |
| --- | --- |
| Built-in simple CNN | Extraction, timing, and short CPU training tested |
| AlexNet / VGG16 | Extraction and operation counts tested; full_frame available |
| Residual networks, including ResNet | Dependency edges can be traced in proxy mode; full_frame rejects residual merges |
| Concatenation-based networks | Full-frame timing rejected; channel-changing merges also rejected in proxy extraction |
| Grouped / depthwise convolutions | Rejected |
| Other torchvision models | Dynamic name lookup; tracing and supported operators are still required |
| Custom models | Python file with build_model(), subject to the same restrictions |

Residual dependency tracing does not model residual-add arithmetic. It must not be described as exact ResNet timing.

A custom model file can return either a model or a model/input pair:

~~~python
import torch
from torch import nn

def build_model():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )
    return model, torch.zeros(1, 3, 16, 16)
~~~

Pass it with --use_cnn --custom_model path/to/model.py. Without a supplied input, the default input is 1 × 3 × 224 × 224. Pretrained weights and datasets are not needed for shape-based extraction.

## Algorithms and comparison discipline

DDPG uses MLP actor/critic networks, actor/critic learning rates 0.0002/0.001, gamma 0.98, and a training minibatch of 64. These networks **do not yet match the paper's CNN policy**.

Each action places up to --batch_z tasks. Coordinates are floored; collisions use the nearest free grid position by Manhattan distance. Nonterminal reward is zero; completed placements receive sqrt(B) − sqrt(L), where B is the best initial random-search cost. OU noise uses assumed theta=0.15 and sigma=0.2 with a fading scale.

Run either baseline on the same small workload:

~~~bash
python src/run_multi_chip.py --algo random --use_cnn \
  --channels_per_partition 128 --timing_model full_frame \
  --iters 100 --seed 0 --report runs/random.json

python src/run_multi_chip.py --algo sa --use_cnn \
  --channels_per_partition 128 --timing_model full_frame \
  --iters 100 --seed 0 --report runs/sa.json
~~~

SA uses a fixed 0.99 cooldown and an approximately 1% task subset (at least two when possible). Relocation to unused cores is an implementation choice; its exact neighborhood is not established by the paper.

The paper uses about one million placements for RS and SA. CLI defaults are much smaller. Matching that budget alone does not establish reproduction. Keep workload, partitioning, topology, objective, units, budgets, and seeds consistent; report runtime and variation over multiple seeds. Do not compare historical proxy scores with the new timing mode or interpret smoke-test gains as the paper's results.

## Important options

Run python src/run_multi_chip.py --help for the complete CLI.

| Option | Default / meaning |
| --- | --- |
| --algo | ddpg; alternatives sa and random |
| --use_cnn | Use an extracted model; otherwise a synthetic DAG |
| --channels_per_partition | 8; set explicitly, since small values can exceed grid capacity |
| --chips_x / --chips_y | 2 / 2 |
| --rows / --cols | 4 / 4 per chip |
| --topology | mesh or torus |
| --timing_model | proxy or full_frame |
| --mac_utilization | 1.0; assumed arithmetic utilization |
| --vva_ops_per_cycle | 1.0; assumed VVA throughput |
| --on_bandwidth_gbs / --off_bandwidth_gbs | 64 / 100; full_frame only |
| --epochs | 1,000 total DDPG episodes; each builds a complete placement |
| --baseline_trials | 1,000 RS trials to establish DDPG reward baseline B |
| --iters | 5,000 trials for standalone RS/SA |
| --batch_z | 3 tasks per action |
| --train_every | 5; set 1 for a training-update attempt every environment step |
| --device | Auto-select CUDA when available; cpu/cuda can be explicit |
| --seed | Unset unless supplied |
| --checkpoint_every | 100 episodes when saving is enabled |
| --report | Optional JSON output path |

Create parent output directories before running. Reports contain configuration, task count, best cost, objective units, chip-major placement IDs, runtime, Torch version, CUDA availability, and limitations. Runtime starts at algorithm dispatch, excluding extraction and environment construction; reports are summaries, not per-episode training logs.

## What has actually been validated

- Original smoke suite, including real CNN extraction.
- Seven regression tests covering operation conservation, timing/byte units, pooling/flatten routing, residual dependencies and timing rejection, core IDs, SA behavior, and parameter updates.
- A 10-episode CPU DDPG run, saved and resumed to episode 12.
- Small SA and random-search CLI runs, including torus mode.
- CLI rejection of unavailable CUDA and invalid timing inputs.
- Real-model extraction at partition size 512:

| Model | Logic tasks | VMM MACs per frame |
| --- | ---: | ---: |
| AlexNet | 252 | 714,188,480 |
| VGG16 | 516 | 15,470,264,320 |

These are implementation checks. H100 execution, 12 GB memory use, long-run convergence, multi-seed paper-scale results, and the paper's reported percentage improvements have not been established.

## Remaining reproduction work

The corrected [project state and next steps](PROJECT_STATE.md) distinguish
implemented behavior from assumptions and historical experiment claims.

For a bounded device check after installing the dependencies above, run:

```bash
python src/validate_device.py --device cpu --output runs/cpu-validation.json
# On the allocated H100 host once SSH access is available:
python src/validate_device.py --device cuda --output runs/h100-validation.json
```

This runs both test suites, checks real DDPG updates and an agent checkpoint
round-trip, and records component timings and CUDA peak training memory in JSON.
CUDA mode fails explicitly if unavailable. The default CPU run passed with 120
steps and 57 updates on 2026-09-11. This short check does not establish convergence,
steady-state memory fit, or GPU speedup; timings include synchronization overhead.

1. Reconstruct and validate block-streaming stages and communication contention with explicit assumptions.
2. Implement compute-aware partitioning and buffer-capacity constraints; the paper's refinement formula is unspecified.
3. Implement and validate merge arithmetic and separate CONV/FC placement regions.
4. Replace the MLP actor/critic with the paper's CNN architecture.
5. Validate CUDA execution and profile actual memory/throughput on the allocated slice.
6. Automate multi-seed, matched-budget experiments and statistical reporting.

Parallel GPU environments are a possible later performance change, not an implemented feature.

## Repository layout

~~~text
src/
  run_multi_chip.py          # Maintained multi-chip CLI, extraction, DDPG, RS, SA
  compute_model.py           # Tile operation counts, arithmetic seconds, traffic bytes
  multi_chip_environment.py  # Placement objective and dependency-level validation
  multi_chip_topology.py     # Mesh/torus costs and physical core IDs
  test_multi_chip.py         # Original smoke suite
  test_reconciliation.py     # Reconciliation regression suite
  validate_device.py         # Bounded CPU/CUDA validation and component timings
  run_multi_chip_fast.py     # Historical alternative; not reconciled or validated
  agent/, env/, runner/      # Original single-chip implementation
RECONCILIATION.md             # Detailed assumptions and reconciliation history
PROJECT_STATE.md              # Corrected progress, evidence and next steps
requirements.txt              # Historical dependency list, not current multi-chip setup
~~~

The original single-chip path is retained separately. The reconciliation and validation described here apply to the maintained multi-chip files above.
