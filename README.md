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

## Installation

Clone the maintained branch and create an isolated environment:

```bash
git clone --branch cpu https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU-only use:

```bash
python -m pip install numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For the H100, install the CUDA build of PyTorch and torchvision selected for the host driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install NumPy. The historical root `requirements.txt` contains old pins and is not the environment specification for this path.

Verify the environment:

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The local CPU checks used Python 3.13, torch 2.14.0+cpu, and torchvision 0.29.0+cpu. Those versions are evidence for this workstation only, not required CUDA pins.

## Tests

Run both maintained suites from the repository root:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
```

Then run the bounded device validator:

```bash
python src/validate_device.py --device cpu --output runs/cpu-validation.json
# On the allocated GPU host:
python src/validate_device.py --device cuda --output runs/h100-validation.json
```

The CUDA validator checks that networks and training tensors are on the GPU, records visible/peak memory, and separates action, environment, replay, and update timing. Environment evaluation, placement repair, replay storage, and experiment control remain CPU-side, so a faster GPU does not by itself improve solution quality.

## Quick functional run

This small CPU command checks extraction, training, checkpointing, and reports. It is not a paper comparison:

```bash
mkdir -p runs
OMP_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model simple \
  --channels_per_partition 128 --timing_model full_frame \
  --device cpu --epochs 10 --baseline_trials 10 \
  --train_every 1 --batch_z 3 --seed 0 \
  --save_checkpoint runs/smoke.pt \
  --checkpoint_every 5 --report runs/smoke.json
```

`--epochs` is a total target when resuming. Checkpoints restore models, optimizers, best placement, baseline, counters, and RNG state. Replay is not persisted, and the noise-fading schedule depends on the requested total, so a resumed run is not bit-exact.

Run standalone ASA on CPU:

```bash
python src/run_multi_chip.py \
  --algo asa --iters 100000 --seed 0 \
  --asa_diagnostics runs/asa-seed0.jsonl \
  --report runs/asa-seed0.json
```

Run DDPG and refine its best placement with ASA. Here DDPG evaluates 8,000 complete placements and ASA evaluates 2,000 candidates:

```bash
python src/run_multi_chip.py \
  --algo ddpg_asa --device cpu --cpu_threads 20 \
  --epochs 8000 --placements_per_epoch 1 --iters 2000 \
  --baseline_trials 10000 --diagnostics_every 100 \
  --seed 0 --report runs/ddpg-asa-seed0.json
```

ASA measures uphill cost changes to set its initial temperature. It then adapts temperature from the observed acceptance ratio, reheats after stagnant windows, and expands or contracts the fraction of moved tasks. The JSON report records accepted/improving moves, reheats, temperature, neighborhood size, initialization, and exact candidate count. `ddpg_asa` always warm-starts ASA from the best DDPG placement, so the final returned solution cannot be worse than that warm start.

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

Same-chip messages take deterministic X-then-Y routes. Inter-chip messages run from the source core to a lower-left chip-periphery gateway, traverse the chip grid X then Y, then travel from the destination gateway to its core. A time phase combines the maximum task compute time with the bottleneck serialized load on any shared directed link. The gateway is inferred from Figure 3; the paper's complete GRS implementation is unavailable.

CONV work and traffic are divided across `--conv_blocks` (default 4, taken from the illustrative Figure 7). FC uses one layer work unit. Workload-specific block counts are not published. ResNet branch dependencies are retained; residual addition is assigned to the destination transformation/VVA path without adding a Figure 6 core.

## Objectives

| Mode | Meaning | Units |
| --- | --- | --- |
| `proxy` | Communication volume × hierarchical distance | Arbitrary score |
| `full_frame` | Ideal arithmetic plus per-task byte-hop serialization | Seconds |
| `paper_pipeline` | Reconstructed block/layer phases with XY shared-link contention | Seconds |

`paper_pipeline` requires `paper_targets`, `workload_region conv|fc`, and `paper_xy`. It uses Table 1's 128 MACs at 400 MHz, 64 GB/s/core on-chip links, 100 GB/s/chip off-chip links, 8-bit activations/weights, and 32-bit partial sums. VVA defaults to one addition/cycle because its throughput is unpublished.

The remaining simulator gaps are material: 64 KB input/activation-buffer stalls, exact multicast/GRS behavior, router startup, transformation costs, compute/communication overlap, and workload-specific block schedules. Results must therefore be called a documented reconstruction, not an exact replay of the authors' simulator.

## DDPG and baselines

Paper-mode DDPG uses the Figure 9 spatial CNN, the 2-D placement grid, batched `2z` continuous coordinates, floor conversion, nearest-free Manhattan repair, actor learning rate 0.0002, critic learning rate 0.001, gamma 0.98, minibatch 64, and sparse terminal reward. `batch_z`, OU parameters, replay capacity, padding, LRN parameters, target networks, and soft-update coefficient are not fully specified by the paper and remain recorded assumptions.

`--reward_mode potential` and the `mlp`/`cnn` agents are improvement conditions. Do not mix their results into the frozen paper-mode comparison. Collision repairs are expected because continuous coordinates may select the same or a masked core; diagnostics separate occupied-core repairs from mask repairs and also evaluate the deterministic policy.

BS fills allowed physical cores in chip-major order. RS samples complete valid placements. SA uses current-cost acceptance, cooling factor 0.99, and roughly 1% placement perturbations that may use free cores.

ASA and DDPG→ASA are project extensions, not features claimed by the source DNN-mapping paper. Keep fixed SA in result tables as the paper-aligned baseline. The runner matches the hybrid's combined candidate count to DDPG. For a direct hybrid-versus-ASA ablation, set `--search_budget` equal to `--epochs × --placements_per_epoch`; otherwise the paper-scale defaults deliberately give SA/ASA one million candidates and DDPG/hybrid 300,000.

## Interpreting results

A successful run proves that the program executed; it does not prove that DDPG learned. Use the JSONL diagnostics to compare noisy and deterministic policy costs, actor/critic losses, unique intended cores, and collision repair counts. Judge convergence across at least five seeds and compare all methods under the declared complete-placement budgets.

Report CONV and FC separately. Normalize latency to BS as in the paper, include seed mean/sample standard deviation/minimum/maximum, and examine hop counts and link loads. Do not compare old proxy scores, full-frame scores, and paper-pipeline seconds as though they were the same metric.

See `PROJECT_STATE.md` for validated evidence and `NEXT_STEPS.md` for the work still required before claiming paper-comparable results.
