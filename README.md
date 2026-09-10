# DNN Mapping with Reinforcement Learning

This project studies placement of DNN computation across multi-chip,
many-core accelerators. It is based on *Core Placement Optimization for
Multi-chip Many-core Neural Network Systems with Reinforcement Learning*
(Wu, Deng, Li, and Xie, ACM TODAES 2020).

The maintained multi-chip path is a tested research implementation with
explicit approximations. It is not a complete reproduction of the paper's
block-streaming simulator or reported percentages.

## What it does

The program traces a PyTorch model, partitions Conv2d and Linear layers into
VMM and VVA logic tasks, builds a communication graph, assigns tasks to
physical cores, and searches for a low-cost placement with DDPG, Random Search,
or Simulated Annealing.

The maintained path includes torch.fx dependency tracing, custom model support,
mesh and torus topologies, channel-overlap routing through pooling and flattening,
batched actions, Manhattan collision resolution, sparse rewards, checkpoints,
JSON reports, and CUDA-aware actor/critic training.

## Present state

CPU validation is complete for both test suites, short DDPG training and
checkpoint resume, SA/random runs, and AlexNet/VGG16 extraction. CUDA selection
and tensor placement are implemented, but H100 execution and a 12 GB slice have
not been tested in this repository.

The DDPG actor and critic are still MLPs rather than the paper's CNNs. The
current full-frame timing mode is an explicit approximation. It does not model
the paper's block-streaming schedule, shared-link contention, router startup,
buffer stalls, communication/computation overlap, or residual merge arithmetic.
The paper's compute-balanced partitioning rule is qualitative and unspecified;
the current uniform partitioner is therefore an interpretation.

See [RECONCILIATION.md](RECONCILIATION.md) for detailed provenance,
assumptions, limitations, and validation evidence.

## Installation

Use Python 3.10 or newer. The root `requirements.txt` contains historical
dependencies for the original single-chip code, including old Torch pins. It
is not the recommended environment for the maintained multi-chip path.

```bash
git clone --branch codex/reconciled-paper-implementation \
  https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

For CPU development:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPUs, install CUDA-enabled PyTorch and a matching torchvision from
the [official PyTorch selector](https://pytorch.org/get-started/locally/).
Choose the CUDA version supported by the host; do not install CPU-only wheels
in the H100 environment.

## Tests

Run from the repository root:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p test_reconciliation.py -v
```

The original suite covers topology, placement, batching, CNN extraction, and
baselines. The reconciliation suite covers operation conservation, timing and
byte units, pooling/flatten routing, residual safeguards, core-ID conversion,
SA behavior, and actual DDPG parameter updates. CUDA checks run automatically
when a CUDA device is available.

## Small CPU run

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

Resume to a total of 12 episodes:

```bash
OMP_NUM_THREADS=2 python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model simple \
  --channels_per_partition 128 --timing_model full_frame \
  --device cpu --epochs 12 --baseline_trials 10 \
  --train_every 1 --batch_z 3 --seed 0 \
  --load_checkpoint runs/smoke.pt --save_checkpoint runs/smoke.pt \
  --report runs/resume.json
```

Checkpoints include a workload/configuration fingerprint and RNG states.
Replay memory is not persisted, so resumed training is not bit-for-bit
identical. Old or incompatible checkpoints are rejected.

## GPU / H100 run

Verify the allocated device before training:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run the tests first, then a short CUDA smoke test:

```bash
python src/run_multi_chip.py \
  --algo ddpg --use_cnn --model alexnet \
  --channels_per_partition 512 --timing_model full_frame \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
  --device cuda --epochs 10 --baseline_trials 10 \
  --batch_z 3 --train_every 1 --seed 0 \
  --save_checkpoint runs/alexnet-cuda.pt \
  --report runs/alexnet-cuda.json
```

The actor, critic, and training minibatches use CUDA. Graph extraction,
environment evaluation, replay storage, and placement search remain on CPU.
Multiple environments are not batched on the GPU yet. Check visible memory on
the actual H100 slice before increasing workload or episode counts.

## Timing modes

`--timing_model proxy` is the default historical score. It multiplies
communication volume by on-chip/off-chip hop weights and has arbitrary units.

`--timing_model full_frame --use_cnn` reports seconds for one full-frame work
unit using these documented assumptions:

- VMM time: `ceil(MACs / (128 * utilization)) / 400 MHz`.
- VVA time: `ceil(additions / (vva_ops_per_cycle * utilization)) / 400 MHz`.
- VMM outputs use 4 bytes per partial sum; VVA outputs use 1 byte per activation.
- Default serialization bandwidth is 64 GB/s on-chip and 100 GB/s off-chip.
- Default VVA throughput is 1 addition/cycle because the paper does not specify it.

Full-frame timing is not end-to-end inference latency. It excludes shared-link
contention, multicast, router startup, buffers, stalls, overlap, input/output
transfer, and block-streaming startup. Residual/concat timing and grouped
convolutions are rejected where their costs cannot be represented safely.

## Workloads and algorithms

Built-in `simple`, torchvision `alexnet`, `vgg16`, and `resnet50` are supported.
Custom models must define `build_model()` returning either a model or
`(model, dummy_input)` and can be passed with `--custom_model`.

`--channels_per_partition` controls channel grouping. Smaller values produce
more logic tasks and may exceed the grid capacity. No particular value is
claimed to reproduce the paper's unspecified compute-balanced allocation.

DDPG uses batched actions, learning rates `0.0002` and `0.001`, discount `0.98`,
minibatch size 64, and fading Ornstein-Uhlenbeck exploration noise. Each
completed placement receives `sqrt(B) - sqrt(L)`; intermediate rewards are zero.
The paper used approximately one million placements for Random Search and SA;
CLI defaults are smaller for development.

For all options:

```bash
python src/run_multi_chip.py --help
```

Use identical workload, timing mode, topology, budget, and seed when comparing
algorithms. Do not compare proxy scores with full-frame seconds.

## Repository layout

```text
src/run_multi_chip.py          maintained multi-chip CLI and algorithms
src/compute_model.py            operation, arithmetic, and traffic estimates
src/multi_chip_environment.py   placement objective and DAG validation
src/multi_chip_topology.py      mesh/torus topology and core IDs
src/test_multi_chip.py          original smoke tests
src/test_reconciliation.py      reconciliation regression tests
RECONCILIATION.md               detailed status and assumptions
```

The historical single-chip PPO implementation remains under `src/agent`,
`src/env`, and `src/runner`; it is separate from the maintained multi-chip path.
