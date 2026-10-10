# DNN Mapping with Reinforcement Learning

This repository maps DNN computation tasks onto a multi-chip many-core accelerator using DDPG, guided DDPG, masked categorical PPO, PPO-guided adaptive simulated annealing, random search, fixed simulated annealing, adaptive simulated annealing (ASA), a DDPG→ASA hybrid, or the sequential baseline (BS). Its reproduction target is Wu et al., **Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning**, ACM TODAES 2020 ([DOI 10.1145/3418498](https://doi.org/10.1145/3418498)).

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
| Guided DDPG | Implemented as an experimental learning repair using legal actions, ASA demonstrations, complete-episode returns, prioritized replay, and twin critics |
| Masked PPO | Implemented as an experimental exact legal-action policy with a paired frozen control |
| PPO-guided ASA | Implemented as an experimental legal-neighbor proposal policy with fixed Metropolis acceptance, the existing ASA temperature controller, and a uniform frozen control |
| Objective sensitivity preflight | Implemented; flat paper-mode objectives stop before long optimization |
| 30 placements/epoch and paper search budgets | Explicitly accounted for by the paper runner |
| XY routing and link contention | Reconstructed and implemented |
| 64 KB weight-buffer constraint | Enforced during paper partition reconstruction |
| 64 KB activation-buffer stalls and exact GRS | Not published in enough detail; not implemented |
| Full paper-scale results | Not run yet |

Paper-mode reports include routed mean hop counts and on/off-chip link-load summaries. Multi-seed summaries report the objective normalized to BS and its inverse ratio. The inverse ratio is useful for comparison, but it is not labeled as measured accelerator throughput.

## Repository layout

- `src/run_multi_chip.py` runs one BS, DDPG, guided-DDPG, masked-PPO, random-search, SA, ASA, or DDPG→ASA experiment.
- `src/run_multiseed_experiment.py` runs selected methods across seeds and can schedule independent CPU jobs concurrently.
- `src/run_paper_experiment.py` runs separate CONV and FC paper-mode suites.
- `src/compute_model.py` reconstructs partitions and converts work/traffic to physical units.
- `src/multi_chip_topology.py` implements physical IDs and mesh/torus routing.
- `src/multi_chip_environment.py` implements the placement objective and diagnostics.
- `src/validate_device.py` performs a bounded CPU/CUDA functionality profile.
- `PROJECT_STATE.md` gives the detailed handoff state and known limitations.
- `NEXT_STEPS.md` tracks the reproduction gates.
- `RESEARCH_FINDINGS.md` audits the failed Kaggle run, Gemini's findings,
  primary literature, forum leads, implemented repairs, and the staged
  learning-validation plan.
- `DDPG_REFERENCE_AUDIT.md` compares this implementation with OpenAI Spinning
  Up, Stable Baselines3, TD3, Wolpertinger, and DDPGfD reference algorithms.
- `scripts/run_kaggle_guided_ablation.sh` runs reproducible trained/control
  Kaggle experiments and packages their reports.
- `scripts/run_kaggle_masked_ppo_ablation.sh` runs the next paired masked-PPO
  learning gate and packages its reports.
- `scripts/run_kaggle_masked_ppo_multichip_gate.sh` tests AlexNet-FC PPO against
  its frozen control and matched-budget RS/ASA/BS baselines.
- `scripts/run_kaggle_ppo_asa_gate.sh` compares learned proposals, an exactly
  uniform frozen proposal control, and ordinary ASA under the same objective
  evaluation budget.
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

## Guided DDPG learning repair

`--algo ddpg_guided` is an experimental extension created after the 3,000-placement diagnostic run showed that the original continuous actor was not learning a useful policy. In that run, almost every requested action collided, the environment silently replaced it with a different core, and replay stored the requested action instead of the action that produced the transition. The deterministic actor collapsed to only a few intended cores while the critic loss grew very large.

Guided DDPG changes the learning path while leaving `--algo ddpg` available as the paper-aligned control:

- the actor emits a proto-action and the twin critics rank nearby legal, collision-free batches;
- the environment executes that exact legal batch and replay stores the same action;
- an ASA placement supplies a valid demonstration for behavior cloning;
- every step receives a discounted complete-placement return instead of waiting for a one-step terminal sample to propagate through replay;
- prioritized replay samples informative transitions and keeps demonstration transitions permanently;
- Huber critic loss, delayed actor updates, gradient clipping, and conservative twin-critic scoring reduce divergence;
- an initial uniformly random legal-action phase broadens replay coverage before actor proposals take over;
- the state contains only the active placement region, so AlexNet CONV uses a 16×16 map instead of padding one active chip to the full 64×64 machine.

A successful guided run should report zero collision repairs, one unique executed core per logic core, finite critic statistics, and a deterministic policy that improves over its untrained control across multiple seeds. These checks show that the learning loop is internally consistent; paper-comparable claims still require the frozen DDPG, BS, RS, and SA baselines under matched placement budgets.

The following bounded AlexNet CONV run performs 100 declared epochs with the paper's 30 complete placements per epoch. The checkpoint path is supplied for both loading and saving, so the same command continues an interrupted run. Diagnostics are appended after a resume.

```bash
mkdir -p runs/alexnet-conv-guided

python src/run_multi_chip.py \
  --algo ddpg_guided --device cuda \
  --use_cnn --model alexnet \
  --partition_mode paper_targets --workload_region conv \
  --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
  --agent_arch paper_cnn --reward_mode sparse \
  --epochs 100 --placements_per_epoch 30 \
  --baseline_trials 1000 --batch_z 3 --train_every 1 \
  --exploration_decay_placements 2400 \
  --guided_top_k 8 --guided_demo_iterations 5000 \
  --guided_pretrain_updates 500 --guided_bc_decay_placements 2000 \
  --guided_random_start_placements 100 \
  --diagnostics_every 30 --checkpoint_every 100 \
  --sensitivity_trials 64 --seed 0 \
  --save_checkpoint runs/alexnet-conv-guided/seed0.pt \
  --load_checkpoint runs/alexnet-conv-guided/seed0.pt \
  --diagnostics runs/alexnet-conv-guided/seed0.jsonl \
  --asa_diagnostics runs/alexnet-conv-guided/seed0-asa.jsonl \
  --report runs/alexnet-conv-guided/seed0-report.json
```

The 3,000 online placements are a learning validation budget, not the paper's final search budget. If this run beats a separately run untrained control and remains stable, repeat it for at least five seeds before scaling the online-placement count.

For the matched untrained control, repeat the same command with
`--guided_disable_learning`, a different checkpoint/diagnostics/report prefix,
and the same seed and budgets. This flag suppresses both demonstration
pretraining and online gradient updates while preserving legal-action handling,
uniform random warm-up, ASA generation, noise scheduling, and evaluation
accounting. Compare `deterministic_cost` between trained and control JSONL files;
best-so-far cost can still improve through random exploration and is not enough
to establish learning.

The evidence and decision rules behind this experiment are documented in
[RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md).

The implementation comparison with maintained public DDPG code is documented
in [DDPG_REFERENCE_AUDIT.md](DDPG_REFERENCE_AUDIT.md). The audit also found and
fixed shared replay/exploration randomness: trained and no-learning conditions
now use identical seeded action randomness while replay sampling has an
independent stream.

On Kaggle, the following repository script runs a paired one-seed check and
creates `runs/kaggle-guided-short.zip`:

```bash
bash scripts/run_kaggle_guided_ablation.sh short
```

For the five-seed follow-up, use:

```bash
bash scripts/run_kaggle_guided_ablation.sh extensive
```

The short preset runs 300 online placements for one trained/control pair. The
extensive preset runs 3,000 online placements for each condition across five
paired seeds. On two Kaggle T4 GPUs the pairs run concurrently, one seed pair
at a time. The runner refreshes its ZIP archive after every completed pair and
again when it exits, so an interrupted session leaves recoverable output. A
second invocation skips pairs carrying a valid completion marker. It restarts
an interrupted pair from the beginning because guided replay is not stored in
the model checkpoint; silently resuming without replay would change the
experiment.

The extensive guided-DDPG run is now complete. Training lost all five paired
late deterministic comparisons. The trained aggregate mean was 39.0857
microseconds versus 38.7322 microseconds for the frozen control. Guided DDPG is
therefore retained as a negative ablation rather than the recommended next run.

## Masked PPO learning experiment

`--algo ppo_masked` matches the actual decision: choose one unused physical
core for the next logic core. Invalid actions receive zero probability before
sampling, so the policy never needs collision repair. It uses PPO clipping,
GAE, a value baseline, entropy regularization, gradient clipping, deterministic
diagnostics, resumable model/optimizer checkpoints, and explicit evaluation
accounting. This is an experimental improvement; it is not Wu et al.'s DDPG.

Run the one-seed paired gate on a Kaggle GPU first:

```bash
bash scripts/run_kaggle_masked_ppo_ablation.sh short
```

The script runs 300 trained placements and 300 frozen-control placements, then
creates `runs/kaggle-masked-ppo-short.zip`. It verifies that both conditions
start from the same deterministic policy, reports zero collision repairs, and
compares their late deterministic costs. Only if this gate is positive, run:

```bash
bash scripts/run_kaggle_masked_ppo_ablation.sh extensive
```

The extensive preset uses five paired seeds and 3,000 placements per condition.
Both presets use the paper-style sparse terminal reward. With the fixed episode
horizon, `gamma=1` and `GAE lambda=1` assign the complete-placement return to
every placement decision without adding partial objective evaluations.

The extensive AlexNet-CONV experiment is complete. Trained PPO beat its frozen
deterministic control in all five seeds. Its aggregate late deterministic mean
was 42.6118 microseconds versus 45.3588 microseconds, a 6.06% improvement; the
exploratory paired 95% interval for absolute improvement was 1.0972 to 4.3969
microseconds. All initial-policy comparisons matched, collision repairs were
zero, and training diagnostics were finite.

This establishes a repeatable learning signal on the reconstructed one-chip
AlexNet-CONV objective. It does not establish paper-level performance. Best
retained solutions improved by only 1.26% on average because random rollout
search remained competitive. The next experiment must use a multi-chip region
and compare all methods under matched evaluation budgets.

The bounded next gate uses AlexNet-FC, whose 932 logic cores span four chips.
It runs 300 trained PPO placements and a frozen control in parallel, then gives
random search and ASA the same 567 complete-placement evaluations available to
each PPO condition (256 reward-baseline trials, 300 training rollouts, one
initial deterministic evaluation, and ten periodic deterministic evaluations):

```bash
bash scripts/run_kaggle_masked_ppo_multichip_gate.sh
```

The resulting `runs/kaggle-masked-ppo-alexnet-fc-short.zip` contains the paired
diagnostics, RS/ASA/BS reports, logs, checkpoints, and `gate-summary.json`.
The completed short gate produced only a 0.163% directional deterministic
improvement over the frozen control. PPO's best cost was 24.2075 microseconds,
worse than matched random search at 24.1472 and ASA at 24.1734 microseconds.
The mechanics passed, but this is not a multi-chip learning or optimizer pass.
The next bounded test is one paired 3,000-placement extension with 4,101
matched evaluations before considering a five-seed run:

```bash
bash scripts/run_kaggle_masked_ppo_multichip_gate.sh extended
```

The extension completed with a 0.558% late deterministic advantage, but no
best-placement benefit: trained PPO reached 24.1080 microseconds and its frozen
control 24.1072. Random search reached 24.1155, while ASA reached 23.4318
microseconds and beat trained PPO by 2.80% under the same 4,101-evaluation
budget. Do not scale the unchanged PPO configuration to five FC seeds. Any
ASA/PPO hybrid must share one fixed total evaluation budget and beat ASA alone.

A subsequent five-seed AlexNet-FC baseline confirmed that ASA's advantage is
repeatable. With one random initialization plus 4,100 ASA proposals per seed,
ASA averaged 23.5001 microseconds (sample standard deviation 0.0653) versus
24.1305 microseconds (0.0408) for 4,101-sample random search. ASA won every
paired seed by 0.5720--0.6837 microseconds, a mean 2.61% improvement, and was
5.70% below the 24.9203-microsecond sequential baseline. This is the minimum
quality threshold for any learned-proposal ASA extension.

## PPO-guided ASA gate

`--algo ppo_asa` is an experimental hybrid based on the verified design of
[Qiu and Liang's RL-Based-SA](https://github.com/nathanqiu07/RL-Based-SA-Public).
PPO ranks a configurable set of legal swap/relocation candidates. Candidate
features include task communication pressure, physical movement, whether the
current bottleneck stage is touched, a cheap communication-distance change,
temperature, progress, the previous energy change, and recent acceptance.
Only the selected proposal evaluates the true objective. Metropolis acceptance
and the existing adaptive temperature controller remain outside the policy.

The proposal head begins at exactly uniform probability. Passing
`--ppo_asa_disable_learning` therefore provides a matched uniform-proposal
control with identical initialization and evaluation accounting. Generated
model files are final snapshots for audit and are explicitly not resumable.

The local 1,001-evaluation smoke produced 23.9986 microseconds for learned
PPO-ASA, 24.1123 for its uniform control, and 23.9629 for ordinary ASA. The
completed 4,101-evaluation Kaggle gate reversed that small early signal:
learned PPO-ASA reached 23.5602 microseconds, compared with 23.4213 for the
uniform control and 23.4318 for ordinary ASA. The learned run was 0.593% worse
than its control and 0.548% worse than ASA. Its policy entropy remained within
0.132% of uniform and 76.7% of selected proposals were objective-neutral.

The command that reproduces the completed gate is:

```bash
bash scripts/run_kaggle_ppo_asa_gate.sh extended
```

The output archive contains both policy snapshots, JSONL diagnostics, all
three reports, logs, and `gate-summary.json`. Do not scale this unchanged
policy to five seeds. The next bounded experiment must first reduce neutral
proposals with a matched bottleneck-focused neighborhood, then beat both its
uniform control and ordinary ASA. The five-seed threshold remains the
established ASA mean of 23.5001 microseconds.

The matched focused neighborhood is enabled with
`--ppo_asa_focus_bottleneck`. It guarantees that each candidate moves a task
from the current maximum-latency pipeline stage, while leaving the policy,
acceptance, temperature controller, and evaluation accounting unchanged. A
local 1,001-evaluation smoke reached 24.0627 microseconds trained, 24.1598 for
the focused uniform control, and 23.9629 for ASA. The completed extended gate
reached 23.7792 trained, 23.7595 control, and 23.4318 ASA. Although focusing
cut neutral proposals to 37.2%, trained PPO lost both comparisons. The command
that reproduces the completed experiment is:

```bash
bash scripts/run_kaggle_ppo_asa_gate.sh focused-extended
```

Do not scale either short-chain PPO-ASA variant to five seeds. The public-code
and training-scale audit in `GITHUB_IMPLEMENTATION_AUDIT.md` shows that the
closest official RL-Based-SA setup uses millions of transitions across many
parallel problem instances. The next RL implementation must make that training
regime possible while retaining the current evaluator and matched controls.

That training-scale path is now implemented. `--ppo_asa_restart_interval`
starts independently initialized placement chains while retaining one shared
proposal policy and optimizer; `--ppo_asa_focus_fraction` mixes bottleneck-
anchored and global proposals in every candidate pool. Chain boundaries are
terminal for advantage estimation, and every new-chain initialization is
included in `total_objective_evaluations`. The paired script automatically
gives ordinary ASA the same number of true objective calls.

A 520-evaluation local smoke passed all restart, finite-update, frozen-control,
and budget checks. Learned and frozen PPO-ASA tied at 24.31712 microseconds and
ASA reached 24.05376 microseconds; this small run validates mechanics only.
The first decision run trains across 1,954 chains and uses exactly 251,954
objective evaluations per condition (250,000 proposals plus initializations):

```bash
bash scripts/run_kaggle_ppo_asa_gate.sh multichain-train
```

On a Kaggle T4 x2 session, trained and frozen PPO-ASA run concurrently, then
ASA runs with the matched budget. The script prints progress every minute and
creates `runs/kaggle-ppo-asa-alexnet-fc-multichain-train.zip`. Run
`multichain-preflight` first only when checking a new environment. The code
uses the training structure described by the official RL-Based-SA project but
does not copy its source; citations and compatibility limits are recorded in
`GITHUB_IMPLEMENTATION_AUDIT.md`.

The training-scale run is complete. Trained PPO-ASA reached 23.96800
microseconds versus 24.08032 for its frozen control, a 0.466% improvement, and
beat control at 3,413 of 3,906 matched checkpoints. Its entropy fell 10.27%
below the uniform maximum and its improving-move rate was higher, so the policy
did learn. Ordinary uninterrupted ASA reached 23.14624 microseconds and beat
the learned search by 3.55%. This is a learning-only result, not an optimizer
win. Reports, the study, and the trained checkpoint are preserved under
`runs/kaggle-ppo-asa-multichain-5ea4c65-2026-10-09/`.

The next gate freezes that checkpoint and evaluates it on five unseen,
uninterrupted chains. Each seed gives the trained policy, a fresh uniform
policy, and ASA exactly 4,101 objective evaluations:

```bash
bash scripts/run_kaggle_ppo_asa_holdout.sh
```

This separation is required because short restarts generate diverse training
data but handicap final placement refinement. The holdout gate reports whether
the learned proposal generalizes and whether it beats ASA as an optimizer.

The five-seed holdout is complete and rejects the saved checkpoint. Frozen
PPO-ASA averaged 23.63050 microseconds, versus 23.51504 for a fresh uniform
policy and 23.50518 for ASA. It lost all five pairs to both controls. A local
five-seed 128-proposal diagnostic showed a weak 0.385% mean advantage over
uniform in three of five seeds, indicating that the policy was specialized to
its short training horizon rather than useful for 4,100-step refinement.

The next experiment corrects this specific mismatch. Training uses 4,100-step
episodes, normalizes progress within each chain, updates PPO at chain
boundaries, and immediately performs the same five-seed frozen holdout:

```bash
bash scripts/run_kaggle_ppo_asa_deployment_experiment.sh
```

That deployment-matched run is complete. Training used 250,000 proposals but
only 61 PPO updates. Its frozen checkpoint averaged 23.51635 microseconds over
five unseen seeds, versus 23.51504 for a fresh uniform policy and 23.50518 for
ASA. It beat uniform in three pairs and ASA in one; both paired intervals
included zero. Final entropy remained within 0.31% of uniform, approximate KL
was near zero, and PPO clipping never activated. The checkpoint is rejected.

The paper audit in `PAPER_SCALE_DECISION.md` shows why this result does not yet
reject the method: the closest RL-Based-SA implementation trains for 1,000 PPO
epochs, while this checkpoint received 61 updates. The sufficient-scale run
uses 1.25 million proposals, exactly 1,250 updates, paper-aligned PPO settings
without an entropy bonus, and a 10-seed frozen holdout:

```bash
bash scripts/run_kaggle_ppo_asa_sufficient_scale.sh
```

If that checkpoint does not beat the fresh uniform policy in at least eight
of 10 holdout seeds with a positive paired mean, the current candidate-scoring
architecture is closed. ASA remains the recommended optimizer unless the
trained policy also beats ASA under the same objective-call budget.

## Interpreting results

A successful run proves that the program executed; it does not prove that DDPG learned. Use the JSONL diagnostics to compare noisy and deterministic policy costs, actor/critic losses, unique intended cores, and collision repair counts. Judge convergence across at least five seeds and compare all methods under the declared complete-placement budgets. Diagnostic policy rollouts are additional objective evaluations; use `total_candidate_evaluations` whenever deterministic retention is enabled.

Report CONV and FC separately. Normalize latency to BS as in the paper, include seed mean/sample standard deviation/minimum/maximum, and examine hop counts and link loads. Do not compare old proxy scores, full-frame scores, and paper-pipeline seconds as though they were the same metric.

See `PROJECT_STATE.md` for validated evidence and `NEXT_STEPS.md` for the work still required before claiming paper-comparable results.

## Bounded local CPU study (September 2026)

After the CPU setup above, this runs five small AlexNet CONV diagnostics:

```bash
source .venv/bin/activate
mkdir -p runs/local-study
for SEED in 0 1 2 3 4; do
  python src/run_multi_chip.py \
    --algo ddpg --device cpu --cpu_threads 2 --cpu_interop_threads 1 \
    --use_cnn --model alexnet --partition_mode paper_targets \
    --workload_region conv --timing_model paper_pipeline --routing_model paper_xy \
    --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
    --agent_arch paper_cnn --reward_mode potential \
    --epochs 12 --placements_per_epoch 1 --baseline_trials 64 \
    --batch_z 3 --train_every 10 --exploration_decay_placements 1000 \
    --diagnostics_every 2 --retain_deterministic_candidates \
    --checkpoint_every 2 --sensitivity_trials 64 --seed "$SEED" \
    --save_checkpoint "runs/local-study/ddpg-seed${SEED}.pt" \
    --diagnostics "runs/local-study/ddpg-seed${SEED}.jsonl" \
    --report "runs/local-study/ddpg-seed${SEED}.json"
done
```

Start with one job and two CPU threads. More threads or simultaneous CNN jobs
can be slower; benchmark on the actual machine before increasing either.
This command starts fresh. To continue an interrupted seed, run its command
with `--load_checkpoint runs/local-study/ddpg-seedN.pt`, retaining the same
exploration horizon and using a new diagnostics filename. Replay is not saved,
so resumed runs are not equivalent to uninterrupted experiments. Checkpoints
from before deterministic-retention accounting are incompatible with this code.

Each uninterrupted seed evaluates 12 training placements, six deterministic
rollouts, and 64 reward-baseline samples: **82 selectable candidates**, plus
64 preflight samples used only to check sensitivity. Potential shaping and
retention are optional project extensions. These very short runs assess
repeatability and execution; they cannot establish convergence or reproduce
the paper's benchmark results.

For comparable small search baselines, run `--algo random --iters 82` or
`--algo sa --iters 82` / `--algo asa --iters 82` with the same workload,
topology, seed, and preflight settings. SA and ASA additionally evaluate one
initial placement (83 candidates including initialization). ASA's 32
calibration proposals are included in its 82 proposals. Record this accounting
instead of calling the experiments exactly budget-matched. DDPG's
`total_candidate_evaluations` excludes its 64 reward-baseline samples.

An untrained-policy control is essential: repeat the same bounded command with
`--train_every 100000000`, fresh output names, and no checkpoint loading.
With only 732 environment steps, this performs zero optimizer updates while
keeping exploration and deterministic evaluation. Compare its best layouts to
the trained policy before attributing an advantage over RS/SA to learning.

Completed results and per-seed settings are in
[runs/local-cpu-5seed-2026-09-30/STUDY.md](runs/local-cpu-5seed-2026-09-30/STUDY.md).
The trained mean was 38.24 µs versus 38.47 µs untrained; this short study does
not establish a reliable learning advantage.

## Target-network normalization fix (September 2026)

`paper_cnn` uses BatchNorm. The DDPG target actor and critic now stay in
evaluation mode when calculating the Bellman target, so a target value does not
change merely because different transitions share its minibatch. Target
BatchNorm buffers are synchronized with their online networks after each
update, and the critic's BatchNorm statistics are not changed during the
actor-only update. This is a correctness and stability repair, not a change to
the paper's stated hyperparameters.

A matched five-seed, 12-placement CPU diagnostic produced a mean best latency
of **38.10 µs**, compared with **38.24 µs** before the repair (0.14 µs, or
0.37%, lower). Among the four seeds run with the same two-thread setting, three
improved and one regressed. The sample is far too small to claim convergence,
but the fix removes a definite source of target noise. Old DDPG checkpoints are
intentionally incompatible; start a new checkpoint after pulling this revision.
The reports and exact command are in
[runs/local-cpu-batchnorm-2026-09-30/STUDY.md](runs/local-cpu-batchnorm-2026-09-30/STUDY.md).
