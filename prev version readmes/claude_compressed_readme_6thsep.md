# DNN Mapping with Reinforcement Learning
## Core Placement Optimization for Multi-Chip Many-Core DNN Accelerators

This project reproduces and extends **"Core Placement Optimization for Multi-Chip Many-Core Neural
Network Systems with Reinforcement Learning"** (Wu, Deng, Li, Xie — ACM TODAES 2020), which uses a
DDPG reinforcement-learning agent to decide where to physically place a neural network's computation
across a multi-chip hardware grid, minimizing communication latency.

It started from the open-source repo *Core Placement with Reinforcement Learning* and has since been
substantially rewritten to match the paper's actual algorithm, then scaled up to run real networks
(AlexNet, VGG16, ResNet50) at the paper's real hardware scale (4,096 cores).

**Status in one line:** the core RL algorithm now matches the paper's math and mechanics (verified
against the paper's text section-by-section, with a passing test suite), and has produced real,
credible improvements over a random-search baseline on full-scale runs — up to ~47% on one AlexNet
run. The paper's *exact* reported numbers haven't been reproduced yet — a handful of concrete,
well-understood gaps remain (Section 10).

---

# 1. What Problem Is This Solving? (Plain-Language Version)

A neural network is a sequence of computations (layers) that pass data to each other. When you run
that network on specialized hardware, each layer's work gets split up and assigned to a physical
"core" — a small chunk of the chip that does math and talks to its neighbors over wires.

**The problem:** wires that go between cores *on the same chip* are fast. Wires that cross to a
*different chip* are much slower (5–10x, per the paper). So if two parts of the network that talk to
each other a lot end up on different chips, everything slows down.

```text
Task 0 ──talks a lot──> Task 1
   |                        |
   v                        v
Core on Chip A      Core on Chip B   <-- BAD: heavy traffic crosses a slow chip boundary
```

The question this project answers: **given a neural network and a grid of chips/cores, where should
each piece of computation go to minimize communication delay?** For any nontrivial network, trying
every possible arrangement is impossible (64 cores alone gives 64! ≈ 10⁸⁹ possibilities), so the
paper trains an RL agent (DDPG) to learn good placements instead of searching exhaustively.

---

# 2. Reproduce First, Improve Second

This project follows a strict two-stage plan:

```text
STAGE 1 — Reproduce            STAGE 2 — Improve
Paper → Faithful code            Validated baseline → Identify a real limitation
      → Controlled experiments        → Propose + implement a change
      → Baseline results               → Compare against the validated baseline
                                        → Only THEN call it an improvement
```

The reason for being strict about this: if you change something and the results improve, that's only
a meaningful finding if you know your *starting point* was actually correct. Otherwise you can't tell
whether you fixed a bug or discovered something real. **We are currently finishing Stage 1.**

---

# 3. Repository Structure

```text
src/
├── multi_chip_topology.py       # Grid of chips, each with a core mesh; latency model
├── multi_chip_environment.py    # RL environment: placement, pipeline-latency objective
├── run_multi_chip.py            # Main entry point: workload extraction, DDPG, SA, RS, CLI
├── test_multi_chip.py           # Smoke tests for everything above
│
├── agent/                       # ORIGINAL single-chip PPO implementation (untouched by
│   ├── model.py                 # this reproduction effort — the paper's system is
│   ├── actor.py                 # multi-chip; single-chip PPO is a separate, earlier
│   ├── agents.py                # part of the starting repo, kept for reference)
│   └── community_detection.py
├── env/
│   ├── core_mapper.py           # Single-chip placement + collision handling
│   └── reward.py                # Single-chip reward, INCLUDING deadlock penalty logic
│                                 # (deadlock handling is NOT part of the multi-chip
│                                 #  pipeline-latency objective described below)
├── runner/
│   ├── exact_mapping.py         # Zigzag/Neighbor heuristics (single-chip only)
│   ├── random_search.py
│   └── simulated_annealing.py
│
├── parsedata.py, structure.py   # Netlist parsing for the single-chip path
├── config/                      # Single-chip config files
└── main.py, compare_algos.py    # Single-chip entry points
```

**Everything in this README past this point is about the multi-chip DDPG path** (top 4 files above)
— that's what the 2020 paper actually describes and what all the reproduction work has targeted.

---

# 4. System Pipeline

```text
Real DNN (AlexNet / VGG16 / [ResNet50, approximate — see Section 10])
        |
        v
Hook-based extraction  (extract_model_task_graph in run_multi_chip.py)
        |
        v
Channel partitioning into VMM / VVA logic cores  (Sec 3.1.1 / Fig. 5-6 of the paper)
        |
        v
Communication DAG (task_graph: which logic core sends how much data to which other)
        |
        v
Multi-chip environment  (multi_chip_environment.py — paper-scale: up to 4,096 cores)
        |
        v
Placement algorithm: DDPG  |  Random Search  |  Simulated Annealing
        |
        v
Pipeline bottleneck-latency evaluation  (Eq. 4 of the paper)
        |
        v
Sparse, baseline-normalized reward  (Algorithm 1 of the paper, DDPG only)
```

---

# 5. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy torch
```

For real network workloads (AlexNet/VGG16/ResNet50), also install:
```bash
pip install torchvision
```

Check everything's available:
```bash
python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

No GPU is required — the implementation auto-detects CUDA if present and otherwise runs on CPU with
all available threads. A modern multi-core CPU (8+ cores) is recommended for anything beyond the
smallest smoke tests. Note: sustained multi-hour, all-core CPU runs can trigger thermal throttling on
desktop-class hardware, which shows up as per-episode time gradually increasing over a long run —
this is a hardware/cooling effect, not a bug, and doesn't affect result correctness (see Section 9).

---

# 6. Quick Start

**Always run the test suite before a long experiment:**
```bash
python3 test_multi_chip.py
```
Expect 7 checks to pass (skips the CNN-partitioning check gracefully if torch isn't installed):
topology, environment, pipeline latency/objective, collision resolution, batched actions, CNN
workload partitioning (including real torchvision models), and the SA-vs-random sanity check.

**Smallest possible DDPG run** (fast, good for checking nothing's broken):
```bash
python3 run_multi_chip.py --algo ddpg --use_cnn --epochs 50
```

---

# 7. Example Commands (Full Reference)

| Goal | Command |
|---|---|
| Legacy whole-layer workload (debugging only) | `python3 run_multi_chip.py --algo ddpg --use_cnn --channels_per_partition 0` |
| Small partitioned demo CNN (906 logic cores) | `python3 run_multi_chip.py --algo ddpg --use_cnn --channels_per_partition 8 --chips_x 4 --chips_y 4 --rows 16 --cols 16` |
| Real AlexNet at paper scale | `python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet --channels_per_partition 128 --chips_x 4 --chips_y 4 --rows 16 --cols 16 --epochs 1000 --seed 0 --batch_z 3` |
| Real VGG16 at paper scale | same as above with `--model vgg16` |
| ResNet50 (see caveat, Section 10) | same as above with `--model resnet50` |
| Reproducible single run | add `--seed 0` (or any integer) to any command above |
| Performance-tuned long run | add `--train_every 5 --device cpu` (or `cuda`) |

**Multi-seed averaging** (the paper itself runs 5 seeds and averages — Fig. 20 — rather than trusting
one fixed seed; this loop does the same):
```bash
for s in 0 1 2 3 4; do
    python3 run_multi_chip.py --algo ddpg --use_cnn --model alexnet \
        --channels_per_partition 128 --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
        --epochs 1000 --seed $s --batch_z 3 --train_every 5 \
        | tee "log_seed${s}.txt"
done
```
(Results still need aggregating by hand/script afterward — no automated averaging yet.)

### Key CLI flags at a glance

| Flag | Meaning |
|---|---|
| `--use_cnn` | Use a real DNN workload instead of a random synthetic task graph |
| `--model {simple,alexnet,vgg16,resnet50}` | Which network to extract (`simple` = small demo CNN) |
| `--channels_per_partition N` | Partition granularity — smaller = more, finer logic cores (0 = legacy one-task-per-layer) |
| `--chips_x/--chips_y/--rows/--cols` | Multi-chip grid shape; total cores = product of all four |
| `--batch_z N` | Logic cores placed per DDPG action (paper's batched action, Sec 3.2) |
| `--epochs N` | Training episodes |
| `--seed N` | Seed random/numpy/torch for reproducibility |
| `--train_every N` | Run one gradient update every N steps instead of every step (big speed lever) |
| `--device {cpu,cuda}` | Force a device; default auto-detects |
| `--baseline_trials N` | Random-search samples used to compute the reward-normalizing baseline B |

---

# 8. How the Implementation Maps to the Paper

This is the core of what "reproduction" means here — each subsection below is one place where the
starting repo either didn't match the paper at all, or matched it incorrectly, and what was done
about it.

## 8.1 Workload extraction & channel partitioning (Sec 3.1.1, Fig. 5-6)

**What the paper does:** each CONV/FC layer's weights are partitioned along input channels (N groups)
and output channels (M groups), producing a grid of **VMM** (vector-matrix-multiply) logic cores that
each compute a partial sum, feeding into one **VVA** (vector-vector-accumulation) core per output
group that combines them. ReLU/pooling are *not* separate logic cores — they happen inside each
core's "transformation unit" (paper Sec 2.2).

**What this repo does:** `extract_model_task_graph()` in `run_multi_chip.py` implements exactly this.
It hooks every `Conv2d`/`Linear` layer (via `model.modules()`, so it works regardless of container
nesting) and builds the VMM/VVA graph with `--channels_per_partition` controlling group size. Works
with a small demo CNN (`simple`) or real `torchvision` models (`alexnet`, `vgg16`, `resnet50`).

**Caveat — ResNet50:** hooks see Conv2d/Linear *call order*, not the `add()` operation that merges a
skip-connection branch back into the main path. Every layer's shape is captured correctly, but the
Bottleneck block's shortcut edges get approximated as a sequential chain, which is topologically
wrong for those specific edges. Fixing this needs real computational-graph tracing (`torch.fx`), not
yet implemented. A runtime warning fires whenever `--model resnet50` is used.

## 8.2 Objective function (Eq. 4)

**Paper:** minimize `L(P) = max_k T(k|P)` — the *slowest pipeline stage*, not the total summed cost,
because a streaming pipeline's throughput is capped by its bottleneck stage.

**Bug found & fixed:** the original code summed communication cost over *every* task pair — a
completely different (and much easier) objective that doesn't reflect throughput at all.
`pipeline_stages()` now derives topological levels from the task DAG (Kahn's algorithm), and
`pipeline_latency()` returns the true bottleneck. The old sum-based metric is kept only as a secondary
diagnostic (`total_communication_cost()`), never used as the actual objective.

## 8.3 Reward function (Algorithm 1, line 11)

**Paper:** sparse reward — `0` on every non-terminal step, and `r_t = √B − √L(P)` only when a full
placement is completed, where `B` is a fixed random-search baseline computed once up front.

**Bug found & fixed:** the original code gave a dense, per-step reward computed a different way on
every intermediate step, with no baseline normalization at all. Now `run_ddpg()` computes `B` via
`run_random()` before training starts, and the mapper returns `0` for every step except the final one
in an episode.

## 8.4 Collision resolution (Sec 3.2, p.11–12)

**Paper:** floor the actor's continuous output to an intended integer core. If free, place there
directly. If occupied, search *only then* for the **minimum Manhattan distance** to the original
intended position, ties broken by first-found.

**Bug found & fixed:** the original code did a full-grid **squared-Euclidean** nearest-unoccupied-core
search on *every single step*, regardless of whether there was a conflict — different metric, wrong
trigger condition, and unnecessarily slow. Now matches the paper's exact mechanism.

## 8.5 Batched actions (Sec 3.2)

**Paper:** one action places `z` logic cores at once — `[x1,y1,...,xz,yz]`.

**Bug found & fixed:** the original code placed one core per action (`action_dim=2`). `--batch_z`
now controls this properly end-to-end (Actor/Critic output dims, mapper step logic, including correct
handling of a final partial batch when `num_tasks` isn't evenly divisible by `z`).

## 8.6 State representation (Sec 3.2, "Representation of Core Placements")

**Paper:** the state matrix encodes occupied cores by the *index of their assigned logic core* — the
agent should be able to see **which** task is where.

**Bug found & fixed, and this one mattered a lot in practice:** the original state only recorded a
binary occupied/free flag, and separately fed the current task's *outgoing* communication volume
(useful only once a successor is placed — which it isn't yet). This meant the agent had no way to see
where an already-placed *predecessor* task ended up, which is exactly the information needed to
minimize communication cost against it. Fixing this (task-indexed occupancy + both incoming and
outgoing volume) produced a measurable, real improvement in training convergence — before the fix,
training plateaued almost immediately and never moved again for hundreds of episodes; after, the
policy visibly converged to a tighter, better-performing placement.

## 8.7 Exploration noise (Sec 3.2)

**Paper:** Ornstein-Uhlenbeck process with a fading factor.

**Current implementation:** plain Gaussian noise. What *was* fixed is the decay schedule — it's now
scaled to reach ~0.01 by 80% of training regardless of episode count (the original fixed `0.995`
decay barely moved over a short run, making it impossible to tell whether a policy had actually
converged versus still being noise-dominated). Replacing Gaussian with real OU noise is still open.

## 8.8 Baselines & fair comparison (Sec 4.2)

Random Search and Simulated Annealing are implemented and, critically, **all three algorithms
(DDPG/RS/SA) evaluate placements through the exact same `env.evaluate()` → `pipeline_latency()`
call** — there's no separate cost function per algorithm, so any measured difference is attributable
to the placement strategy, not to inconsistent scoring.

## 8.9 Multi-chip topology & hardware config (Table 1)

The grid is fully configurable (`--chips_x/--chips_y/--rows/--cols`) and has been run at the paper's
actual system size: **4×4 chips × 16×16 cores/chip = 4,096 total cores** (this is 16 chips of 256
cores each — matches Table 1 exactly; an earlier internal note mistakenly called this a discrepancy
due to an arithmetic slip, which has been corrected). The hierarchical latency model uses Manhattan
routing with configurable on-chip (α) / off-chip (β) latencies, matching the paper's ratio by default
(α=1.0, β=5.0), though not yet derived from the paper's specific measured bandwidth figures (100 GB/s
off-chip, 64 GB/s per-core NoC) — see Section 10.

---

# 9. Performance Engineering

None of this changes any math — it's entirely about making the correct algorithm run in practical
time. This mattered a lot in practice: an early full-scale run was on pace to take ~30+ hours before
these fixes.

* **Sparse pipeline evaluation**: `pipeline_latency()` originally re-scanned the *entire* task
  matrix for every task on every call — for a partitioned workload that's mostly sparse (a few real
  edges per task among thousands of possible pairs), this was needless O(n²) work. Switched to a
  cached sparse adjacency list. **Measured 19.4x speedup** on a 906-task workload.
* **Reduced training frequency** (`--train_every`): the DDPG agent originally ran a full gradient
  update on *every* environment step — for a large batched workload that's hundreds of expensive
  updates per episode. `--train_every N` spaces these out (experience is still recorded every step,
  only the update frequency changes).
* **GPU auto-detection + CPU thread fix**: the agent never checked for a GPU at all, even when one
  was available. Now auto-detects CUDA; on CPU-only machines, explicitly sets PyTorch to use all
  available cores (some environments silently default to a single thread).
* **Live ETA reporting**: every progress line now reports measured seconds/episode and a live
  estimated time to completion, computed from actual elapsed time — no more guessing whether a run
  will take 10 minutes or 30 hours.
* **Pre-flight validation**: if a workload's logic-core count exceeds the configured grid's total
  core count, the program now fails immediately with a clear fix (raise the grid size or
  `channels_per_partition`) instead of crashing deep inside placement code.

**Known remaining bottleneck:** `_occ_map()` (state construction) still contains an unvectorized
Python loop over `num_tasks` on every step. At AlexNet's 1,445-task scale this is a likely contributor
to per-step cost — not yet fixed.

**Observed but not yet root-caused:** on long CPU-only runs (4+ hours), per-episode time has been
observed to climb steadily over the course of a run (e.g. 9.4s/ep → 18.7s/ep across ~870 episodes on
a single AlexNet run). The replay buffer and training loop were checked and are correctly bounded
(no unbounded list growth found in code), so the leading hypothesis is CPU thermal throttling under
sustained 20-thread load rather than a software bug — worth confirming per-machine via
`watch -n1 "cat /proc/cpuinfo | grep MHz"` during a run. Doesn't affect result correctness, only
wall-clock time; flag it if the trend fails to plateau, since sustained thermal throttling should
level off rather than climb indefinitely.

---

# 10. Known Limitations / Open Gaps

| Gap | Status |
|---|---|
| ResNet50 skip connections | Approximated as sequential; needs `torch.fx` graph tracing for a true fix |
| Exploration noise | Gaussian, not the paper's Ornstein-Uhlenbeck process |
| Learning rates / γ | `1e-4 / 1e-3 / 0.99` vs. paper's `0.0002 / 0.001 / 0.98` — close, not identical |
| On/off-chip latency values (α/β) | Using the paper's default *ratio*, not values derived from their measured bandwidth figures |
| `_occ_map()` performance | Unvectorized Python loop, likely bottleneck at real-workload scale |
| Per-episode time drift on long CPU runs | Observed, not yet root-caused (likely thermal, see Section 9) |
| Multi-seed averaging | `--seed` works per-run; no automated aggregation across seeds yet |
| Actor/Critic architecture | MLP, not the paper's CNN (relevant once workloads have real spatial/graph structure) |
| GA baseline, other topologies (2D torus/HNoC/dragonfly), RNN workload | Not implemented |
| Exact reported percentages (50.5% / 38.4% / 18.6%) | Not yet matched — see Section 11 |
| Current Cost doesn't tighten near Best Cost | Seen on every run so far, even after hundreds of episodes — the policy finds one good placement but doesn't reliably reproduce/improve on it; likely tied to the MLP-capacity and exploration-mechanism gaps above |

---

# 11. Experimental Results So Far

| Run | Workload | Cores | Episodes | Baseline B | Best DDPG cost | Result |
|---|---|---|---|---|---|---|
| ✅ Completed | SimpleCNN, 906 logic cores (`channels_per_partition=8`) | 4,096 | 10,000 (~8.7 hrs) | 444,416 | 340,992 | **~23.3%** better than random-search baseline |
| ✅ Completed | AlexNet, 1,445 logic cores (`channels_per_partition=128`) | 4,096 | ~870+/1,000 | 12,697,984 | 6,749,184 | **~46.8%** better than random-search baseline |
| ⏳ Extraction validated | VGG16, 1,616 logic cores (`channels_per_partition=128`) | 4,096 | — | — | — | not yet trained |
| ❌ Not attempted | ResNet50 | — | — | — | — | blocked on skip-connection caveat |

**Read this carefully:** these results are evidence the RL pipeline is mechanically correct and
learns effectively — they are **not** a reproduction of any specific paper percentage. The
exploration mechanism, exact hyperparameters, and exact latency values all still differ from the
paper (Section 10), and in every run so far `Current Cost` doesn't reliably tighten toward
`Best Cost`, meaning the policy hasn't fully converged even when it found a good result. Paper
reference targets, for comparison:

```text
DDPG vs. sequential baseline : 50.5% latency reduction
DDPG vs. random search       : 38.4% latency reduction   (reported as improvement over RS)
DDPG vs. simulated annealing : 18.6% latency reduction
```

The AlexNet result (46.8%) is now close to the paper's headline number, but should not yet be read as
"reproduced" — it wasn't achieved under matching hyperparameters, exploration mechanism, or exact
latency values, and hasn't been confirmed to be stable across multiple seeds.

---

# 12. Roadmap

**To finish Stage 1 (faithful reproduction):**
1. Root-cause and fix the per-episode time drift on long CPU runs (Section 9).
2. Fix the `_occ_map()` performance bottleneck.
3. Complete AlexNet, then VGG16 training runs to their reported convergence point (paper uses
   ~300K–400K total placements for its full-scale experiments — plan training budgets accordingly).
4. Implement real Ornstein-Uhlenbeck exploration noise.
5. Align learning rates/γ exactly with Table 1.
6. Fix ResNet50 extraction via `torch.fx` graph tracing.
7. Derive α/β latency values from the paper's actual bandwidth numbers instead of the default ratio.
8. Automate the 5-seed averaging methodology (Fig. 20) instead of running seeds one at a time by hand.

**Stage 2 (only after Stage 1 is solid) — genuine improvements, in the paper's own suggested
directions:**
* **Graph-based RL**: embed the full task DAG with a GCN instead of a flattened MLP state. (Note:
  the repo's single-chip PPO path already contains a `GraphConv`/GCN implementation in `agent/model.py`
  — that's unrelated to this multi-chip DDPG work, which still uses a plain MLP; reusing/adapting that
  code is a reasonable starting point.)
* **Attention-based placement**: let the policy learn which tasks/regions matter most for the current
  decision, rather than a fixed-size flattened representation.
* **Topology-generalized placement**: extend beyond the 2D mesh to 2D torus, HNoC, dragonfly, etc.
* **Modern architectures**: evaluate on networks with real branching structure (ResNet, Transformers)
  once extraction correctly captures that structure — this is naturally linked to the GCN direction,
  since a chain-structured workload gives a GCN little advantage over an MLP.

---

# 13. Acknowledgement

Built on top of the open-source *Core Placement with Reinforcement Learning* repository, extended
toward a faithful reproduction of the methodology in Wu, Deng, Li, and Xie, "Core Placement
Optimization for Multi-Chip Many-Core Neural Network Systems with Reinforcement Learning," ACM
Transactions on Design Automation of Electronic Systems, 2020. This repository should be treated as
an ongoing reproduction-and-extension effort, not a finished or independently verified reproduction
of the original paper's results.
