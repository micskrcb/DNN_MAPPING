# DNN Mapping with Reinforcement Learning & Heuristics

> Optimizes the placement of partitioned Deep Neural Network (DNN) workloads (AlexNet, VGG16, ResNet50) onto multi-chip many-core hardware accelerators (4×4 chips, 4,096 cores) to minimize bottleneck pipeline stage execution latency.
> Implements and evaluates contextual-bandit-guided simulated annealing, classical simulated annealing (SA), adaptive simulated annealing (ASA), masked categorical PPO, and DDPG based on Wu et al. (ACM TODAES 2020, [DOI 10.1145/3418498](https://doi.org/10.1145/3418498)).

---

## Quickstart: Setup & Run from Scratch

### 1. Prerequisites & Environment Setup
Requires Python 3.8+ on Linux (Ubuntu/Debian recommended).

```bash
# Clone the repository
git clone https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU PyTorch)
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Verify Installation
Run the test suites to ensure hardware modeling, topologies, and routing models are functional:

```bash
# Run all smoke and regression tests (38 tests)
python tests/run_all_tests.py
```

### 3. Run Your First Placement
Run a quick placement optimization using the primary active method (Contextual-Bandit Fixed SA) on the AlexNet workload:

```bash
python src/run_multi_chip.py \
  --algo ppo_asa \
  --ppo_asa_learner bandit \
  --ppo_asa_annealing fixed \
  --ppo_asa_candidates 16 \
  --ppo_asa_focus_fraction 0.5 \
  --model alexnet \
  --workload_region fc \
  --partition_mode paper_targets \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --iters 500 \
  --seed 0 \
  --report runs/quickstart_report.json
```

---

## Documentation Roadmap

- **[`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md)**: **The complete technical manual.** Repository structure, file-by-file purpose, hardware modeling theory, and an exhaustive reference of all CLI flags and operational modes.
- **[`docs/IMP_README.md`](docs/IMP_README.md)**: Essential onboarding rules, branch hygiene, and contextual bandit execution guide.
- **[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)**: Current verified benchmarks, statistical intervals, and hardware reconciliation details.
- **[`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md)**: Full chronological research narrative from paper reproduction through DDPG failure analysis, PPO, and contextual-bandit SA.
- **[`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md)**: Active 66-item reproduction checklist and experiment backlog.
- **[`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md)**: Theoretical insights, VVA throughput sensitivity, and literature review.

---

## Repository Structure Overview

```
DNN_MAPPING/
├── CODEBASE_GUIDE.md              # Technical manual: architecture, file index, CLI flags
├── README.md                      # Entry point: specifications, quickstart, and workflow
├── src/                           # Active multi-chip placement engine & algorithms
│   ├── run_multi_chip.py          # Primary placement engine (CLI, models, evaluators)
│   ├── multi_chip_topology.py     # NoC mesh/torus topology & XY routing
│   ├── multi_chip_environment.py  # Pipeline DAG staging & bottleneck latency evaluator
│   ├── compute_model.py           # VMM/VVA layer partitioning & MAC unit conversions
│   ├── run_multiseed_experiment.py# Multi-seed concurrent runner
│   ├── run_paper_experiment.py    # Paper reproduction suite orchestrator
│   ├── train_oracle_candidate_ranker.py # Supervised candidate ranking trainer
│   ├── validate_device.py         # Hardware profiling & sanity verification
│   ├── analysis/                  # Statistical analysis & gate scripts (analyze_*.py)
│   └── experiments/               # Exploratory ablations & historical comparators
├── tests/                         # Unit and regression test suites (run_all_tests.py)
├── scripts/                       # Automated Kaggle & SLURM experiment launcher scripts
├── docs/                          # Specialized technical guides & research journey
├── legacy/                        # Archived upstream single-chip GCN PPO framework
└── runs/                          # Benchmark logs, study reports, and checkpoints
```

---

## Supported Optimization Methods

| Algorithm | CLI Argument | Status & Description |
| :--- | :--- | :--- |
| **Contextual-Bandit Fixed SA** | `--algo ppo_asa --ppo_asa_learner bandit --ppo_asa_annealing fixed` | **Primary Active Method.** Online MLP ranks legal neighbor candidates with bottleneck focusing inside fixed geometric SA. |
| **Fixed Simulated Annealing** | `--algo sa` | Classical geometric cooling ($\alpha=0.99$) with calibrated initial temperature. Strongest non-learned baseline. |
| **Adaptive Simulated Annealing (ASA)** | `--algo asa` | Dynamic temperature and neighborhood adjustment based on windowed acceptance rates. |
| **Random Search** | `--algo random` | Uniform random permutation placement over valid cores. |
| **Sequential Baseline (BS)** | `--algo bs` | Deterministic placement in chip-major physical core order. |
| **Masked Categorical PPO** | `--algo ppo_masked` | Autoregressive constructive placement with exact invalid-action masking. |
| **Continuous DDPG** | `--algo ddpg` | Paper reproduction path (Wu et al. 2020) using spatial CNN and collision repair. |

---

## Standard Workflows & Examples

### 1. Run Contextual-Bandit Guided SA (Active Benchmark)
Run 5,000 iterations of learned SA on the VGG16 fully-connected region:

```bash
python src/run_multi_chip.py \
  --algo ppo_asa \
  --ppo_asa_learner bandit \
  --ppo_asa_annealing fixed \
  --ppo_asa_candidates 16 \
  --ppo_asa_focus_fraction 0.5 \
  --model vgg16 \
  --workload_region fc \
  --partition_mode paper_targets \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --iters 5000 \
  --seed 0 \
  --report runs/vgg16_bandit_report.json
```

### 2. Run Classical Baselines for Comparison
Compare directly against Fixed Simulated Annealing under the exact same hardware configuration:

```bash
python src/run_multi_chip.py \
  --algo sa \
  --model vgg16 \
  --workload_region fc \
  --partition_mode paper_targets \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --chips_x 4 --chips_y 4 \
  --rows 16 --cols 16 \
  --iters 5000 \
  --seed 0 \
  --report runs/vgg16_sa_report.json
```

### 3. Run Multi-Seed Statistical Experiments
Execute multiple seeds concurrently across available CPU cores:

```bash
python src/run_multiseed_experiment.py \
  --jobs 2 \
  --cpu_threads 4 \
  --seeds 0,1,2,3,4 \
  --algorithms bs,sa,random \
  --model alexnet \
  --workload_region fc \
  --timing_model paper_pipeline \
  --routing_model paper_xy \
  --search_budget 5000 \
  --output_dir runs/alexnet_multiseed
```

---

## Verified Empirical Evidence

Under strictly matched true-objective evaluation budgets:
- **AlexNet-FC (Held-Out Seeds 3–5)**: Contextual-bandit RL+SA outperformed its matched focused-uniform control in 3/3 seeds by **1.965%** on average ($p < 0.05$), beat ASA by **9.288%**, and achieved parity with unrestricted fixed SA (+0.669%).
- **VGG16-FC (All 6 Seeds)**: Contextual-bandit RL+SA outperformed its matched focused-uniform control in 6/6 seeds by **1.946%** on average (exact sign-test $p = 0.03125$), beat ASA by **4.605%**, and matched fixed SA.
- **AlexNet-CONV (Single-Chip)**: Learned RL+SA, frozen control, and fixed SA all reached the identical compute floor ($37.02117\,\mu\text{s}$), demonstrating a valid final-quality tie.

For complete numerical tables, confidence intervals, and hardware reconciliation details, see **[`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)**. For the chronological research and debugging narrative, see **[`docs/PROJECT_JOURNEY.md`](docs/PROJECT_JOURNEY.md)**.
