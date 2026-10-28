Latest: see [NEUTRAL_WEIGHT.md](NEUTRAL_WEIGHT.md). Run `bash scripts/run_kaggle_neutral_weight.sh runs/kaggle-neutral-weight`. Stage24 rejected; pool18/16 retained. The new loss weighting is experimental; older protocols below are historical.

## Current workflow: three seeds per algorithm experiment

Read [IMPROVEMENT_PRIORITY.md](IMPROVEMENT_PRIORITY.md). Edit `src/run_multi_chip.py` for the local target and outcome diagnostics; `src/analyze_priority_seeds.py` for aggregate/progress reports. Tests: `src/test_improvement_priority.py`. Run from the repository root: `bash scripts/run_kaggle_priority_three_seed.sh runs/priority-three-seed`. Seeds25–27, three methods, estimated9–12h CPU. Completed seeds can be skipped on recovery; interrupted seed training restarts. Historical experiment entries follow.

## Latest experiment: communication-aware relocation (25 October)

Read [ATTRACTION_BANDIT.md](ATTRACTION_BANDIT.md). The new helper is in `src/structured_moves.py`; pool/learning/SA integration is in `src/run_multi_chip.py`; tests are in `src/test_attraction_moves.py`. Run from repository root: `bash scripts/run_kaggle_attraction_bandit.sh runs/attraction-bandit-UNIQUE`. CPU, estimated4–6h. This supersedes the structured reordering experiment as the next run, not as a proven better mapper.

## Latest algorithm experiment (24 October)

For the structured online RL+SA change, read [STRUCTURED_BANDIT.md](STRUCTURED_BANDIT.md). The main files are `src/structured_moves.py` (new moves), `src/run_multi_chip.py` (learning and SA integration), `src/analyze_online_rapid.py` (comparisons) and `src/test_structured_moves.py` (correctness). Run from the repository root: `bash scripts/run_kaggle_structured_bandit.sh runs/structured-bandit-UNIQUE`. This uses CPU and takes an estimated4–6h. Previous experiments below are historical context; do not assume they are the active default.

# Important: Read This Before Working on the Project

The project has changed a lot from the older DDPG+ASA CPU version. The current
main research method is **contextual-bandit-guided fixed simulated annealing**,
referred to as RL+SA in the project notes.

Use the `cleaned-up` branch as the base. Do not push directly to `cleaned-up`. Make a separate
branch for every task so the current tested implementation stays reproducible.

## Get the current code

```bash
git clone https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
git fetch origin
git switch cleaned-up
git pull --ff-only origin cleaned-up
git switch -c work-YOURNAME-TASK
```

Examples of task branches:

```bash
git switch -c work-statistics
git switch -c work-method-audit
git switch -c work-paper-theory
```

Do not merge old DDPG+ASA changes into the current method unless the exact
change has been reviewed first. Do not change the frozen RL+SA hyperparameters
while the cross-workload experiments are running.

## Set up the CPU environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The root `requirements.txt` contains historical version pins and is not the
environment specification for the current branch.

## What to read

Do not try to read every project document before starting. First read this
file, then read:

- The part of `README.md` related to your task.
- `PROJECT_STATE.md` for the current results and the present state of the code.

Use the other documents only when they are relevant to your task:

- Read `NEXT_STEPS.md` when selecting or planning the next experiment or task.
- Read `RESEARCH_FINDINGS.md` for literature review, result interpretation, or
  paper writing.
- Read `RECONCILIATION.md` when comparing the old implementation with the
  current reconciled code.
- Read `PPO_REFERENCE_ADAPTER.md` only when working on PPO or learned-SA
  internals.
- Read `THIRD_PARTY_NOTICES.md` before using or modifying external code.

## Files to understand before changing RL+SA

Most RL+SA behavior is intentionally kept in a small set of files:

- `src/run_multi_chip.py` contains the implementation. Start with
  `run_ppo_asa`, despite the historical function name. Within it,
  `candidate_features` defines the nine inputs including `proxy_gain`,
  `make_proposal_pool` creates the 16 legal candidates, `update_bandit` trains
  the signed-gain regressor, and the epsilon-greedy block selects a move. This
  file also owns SA temperature, Metropolis acceptance, checkpoint metadata,
  reports, and command-line arguments.
- `scripts/run_kaggle_rl_fixed_sa_gate.sh` is the canonical experiment runner.
  It freezes the workload and algorithm flags, launches learned and uniform
  control runs with matched seeds and objective budgets, runs fixed SA, ASA,
  and random-search context baselines, and creates the result ZIP.
- `src/test_reconciliation.py` contains the invariants that must continue to
  pass. Relevant tests cover learned/control separation, legal candidates,
  exact incremental-versus-full objective agreement, and fixed-SA agreement.
- `PROJECT_STATE.md` records completed results and their limitations.
  `NEXT_STEPS.md` records the active experiment order, and
  `RESEARCH_FINDINGS.md` explains why each algorithmic change was made.

For the next RL improvement, inspect the `proxy_gain` feature and epsilon-
greedy selection in `src/run_multi_chip.py`. Add the proxy-greedy comparator as
a new explicit control mode; do not silently change the frozen learner.

## Run tests before pushing

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/validate_device.py \
  --device cpu \
  --output runs/cpu-validation.json
git diff --check
```

Run a small local CPU smoke test after changing the RL code:

```bash
python src/run_multi_chip.py \
  --algo ppo_asa --ppo_asa_learner bandit \
  --ppo_asa_annealing fixed --ppo_asa_candidates 16 \
  --ppo_asa_focus_fraction 0.5 \
  --ppo_asa_update_style reference_separate \
  --ppo_asa_reward_mode positive_delta \
  --ppo_asa_rollout_steps 256 --ppo_update_epochs 4 \
  --ppo_minibatch_size 256 --ppo_hidden_dim 128 \
  --ppo_learning_rate 3e-4 --ppo_asa_weight_decay 1e-4 \
  --ppo_gamma 0 --ppo_gae_lambda 0 --ppo_entropy_coef 0.001 \
  --ppo_asa_bandit_epsilon_final 0.5 \
  --ppo_asa_bandit_epsilon_decay 1.0 \
  --use_cnn --model alexnet \
  --partition_mode paper_targets --workload_region fc \
  --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 \
  --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1 \
  --device cpu --seed 0 --iters 1000 \
  --asa_diagnostics runs/local-bandit-smoke.jsonl \
  --report runs/local-bandit-smoke-report.json
```

Run the canonical short Kaggle gate from the repository root:

```bash
MODEL_OVERRIDE="vgg16" \
WORKLOAD_REGION_OVERRIDE="fc" \
SEEDS_OVERRIDE="0" \
bash scripts/run_kaggle_rl_fixed_sa_gate.sh \
  bandit-short runs/kaggle-bandit-vgg16-fc-short
```

Run the 100,000-evaluation held-out VGG16-FC gate with random, fixed-SA, and
ASA baselines:

```bash
MODEL_OVERRIDE="vgg16" \
WORKLOAD_REGION_OVERRIDE="fc" \
SEEDS_OVERRIDE="3 4 5" \
bash scripts/run_kaggle_rl_fixed_sa_gate.sh \
  bandit-extended runs/kaggle-bandit-vgg16-fc-heldout
```

Run the three-seed, 20,000-evaluation learned-versus-proxy diagnostic without
repeating the context baselines:

```bash
MODEL_OVERRIDE="vgg16" \
WORKLOAD_REGION_OVERRIDE="fc" \
SEEDS_OVERRIDE="6 7 8" \
SKIP_CONTEXT_BASELINES="1" \
bash scripts/run_kaggle_rl_fixed_sa_gate.sh \
  bandit-proxy-short runs/kaggle-bandit-vgg16-fc-proxy-short
```

When the work is ready:

```bash
git status
git add <only-the-files-you-changed>
git commit -m "Explain the specific change"
git push -u origin work-YOURNAME-TASK
```

Share the branch name, commit hash, changes made, reason for the changes, test
results, and any limitation found. Do not commit `.venv`, model checkpoints,
or generated experiment folders. Only commit a result ZIP when it is selected
as compact research evidence and its SHA-256 is recorded.

## What the current method does

A DNN is split into logical cores. The placement problem assigns every logical
core to a unique physical core across multiple chips. The objective is the
reconstructed pipeline latency, so a lower cost is better.

Fixed simulated annealing creates legal neighboring placements. If the new
placement is better, it is accepted. A worse placement may still be accepted
with probability:

```text
exp(-cost_increase / temperature)
```

This lets SA escape local minima. The temperature follows the fixed geometric
cooling schedule.

The contextual bandit receives the current search context and 16 legal
candidate moves. It predicts the signed immediate cost improvement of each
candidate and chooses one using epsilon-greedy exploration. Only that one move
uses the true objective evaluation. SA then decides whether to accept it.

So the responsibilities are:

- RL chooses which legal move SA should try.
- SA accepts or rejects that move and controls cooling.
- The frozen control selects uniformly from the same candidate pool.
- Learned versus frozen is the main test of whether learning helps.

This is a contextual bandit rather than DDPG or PPO because it learns immediate
proposal quality and does not estimate a long-term value function.

## Current evidence

For held-out AlexNet-FC seeds 3--5, RL+SA beat its frozen control in all three
seeds by 1.965% on average. It beat ASA by 9.288% on average. Its mean advantage
over fixed SA was 0.669%, but that comparison is not yet statistically settled.

In the first frozen AlexNet-CONV short test, RL+SA reached 37.02117
microseconds, compared with 37.44656 for the frozen control and 37.87195 for
fixed SA. In the later three-seed 100,000-proposal test, learned RL+SA, frozen
control, and fixed SA all reached the same 37.02117-microsecond minimum. CONV
therefore shows a short-budget signal but a final-cost tie.

Do not describe the method as proven across all workloads yet. The supported
claim is that the frozen learned proposal policy showed a positive held-out
benefit on AlexNet-FC. AlexNet-CONV currently shows no final-cost advantage.

The frozen VGG16-FC short test was positive: learned RL+SA beat its frozen
control by 1.834%, fixed SA by 1.490%, and ASA by 2.739% at seed zero. It made
77 finite updates and justified the completed extended gate below.

The extended VGG16-FC test passed its exploratory gate: RL+SA beat its frozen
control in all three seeds by 2.255% on average, with paired 95% interval
[1.107%, 3.403%]. It beat
fixed SA in all three by 0.908% on average, but that interval crosses zero,
and beat ASA in all three by 5.418%. A methodology audit identified a useful
next check: compare against non-learned proxy-greedy selection under the same
pool and exploration schedule. This does not erase the positive result or
require every suggested ablation before work continues.

The later VGG16-FC seeds 3--5 confirmed the direction: RL+SA beat focused-
uniform control 3/3 by 1.637% on average, ASA by 3.793%, and random by 22.405%,
while remaining effectively tied with fixed SA. Across all six VGG seeds it
beat control 6/6 by 1.946%. The next short command uses `bandit-proxy-short`
to compare learning with direct proxy-greedy selection without changing the
trained method.

## Main theory references

1. *Core Placement Optimization for Multi-chip Many-core Neural Network
   Systems with Reinforcement Learning* defines the multi-chip placement
   problem and the original RL setting.
2. *Placement in Integrated Circuits using Cyclic Reinforcement Learning and
   Simulated Annealing* combines learned placement with SA refinement.
3. Qiu and Liang's *Reinforcement Learning Based Simulated Annealing* keeps
   Metropolis acceptance and cooling fixed while learning proposal behavior.

Properly cite external papers and repositories. Before using external code,
check its license and record the source and exact commit in
`THIRD_PARTY_NOTICES.md`.

## Useful work that does not change the frozen method

- Audit the learned-versus-frozen objective accounting and seeding.
- Add paired statistical analysis and publication-quality plots.
- Check manifests, commit hashes, budgets, workload settings, and final epsilon.
- Improve documentation and reproducibility commands.
- Review related work and write theory with proper citations.
- Add tests for real invariants or confirmed bugs.

Do not tune the method using completed seeds 0--5. New method ideas must be
developed separately from the frozen validation results.
Current development result: two completed 200k VGG16-FC seeds show learned RL+SA beats the matched frozen policy by 1.77% mean and fixed SA by 2.70% mean. The next controlled screen is `TRAIN32_HORIZON.md`; run `bash scripts/run_kaggle_train32_horizon.sh runs/kaggle-train32-horizon`.
