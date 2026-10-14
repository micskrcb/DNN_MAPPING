# Important: Read This Before Working on the Project

The project has changed a lot from the older DDPG+ASA CPU version. The current
main research method is **contextual-bandit-guided fixed simulated annealing**,
referred to as RL+SA in the project notes.

Use the `cpu` branch as the base. Do not push directly to `cpu`. Make a separate
branch for every task so the current tested implementation stays reproducible.

## Get the current code

```bash
git clone https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
git fetch origin
git switch cpu
git pull --ff-only origin cpu
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

## Run tests before pushing

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p 'test_reconciliation.py' -v
python src/validate_device.py \
  --device cpu \
  --output runs/cpu-validation.json
git diff --check
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
large result ZIPs, or generated experiment folders.

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

The frozen VGG16-FC short test is also positive: learned RL+SA beat its frozen
control by 1.834%, fixed SA by 1.490%, and ASA by 2.739% at seed zero. It made
77 finite updates. Treat this as a passed screening test, not a final result;
the unchanged three-seed 100,000-evaluation VGG16-FC test is still required.

The extended VGG16-FC test passed its exploratory gate: RL+SA beat its frozen control in all three
seeds by 2.255% on average, with paired 95% interval [1.107%, 3.403%]. It beat
fixed SA in all three by 0.908% on average, but that interval crosses zero,
and beat ASA in all three by 5.418%. A methodology audit identified a useful
next check: compare against non-learned proxy-greedy selection under the same
pool and exploration schedule. This does not erase the positive result or
require every suggested ablation before work continues.

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
