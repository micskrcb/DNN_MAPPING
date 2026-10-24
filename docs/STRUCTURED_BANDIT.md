# Structured communication-group proposals for online RL+SA

## Why this experiment

The latest received result is seed21 learning intensity, not seed22 confirmation:16-step training beat the previous learner on both FC workloads, but its advantage over fixedSA was3.11114% on VGG16 and0.07294% on AlexNet. The user requested faster algorithm development instead of additional confirmation loops. The seed22 runner remains available; this development run does not depend on it. We have not established5% over a publishedDDPG method. An earlier4.55% win over the local reconstruction retained a random-baseline DDPG solution and is not that claim.

## What changes

`src/structured_moves.py` precomputes communication-connected groups by bounded breadth-first traversal, visiting higher-volume edges first and using task ID to break ties. Each group contains at most8 tasks. Treating edges as undirected here groups communicating tasks; it does not change the directed workload used by the evaluator.

A structured candidate takes one group, sorts its currently occupied cores by one of two chip-major row/column orders, optionally reverses that order, and assigns those slots to the group's traversal order. An identity permutation is rotated once to avoid wasting a candidate. No new core is allocated; the occupied core set stays unchanged. Existing random moves can still relocate tasks into free cores. Groups are static, but the physical slots adapt to the current placement.

`--ppo_asa_pool_structured_fraction .5` replaces8 of16 candidates with these moves. Remaining candidates use the existing random neighborhood. Candidate scoring uses the unchanged legacy9 features and signed proposed_delta target; the16-step bandit chooses among them. One selected candidate gets an exact evaluation and ordinary fixedSA Metropolis acceptance. Calibration remains unchanged. No oracle labels or extra objective queries are added. Empty communication graphs fall back to the original neighborhood and report zero effective structured candidates. Simultaneous small-move and structured mixtures are rejected to keep this experiment interpretable.

This is an original, limited adaptation of the structured-search idea in Wang etal, *A Dimensional Order Search Based Mapping Algorithm for Deploying DNNs onto Many-Core Neuromorphic IC*, DOI https://doi.org/10.1109/TCAD.2025.3640508. It is **not DOS reproduction**: groups do not encode H/W/Cin/Cout tensor coordinates, and no TianjicX time-sharing, optimality or deadlock guarantees are implemented. See DOS_PAPER_ASSESSMENT.md. No external source code was copied.

## Full-budget development run

```bash
bash scripts/run_kaggle_structured_bandit.sh runs/structured-bandit-UNIQUE
```

VGG16-FC, seed23,100000 proposals plus initialization per arm. Four arms:

| Arm | Learning | Structured share |
|---|---|---:|
| train_more |16 minibatch steps/update|0%|
| structured |16 minibatch steps/update|50%|
| structured_uniform |Disabled; uniform candidate selection|50%|
| fixed_sa |None; original SA|None|

All share the objective, topology, partitioning, initial placement and evaluation budget. Learned arms retain epsilon endpoint.02, replay50000, batch256, rollout256 and existing learning rate. Default4-step settings elsewhere are not silently changed. The structured uniform arm is the same pool-generation rule, not an identical sequence of visited states once trajectories diverge.

Compare structured against all three controls. Improvement over train_more tests the new action space in the learned method; improvement over structured_uniform checks the value of learned selection; fixedSA remains the practical baseline. Selection frequency is recorded. Report mixed outcomes and runtime overhead explicitly. This one-seed development experiment does not establish significance or a5% published-method win. No automatic promotion threshold is relaxed.

Plan approximately4–6h on Kaggle CPU, estimated from the prior VGG16 timings; the new proposal overhead is not yet measured at full scale. Use Internet on and Save Version → Save & Run All. Each completed arm is archived. The ZIP contains validated summary, reports, diagnostics, tests and provenance. No Kaggle run is automatically submitted by this repository script from the local computer.

## Files to understand/change

- `src/structured_moves.py`: group construction and legal spatial permutations.
- `src/run_multi_chip.py`: candidate-pool integration, exact evaluation, learning/SA interaction, metadata and checkpoint compatibility.
- `src/analyze_online_rapid.py`: budgets, controls, gains and structured selection counts.
- `scripts/run_kaggle_structured_bandit.sh`: pinned experimental settings.
- `src/test_structured_moves.py`: legality, reproducibility, exact cost checks, default preservation, accounting and checkpoints.

## Local validation

Standard and focused tests passed. A synthetic64-task,601-evaluation four-arm CLI run completed with valid summaries and ZIP. This checks orchestration and accounting, not the full DNN result. Evidence: `artifacts/structured-bandit-local-validation-20261024/`.
