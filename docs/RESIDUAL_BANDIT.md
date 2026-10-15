# Bounded proxy correction inside fixed SA

## Why this experiment

The `a00784a` VGG16-CONV gate gave mean costs of 728.898 us for learned
proposals, 558.921 us for proxy-greedy, and 609.994 us for fixed SA. The
learner lost both comparisons in all three seeds. Weight changes and
nonuniform scores alone do not prove useful learning; the paired optimizer
outcome remains the test. The previous gate failed at 20,000 proposals. That
does not establish that additional training could never help.

## Small, explicit method change

`--ppo_asa_bandit_scoring proxy_residual` replaces direct scoring with

```
score(state, candidate) = proxy_gain(candidate) + 0.25 * tanh(network_output)
target = tanh(selected_training_reward)
loss = smooth_L1(score, target)
```

The last network layer starts at zero, so initial scores equal proxy-greedy
exactly. The score correction is bounded by `--ppo_asa_bandit_residual_limit`
(default 0.25). This bounds score displacement, not the resulting placement
cost: it cannot guarantee that the learned search beats proxy-greedy.
The target transform preserves sign and ordering and puts rewards on a
bounded scale. It is a declared heuristic normalization, not a physical
equivalence between proxy byte-hop reduction and true latency reduction.

`--ppo_asa_state_encoding relative_cost` replaces two early-saturating inputs
with current/initial and best/initial cost ratios. The historical inputs
clipped at a 10% cost reduction. Eight state dimensions are retained; loading
a checkpoint with a different encoding or scoring mode raises an error.

The bandit now trains on the selected reward mode, matching its reported
training reward. Previously it always stored signed proposed gain even when
another mode was requested. Historical proposed-delta runs are unaffected.
Old direct scoring and clipped state remain the defaults. New residual
checkpoints are inference exports, not exact optimizer resumes.

Shared bottleneck stage/task lookups are cached once per candidate pool.
Nonlearning uniform controls skip the unused neural forward pass. SA/ASA
top-level report counts now include initialization, agreeing with their
existing algorithm metadata; the search itself has not changed.

No new external source code was copied. This experiment extends the existing
local contextual bandit; earlier references remain in THIRD_PARTY_NOTICES.

## Frozen short protocol

- VGG16-CONV, fresh seeds 3, 4, 5; 20,001 optimization objective calls per
  method/seed. Calibration proposals are included. Each run additionally
  performs the same 256 nonselectable objective-sensitivity samples.
- Learned residual + fixed SA, matched proxy-greedy + fixed SA, plain fixed
  SA, and random search. Sixteen candidates, contention12 features, 50%
  focus, 5% stage slack, and the previous top-four exploration schedule stay
  fixed. Replay, neighborhoods, acceptance and cooling are unchanged.
- The directional gate requires a positive mean paired improvement and wins
  in at least two of three seeds against **both** proxy-greedy and fixed SA.
  Random search provides context. Report all effects; three seeds do not
  establish statistical significance or broad generalization.
- No automatic extension. Even a positive short result requires independent
  validation. This compares the combined residual/state/target package; it
  does not identify each component's separate contribution.

## Run

Use the existing environment with NumPy, PyTorch and torchvision. Kaggle's
GPU environment already supplies these dependencies; do not install the
legacy pinned requirements.txt for this experiment.

```bash
git clone --branch cleaned-up https://github.com/micskrcb/DNN_MAPPING.git
cd DNN_MAPPING
PYTHONPATH=src python -m unittest discover -s src -p 'test_*.py'
bash scripts/run_kaggle_residual_bandit_gate.sh short runs/residual-first
```

For an existing checkout, first use `git fetch origin cleaned-up` and check
out the experiment's specified commit. Record that commit when sharing a ZIP.
The runner requires a fresh output path, runs reconciliation tests first,
runs learned/proxy concurrently on two GPUs (sequentially on one), and
archives logs, reports, provenance, checkpoint and summary after stages and
on exit. The result is `runs/residual-first.zip`. A hard platform termination
cannot guarantee execution of the exit trap; completed-stage ZIPs remain.

Small local CLI check (not learning evidence):

```bash
OMP_NUM_THREADS=2 DEVICE_OVERRIDE=cpu BUDGET_OVERRIDE=64 SEEDS_OVERRIDE=3 \
MODEL_OVERRIDE=alexnet WORKLOAD_REGION_OVERRIDE=fc \
bash scripts/run_kaggle_residual_bandit_gate.sh short runs/residual-smoke
```

The default Kaggle protocol is estimated at roughly 3--5 hours on two T4s,
based on the earlier ~40--47-minute candidate-selector runs per seed plus
baseline times. One GPU runs the pair sequentially and will take longer.
This is an estimate, not a GPU benchmark of the new code.

## Local validation

The CPU suite tests exact initial proxy equivalence, a held-out synthetic
correction signal, bounded scores, selected-reward routing, checkpoint
compatibility, and preservation of progress after a 10% improvement.
Real VGG16-CONV performance is pending the next uploaded archive.

## Selected-only result and pool-supervised follow-up

The seed 3--5 run completed cleanly but failed its directional gate. Learned
residual SA averaged 541.885 us versus 507.021 us for proxy-greedy, a 6.907%
mean loss with only one win. It beat fixed SA in all three seeds by 23.788%.
The model updated, but selected-only feedback did not add reliable value over
the proxy.

The opt-in `--ppo_asa_bandit_pool_supervision_stride N` evaluates every legal
candidate before applying the already-selected move every N proposal steps.
Selection occurs before exact pool costs are available. All candidate calls
are charged to the same optimization budget, and all exact proposed gains are
added to replay. Thus supervision buys better labels by taking fewer SA moves;
it receives no free oracle calls. The next frozen screen uses stride 32 and
fresh seeds 6--8. This is still online contextual-bandit supervision, not PPO.
