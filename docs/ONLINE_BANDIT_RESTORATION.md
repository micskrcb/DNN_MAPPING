# Restore the demonstrated online bandit before replacing it

The earlier ~23% result is VGG16-FC held-out seeds 3–5: 22.405% mean
paired improvement over random search, 1.637% over focused-uniform,
3.793% over ASA, and 0.122% over plain fixed SA. It is not a 23% causal
learning contribution. Across six seeds the learning control was beaten 6/6.
These results remain valid under their documented reconstructed evaluator.

## Why the subsequent experiments changed

The earlier control did not establish superiority to a cheap proxy heuristic.
Subsequent work tested stronger controls, CONV workloads, VVA=1 rather than 16,
new features/exploration, and eventually offline frozen ranking. Those are
separate methods/settings; their failures do not erase the online result.

A concrete compatibility issue: bacc334 stored signed proposed reward in bandit
replay despite the command flag positive_delta (which controlled PPO). Current
code honors the flag for bandit too. The historical gate now explicitly uses
proposed_delta for bandit and retains positive_delta for PPO. No global reward
behavior is reverted. Do not silently reuse an old command under new semantics.

## Protocol

Run `bash scripts/run_kaggle_online_restoration.sh runs/online-restoration`.
Default: VGG16-FC, historical seed 3, 100000 proposals (100001 objective calls),
execution VVA16/partition VVA1, old logic ordering, fixed SA. Six arms:

1. Exact archived source at bacc334, original settings and online updates.
2. Current online bandit, explicit signed target, legacy9/clipped state.
3. Same current online learner with only relative_cost state changed.
4. Matched focused-uniform control.
5. Matched proxy-scoring control using the same exploration schedule.
6. Plain fixed SA, a practical baseline with different proposal mechanics.

No offline dataset/pretraining, no frozen learned checkpoint, no oracle labels.
Original online learner: 16 candidates, focus .5, replay 50000, warmup512,
4 gradient steps every256 observations, lr .0003, weight decay .0001,
epsilon_all decaying to .5 over full budget. Historical defaults retained.
All settings are explicit in the new runner. Reports/checkpoints from earlier
runs are not overwritten. Historical source is archived with the output.

Seed 3 archived costs: learned4.45568, uniform4.51376, fixedSA4.4464,
random5.69456 microseconds. Modern library versions/hardware can affect
reproduction. A local 1024-proposal toy replay matched old/current best cost,
update count and epsilon, but not all training diagnostics; exact parity is
not asserted. The archived source arm prevents mislabeling current code as exact.

This is a developmental restoration, not a fresh confirmation. Keep historical
and current reference arms even if the state-encoding variant wins. Only extend
a variant after final costs beat the original and strong controls; do not
replace a working method on candidate-ranking metrics alone.

Estimated CPU duration 5–7 hours from historical ~59-minute learned runs;
not measured for this script. For a quick plumbing check set BUDGET_OVERRIDE=1024;
that cannot establish learning quality. SEEDS_OVERRIDE="3 4 5" expands coverage
but roughly triples runtime. All runs use CPU to avoid GPU quota dependency.

## Local checks

The full 55-test suite passed with CPU PyTorch. Archived/current toy runs used
1024 proposals and both performed two online updates with matching epsilon and
best cost; training diagnostics differed, so the exact historical arm is retained.
Additional small-graph checks exercise current original/relative/uniform/proxy
modes, enforcing budgets and online-versus-disabled updates. Historical CLI
startup was checked on the actual workload, but full local workload replays
were stopped to conserve compute. No full-run reproduction result is claimed.
