# Communication-aware relocation for online RL+SA

## Evidence and decision

Received structured-pool seed23: the existing16-step learner4.35504µs remained better than structured4.39952µs, fixedSA4.42480µs and structured uniform4.51040µs. Structured learning beat uniform2.45832%, but lost1.02135% against the working learner. Archive: `artifacts/structured-bandit-seed23-20261025/`. Keep the original16-step mapper as reference. The5% publishedDDPG target remains unproven.

The group reordering could only permute existing occupied slots. A plausible limitation is its inability to bring scattered communicating tasks into closer slots. The new method tests that explanation; it is not a proven diagnosis or guaranteed improvement.

## New proposal

Opt-in `--ppo_asa_pool_attraction_fraction .5` allocates8 of16 candidate slots to communication-aware relocation/swap proposals. The other8 retain the original neighborhood. All other learning and annealing settings remain unchanged.

For each task, precompute its strongest positive-volume communication partner (incoming and outgoing volumes are combined). Sample one such pair and randomly select which endpoint to move. Consider up to8 allowed neighboring cores on the partner's chip, with torus wrapping only on torus hardware. Only slots strictly closer to the partner under the existing distance/latency proxy qualify. Prefer a free slot; otherwise swap with its occupant. Try at most4 pairs; if no qualifying slot exists, use an ordinary proposal. Empty communication graphs also fall back. Neighbor slots are cached and occupancy is indexed once per candidate pool to avoid repeated full-grid scans.

The move improves only a selected pair's distance proxy, not necessarily overall communication or exact cost. Other edges and the displaced task can get worse. The existing learned scorer evaluates candidate features, then exact incremental evaluation and fixedSA Metropolis acceptance decide what happens. No extra objective labels, altered routing, reduced accounting or oracle supervision are introduced. Accepted relocation updates the free-core pool; merely generating or rejecting one never does.

Track generated and selected relocation/swap/fallback counts. A mixture can fail because attraction is rarely feasible or because useful-looking proxy moves damage the full objective. These counters help distinguish those cases. Existing structured reordering remains separately available; mixtures with its flag or the small-move flag are rejected for this experiment. Defaults and old checkpoint behavior remain unchanged at zero attraction fraction; incompatible checkpoint pools are rejected.

Original project code; no third-party implementation copied. This follows our analysis of the DOS-inspired structured experiment, but is not DOS or an implementation of its time-sharing/4D mapping. See DOS_PAPER_ASSESSMENT.md and THIRD_PARTY_NOTICES.md for the underlying paper attribution. No novelty or published-performance claim is made.

## Next full run

```bash
bash scripts/run_kaggle_attraction_bandit.sh runs/attraction-bandit-UNIQUE
```

VGG16-FC seed24,100000 proposals+initialization=100001 evaluations per arm:

|Method|Learning|Attraction slots|
|---|---|---:|
|train_more|Existing16-step bandit|0/16|
|attraction|16-step bandit|8/16|
|attraction_uniform|Disabled; uniform selection|8/16|
|fixed_sa|None|OriginalSA|

Compare the learned candidate against all three controls. Improvement over the same unlearned pool is needed to attribute a gain to learned selection; improvement over train_more is needed to call the new mapper better. Report all outcomes, including ties, runtime and fallback frequency. This is one development seed, not a statistical confirmation or publishedDDPG comparison. No promotion of the unsuccessful seed23 method.

Estimated4–6h on Kaggle CPU from the previous4.32h four-arm run; new proposal overhead is unmeasured at full scale. Enable Internet. Use Save Version → Save & Run All. ZIP snapshots are written after completed arms and on ordinary exit/failure. Local tests do not replace the full DNN run.

## Relevant files

- `src/structured_moves.py`: cached nearby slots and attraction proposal helper.
- `src/run_multi_chip.py`: candidate pool, exact acceptance/free-pool handling, telemetry and checkpoints.
- `src/test_attraction_moves.py`: free relocation, full-grid swaps, masks, torus neighbors, incremental correctness and accounting.
- `scripts/run_kaggle_attraction_bandit.sh`: next-run configuration.
- `src/analyze_online_rapid.py`: matched-run validation and percentages versus existing mapper, uniform control andSA.

## Validation

Standard and focused tests passed, including free-slot relocation and full-grid swap checks. A synthetic64-task four-arm CLI run completed at601 evaluations each with validated summary and ZIP; its full grid exercised swaps, not free relocations. Evidence: `artifacts/attraction-local-validation-20261025/`. Full-scale VGG16 performance remains unmeasured for this variant.
