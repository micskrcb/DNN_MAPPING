# Reconciliation and validation status

This branch reconciles GitHub commit `41f9d02` with the user-supplied Gemini
archives `files (4).zip` and `files (5).zip`. Their three shared files are
byte-identical; archive 4 also contains the topology and environment.
Archive prose is historical evidence, not proof that its implementation is correct.

## Restored and corrected

- Restored FX dependency tracing, custom models, torus topology, vectorized
  occupancy/collision handling, and SA current-state acceptance with fixed 0.99 cooling.
- Corrected the archive's cross-layer all-to-all broadcast: route overlapping
  channel ranges, using consumer tensor shapes after pooling and flattening.
- Partition remainder channels exactly; per-layer VMM operations conserve the
  unsplit layer MAC count. Grouped convolutions are explicitly unsupported.
- Corrected the previous changes' addition of seconds to arbitrary communication
  scores. An untyped `--compute_ops` vector is now rejected.
- Corrected row-major policy coordinates versus chip-major physical core IDs.
- SA can relocate into unused cores and evaluates the requested neighbor budget.
  The relocation choice is a documented interpretation of the paper's neighborhood,
  not an exact reconstruction. SA temperatures remain configurable API assumptions.
- Reject cyclic pipeline graphs, invalid capacity, absent PyTorch/CUDA, and
  incompatible checkpoints rather than silently returning misleading results.
- Record optional JSON run reports. Checkpoints include configuration fingerprint
  and RNG states. Replay is not saved, so resumed training is not bit-exact.

## Objective units and assumptions

Default `--timing_model proxy` retains the communication score for historical
comparison. It contains no compute term and has no physical time unit.

`--timing_model full_frame --use_cnn` evaluates seconds for one complete frame:

1. VMM arithmetic: ceil(MAC count / (128 * utilization)) / 400 MHz.
2. VVA arithmetic: ceil(add count / (assumed additions/cycle * utilization)) / 400 MHz.
   Default VVA throughput is **1 addition/cycle**, an explicit assumption, not
   specified by Table 1. Set `--vva_ops_per_cycle` for sensitivity experiments.
3. VMM output traffic uses 4 bytes per partial sum; VVA output uses 1 byte per
   activation, following Table 1 precision. Serialization uses 64 GB/s on-chip
   and 100 GB/s off-chip by default (decimal GB). Each byte-hop is charged at
   the corresponding inverse bandwidth. These are configurable assumptions.
4. A task's service time is its arithmetic plus summed outgoing transfer costs.
   The objective is the maximum task service time. Grouping tasks by DAG depth
   before taking nested maxima does **not** reconstruct the paper's schedule.

This is a dimensionally consistent **full-frame approximation**, not the paper's
intra-frame block-streaming simulator or end-to-end inference latency. Bandwidth
alone is not hop latency. The model has no shared-link contention, multicast,
router startup, buffers, bandwidth sharing, or overlap. It omits input injection,
output collection, bias/activation/pooling arithmetic, and buffer-capacity checks.
CONV and FC do not yet occupy independently optimized masked regions.

Residual/concat timing is rejected because merge arithmetic is not represented.
Proxy mode traces residual dependency edges but does not provide exact merge
semantics; channel-changing concatenations are rejected. The uniform partitioner
is not the paper's qualitatively described compute-balanced allocation.

## GPU status and tests

The DDPG actor/critic remain **MLPs**, not the paper's CNNs. Learning rates are
0.0002/0.001 and gamma is 0.98. OU noise uses assumed theta=0.15, sigma=0.2,
unit time step, reset each placement, with fading scale. These OU parameters are
implementation choices. Networks/replay minibatches use the selected Torch device;
the environment, replay storage and placement search remain on CPU.

CPU PyTorch is installed only in `/tmp/dnn-validation-venv` for local validation.
No local CUDA device is available. Neither H100 execution nor 12 GB GPU memory
usage is certified. This reconciliation does not claim paper-result reproduction.

Validation performed with torch 2.14.0+cpu and torchvision 0.29.0+cpu:

- Original smoke suite passed, including the previously skipped CNN extraction.
- Seven additional regression tests passed: arithmetic conservation, timing units,
  pooling/flatten traffic, residual dependencies and rejection, topology ID mapping,
  SA budget/relocation, and actual actor parameter updates.
- A 10-episode CPU DDPG run completed, saved its checkpoint and resumed to episode 12.
  This is a functional smoke test, not convergence evidence.
- Real AlexNet and VGG16 extraction completed at partition size 512: respectively
  252 and 516 tasks; VMM MAC totals 714,188,480 and 15,470,264,320 per frame.
  Those partition settings are test configurations, not paper-matched allocations.

Use `src/run_multi_chip.py`. The separate historical `run_multi_chip_fast.py`
has not been reconciled or validated in this change.

Run from the repository root, using a Python environment with NumPy, PyTorch
and torchvision installed:

```bash
python src/test_multi_chip.py
python -m unittest discover -s src -p test_reconciliation.py -v
python src/run_multi_chip.py --algo ddpg --use_cnn \
  --channels_per_partition 128 --timing_model full_frame \
  --device cpu --epochs 10 --baseline_trials 10 --train_every 1 \
  --seed 0 --save_checkpoint smoke.pt --report smoke.json
```

On the H100 host, use its CUDA-enabled PyTorch environment and run the tests
first. The training-update test exercises CUDA automatically when available.
Then run the same small smoke command with `--device cuda`. Check the report
and GPU allocation before increasing workload/budgets. Do not reuse old
checkpoints or compare costs across different timing models.

The reconciled partition counts can differ considerably from the old hook
extractor. Use extraction and capacity checks before long runs; no particular
channel partition size is asserted to reproduce the paper.

Git changes are local until committed and pushed to a remote branch. Nothing
in this reconciliation changes the remote repository automatically.
