# Project state: DNN core placement

Updated 2026-09-11. This replaces the contradictory current/historical sections
in the supplied Claude briefing. Historical statements are not current validation.

## Goal and source of truth

Reproduce, then improve upon Wu, Deng, Li and Xie's *Core Placement Optimization
for Multi-chip Many-core Neural Network Systems with Reinforcement Learning*
(ACM TODAES, 2020; DOI 10.1145/3418498).

Branch: `codex/reconciled-paper-implementation` in
https://github.com/micskrcb/DNN_MAPPING.

At the start of this audit, local HEAD and the remote-tracking branch both
pointed to `2ddc316`, following the confirmed GitHub push. The implementation
commit is `c655138`. README commits `552b611`, `d93362a`, `e7a05d0`, and
`2ddc316` represent rewrite/revert/replacement/expansion history; the old action
to push `552b611` is obsolete. No changes were merged into main.

Code and tests are authoritative for behavior. The paper is authoritative for
reproduction requirements. A statement in an AI briefing is neither proof of
correctness nor evidence of an experiment having run.

## Corrections to Claude's supplied document

| Claim | Corrected assessment |
| --- | --- |
| Compute latency is resolved | A unit-consistent full-frame approximation is implemented; the paper's timing model is not reproduced |
| VMM/VVA both use 128 MACs/cycle | VMM uses 128 MACs/cycle; VVA defaults to a separately assumed 1 addition/cycle |
| JSON resolves multi-seed reporting | Per-run JSON exists; aggregation, confidence intervals, and automated multi-seed comparisons do not |
| Two sessions agreeing on SA proves correctness | Only code tests and a methodology comparison establish evidence; SA relocation remains an interpretation |
| Parallel GPU environments are required for speedup | This is a candidate optimization; measure actor/update/CPU costs first |
| Add-based merges are exact | Proxy tracing preserves dependencies but omits merge arithmetic; full-frame residual timing is rejected |
| Any FX-traceable model is supported | Tracing is necessary but insufficient; grouped convolutions and several merge patterns are unsupported |
| Independent input partitions preserve old counts | They can change counts substantially, especially at Conv-to-FC boundaries; old allocations cannot be reused as evidence |
| Reaching a lower search cost proves mechanical correctness | It demonstrates a lower score under that implementation; independent correctness checks are still required |
| Hyperparameters are Table 1 entries | The actor/critic rates and gamma are stated in Section 4.1; Table 1 specifies hardware |

The earlier noise, zero-compute, all-to-all routing, and missing-kernel-size
claims in the historical half are superseded. Avoid retaining them as a second
competing list of current TODOs.

## Current implementation

Maintained files live under `src/`:

- `run_multi_chip.py`: FX extraction, dynamic/custom models, MLP DDPG, mapper,
  random search, SA, checkpointing and JSON run summaries.
- `compute_model.py`: disjoint channel ranges, VMM MAC/VVA addition counts,
  arithmetic seconds and edge byte volumes.
- `multi_chip_topology.py`: mesh/torus costs and chip-major physical IDs.
- `multi_chip_environment.py`: graph/capacity validation and maximum task
  service-cost objective.
- `test_multi_chip.py`: eight smoke checks, including extraction with Torch.
- `test_reconciliation.py`: seven regression tests, including a real update.
- `validate_device.py`: new bounded validation/profiling runner described below.

Single-chip PPO/GCN files and the historical `run_multi_chip_fast.py` are not
part of this validated multi-chip path. Single-chip deadlock penalties are not
included in the multi-chip objective.

### Extraction and partitioning

FX discovers Conv2d/Linear dependencies. Weights are partitioned uniformly along
input and output channels. VMM outputs feed VVA reduction tasks. Remainder
tiles conserve layer MAC counts, and overlapping channel ranges replace
unconditional cross-layer broadcasts. Pool/flatten traffic uses consumer shapes.

This is not compute-balanced partitioning. Weight/activation-buffer capacity is
not enforced. Residual add/concat timing is rejected; proxy residual dependency
edges do not model the arithmetic of a merge. Grouped/depthwise convolutions
and unsupported channel-changing merges are rejected.

### Objective

`proxy` is the default communication-only score in arbitrary units.
`full_frame --use_cnn` uses seconds, 128 MACs/cycle at 400 MHz for VMM, a
configurable VVA addition rate (default 1/cycle), and 4-byte partial sums versus
1-byte activations. Defaults are 64 GB/s on-chip and 100 GB/s off-chip.

Both modes take the maximum per-task service cost. Nested maxima over DAG
levels do not reconstruct pipeline stages from the paper. Full-frame arithmetic
plus byte-hop serialization omits contention, shared resources, multicast,
router startup, buffers/stalls, overlap, input/output costs, transformation-unit
arithmetic and the block-streaming schedule. CONV/FC regions are not separated.
Thus physical units are resolved; physical simulator fidelity is not.

### DDPG and baselines

Actor/critic are MLPs. Learning rates are 0.0002 and 0.001; gamma is 0.98;
minibatch is 64. Target networks and soft updates are implementation choices
that also need comparison with the paper's detailed algorithm.

Sparse terminal reward is sqrt(B)-sqrt(L); batched coordinates and Manhattan
collision handling are implemented, including grid-to-physical ID conversion.
Task-index occupancy is supplemented with communication features. The state
is not identical to the paper's grid-only representation.

OU noise uses assumed theta=0.15, sigma=0.2, unit timestep, per-placement reset,
and fading scale. The mechanism is present; unspecified parameters are not
claimed to be reproduced exactly.

RS samples placements. SA uses current-cost acceptance, fixed 0.99 cooling,
roughly 1% perturbations and relocation to unused cores. Cooling/budget and
relocation tests pass, but the exact neighborhood is a documented choice.
Paper-scale million-placement experiments have not been completed here.

### Checkpoints, reports and GPU

Checkpoints restore models/optimizers, baseline, best placement, counters and
RNGs, with a workload/configuration fingerprint. Replay is not persisted and
changing the target episode count changes the fading schedule; resume is not
bit-exact. JSON reports contain summaries, not per-episode learning curves or
multi-seed statistical analysis.

Networks and sampled training tensors can use CUDA. The environment and replay
storage remain CPU-side. Full H100 execution, slice-memory fit and speedup are
unverified. User accesses the allocation via SSH but has no access today
(2026-09-11). No host credentials were provided or required for local work.

## Evidence, without overclaiming

Prior CPU validation used torch 2.14.0+cpu and torchvision 0.29.0+cpu:

- Both suites passed, including real extraction and actor parameter updates.
- A 10-episode CPU DDPG run saved and resumed to episode 12.
- Small RS/SA runs and torus mode completed with valid placements/finite scores.
- AlexNet at partition 512: 252 tasks; 714,188,480 VMM MACs/frame.
- VGG16 at partition 512: 516 tasks; 15,470,264,320 VMM MACs/frame.

These are functionality checks, not convergence experiments. The historical
906-task SimpleCNN / 1,445-task AlexNet runs and their reported 23.3% / 46.8%
improvements were supplied by earlier conversations, not rerun against this
reconciled objective. They must not be combined with current scores or used as
proof of paper-result reproduction. Old 19.4x/4.8x speedup claims likewise are
historical measurements, not fresh benchmarks on this branch.

## Work started from this corrected plan

Added `src/validate_device.py` to make the next GPU check concrete. It defaults
to CUDA and explicitly fails without it, writing a JSON failure report rather
than silently substituting CPU. With a working device it:

1. Runs the two maintained test suites.
2. Builds an extracted full-frame workload and RS reward baseline.
3. Runs bounded episodes using the existing mapper, replay and DDPG methods.
4. Checks valid placements, terminal rewards, finite/changed actor and critic
   parameters, correct network device, and an agent checkpoint round-trip.
5. Reports synchronized action/placement/state/replay/update wall times,
   separately labeled warmup updates, visible CUDA memory, peak allocated and
   reserved training memory, configuration and Git revision.

The profiler uses a fixed exploration scale for measurement, not the production
run's fading schedule. Synchronization adds overhead; do not interpret its
timings as uninstrumented throughput. Its replay cap is intentionally bounded.
Short-run peak memory does not establish steady-state memory fit. Agent
checkpoint round-trip is not a whole-run deterministic-resume test.

Commands from the repository root, after installing matching dependencies:

```bash
python src/validate_device.py --device cpu --output runs/cpu-validation.json
# Once SSH GPU access returns, execute on the allocated GPU host:
python src/validate_device.py --device cuda --output runs/h100-validation.json
```

For a larger measurement, explicitly increase partition/grid/episode settings
after the small validation passes. Compare CPU and CUDA with the same config;
the profiler does not automatically prove or compute a speedup.

## Next priorities

CPU validation completed on 2026-09-11 with Torch 2.14.0+cpu: both maintained
suites passed, and the default 12-episode workload completed 120 steps and 57
training updates. Actor and critic parameters changed and remained finite;
placement/reward checks and the agent checkpoint round-trip passed. The workload
had 29 tasks on 64 physical cores. Instrumented training took about 0.60 seconds
on this local CPU; this is not a GPU speed or convergence result. An explicit
CUDA request correctly produced a failure report because CUDA is unavailable.

1. Preserve this CPU smoke check as a prerequisite for GPU experiments.
2. When SSH access returns, run the CUDA command on the allocated slice and
   inspect component timings and memory; a lack of access is not a failed GPU test.
3. Implement the paper's CNN actor/critic and grid state as an explicit,
   testable configuration; keep old checkpoints distinguishable.
4. Specify and validate the stage/communication model, buffer constraints and
   compute-aware partition interpretation before long paper-comparison runs.
5. Add structured per-episode metrics and multi-seed aggregation, then run
   comparable search budgets with measured runtime and statistical uncertainty.

Parallel GPU environments should be scoped only if measurements justify the
rewrite. Undefined paper details must remain labeled assumptions.
