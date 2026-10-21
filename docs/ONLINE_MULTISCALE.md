# Online mapper: mixed-size candidate pool

The earlier code perturbed about1% of tasks per candidate (about19 tasks on
VGG16-FC). New opt-in flag `--ppo_asa_pool_small_move_fraction .5` allocates
8/16 candidates to two-task moves and8/16 to the existing1% moves. Both are
legal permutation/optional-relocation neighbors. On small workloads where1%
already rounds to2, these move families coincide; do not claim an improvement
from scale diversity in that regime.

Hypothesis: small moves can refine a good placement without disturbing many
unrelated tasks, while larger moves retain search diversity. The online bandit
learns from selected signed gains which candidates to choose. This is not a
new reward, offline supervision or claim that small moves will always win.

Feature dimension remains9. Existing changed-task feature remains normalized
by the ordinary move size, so it exposes the smaller scope. No exact objective
is evaluated for unselected candidates. Calibration, fixed temperature schedule,
exploration, replay and optimizer remain unchanged. A two-task neighbor may
also relocate one of those tasks to a free core; it is not always a pure swap.

Default fraction0 preserves the old mapper. Checkpoints record the fraction
and reject incompatible loading; older checkpoints imply fraction0. Reports
record generated small-candidate count and selected-small proposal count.

Run:
```bash
bash scripts/run_kaggle_online_multiscale.sh runs/online-multiscale
```
Default seed10, VGG16-FC VVA16/1,100001 calls per arm:
1. Original50%-focused online mapper.
2. Existing unfocused online mapper.
3. New unfocused mixed-size online mapper.
4. Same mixed pool, uniform choice.
5. Same mixed pool, proxy choice.
6. Plain fixed SA.

All candidate-arm calibrations retain focus.5 and ordinary move size.
No policy is pretrained. An additional256 sensitivity calls apply to each arm.
Budget and initial/calibration checks are enforced. Approximate CPU duration
7–9 hours for six arms, based on recent measured runtime; output archives are
updated after each completed arm.

Seed10 here is a new algorithm-development run. It must not be combined with
seed9 as unchanged confirmation of the old pool-only experiment. That planned
confirmation was superseded by the user's request to prioritize improvements.
Keep seed8 screening and seed9 near-tie in the record. A positive result needs
new-seed checks before adopting the variant. No significance claim from one seed.

Tests check exact candidate sizes, unchanged default behavior/calibration,
legal-neighbor infrastructure, objective accounting, continued online updates,
and checkpoint mismatch rejection. No full benchmark improvement is claimed
from these mechanics checks.

Local CPU validation: all64 unit tests and functional smoke tests passed with
PyTorch enabled. Benchmark quality remains unmeasured for this new variant.
