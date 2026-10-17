# External research review for bandit-guided Proxy-SA

## Purpose and decision

This note records the 17 October 2026 internet review and the independent
Gemini Deep Research report supplied as `DNN Mapping Bandit SA Review.docx`.
It separates findings accepted for this project from claims that are stale,
unsupported, or still experimental. The source DOCX is not committed because
it contains unresolved `[span_...]` citation artifacts and relies partly on
secondary sources.

The current full-information operator gate remains a useful diagnostic. It is
not the final candidate method. Any positive result must also beat an ordinary,
unthrottled Proxy-SA run with the same total true-objective budget. If the gate
fails, the next bounded experiment is selected-action, proxy-gated, discounted
Linear Thompson Sampling with exactly one true-objective call per step.

## Relevant primary software and papers

- [N-Wouda ALNS](https://github.com/N-Wouda/ALNS) is a tested MIT-licensed
  Python framework for operator selection and acceptance criteria. Its
  `MABSelector`, `AlphaUCB`, segmented roulette selection, outcome categories,
  and simulated-annealing interfaces are useful references. Its routing
  operators are not applicable to this placement representation.
- [MABWiser](https://github.com/fidelity/mabwiser) is an Apache-2.0 contextual
  bandit library containing LinUCB, LinTS, UCB1, Softmax, and neighborhood
  policies. It is suitable for reference tests or an optional implementation;
  importing it is not required for a six-dimensional four-arm controller.
- [Balans](https://github.com/coin-or/balans), published at IJCAI 2025, is an
  Apache-2.0 bandit-guided adaptive large-neighborhood search system. Its
  online operator-selection pattern is relevant, while its MIP neighborhoods
  and solver dependencies are not reusable here.
- Garivier and Moulines,
  [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems](https://arxiv.org/abs/0805.3415),
  motivates discounted and sliding-window evidence for changing operator
  utilities.
- Fialho et al.,
  [Extreme Value Based Adaptive Operator Selection](https://inria.hal.science/inria-00287355),
  motivates recent-window credit and change detection when useful operators
  change during a search.
- Ropke and Pisinger's ALNS work motivates distinct credit for a new global
  best, an improving move, an accepted worsening move, and a rejected move.
- Vashisht et al.,
  [Placement in Integrated Circuits using Cyclic Reinforcement Learning and Simulated Annealing](https://arxiv.org/abs/2011.07577),
  applies PPO to produce an SA initialization. It uses sequence-pair
  floorplanning rather than fixed-grid DNN core placement and has no verified
  official implementation available in this review.
- The supplied report relies heavily on the 2026 SERENE-MH result and DOI
  `10.1109/ACCESS.2026.3742882`. Its existence was corroborated, but the full
  primary text and official source code were not available during this audit.
  Detailed numerical and pseudocode claims from the report must therefore be
  verified before being cited or implemented as a reproduction.

## Accepted findings

1. Operator rewards are non-stationary because useful moves change between
   high-temperature exploration and late refinement.
2. Raw signed immediate delta can punish an accepted worsening move even when
   that move helps SA escape a basin. Outcome-aware credit is a credible
   ablation, although positive credit for every accepted worsening move can
   also reward unproductive wandering.
3. Full-information labels improve identifiability but reduce the number of
   sequential state transitions under a fixed objective-call budget.
4. The current matched proxy control isolates learned selection under the same
   number of transitions. It does not replace an unthrottled Proxy-SA baseline.
5. Cheap proxy screening of a small operator slate can reject catastrophic
   candidates without consuming additional true-objective calls.
6. Anytime best-cost curves and their normalized area should accompany final
   best cost and wall-clock time.

## Corrections to the supplied report

- The active operator selector is not stationary infinite-memory LinUCB. It
  already uses three phase-specific models and a configurable evidence
  discount of `0.995`.
- A full-information stride of 20 does not make the complete experiment four
  times as expensive. Nineteen single-call decisions plus one four-call
  collection decision consume 23 calls instead of 20, a 15% call overhead and
  about 13% fewer sequential transitions under a fixed budget.
- Regret bounds do not prove that counted full-information collection is
  strictly suboptimal for this simulator. The usefulness of the extra labels
  depends on model error, reward variance, and the value of additional chain
  transitions. The gate is therefore diagnostic evidence, not the final
  benchmark.
- MABWiser radius or nearest-neighbor policies do not automatically solve
  temporal non-stationarity; old observations from similar contexts can still
  be stale.
- LinTS is not universally superior to LinUCB. Discount `0.98`, window length
  `50`, sampling scale `0.25`, reward weights, and the proposed proxy rank
  bonus are hypotheses requiring matched ablations.
- Thirty seeds are not a universal prerequisite for every development gate.
  Use three paired seeds for rejection, then fresh workloads/seeds and larger
  samples only after a method clears the bounded gate.
- The proposed discounted LinTS method is a contextual bandit. It should not be
  described as long-horizon RL unless actions affect multi-step returns and
  the learner is trained on those delayed returns.

## Next bounded experiment after the current gate

Keep the evaluator, fixed-SA temperature schedule, Metropolis acceptance, and
four existing operators unchanged. Compare under identical seeds and true
evaluation counts:

1. unthrottled exact eight-candidate Proxy-SA;
2. uniform selection over the four operators;
3. discounted LinUCB with selected-action feedback;
4. proxy-gated discounted Cholesky LinTS;
5. optionally, a context-free discounted-UCB or segmented-roulette control.

For the proposed method, sample operator values from discounted linear
posteriors, retain the top two operators, generate one candidate from each,
screen both with the cheap communication-distance proxy, and spend one true
evaluation on the screened winner. Update only the executed operator. Test
raw signed reward and an outcome-aware bounded reward as a prespecified
ablation rather than changing several mechanisms after observing results.

The short advancement rule remains paired and conservative: beat unthrottled
Proxy-SA in at least two of three seeds and improve mean final cost by at least
1%, with valid accounting and finite diagnostics. Also report normalized area
under the best-so-far curve. Failure closes the configuration; passing permits
fresh seeds or another workload before a 100,000-call extension.

## Path to genuine long-horizon RL

If the project requires an RL rather than contextual-bandit contribution, the
next state/action design must expose delayed consequences. One action should
select an operator or operator mixture for a block of SA moves, and its return
should include improvement in global best cost over that block. Training would
need multiple independent chains and preferably multiple DNN/hardware
instances, with workload-level validation splits. Supervised full-information
operator labels may initialize the policy, followed by discrete PPO or Double
DQN fine-tuning. This is a larger experiment and should follow, not replace,
the cheaper online-bandit gate.

## Documentation coverage

- `PROJECT_JOURNEY.md` gives the chronological narrative.
- `PROJECT_STATE.md` is the current empirical source of truth.
- `NEXT_STEPS.md` contains gates and stop conditions.
- `RESEARCH_FINDINGS.md` records literature interpretation and failure
  analyses.
- `RECONCILIATION.md` records paper/evaluator assumptions.
- `runs/` preserves committed summaries, reports, commands, and archive hashes.
- `archive/` preserves superseded audits and historical READMEs.

Chat discussion is not treated as evidence by itself. A result enters the
project record only when its configuration, revision, budget, controls, and
limitations can be stated accurately.
