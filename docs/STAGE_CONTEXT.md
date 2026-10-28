# Cached stage context for the online learner

The combined pool18/32 candidate lost seed30: 4.29808 us against pool18/16 at 4.28928 us and legacy9/16 at 4.29152 us. All completed 200001 evaluations. Combined lost 0.20516% to context16 and 0.15286% to the established learner. Context16 won by 0.05220%; across seeds29–30 its mean paired gain is 0.08010%. This small two-seed direction supports retaining context16 as a development candidate, not claiming significance. No SA or DDPG arm was present. Reports: artifacts/pool-combined-seed30-20261028.

## Next feature hypothesis

Earlier expensive topology/contention inputs and stronger training did not yield reliable gains. The new stage24 option adds only pre-move information from the evaluator's cached stage costs. For all stages affected by moved tasks (including predecessors whose outgoing communication changes), it adds minimum normalized slack, mean normalized slack, and fraction of stages affected. Slack is (maximum current stage cost - stage cost) / maximum current stage cost. Zero slack means a tied/current bottleneck. Empty affected sets use [1,1,0]. All-zero stage costs produce zero slack for affected stages.

Raw inputs are legacy9 plus these three fields. Their 12 relative-pool versions are appended by the existing normalization, giving 24 inputs. No candidate's post-move latency or reward enters the features. Each proposal still evaluates only the selected candidate. Replay stores observed features; checkpoint feature checks reject mixing stage24 with earlier input dimensions.

This exposes current bottleneck proximity but does not predict the actual direction of communication changes. If almost every proposal touches the same stages, these inputs can be constant and unhelpful; the existing mean_feature_std diagnostics show this. More inputs add encoder parameters, so this is not a capacity-matched feature attribution claim. The added affected-stage lookup can add runtime, despite avoiding route simulation for every candidate. No improvement is assumed.

## Experiment

Run: bash scripts/run_kaggle_stage_context.sh runs/kaggle-stage-context

Fresh seed31, VGG16-FC, 200000 proposals plus initialization per method: pool_context (pool18/16), stage_context (stage24/16), fixed_sa. Same reward, candidate generation, annealing, exploration, update schedule and objective-call budget. The analyzer rejects differences beyond input features and output paths, and checks matching update counts. Report final cost versus both controls and runtime. One-seed development screen, not a published-DDPG comparison. Prior negative variants remain available.

Estimated CPU duration 7–10 hours, uncertain until measured. Internet enabled; no input ZIP needed. Completed arms are reusable with the same commit and retained directory; unfinished arms restart. Output: kaggle-stage-context.zip.

Implementation: src/run_multi_chip.py (cached_stage_features and stage24 input), src/test_stage_context.py, src/analyze_stage_context.py, scripts/run_kaggle_stage_context.sh. Existing mechanics are reused; no external code copied.
