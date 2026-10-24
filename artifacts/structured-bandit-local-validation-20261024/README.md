# Structured move implementation validation

Synthetic64-task graph on1×1 chip,8×8 cores,proxy timing,paper_xy routing,seed23,601 placement evaluations per arm. All four arms completed, analyzer and ZIP generation passed. This is not a DNN benchmark or an improvement claim.51 standard tests plus36 focused test invocations passed (some imported test cases repeat). Includes legal occupancy, changed-task detection, reproducibility, incremental versus full evaluator checks, budget accounting and checkpoint mismatch rejection.

The first smoke harness incorrectly requested paper_pipeline without extracted CNN metadata and exited before optimization; the corrected synthetic harness uses proxy timing. The real Kaggle script retains extracted VGG16-FC and paper_pipeline.
