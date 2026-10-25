Local validation, 2026-10-25
============================

15 focused tests and 51 regression tests passed. A real CLI synthetic smoke ran all three methods on seeds 25, 26, 27 with 600 proposals per arm, a 1x1 chip / 8x8 grid and proxy timing. This verifies workflow mechanics, not DNN placement performance. The temporary smoke copies changed only workload/hardware/timing and budget; the committed Kaggle runner retains VGG16-FC and 100000 proposals. Rerunning the smoke validated and skipped all completed seeds. No full-budget outcome is claimed.
