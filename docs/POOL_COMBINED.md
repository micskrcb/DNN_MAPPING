# Combined pool-context and 32-step training screen

Seed29's context feature gain was 0.108% over the established learner. Seed28's 32-step gain was 0.094%. These separate-seed gains cannot be added. The established default stays legacy9 with 16 optimizer steps.

The new opt-in pool_context32 method combines the existing pool18 descriptors with 32 optimizer steps at each update event. No new exact evaluations or rewards are introduced. Test all arms from scratch on fresh seed30: train_more (legacy9/16), pool_context (pool18/16), pool_context32 (pool18/32). All use VGG16-FC and 200000 proposals plus initialization, matched initial placement and annealing, exploration and reward settings.

The direct context16/context32 comparison isolates fitting intensity at the same representation. The established/context16 comparison tests the feature change. The analyzer requires identical update-event counts and a twofold optimizer-step ratio only for the combined arm. More fitting may overfit replay; a win is not assumed.

This is a one-seed development screen, not a statistical confirmation. Without legacy9/32 on the same seed it cannot identify a formal interaction effect. Fixed SA is omitted to keep three arms within a session; do not claim a new SA-relative percentage using a different seed's baseline. The 5% published-DDPG target remains unproven.

Run: bash scripts/run_kaggle_pool_combined.sh runs/kaggle-pool-combined

Allow roughly 7–9 hours on CPU, based on prior runs; runtime varies. Enable Internet, no input archive needed. Same-commit reruns with retained output skip completed arms and restart unfinished arms. The output ZIP includes the strict comparison summary. Files: src/analyze_pool_combined.py, scripts/run_kaggle_pool_combined.sh, and the pool_context32 case in scripts/run_kaggle_online_rapid_stage.sh. Existing descriptor and training implementations are reused; no external code copied.
