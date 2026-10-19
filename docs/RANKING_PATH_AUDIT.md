# Ranking-path audit

Run `bash scripts/run_kaggle_ranking_path_audit.sh runs/ranking-path-audit`.
CPU; estimated 1.5–2.5 hours from the previous 92-minute nine-run experiment,
with additional audit cost and training. Library/hardware variance applies.

Reuse identical training pools (30–33), 100 epochs/1,600 updates, frozen model.
Cross learned/proxy scores with exploratory/greedy selection on seeds 40–41.
All other workload, temperature, proposal and objective settings match the
previous rollout. These reused seeds are diagnostic, not a new held-out claim.

Greedy uses warmup=0, top-m=1 and global exploration=0, making selection
probability one on the highest score. SA still accepts uphill moves using its
unchanged temperature rule. Exploratory reproduces the previous schedule.

Each run gets 4,101 search calls plus up to 960 diagnostic calls (64 pools,
15 additional evaluations each), plus 16 sensitivity calls. Oracle results
never choose actions or train weights. Exports add policy scores, probabilities
and selected indices. A checkpoint test verifies auditing preserves final cost.

The summary reports full-search costs and all/late-pool diagnostics:
- Greedy minus proxy exact gain on the same pool: ranking generalization.
- Expected selected gain minus greedy gain: effect of exploration in that pool.
- Realized selected gain minus proxy gain: observed sampled choices.

Candidate gains are immediate, not long-horizon values. Positive same-pool
ranking with worse final costs would motivate investigating future consequences
and state distribution, not prove either cause. No confidence intervals treat
correlated pools as independent. No automatic advancement or retraining occurs.
If an arm improves, confirm with fresh seeds before declaring progress.
