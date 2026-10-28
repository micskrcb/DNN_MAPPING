# Neutral-example regression weighting

## Received result

Stage24 failed on VGG16-FC seed31: 4.33872 us versus pool18/16 at 4.28048 us and fixed SA at 4.37616 us. All completed 200001 evaluations at revision e562ee6. Stage24 lost 1.36060% to pool18, while pool18 beat SA 2.18639%. Stage24 selected 103477 neutral proposals versus 77296 for pool18; improving proposals were 2770 versus 2913. This association does not prove why stage24 lost. Reject stage24 and retain pool18/16. Source: artifacts/stage-context-seed31-20261029.

## Next hypothesis

Neutral observations constitute a substantial part of the selected-move replay. The opt-in bandit_neutral_weight=.25 reduces their relative influence on regression. Targets remain the signed proposed gain. Uniform replay sampling, candidate selection, exact objective, SA acceptance, exploration and 16-step update schedule remain unchanged.

For a minibatch, compute SmoothL1 per observation. Give normalized targets with absolute value <=1e-9 weight .25; improving and worsening observations both receive weight1. Loss is sum(weight*loss)/sum(weight). The default1 follows the exact previous loss path. All-neutral minibatches therefore have the same loss as before; the change is their relative influence in mixed minibatches. No positive oversampling, gain bonus, altered reward or extra exact-cost labels are introduced. Additional sampled_neutral_fraction diagnostics describe minibatch composition.

Earlier improvement_priority changed labels and did not beat the existing learner across three seeds. Earlier positive oversampling was harmful. This changes loss weighting instead, but is still a biased regression objective: it may overpredict gains on neutral moves, worsen calibration or overfit nonneutral outcomes. Raw loss values are not directly comparable between arms. The hypothesis is unproven and a decline must be retained, not tuned away.

## Protocol and files

Run: bash scripts/run_kaggle_neutral_weight.sh runs/kaggle-neutral-weight

Fresh seed32, VGG16-FC, 200000 proposals plus initialization. Compare pool_context (weight1), neutral_weight (weight.25), and fixed_sa. The analyzer rejects hidden configuration changes, budget/update mismatches and wrong weights. Checkpoint loading rejects weight mismatches. Only direct signed selected-label bandit regression supports the nondefault option.

This is a one-seed development screen, not published-DDPG or significance evidence. A useful result must improve final placement cost over the unweighted learner, with runtime reported. CPU estimate 7–10 hours; enable Internet, no dataset needed. Completed arms can be skipped on same-commit reruns with output preserved; unfinished arms restart.

Code: src/run_multi_chip.py (loss, option, guard, diagnostics), src/test_neutral_weight.py, src/analyze_neutral_weight.py, scripts/run_kaggle_neutral_weight.sh. Original project implementation; no external code copied.
