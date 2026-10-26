# Two-seed 200k learning-attribution result

Source archives: `kaggle-learning-horizon.zip` (seed 25) and `kaggle-learning-horizon-seed26.zip` (seed 26), both revision `acabd3ccce172fe23654136ddaafbf12161c0793`. Each completed 200001 objective evaluations for VGG16-FC with the same initial cost within its paired seed.

| Seed | Learned, 16 steps | Frozen, zero updates | Fixed SA | Learned gain over frozen | Learned gain over SA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 | 4.29264 us | 4.35584 us | 4.39504 us | 1.45093% | 2.32990% |
| 26 | 4.28736 us | 4.37920 us | 4.42320 us | 2.14211% | 3.07108% |
| mean paired gain | | | | **1.79652%** | **2.70049%** |

Both learned runs made 780 update events and 12480 optimizer steps. Both frozen runs made zero updates. Both learned runs beat both controls. Seed 27 was not run and is not assumed. This is positive development evidence for online learning within the reconstructed evaluator, but it is not a significance test, convergence result, paper-equivalent DDPG comparison, or 5% published-method improvement claim.
