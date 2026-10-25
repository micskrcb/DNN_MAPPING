# Attraction result: seed24

Source: `kaggle-attraction-bandit-20261024T211016Z.zip`
SHA256: `8e31cc4464a5c849f3cc69f7bdedcd719fe7a9ef48a6525e3a5f6b1796dd2c56`

Revision9c1e686, complete, exit0. VGG16-FC,100001 evaluations each, initial6.864µs.

|Method|Best cost µs|Gain vsSA|
|---|---:|---:|
|Working16-step learner|4.38496|1.03279%|
|Attraction learner|4.56144|-2.95031%|
|Attraction uniform|4.47360|-0.96779%|
|FixedSA|4.43072|0%|

Attraction lost4.02467% to the working learner and1.96352% to its uniform control. Both trained arms389 updates/6224 optimizer steps. Selected attraction moves73341 versus uniform49883; zero fallbacks. Final logged acceptance0.91176 for attraction versus0.26471 for reference. Acceptance alone does not prove neutral moves; the preference for low-impact moves is a hypothesis. All continued improving, while attraction lagged both controls near25k,50k,75k,100k. One seed does not establish convergence or rule out this method. Retain old mapper. Full JSONL diagnostics remain in source ZIP. No checkpoints loaded.
