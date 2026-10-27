# 32-step online learner screen

## Why this is the next change

At 200001 placement evaluations on VGG16-FC, two completed paired seeds show that the established 16-step learned proposal policy outperforms both an identical frozen policy and fixed SA. The measured mean gains are 1.77% over frozen and 2.70% over fixed SA. This is development evidence from seeds 25 and 26, not a third assumed seed or a published-DDPG comparison.

The learner was still finding retained-best placements at the 200k endpoint. The only new algorithmic variable is therefore optimizer-step intensity: `train_32` performs 32 minibatch updates at each of the same update events where `train_more` performs 16. Candidate generation, exact cost, SA acceptance, seed, initialization, replay capacity, labels, features, epsilon schedule, temperature schedule, objective-call budget, and hardware parameters are unchanged.

Increasing from 4 to 16 updates was a prior positive result. Increasing to 32 may reduce regression error, but it may also overfit a drifting replay buffer or simply consume more wall time. This is a one-seed screen, not a claim that 32 is better.

## Run

```bash
bash scripts/run_kaggle_train32_horizon.sh runs/kaggle-train32-horizon
```

The default is fresh seed 28, VGG16-FC, 200000 proposals plus initialization per arm. It compares `train_more` (16 steps), `train_32` (32 steps), and fixed SA. The analyzer rejects hidden changes other than update epochs and output paths, requires equal update-event counts, and requires exactly twice as many optimizer steps in `train_32`.

Allow roughly 7–9 hours on Kaggle CPU. The stage preserves completed arms and checks their saved command before skipping them on rerun. An unfinished arm restarts from the beginning; this is not model-state resume. A positive screen means `train_32` has a lower best cost than both controls. It does not establish a 5% result over TODAES DDPG.
