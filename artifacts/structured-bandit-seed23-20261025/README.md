# Structured bandit: seed23

Source: `kaggle-structured-bandit-20261024T154129Z.zip`
SHA256: `5ae2a3c9a00bf0fea35a00f85290904cc33210a86aeb513db0b7b5bd74fbd6d0`

Revision57a65ef, complete, exit0. VGG16-FC;100001 evaluations per arm, initial6.51344µs.

|Method|Best cost µs|Gain over fixedSA|
|---|---:|---:|
|Working16-step RL+SA|4.35504|1.57657%|
|Structured RL+SA|4.39952|0.57133%|
|Uniform structured pool|4.51040|-1.93455%|
|FixedSA|4.42480|0%|

Structured learning beat its uniform control2.45832%, but lost to the working learner1.02135%. Both trained arms389 updates/6224 optimizer steps. Structured selections38499 versus uniform49821. Total reported runtime4.32301h. Do not promote the structured variant. This is one seed, not evidence of5% over publishedDDPG. Original diagnostics remain in the source ZIP; no checkpoints loaded.
