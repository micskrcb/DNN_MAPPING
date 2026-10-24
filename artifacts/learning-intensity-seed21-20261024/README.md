# Learning intensity: seed21

Source: `kaggle-learning-intensity-20261024T095504Z.zip`
SHA256: `7aae9b1e5d3e2a78a8bcf254fa6f86f7db67aa91bcb9d88ac89825b082305038`

Revision b91fece; both stages completed with exit0,73 preflight tests each. All arms100001 evaluations with matched initial placements within workload. Four-step vs16-step training:1556 vs6224 optimizer steps,389 update events each.

|Workload|4-step RL+SA µs|16-step RL+SA µs|SA µs|16-step gain vs4-step|16-step gain vsSA|
|---|---:|---:|---:|---:|---:|
|VGG16-FC|4.35488|4.33504|4.47424|0.45558%|3.11114%|
|AlexNet-FC|2.24000|2.19200|2.19360|2.14286%|0.07294%|

Positive developmental result on both workloads; AlexNet effectively ties SA. Seed21 was previously used for a1025-evaluation mechanics smoke, so it is not untouched held-out evidence. Total reported optimizer runtimes5.12787h, excluding setup. More training did not materially increase observed runtime; this does not establish that it is universally free. No comparison against publishedDDPG or5%SOTA claim. No checkpoint deserialization. Full diagnostics remain in the source ZIP.
