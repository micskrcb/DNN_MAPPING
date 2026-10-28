# Pool-context result

Source: kaggle-pool-context.zip; SHA256 4cd3cef39957a130fa80c48afaf88f40c5865cf897905f75cb45df55d788a852.

Seed29, VGG16-FC, revision 2f98bc1, complete with exit 0. All arms: 200001 evaluations, initial cost 8.144 us. Final best: train_more 4.29648 us, pool_context 4.29184 us, fixed_sa 4.39248 us. Context improves over reference by 0.107995% and over SA by 2.291189%. Both learners: 780 updates, 12480 optimizer steps. Total runtime 6.39405 hours.

The only learning-configuration difference is feature set (plus output paths). Context ahead at checkpoints near 50k/100k/150k/180k/200k. Tiny one-seed development win; not statistically established or a published-DDPG comparison. Reports and tests are retained here; raw traces remain in the source ZIP.
