# Received 32-step horizon experiment

Source ZIP: kaggle-train32-horizon.zip. SHA256: 8d958ffe7ece494ef8e18988821d4f009e61b7c06aed54cc97bacd518ec531d5.

Revision ea00c34bf0cf701b953236e86b705cdf418a097b, VGG16-FC, seed 28. All arms completed 200001 exact placement evaluations. Initial cost 9.552 us for all arms.

| Method | Best cost (us) | Gain over fixed SA |
| --- | ---: | ---: |
| 16-step learner | 4.27584 | 2.08120% |
| 32-step learner | 4.27184 | 2.17280% |
| Fixed SA | 4.36672 | 0% |

Gain of 32 over 16: 0.09355% (4 ns). Both made 780 update events; optimizer steps 12480 and 24960. Total measured runtime: 5.88689 hours. Configurations differed only in update epochs and output paths. The new arm was behind at intermediate checkpoints near 50k, 100k, 150k and 180k. Retain 16 as established default. This one-seed small win does not establish significance, a general intensity benefit or a 5% gain over published DDPG. Raw JSONL/logs remain in the original source ZIP; compact original reports and source summary are retained here.
