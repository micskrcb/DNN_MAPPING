# Sufficient-scale PPO-ASA decision result

Source revision: `bdcd801478042ff59a901b57d347db3e6e4e9a7b`  
Uploaded archive SHA-256: `4734d212bb5f7bf9bb0d208f595704ae716b55e4e5bacc87a9674a8839194b78`

## Protocol integrity

The run completed the prespecified protocol on Kaggle:

- 1,250,000 evaluated training proposals;
- 305 independently initialized 4,100-step chains;
- exactly 1,250 PPO updates with 10 optimization passes per update;
- 1,250,305 total training objective calls including chain initializations;
- 10 unseen frozen-policy holdout seeds;
- 4,101 objective calls for trained PPO-ASA, uniform control, and ASA in every
  holdout seed;
- all holdout mechanics checks passed.

Training took 23,294 seconds (6.47 hours). The report records the intended
learning rate, weight decay, gamma, GAE lambda, clipping ratio, and zero entropy
bonus. The dirty Git flag comes from generated run output inside the checkout;
the recorded source revision is the expected commit.

## Holdout result

| Condition | Mean best latency (microseconds) | Sample SD |
| --- | ---: | ---: |
| Trained PPO-ASA | 23.502576 | 0.041239 |
| Uniform frozen proposal policy | 23.507216 | 0.034378 |
| Ordinary ASA | 23.489392 | 0.062558 |

Against uniform, trained PPO won three seeds, tied five, and lost two. Its mean
advantage was only 0.004640 microseconds (`0.01974%`), with exploratory paired
95% interval `[-0.007146, 0.016426]`. This fails the prespecified positive
mean plus eight-of-10 policy gate.

Against ASA, trained PPO won four of 10 and was 0.013184 microseconds worse on
average (`-0.05613%` improvement). Its paired interval also includes zero. It
therefore fails the optimizer gate.

## Learning diagnosis and decision

The policy did not merely need more epochs. Entropy finished at `2.77258849`,
effectively identical to the uniform maximum `ln(16) = 2.77258872`.
Approximate KL was `8.65e-9`, clip fraction was zero, and policy loss was near
zero at update 1,250. Entropy was already effectively uniform around update
312 and stayed there while critic value loss continued to fall. The optimizer
learned the return scale but found no stable action preference.

This closes the current candidate-set PPO proposal scorer. More epochs of the
same configuration are not justified. It does not close PPO generally: the
constructive masked-PPO CONV experiment remains a positive learning result.
Any further RL+ASA work must change the representation or training structure,
such as a recurrent full-rollout policy, vectorized complete-placement policy,
or cyclic learned initialization. ASA remains the recommended AlexNet-FC
optimizer for immediate results.
