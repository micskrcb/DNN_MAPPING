# Paper-informed PPO-ASA training decision

Updated 2026-10-10.

## What the published budgets mean

The related papers count different operations, so their numbers cannot be
treated as interchangeable epochs.

| Work | Reported training/evaluation scale | Meaning |
| --- | --- | --- |
| Wu et al., DDPG core placement | 30 complete placements per epoch; convergence around 300,000--400,000 complete placements; RS and SA search about one million placements | A complete placement assigns every logic core. This corresponds to about 10,000--13,333 declared DDPG epochs, not local swap moves. |
| Myung et al., PPO core placement | 15,000--70,000 placements for smaller cases; 150,000/380,000 samples without community detection and 80,000/180,000 with it on two large cases | A one-step policy generates a complete placement. Convergence requires 1,000 consecutive rewards within 1%--5% of the best training reward. |
| Vashisht et al., cyclic PPO+SA | 10 epochs for ami49 or 15 for lattice; each epoch contains 200 RL steps followed by 5,000 SA steps; results average 10 experiments | PPO learns an SA initializer through a delayed global reward. It is structurally different from the learned proposal policy here. |
| Qiu and Liang, RL-Based-SA | 1,000 training epochs; the paper uses 128 training problems for Knap50, while the public default batches 256 problems for 40 SA steps; PPO uses 10 optimization passes | PPO learns an SA proposal distribution. This is the closest design to the current hybrid. The public default produces about 10.24 million transitions and 1,000 update cycles. |

Primary sources: [Wu et al.](https://doi.org/10.1145/3418498),
[Myung et al.](https://doi.org/10.1109/TNNLS.2021.3117878),
[Vashisht et al.](https://arxiv.org/abs/2011.07577), and
[Qiu and Liang](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p1718.pdf).
The public RL-Based-SA implementation is
[RL-Based-SA-Public](https://github.com/nathanqiu07/RL-Based-SA-Public).

## What the completed 250,000-proposal run established

The run evaluated 250,000 local neighbor proposals, not 250,000 complete
placements. With one update after each 4,100-step chain, it performed only 61
PPO updates. Five unseen holdout seeds were enough for a screening gate, and
they rejected that checkpoint: its mean was 0.00558% worse than uniform and
0.04751% worse than ASA. Its entropy remained within 0.31% of the uniform
maximum, approximate KL was near zero, and PPO clipping never activated.

The raw sample count is substantial, but the optimizer-update count is far
below the closest paper. It is therefore honest to say that the checkpoint did
not learn a useful proposal preference. It is premature to say that the
PPO-ASA idea cannot learn.

Five seeds match the seed count used in several related RL experiments and are
adequate for a gate. They are not enough for a final claim when the measured
difference is close to zero. Vashisht et al. report improvements of roughly
0.85%--4.13% on their lattice tests and 6.52%--7.55% on ami49 over random SA
initialization, averaged across 10 experiments. Our present effect is two
orders of magnitude smaller and has the wrong sign.

## Next decision run

`scripts/run_kaggle_ppo_asa_sufficient_scale.sh` performs the next and last
test of the current candidate-scoring architecture before an architectural
change:

- 1,000,000 evaluated proposals across 244 independent 4,100-step chains;
- exactly 1,000 PPO update cycles with 10 optimization passes each;
- 1,000-transition on-policy batches, with chain boundaries terminating GAE;
- learning rate `2e-4`, weight decay `0.01`, gamma/GAE lambda `0.9`, clip
  ratio `0.25`, and zero entropy bonus, following the closest public setup;
- a frozen 10-seed holdout against a freshly initialized exactly-uniform
  proposal policy and ordinary ASA, each with identical 4,101-call budgets.

This still uses fewer total transitions than the vectorized public benchmark,
but it removes the 61-versus-1,000 update discrepancy and the entropy pressure
that kept the previous policy close to uniform. On Kaggle T4 x2, the expected
wall time is approximately 6--9 hours; the script archives partial output if
the process fails.

If the checkpoint beats uniform on at least eight of 10 holdout seeds with a
positive paired mean, it passes the policy gate. It must also beat ASA on at
least eight seeds to claim an optimizer improvement. A passing one-training-
seed gate must then be repeated with at least three independently trained
checkpoints before publication. If it fails the uniform gate while remaining
near-uniform, close this candidate-scorer architecture and move to an LSTM or
vectorized complete-placement policy; do not conclude that all RL+SA hybrids
are impossible.
