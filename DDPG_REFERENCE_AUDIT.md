# DDPG Reference-Implementation Audit

Updated: 8 October 2026

## Decision

The project should not paste Stable Baselines3 or OpenAI Spinning Up DDPG into
the placement loop unchanged. Both reference implementations require a truly
continuous `Box` action space. Core placement executes a discrete, shrinking
set of legal grid positions, so a direct substitution would preserve the main
action mismatch that caused the failed Kaggle run.

The useful approach is to compare the training loop line by line, retain the
paper-aligned implementation as a control, and adopt reference practices that
remain valid after legal discrete action selection. Guided DDPG is therefore a
hybrid of DDPG, Wolpertinger-style legal candidate selection, demonstration
learning, and selected TD3 stability mechanisms. It is not unmodified DDPG,
DDPGfD, Wolpertinger, or TD3.

## Reference code reviewed

- [OpenAI Spinning Up PyTorch DDPG](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)
- [OpenAI Spinning Up DDPG networks](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/core.py)
- [Stable Baselines3 DDPG](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ddpg/ddpg.py)
- [Stable Baselines3 TD3 training loop](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py)
- [Original TD3 implementation](https://github.com/sfujim/TD3)
- [Wolpertinger paper](https://arxiv.org/abs/1512.07679)
- [DDPG from Demonstrations paper](https://arxiv.org/abs/1707.08817)

The GitHub DDPGfD repositories found during the search are independent course
or reproduction projects rather than code released by the paper's authors.
They are useful for inspection but are not a safer source than the published
algorithm and maintained reference libraries.

## Line-by-line algorithm comparison

| Concern | Established implementation | This project | Assessment |
| --- | --- | --- | --- |
| Action space | Spinning Up and SB3 DDPG use continuous bounded actions | Paper mode emits continuous coordinates and then floors/repairs them | This is the paper's stated method, but it creates a difficult discrete execution mismatch |
| Replayed action | Store the action that produced the transition | Guided mode stores the selected legal bin-centre action | Corrected |
| Random warm-up | Spinning Up uses `start_steps`; SB3 uses `learning_starts` | Guided mode uses uniformly sampled legal complete placements first | Equivalent exploration principle, adapted to legal placement |
| Replay randomness | Library replay buffers own their sampling behavior | Replay and exploration previously shared global NumPy state | Corrected on 8 October: independent seeded streams now make trained/control exploration genuinely paired |
| Critic target | DDPG uses a one-step Bellman target from target actor/critic | Paper mode does this; guided mode regresses discounted full-episode returns | Guided mode solves terminal credit propagation but is no longer literal DDPG critic training |
| Actor loss | Maximize the critic's value of the actor action | Both modes use `-Q(s, actor(s))` | Structurally consistent |
| Critic during actor update | Freeze critic parameters | Both modes freeze the critic | Consistent |
| Target updates | Polyak update target parameters; SB3 copies BatchNorm statistics | Implemented, including BatchNorm buffers | Consistent in paper mode; guided target networks are currently unused by its Monte Carlo critic target |
| Twin critics | TD3 takes the smaller target value | Guided mode has twin critics and conservatively ranks legal candidates | Partial TD3 adaptation |
| Delayed actor | TD3 updates actor less often | Guided mode uses policy delay 2 | Consistent |
| Target smoothing | TD3 adds clipped noise to target actions | Not implemented | Do not call guided mode full TD3 |
| Discrete execution | Wolpertinger finds nearby discrete actions and ranks them with a critic | Guided mode constructs up to `k` nearby legal action batches and ranks them | Same design pattern; current joint-candidate generator is a deterministic heuristic, not an exact approximate-nearest-neighbour index |
| Demonstrations | DDPGfD retains demonstrations and combines prioritized one-step/n-step and supervised losses | Guided mode permanently retains one ASA trajectory, prioritizes it, and adds behavior cloning | Useful subset, not exact DDPGfD |
| Evaluation | Separate deterministic, noise-free episodes | Guided diagnostics use zero-noise policy rollouts | Consistent |

## What was actually wrong

The failed 3,000-placement run was not failing because its Adam update or basic
actor loss differed radically from public DDPG. The dominant problems were at
the boundary between DDPG and the placement environment:

1. Almost every continuous request collided and the environment executed a
   different action.
2. Replay associated the requested action with the repaired transition.
3. Sparse terminal feedback had to propagate through a long placement horizon.
4. The deterministic actor collapsed while noisy exploration continued to find
   occasional good candidates.
5. Training and the no-learning control consumed a shared random stream, so the
   intended paired comparison was still confounded.

Items 1 through 4 motivated guided DDPG. Item 5 was found in this reference
audit and is now fixed with independent, checkpointed action and replay random
streams.

## Remaining research risks

- The guided critic is trained on Monte Carlo return targets while actor
  gradients are evaluated at continuous proto-actions. The critic mostly sees
  discrete bin-centre actions, so its gradients between those points may still
  be unreliable.
- The target actor and target critics do not participate in guided Monte Carlo
  targets. Keeping them has no stabilizing effect in this mode.
- Forced demonstration sampling makes the current importance weights an
  approximation to the complete mixture-sampling probability.
- A single ASA demonstration covers little of the state distribution.
- The legal top-k batch generator is heuristic and can miss a better joint
  combination of the individually nearby cores.

These are reasons to test the current intervention before adding more
mechanisms. If the trained policy cannot beat the identically seeded
no-learning control across seeds, masked categorical PPO is a cleaner match to
the discrete legal action space.

## Validation rule

A short paired run checks mechanics and can reveal a directional signal. It is
not evidence by itself. The multi-seed run is considered promising only when:

- random warm-up placement costs match exactly between each trained/control
  pair;
- both conditions have zero collision repairs;
- all trained losses, Q values, and gradient norms remain finite, absolute Q
  remains below the analyzer's conservative normalized-return guard of 100,
  and late proto-action saturation remains below 95%;
- trained late deterministic cost is lower in at least four of five seeds; and
- the exploratory paired 95% interval for the improvement excludes zero.

The repository script `scripts/run_kaggle_guided_ablation.sh` runs both
conditions, preserves checkpoints and logs, analyzes these gates, and creates a
ZIP archive for download.
