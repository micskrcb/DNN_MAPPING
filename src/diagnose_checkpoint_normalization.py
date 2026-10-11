"""Compare a saved paper CNN's BatchNorm modes on fixed partial layouts.

This is a synthetic distribution probe, not a policy-quality evaluation.
Run with a trusted, local AlexNet-CONV checkpoint only (torch pickle loading).
"""
import argparse
import json
import numpy as np
import torch
from run_multi_chip import DDPGAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    agent = DDPGAgent(4279, 6, device='cpu', agent_arch='paper_cnn',
                      rows=64, cols=64, num_tasks=183)
    agent.load_state_dict(checkpoint['agent'])
    physical = np.asarray(checkpoint['best_placement'])
    if len(physical) != 183 or np.any(physical < 0) or np.any(physical >= 256):
        raise ValueError('Probe requires a 183-task AlexNet CONV layout on the first 16x16 chip')
    grid_ids = (physical // 16) * 64 + physical % 16
    allowed = np.array([y * 64 + x for y in range(16) for x in range(16)])
    states = np.full((64, 4279), -1, dtype=np.float32)
    states[:, 4096:] = 0  # paper_cnn does not read the appended communication vector
    states[:, allowed] = 0
    for i in range(64):
        count = i * 183 // 64
        states[i, grid_ids[:count]] = np.arange(1, count + 1) / 183
    states = torch.tensor(states)
    actions = torch.zeros(64, 6)
    result = {'checkpoint_episode': checkpoint['episode'],
              'probe': 'prefixes of saved best placement; zero actions; not replay samples'}
    with torch.no_grad():
        for name in ('actor', 'actor_target'):
            model = getattr(agent, name)
            model.eval()
            inference = model(states)
            model.train()
            batch = model(states)
            result[name] = {
                'mean_absolute_action_difference': (inference - batch).abs().mean().item(),
                'max_absolute_action_difference': (inference - batch).abs().max().item(),
                'inference_saturated_fraction': (inference.abs() >= 0.999).float().mean().item(),
                'batch_saturated_fraction': (batch.abs() >= 0.999).float().mean().item(),
            }
        for name in ('critic', 'critic_target'):
            model = getattr(agent, name)
            model.eval()
            inference = model(states, actions)
            model.train()
            batch = model(states, actions)
            result[name] = {
                'running_statistics_q_mean': inference.mean().item(),
                'batch_statistics_q_mean': batch.mean().item(),
                'mean_absolute_difference': (inference - batch).abs().mean().item(),
            }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
