"""Experimental DDPG normalization ablation; not the paper reproduction entrypoint.

Calibrate online BatchNorm once per replay batch, then use running statistics
for both gradient objectives and action selection. All normal runner arguments
apply. Checkpoints carry a separate normalization marker and cannot be loaded
by the regular runner without an explicit compatibility error.
"""
import torch
import torch.nn.functional as F
import run_multi_chip as runner


class ConsistentBNAgent(runner.DDPGAgent):
    normalization_experiment = 'consistent-inference-v1'

    def state_dict(self):
        state = super().state_dict()
        state['normalization_experiment'] = 'consistent-inference-v1'
        return state

    def load_state_dict(self, state):
        if state.get('normalization_experiment') != 'consistent-inference-v1':
            raise ValueError('Expected a consistent-inference experiment checkpoint')
        super().load_state_dict(state)

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return None
        arrays = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = [
            torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrays]
        rewards, dones = rewards[:, None], dones[:, None]
        # Update each online network's statistics exactly once, without
        # fitting a different batch-normalized function than the one deployed.
        self.actor.train()
        self.critic.train()
        with torch.no_grad():
            self.actor(states)
            self.critic(states, actions)
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        with torch.no_grad():
            target = rewards + (1 - dones) * self.gamma * self.critic_target(
                next_states, self.actor_target(next_states))
        predicted = self.critic(states, actions)
        critic_loss = F.mse_loss(predicted, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        try:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for online, slow in ((self.actor, self.actor_target),
                                 (self.critic, self.critic_target)):
                for value, destination in zip(online.parameters(), slow.parameters()):
                    destination.lerp_(value, self.tau)
                for value, destination in zip(online.buffers(), slow.buffers()):
                    if destination.is_floating_point():
                        destination.lerp_(value, self.tau)
                    else:
                        destination.copy_(value)
        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
                    replay_q_mean=predicted.detach().mean().item(),
                    replay_q_eval_mean=predicted.detach().mean().item(),
                    bellman_target_mean=target.mean().item(),
                    bellman_target_abs_max=target.abs().max().item(),
                    sampled_reward_abs_max=rewards.abs().max().item())


if __name__ == '__main__':
    runner.DDPGAgent = ConsistentBNAgent
    runner.main()
