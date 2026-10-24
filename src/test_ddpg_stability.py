import unittest
import numpy as np
import torch
from run_multi_chip import DDPGAgent,ReplayBuffer

class StabilityTests(unittest.TestCase):
    def test_clipping_and_value_diagnostics(self):
        torch.set_num_threads(2);torch.manual_seed(0)
        agent=DDPGAgent(8,2,device='cpu');agent.ddpg_grad_clip=.1
        replay=ReplayBuffer()
        for i in range(8):replay.add(np.ones(8,dtype=np.float32)*i,np.zeros(2,dtype=np.float32),1000.,np.ones(8,dtype=np.float32),True)
        result=agent.train(replay,4)
        for k in ('q_abs_mean','target_abs_mean','td_abs_mean','reward_abs_mean','actor_grad_norm','critic_grad_norm'):
            self.assertTrue(np.isfinite(result[k]))
        self.assertEqual(result['target_abs_mean'],1000.)
        for network in (agent.actor,agent.critic):
            norm=torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None]))
            self.assertLessEqual(float(norm),.10001)
if __name__=='__main__':unittest.main()
