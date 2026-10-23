import contextlib,io,unittest
import numpy as np
import torch
from run_multi_chip import run_ddpg
from multi_chip_environment import MultiChipEnvironment

class HorizonTests(unittest.TestCase):
    def run_arm(self,freeze,gamma):
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        env=MultiChipEnvironment(1,1,3,3,task_graph=np.triu(np.ones((4,4)),1),num_tasks=4)
        meta={}
        with contextlib.redirect_stdout(io.StringIO()):
            run_ddpg(env,n_episodes=3,batch_size=2,baseline_trials=2,batch_z=2,train_every=1,
                     device='cpu',run_metadata=meta,ddpg_gamma=gamma,ddpg_disable_learning=freeze)
        return meta
    def test_discount_and_real_training(self):
        m=self.run_arm(False,.999)
        self.assertEqual(m['placement_steps'],2)
        self.assertAlmostEqual(m['first_step_terminal_discount'],.999)
        self.assertGreater(m['invocation_gradient_updates'],0)
        self.assertEqual(m['search_evaluations_including_baseline'],5)
    def test_frozen_control_has_no_updates(self):
        m=self.run_arm(True,.98)
        self.assertEqual(m['invocation_gradient_updates'],0)
        self.assertFalse(m['learning_enabled'])
        self.assertEqual(m['search_evaluations_including_baseline'],5)
if __name__=='__main__':unittest.main()
