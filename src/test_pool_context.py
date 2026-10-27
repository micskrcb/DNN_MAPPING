import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import run_multi_chip as rm
from multi_chip_environment import MultiChipEnvironment


class ContextTests(unittest.TestCase):
    def test_order_invariance_ties_and_raw_preservation(self):
        x = np.arange(36, dtype=np.float32).reshape(4, 9)
        x[:, 2] = 3
        y = rm.pool_relative_features(x)
        np.testing.assert_array_equal(y[:, :9], x)
        np.testing.assert_allclose(y[::-1], rm.pool_relative_features(x[::-1]))
        np.testing.assert_array_equal(y[:, 11], 0)
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(np.abs(y[:, 9:]).max(), 1)
        np.testing.assert_array_equal(rm.pool_relative_features(np.ones((4, 9)))[:, 9:], 0)

    def arm(self, feature_set, **kw):
        torch.set_num_threads(2)
        graph = np.array([[0,3,2,0],[0,0,1,4],[0,0,0,2],[0,0,0,0]], float)
        meta = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rm.run_ppo_asa(MultiChipEnvironment(1,1,3,3,task_graph=graph,num_tasks=4),
                n_iter=24,calibration_trials=4,candidate_count=4,hidden_dim=16,
                device='cpu',seed=3,proposal_learner='bandit',reward_mode='proposed_delta',
                annealing_schedule='fixed',focus_fraction=.5,pool_focus_fraction=0,
                bandit_warmup=4,rollout_steps=4,minibatch_size=4,update_epochs=16,
                bandit_feature_set=feature_set,metadata=meta,**kw)
        return meta

    def test_training_budget_and_checkpoint_contract(self):
        a = self.arm('legacy9')
        with tempfile.TemporaryDirectory() as d:
            checkpoint = str(Path(d) / 'model.pt')
            b = self.arm('pool18', save_checkpoint=checkpoint)
            for k in ('total_objective_evaluations','initial_cost','initial_temperature',
                      'update_count','bandit_optimizer_steps_this_invocation'):
                self.assertEqual(a[k], b[k])
            self.assertGreater(b['update_count'], 0)
            with self.assertRaises(ValueError):
                self.arm('legacy9', load_checkpoint=checkpoint)


if __name__ == '__main__':
    unittest.main()
