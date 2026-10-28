import unittest
from unittest.mock import patch
import numpy as np
import run_multi_chip as rm
from test_pool_context import ContextTests

class StageTests(unittest.TestCase):
    def test_slack_coverage_ties_and_empty(self):
        np.testing.assert_allclose(rm.cached_stage_features([10,8,4], {1,2}), [.2,.4,2/3])
        np.testing.assert_allclose(rm.cached_stage_features([10,10,4], {0,1}), [0,0,2/3])
        np.testing.assert_allclose(rm.cached_stage_features([0,0], {0}), [0,0,.5])
        self.assertEqual(rm.cached_stage_features([10,8], set()), [1,1,0])

    def test_training_budget_and_descriptor_dimension(self):
        helper = ContextTests()
        a = helper.arm('pool18')
        with patch.object(rm, 'pool_relative_features', wraps=rm.pool_relative_features) as transform:
            b = helper.arm('stage24')
        self.assertGreater(transform.call_count, 0)
        self.assertTrue(all(call.args[0].shape[1] == 12 for call in transform.call_args_list))
        for k in ('total_objective_evaluations','initial_cost','initial_temperature',
                  'update_count','bandit_optimizer_steps_this_invocation'):
            self.assertEqual(a[k], b[k])

if __name__ == '__main__': unittest.main()
