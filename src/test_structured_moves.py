import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_multi_chip as rm
import numpy as np
from structured_moves import flow_groups, axis_ranks, flow_axis_neighbor
from test_online_pool import PoolTests

class StructuredTests(unittest.TestCase):
    def test_groups_and_legal_changes(self):
        groups=flow_groups(10,np.arange(9),np.arange(1,10),np.ones(9))
        self.assertTrue(all(2<=len(g)<=8 and len(set(g))==len(g) for g in groups))
        self.assertEqual(flow_groups(3,[],[],[]),[])
        ids=np.arange(30);ranks=axis_ranks(ids//15,ids//30,(ids%15)//5,ids%5)
        placement=np.array([29,4,17,3,20,8,25,7,13,1]);original=placement.copy()
        a=random.Random(3);b=random.Random(3)
        for _ in range(40):
            c,changed,reloc=flow_axis_neighbor(placement,groups,ranks,a)
            d,_,_=flow_axis_neighbor(placement,groups,ranks,b)
            np.testing.assert_array_equal(c,d)
            np.testing.assert_array_equal(np.sort(c),np.sort(placement))
            np.testing.assert_array_equal(np.sort(changed),np.flatnonzero(c!=placement))
            self.assertGreaterEqual(len(changed),2);self.assertLessEqual(len(changed),8)
            self.assertIsNone(reloc)
        np.testing.assert_array_equal(placement,original)

    def test_default_and_incremental_evaluator(self):
        runner=PoolTests()
        a,ma=runner.run_arm(pool_focus_fraction=0)
        b,mb=runner.run_arm(pool_focus_fraction=0,pool_structured_fraction=0)
        self.assertEqual(a,b);self.assertEqual(ma['final_update_metrics'],mb['final_update_metrics'])
        original=rm.IncrementalPipelineEvaluator.candidate_cost
        checked=[]
        def verify(evaluator,changed):
            value=original(evaluator,changed)
            self.assertAlmostEqual(value,float(evaluator.env.evaluate()),places=10)
            checked.append(value)
            return value
        with patch.object(rm.IncrementalPipelineEvaluator,'candidate_cost',verify):
            c,mc=runner.run_arm(pool_focus_fraction=0,pool_structured_fraction=.5)
        self.assertGreater(len(checked),0)
        self.assertEqual(mc['total_objective_evaluations'],25)
        self.assertEqual(mc['initial_temperature'],ma['initial_temperature'])
        self.assertEqual(mc['structured_candidates_per_pool'],2)
        self.assertGreater(mc['selected_structured_proposals'],0)
        _,uniform=runner.run_arm(pool_focus_fraction=0,pool_structured_fraction=.5,disable_learning=True)
        self.assertEqual(uniform['total_objective_evaluations'],25)
        self.assertEqual(uniform['update_count'],0)

    def test_checkpoint_rejects_changed_pool(self):
        runner=PoolTests()
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'model.pt')
            runner.run_arm(pool_structured_fraction=.5,save_checkpoint=path)
            with self.assertRaisesRegex(ValueError,'structured fraction'):
                runner.run_arm(load_checkpoint=path,disable_learning=True,bandit_control='checkpoint')

if __name__=='__main__':unittest.main()
