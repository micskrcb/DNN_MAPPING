import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_multi_chip as rm
import test_online_pool
from structured_moves import nearby_allowed_cores, attraction_neighbor

class AttractionTests(unittest.TestCase):
    @staticmethod
    def distance(a,b):
        return np.abs(np.asarray(a)//4-np.asarray(b)//4)+np.abs(np.asarray(a)%4-np.asarray(b)%4)

    def test_free_relocation_and_pool_commit(self):
        placement=np.array([0,15]);allowed=np.arange(16)
        free=rm._FreeCorePool(allowed,placement);before=free.items.copy()
        inverse=np.full(16,-1);inverse[placement]=np.arange(2)
        result=attraction_neighbor(placement,[np.array([0,1])],nearby_allowed_cores(allowed,4,4),inverse,free,self.distance,random.Random(4))
        candidate,changed,relocation,kind=result
        self.assertEqual(kind,1);self.assertEqual(len(changed),1)
        self.assertLess(self.distance(*candidate),self.distance(*placement))
        self.assertEqual(free.items,before)  # rejected/unselected proposals cannot mutate it
        free.accept_relocation(*relocation)
        self.assertEqual(set(free.items),set(allowed)-set(candidate))
        self.assertEqual(free.positions,{c:i for i,c in enumerate(free.items)})
        np.testing.assert_array_equal(placement,[0,15])

    def test_full_grid_swap_and_mask_fallback(self):
        placement=np.arange(16);free=rm._FreeCorePool(np.arange(16),placement)
        result=attraction_neighbor(placement,[np.array([0,15])],nearby_allowed_cores(np.arange(16),4,4),placement.copy(),free,self.distance,random.Random(4))
        candidate,changed,relocation,kind=result
        self.assertEqual(kind,2);self.assertIsNone(relocation)
        np.testing.assert_array_equal(np.sort(candidate),placement)
        np.testing.assert_array_equal(np.sort(changed),np.flatnonzero(candidate!=placement))
        self.assertLess(self.distance(candidate[0],candidate[15]),self.distance(0,15))
        placement=np.array([0,15]);inverse=np.full(16,-1);inverse[placement]=[0,1]
        self.assertIsNone(attraction_neighbor(placement,[np.array([0,1])],nearby_allowed_cores([0,15],4,4),inverse,rm._FreeCorePool([0,15],placement),self.distance,random.Random(4)))
        self.assertIn(3,nearby_allowed_cores(range(16),4,4,True)[0])
        self.assertNotIn(3,nearby_allowed_cores(range(16),4,4,False)[0])

    def test_integration_exact_cost_budget_and_default(self):
        runner=test_online_pool.PoolTests()
        a,ma=runner.run_arm(pool_focus_fraction=0)
        b,mb=runner.run_arm(pool_focus_fraction=0,pool_attraction_fraction=0)
        self.assertEqual(a,b);self.assertEqual(ma['final_update_metrics'],mb['final_update_metrics'])
        original=rm.IncrementalPipelineEvaluator.candidate_cost
        checked=[]
        def verify(evaluator,changed):
            cost=original(evaluator,changed)
            self.assertAlmostEqual(cost,float(evaluator.env.evaluate()),places=10)
            checked.append(cost);return cost
        with patch.object(rm.IncrementalPipelineEvaluator,'candidate_cost',verify):
            _,more=runner.run_arm(pool_focus_fraction=0,pool_attraction_fraction=.5)
        self.assertGreater(len(checked),0)
        self.assertEqual(more['total_objective_evaluations'],25)
        self.assertEqual(more['initial_temperature'],ma['initial_temperature'])
        self.assertEqual(more['attraction_candidates_per_pool'],2)
        self.assertGreater(more['attraction_generated']['relocation'],0)
        _,uniform=runner.run_arm(pool_focus_fraction=0,pool_attraction_fraction=.5,disable_learning=True)
        self.assertEqual(uniform['total_objective_evaluations'],25);self.assertEqual(uniform['update_count'],0)

    def test_checkpoint_and_invalid_mixture(self):
        runner=test_online_pool.PoolTests()
        with self.assertRaisesRegex(ValueError,'separately'):
            runner.run_arm(pool_attraction_fraction=.5,pool_structured_fraction=.5)
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'model.pt');runner.run_arm(pool_attraction_fraction=.5,save_checkpoint=path)
            with self.assertRaisesRegex(ValueError,'attraction fraction'):
                runner.run_arm(load_checkpoint=path,disable_learning=True,bandit_control='checkpoint')

if __name__=='__main__':unittest.main()
