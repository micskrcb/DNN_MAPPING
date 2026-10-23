import unittest
import torch
from run_multi_chip import within_pool_ranking_loss
import test_online_pool

class RankingTests(unittest.TestCase):
    def test_correct_order_has_lower_loss_and_ties_zero(self):
        gains=torch.tensor([[1.,0.,-1.]])
        good=torch.tensor([[1.,0.,-1.]],requires_grad=True)
        self.assertLess(within_pool_ranking_loss(good,gains),within_pool_ranking_loss(-good,gains))
        loss=within_pool_ranking_loss(good,torch.zeros_like(gains));loss.backward()
        self.assertEqual(loss.item(),0);self.assertTrue(torch.isfinite(good.grad).all())
    def test_online_ranking_charges_pool_labels(self):
        _,m=test_online_pool.PoolTests().run_arm(pool_focus_fraction=0,bandit_pool_supervision_stride=4,bandit_ranking_weight=.25)
        self.assertEqual(m['total_objective_evaluations'],25)
        self.assertGreater(m['supervised_pools'],0)
        self.assertGreater(m['final_update_metrics']['ranking_loss'],0)
        self.assertEqual(m['bandit_ranking_weight'],.25)

if __name__=='__main__':unittest.main()
