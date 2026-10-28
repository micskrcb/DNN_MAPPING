import tempfile
import unittest
from pathlib import Path
import torch
import run_multi_chip as rm
import test_pool_context as helper

class NeutralTests(unittest.TestCase):
    def test_weighted_gradient_and_default(self):
        y = torch.tensor([0., 1., -1.])
        a = torch.tensor([.5,.5,.5], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        old = torch.nn.functional.smooth_l1_loss(a,y)
        self.assertEqual(old.item(),rm.neutral_weighted_regression_loss(a,y).item())
        old.backward()
        rm.neutral_weighted_regression_loss(b,y,.25).backward()
        self.assertAlmostEqual((b.grad[0]/b.grad[1]).item(), .25*(a.grad[0]/a.grad[1]).item())
        for targets in (torch.zeros(3), torch.ones(3)):
            self.assertTrue(torch.isfinite(rm.neutral_weighted_regression_loss(a,targets,.25)))
            self.assertAlmostEqual(rm.neutral_weighted_regression_loss(a,targets,.25).item(),
                                   rm.neutral_weighted_regression_loss(a,targets,1).item())

    def test_real_training_and_checkpoint(self):
        arm = helper.ContextTests().arm
        a = arm('pool18')
        with tempfile.TemporaryDirectory() as d:
            checkpoint = str(Path(d)/'model.pt')
            b = arm('pool18', bandit_neutral_weight=.25, save_checkpoint=checkpoint)
            for key in ('total_objective_evaluations','initial_cost','initial_temperature',
                        'update_count','bandit_optimizer_steps_this_invocation'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(b['bandit_neutral_weight'], .25)
            with self.assertRaisesRegex(ValueError,'neutral weight'):
                arm('pool18', load_checkpoint=checkpoint, allow_finetune_loaded=True)
        for weight in (0, -1, float('nan'), 2):
            with self.assertRaises(ValueError): arm('pool18',bandit_neutral_weight=weight)

if __name__ == '__main__': unittest.main()
