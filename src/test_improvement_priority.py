import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import run_multi_chip as rm
from multi_chip_environment import MultiChipEnvironment
from analyze_priority_seeds import analyze

class PriorityTests(unittest.TestCase):
    def arm(self,mode,**kw):
        torch.set_num_threads(2);meta={}
        graph=np.array([[0,3,2,0],[0,0,1,4],[0,0,0,2],[0,0,0,0]],float)
        with contextlib.redirect_stdout(io.StringIO()):
            rm.run_ppo_asa(MultiChipEnvironment(1,1,3,3,task_graph=graph,num_tasks=4),
                n_iter=24,calibration_trials=4,candidate_count=4,hidden_dim=16,
                seed=3,device='cpu',proposal_learner='bandit',reward_mode=mode,
                annealing_schedule='fixed',focus_fraction=.5,pool_focus_fraction=0,
                bandit_warmup=4,rollout_steps=4,minibatch_size=4,metadata=meta,**kw)
        return meta
    def test_target_separates_outcomes_and_bounds_penalty(self):
        f=lambda g:rm.proposal_training_reward('improvement_priority',g,0,0)
        self.assertEqual(f(0),0);self.assertEqual(f(1e-10),0)
        self.assertLess(f(-.1),0);self.assertEqual(f(-10),-.1)
        self.assertGreater(f(.01),1);self.assertEqual(f(10),2)
        self.assertEqual(rm.proposal_training_reward('proposed_delta',-4,0,0),-4)
    def test_budget_calibration_and_outcome_accounting(self):
        a=self.arm('proposed_delta');b=self.arm('improvement_priority')
        for k in ('total_objective_evaluations','initial_cost','initial_temperature','update_count'):
            self.assertEqual(a[k],b[k])
        self.assertEqual(sum(b['selected_move_outcomes'].values()),20)
        self.assertEqual(b['pool_attraction_fraction'],0)
    def test_checkpoint_rejects_target_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'model.pt');self.arm('improvement_priority',save_checkpoint=p)
            with self.assertRaisesRegex(ValueError,'priority reward'):
                self.arm('proposed_delta',load_checkpoint=p,disable_learning=True,bandit_control='checkpoint')
    def test_empty_summary_is_provisional_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=analyze(d,[25,26,27]);self.assertTrue(r['provisional']);self.assertEqual(r['completed_seeds'],[])
            self.assertIsNone(r['comparisons']['fixed_sa']['mean_paired_gain_percent'])
            with self.assertRaises(ValueError):analyze(d,[25,25])

    def test_32_step_bandit_preserves_update_events_and_doubles_steps(self):
        standard = self.arm('proposed_delta', update_epochs=16)
        candidate = self.arm('proposed_delta', update_epochs=32)
        self.assertEqual(standard['update_count'], candidate['update_count'])
        self.assertEqual(candidate['bandit_optimizer_steps_this_invocation'],
                         2 * standard['bandit_optimizer_steps_this_invocation'])
        self.assertEqual(standard['total_objective_evaluations'],
                         candidate['total_objective_evaluations'])

if __name__=='__main__':unittest.main()
