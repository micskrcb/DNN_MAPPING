"""Candidate-pool ablation preserves the existing calibration and default path."""
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import run_multi_chip as rm
from multi_chip_environment import MultiChipEnvironment

class PoolTests(unittest.TestCase):
    def run_arm(self,**extra):
        graph=np.array([[0,3,2,0],[0,0,1,4],[0,0,0,2],[0,0,0,0]],dtype=float)
        meta={}
        with contextlib.redirect_stdout(io.StringIO()):
            cost=rm.run_ppo_asa(MultiChipEnvironment(1,1,3,3,task_graph=graph,num_tasks=4),
                n_iter=24,calibration_trials=4,candidate_count=4,hidden_dim=16,
                device='cpu',seed=3,proposal_learner='bandit',reward_mode='proposed_delta',
                annealing_schedule='fixed',focus_fraction=.5,bandit_warmup=4,
                rollout_steps=4,minibatch_size=4,metadata=meta,**extra)
        return cost,meta

    def test_default_unchanged_and_calibration_preserved(self):
        torch.set_num_threads(2)
        a,ma=self.run_arm()
        b,mb=self.run_arm(pool_focus_fraction=.5)
        self.assertEqual(a,b)
        self.assertEqual(ma['final_update_metrics'],mb['final_update_metrics'])
        with patch.object(rm,'_placement_neighbor',wraps=rm._placement_neighbor) as neighbor:
            _,mc=self.run_arm(pool_focus_fraction=0)
        self.assertTrue(all(c.kwargs.get('anchor_tasks') is None for c in neighbor.call_args_list[4:]))
        self.assertEqual(ma['initial_temperature'],mc['initial_temperature'])
        self.assertEqual(mc['total_objective_evaluations'],25)
        self.assertEqual(mc['pool_focus_fraction'],0)
        self.assertEqual(mc['calibration_focus_fraction'],.5)
        self.assertGreater(mc['update_count'],0)

    def test_mixed_sizes_preserve_calibration_and_budget(self):
        a,ma=self.run_arm(fixed_perturb_fraction=1,pool_focus_fraction=0)
        b,mb=self.run_arm(fixed_perturb_fraction=1,pool_focus_fraction=0,pool_small_move_fraction=0)
        self.assertEqual(a,b)
        self.assertEqual(ma['final_update_metrics'],mb['final_update_metrics'])
        with patch.object(rm,'_placement_neighbor',wraps=rm._placement_neighbor) as neighbor:
            _,mc=self.run_arm(fixed_perturb_fraction=1,pool_focus_fraction=0,pool_small_move_fraction=.5)
        sizes=[c.args[1] for c in neighbor.call_args_list]
        self.assertEqual(sizes[:4],[4]*4)
        self.assertEqual(sizes[4:],[2,2,4,4]*20)
        self.assertEqual(ma['initial_temperature'],mc['initial_temperature'])
        self.assertEqual(mc['total_objective_evaluations'],25)
        self.assertEqual(mc['small_move_candidates_per_pool'],2)
        self.assertGreater(mc['update_count'],0)
        self.assertTrue(0<=mc['selected_small_move_proposals']<=20)

    def test_checkpoint_rejects_changed_move_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'model.pt')
            self.run_arm(pool_small_move_fraction=.5,save_checkpoint=path)
            with self.assertRaisesRegex(ValueError,'small-move fraction'):
                self.run_arm(load_checkpoint=path,disable_learning=True,bandit_control='checkpoint')

    def test_checkpoint_rejects_changed_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'model.pt')
            self.run_arm(pool_focus_fraction=0,save_checkpoint=path)
            with self.assertRaisesRegex(ValueError,'pool focus fraction'):
                self.run_arm(pool_focus_fraction=.5,load_checkpoint=path,disable_learning=True,bandit_control='checkpoint')

if __name__=='__main__':unittest.main()
