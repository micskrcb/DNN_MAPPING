"""Attribution metric and planted ranking checks for the offline diagnostic."""
import unittest
import numpy as np
from diagnose_bandit_ranking import metrics
try:
    import torch
except ImportError:
    torch = None

class RankingRepairTests(unittest.TestCase):
    def test_proxy_reference_and_ties(self):
        x=np.zeros((2,3,9)); x[:,:,7]=[3,2,1]
        g=np.array([[0.,1.,2.],[2.,1.,0.]])
        self.assertEqual(metrics(g,g,x)['oracle_capture_over_proxy'],1)
        self.assertEqual(metrics(x[:,:,7],g,x)['advantage_over_proxy_seconds'],0)
        self.assertIsNone(metrics(g,np.zeros_like(g),x)['oracle_capture_over_proxy'])

    @unittest.skipIf(torch is None, 'PyTorch unavailable')
    def test_pairwise_gradient_learns_planted_order(self):
        from run_multi_chip import ProposalBanditRegressor
        import torch.nn.functional as F
        torch.set_num_threads(2)
        torch.manual_seed(17)
        x=torch.randn(96,4,9); g=torch.zeros(96,8)
        model=ProposalBanditRegressor(8,9,hidden_dim=16,scoring='direct')
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        a,b=torch.triu_indices(4,4,offset=1)
        y=x[:,:,0]
        for _ in range(100):
            s=model(g[:64],x[:64]); delta=y[:64,a]-y[:64,b]
            loss=F.softplus(-delta.sign()*(s[:,a]-s[:,b])).mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            pred=model(g[64:],x[64:]).argmax(1)
        self.assertGreater((pred==y[64:].argmax(1)).float().mean().item(),.75)

    @unittest.skipIf(torch is None, 'PyTorch unavailable')
    def test_export_loads_as_frozen_proposer(self):
        import tempfile, contextlib, io
        from pathlib import Path
        from diagnose_bandit_ranking import inference_checkpoint
        from run_multi_chip import ProposalBanditRegressor, run_ppo_asa
        from multi_chip_environment import MultiChipEnvironment
        graph=np.zeros((4,4),dtype=np.float32)
        graph[0,1]=graph[1,2]=graph[2,3]=1
        model=ProposalBanditRegressor(8,9,hidden_dim=16,scoring='direct')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path=Path(directory)/'ranker.pt'
            torch.save(inference_checkpoint(model,4,100,feature_set='legacy9',focus=0,slack=0),path)
            meta={}
            cost=run_ppo_asa(MultiChipEnvironment(1,1,2,3,task_graph=graph,num_tasks=4),
                n_iter=24,candidate_count=4,hidden_dim=16,calibration_trials=4,
                device='cpu',seed=17,proposal_learner='bandit',bandit_control='checkpoint',
                disable_learning=True,annealing_schedule='fixed',state_encoding='relative_cost',
                load_checkpoint=str(path),metadata=meta)
            audit_meta={}
            audited_cost=run_ppo_asa(MultiChipEnvironment(1,1,2,3,task_graph=graph,num_tasks=4),
                n_iter=24,candidate_count=4,hidden_dim=16,calibration_trials=4,
                device='cpu',seed=17,proposal_learner='bandit',bandit_control='checkpoint',
                disable_learning=True,annealing_schedule='fixed',state_encoding='relative_cost',
                load_checkpoint=str(path),metadata=audit_meta,oracle_pools=4,oracle_stride=4,
                oracle_dataset_path=str(Path(directory)/'audit.npz'),
                oracle_diagnostics_path=str(Path(directory)/'audit.json'))
            with np.load(Path(directory)/'audit.npz') as audit:
                self.assertEqual(audit['policy_scores'].shape,(4,4))
                self.assertTrue(np.allclose(audit['selection_probabilities'].sum(1),1))
                self.assertEqual(audit['selected_indices'].shape,(4,))
            self.assertEqual(audited_cost,cost)
        self.assertEqual(meta['update_count'],0)
        self.assertEqual(meta['total_objective_evaluations'],25)
        self.assertEqual(meta['source_training_updates'],100)
        self.assertEqual(meta['model_initialization'],'loaded_checkpoint')

if __name__=='__main__': unittest.main()
