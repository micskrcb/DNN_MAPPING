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

if __name__=='__main__': unittest.main()
