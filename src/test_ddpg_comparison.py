"""Synthetic contract fixtures; these are not experimental results."""
import json,tempfile,unittest
from pathlib import Path
from analyze_ddpg_comparison import summarize

class ComparisonTests(unittest.TestCase):
    def reports(self,root):
        common={'seed':17,'model':'vgg16','train_every':1,'batch_z':3}
        for name in ('ddpg','exploit','fixed_sa','random'):
            meta={}
            if name=='ddpg':
                meta=dict(agent_arch='paper_cnn',reward_mode='sparse',start_episode=0,
                    retain_deterministic_candidates=True,invocation_gradient_updates=10,
                    search_evaluations_including_baseline=1031,baseline_search_evaluations=1000,
                    training_candidate_evaluations=30,deterministic_candidate_evaluations=1,
                    best_candidate_source='random_baseline')
            if name=='exploit':
                meta=dict(proposal_learner='bandit',learning_enabled=True,bandit_epsilon_final=.02,
                    pool_focus_fraction=0,bandit_feature_set='legacy9',annealing_schedule='fixed',
                    bandit_pool_supervision_stride=0,update_count=2,bandit_ranking_weight=0,bandit_recent_fraction=0)
            r=dict(config=common,algorithm_metadata=meta,best_cost=1e-6,seconds_elapsed=1,
                   complete_placement_evaluations=30 if name=='ddpg' else 1031)
            (root/f'{name}.json').write_text(json.dumps(r))
    def test_inclusive_budget_not_training_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.reports(p)
            self.assertEqual(summarize(p)['inclusive_search_evaluations_per_method'],1031)
            self.assertFalse(summarize(p)['published_five_percent_claim_supported'])
    def test_reject_shorter_comparator(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.reports(p)
            f=p/'random.json';r=json.loads(f.read_text());r['complete_placement_evaluations']=31;f.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):summarize(p)

if __name__=='__main__':unittest.main()
