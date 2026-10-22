import random
import unittest
from run_multi_chip import sample_bandit_replay_indices

class ReplayTests(unittest.TestCase):
    def test_uniform_path_preserves_rng(self):
        a=random.Random(17);b=random.Random(17)
        self.assertEqual(sample_bandit_replay_indices(a,500,64),b.sample(range(500),64))
        self.assertEqual(a.random(),b.random())

    def test_mixture_is_unique_recent_and_reproducible(self):
        a=sample_bandit_replay_indices(random.Random(17),50000,256,.75,10000)
        b=sample_bandit_replay_indices(random.Random(17),50000,256,.75,10000)
        self.assertEqual(a,b);self.assertEqual(len(set(a)),256)
        self.assertGreaterEqual(sum(i>=40000 for i in a),192)
        self.assertTrue(all(0<=i<50000 for i in a))
        self.assertEqual(len(sample_bandit_replay_indices(random.Random(1),4,4,1,2)),4)

if __name__=='__main__':unittest.main()
