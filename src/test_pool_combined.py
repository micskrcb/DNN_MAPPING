"""Analyzer contract checks; synthetic 32-step report is not performance evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from analyze_pool_combined import analyze

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/pool-context-seed29-20261028'

class CombinedTests(unittest.TestCase):
    def fixture(self, root):
        for method in ('train_more', 'pool_context'):
            (root/f'seed29-{method}.json').write_bytes((SOURCE/f'seed29-{method}.json').read_bytes())
        candidate = json.loads((root/'seed29-pool_context.json').read_text())
        candidate['config']['ppo_update_epochs'] = 32
        candidate['algorithm_metadata']['bandit_update_epochs'] = 32
        candidate['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] *= 2
        (root/'seed29-pool_context32.json').write_text(json.dumps(candidate))
        return candidate

    def test_percentage_and_confounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); original = self.fixture(root)
            result = analyze(root,29)
            self.assertEqual(result['combined_gain_percent']['pool_context'], 0)
            self.assertFalse(result['beats_both'])
            self.assertAlmostEqual(result['combined_gain_percent']['train_more'], .1079953822664)
            bad = copy.deepcopy(original)
            bad['config']['ppo_learning_rate'] *= 2
            (root/'seed29-pool_context32.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError): analyze(root,29)
            bad = copy.deepcopy(original)
            bad['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] -= 1
            (root/'seed29-pool_context32.json').write_text(json.dumps(bad))
            with self.assertRaises((ValueError, AssertionError)): analyze(root,29)

if __name__ == '__main__': unittest.main()
