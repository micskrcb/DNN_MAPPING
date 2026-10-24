import json
import shutil
import tempfile
import unittest
from pathlib import Path
from analyze_learning_intensity import analyze

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/learning-intensity-seed21-20261024'

class IntensityTests(unittest.TestCase):
    def test_received_results_and_runtime(self):
        r = analyze(SOURCE, 21)['workloads']
        self.assertAlmostEqual(r['vgg16']['gain_over_fixed_sa_percent']['train_more'], 3.111142898)
        self.assertAlmostEqual(r['alexnet']['gain_over_reference_percent']['train_more'], 2.142857143)
        self.assertTrue(all(s['candidate_beats_both_controls'] for s in r.values()))
        self.assertGreater(r['vgg16']['training_runtime_change_percent'], 0)
        self.assertLess(r['alexnet']['training_runtime_change_percent'], 0)

    def test_rejects_changed_budget_and_step_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'data';shutil.copytree(SOURCE, root)
            p = root / 'vgg16-fc/seed21-train_more.json'
            original = json.loads(p.read_text())
            r = json.loads(p.read_text());r['complete_placement_evaluations'] -= 1;p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):analyze(root, 21)
            r = original;r['algorithm_metadata']['bandit_optimizer_steps_this_invocation'] -= 1;p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):analyze(root, 21)

    def test_rejects_learning_rate_confound(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'data';shutil.copytree(SOURCE, root)
            p = root / 'vgg16-fc/seed21-train_more.json'
            r = json.loads(p.read_text());r['config']['ppo_learning_rate'] *= 2;p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'beyond training intensity'):analyze(root, 21)

    def test_rejects_wrong_seed(self):
        with self.assertRaises(FileNotFoundError):analyze(SOURCE, 22)

if __name__ == '__main__':unittest.main()
