"""Validate actual seed8 results, aggregation, and comparison guards."""
import json
import tempfile
from pathlib import Path
import unittest
from analyze_online_pool import summarize,METHODS

SOURCE=Path(__file__).resolve().parents[1]/'artifacts/online-pool-seed8-20261020'

class PoolSummaryTests(unittest.TestCase):
    def test_recomputes_real_result(self):
        result=summarize(SOURCE,[8])
        self.assertAlmostEqual(result['aggregate']['original']['mean_paired_gain_percent'],.853707269226589)
        self.assertAlmostEqual(result['aggregate']['fixed_sa']['mean_paired_gain_percent'],1.5440115440115587)
        self.assertTrue(result['pairs'][0]['directional_screen_positive'])

    def test_rejects_duplicate_seed(self):
        with self.assertRaises(ValueError):summarize(SOURCE,[8,8])

    def test_aggregate_and_workload_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for seed in (9,10):
                for method in METHODS:
                    r=json.loads((SOURCE/f'seed8-{method}.json').read_text());r['config']['seed']=seed
                    (root/f'seed{seed}-{method}.json').write_text(json.dumps(r))
            result=summarize(root,[9,10])
            self.assertEqual(result['aggregate']['original']['wins'],2)
            path=root/'seed10-original.json';r=json.loads(path.read_text());r['config']['model']='alexnet';path.write_text(json.dumps(r))
            with self.assertRaisesRegex(AssertionError,'Workload'):summarize(root,[9,10])

if __name__=='__main__':unittest.main()
