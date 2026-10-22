"""Protocol checks using archived real reports, never fabricated outcome evidence."""
import json,shutil,tempfile,unittest
from pathlib import Path
from analyze_online_exploration import summarize

class ExplorationTests(unittest.TestCase):
    def test_real_screen_summary(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/online-exploration-seed12-20261023'
        result=summarize(root,[12])
        self.assertTrue(result['pairs'][0]['directional_screen_positive'])
        self.assertAlmostEqual(result['aggregate']['pool_online']['mean_paired_gain_percent'],.8342786051447115)
        with self.assertRaises(AssertionError):summarize(root,[12,13])

    def test_reject_changed_control(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/online-exploration-seed12-20261023'
        with tempfile.TemporaryDirectory() as d:
            for p in root.glob('seed12-*.json'):shutil.copy(p,d)
            p=Path(d)/'seed12-pool_proxy.json';r=json.loads(p.read_text())
            r['algorithm_metadata']['bandit_control']='uniform';p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):summarize(Path(d),[12])

if __name__=='__main__':unittest.main()
