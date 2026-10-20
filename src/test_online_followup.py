"""Recompute the real restoration result and reject invalid comparisons."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze_online_followup import summarize,METHODS

SOURCE=Path(__file__).resolve().parents[1]/'artifacts/online-restoration-20261019'

class FollowupTests(unittest.TestCase):
    def test_real_restoration_numbers(self):
        result=summarize(SOURCE,[3])
        self.assertAlmostEqual(result['aggregate']['proxy']['mean_paired_gain_percent'],2.9497591674577217)
        self.assertEqual(result['aggregate']['uniform']['wins'],1)

    def test_rejects_mismatched_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for method in METHODS:
                r=json.loads((SOURCE/f'seed3-{method}.json').read_text())
                if method=='proxy':r['complete_placement_evaluations']-=1
                (root/f'seed3-{method}.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(AssertionError,'Unequal evaluation'):
                summarize(root,[3])

if __name__=='__main__':unittest.main()
