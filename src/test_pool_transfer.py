import copy
import json
from pathlib import Path
import tempfile
import unittest
from analyze_pool_transfer import analyze
from test_pool_context import ContextTests


class TransferTests(unittest.TestCase):
    def test_frozen_pool_has_no_updates(self):
        meta = ContextTests().arm('pool18', disable_learning=True)
        self.assertEqual(meta['update_count'], 0)
        self.assertEqual(meta['bandit_optimizer_steps_this_invocation'], 0)
        self.assertFalse(meta['learning_enabled'])

    def test_analyzer_rejects_hidden_config_change(self):
        source = Path(__file__).resolve().parents[1] / 'artifacts/neutral-weight-seed33-20261029'
        trained = json.loads((source/'seed33-pool_context.json').read_text())
        frozen = copy.deepcopy(trained)
        frozen['config']['ppo_asa_disable_learning'] = True
        frozen['algorithm_metadata'].update(learning_enabled=False, update_count=0,
                                            bandit_optimizer_steps_this_invocation=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, report in [('pool_context', trained), ('pool_frozen', frozen),
                                 ('fixed_sa', json.loads((source/'seed33-fixed_sa.json').read_text()))]:
                (root/f'seed33-{name}.json').write_text(json.dumps(report))
            result = analyze(root, 33)
            self.assertEqual(result['gain_over_matched_frozen_percent'], 0)
            frozen['config']['ppo_learning_rate'] *= 2
            (root/'seed33-pool_frozen.json').write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):
                analyze(root, 33)
