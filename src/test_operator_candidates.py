import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import run_multi_chip as rm
from operator_candidates import operator_neighbor, append_operator_features
from structured_moves import axis_ranks
from test_pool_context import ContextTests as _ContextTests


class OperatorTests(unittest.TestCase):
    def test_legality_and_exact_changed_set(self):
        placement = np.array([1, 3, 5, 7])
        groups = [np.array([0, 1, 2])]
        ranks = [np.arange(10)]
        for free in (True, False):
            pool = rm._FreeCorePool(np.arange(10) if free else placement, placement)
            counts = [0]*4
            for i in range(16):
                candidate, changed, relocation, kind = operator_neighbor(
                    placement, 3, pool, random.Random(i), i, 16, groups, ranks, rm._placement_neighbor)
                self.assertEqual(len(set(candidate)), len(candidate))
                np.testing.assert_array_equal(np.sort(changed), np.flatnonzero(candidate != placement))
                self.assertTrue(set(candidate).issubset(set(range(10)) if free else set(placement)))
                if relocation is not None:
                    used, freed = relocation
                    self.assertIn(used, pool.positions)
                    self.assertNotIn(freed, candidate)
                counts[kind] += 1
            if free:
                self.assertEqual(counts, [8, 3, 3, 2])
            else:
                self.assertEqual(counts[1], 0)
        x = append_operator_features(np.zeros((4,18)), [0,1,2,3])
        self.assertEqual(x.shape, (4,22))
        np.testing.assert_array_equal(x[:,-4:], np.eye(4))

    def test_training_frozen_and_checkpoint_contract(self):
        harness = _ContextTests()
        a = harness.arm('operator22')
        b = harness.arm('operator22', disable_learning=True)
        self.assertGreater(a['update_count'], 0)
        self.assertEqual(b['bandit_optimizer_steps_this_invocation'], 0)
        self.assertEqual(a['total_objective_evaluations'], b['total_objective_evaluations'])
        for m in (a,b):
            self.assertEqual(sum(m['operator_selected'].values()), m['policy_steps'])
            self.assertEqual(len(m['candidate_feature_names']), 22)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = str(Path(d)/'model.pt')
            harness.arm('operator22', save_checkpoint=checkpoint)
            with self.assertRaises(ValueError):
                harness.arm('pool18', load_checkpoint=checkpoint)
