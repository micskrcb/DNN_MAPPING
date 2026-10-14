"""Test runner invoking all functional smoke tests and reconciliation unit tests."""
import os
import sys
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from test_multi_chip import (
    test_topology,
    test_environment_reset_step,
    test_pipeline_stages_and_latency,
    test_compute_latency_estimate,
    test_collision_resolution,
    test_batched_actions,
    test_cnn_workload_partitioning,
    test_sa_improves_random,
)
from test_reconciliation import TimingTests, WorkloadTests


def run_smoke_tests():
    print("--- Running Functional Smoke Tests ---")
    test_topology()
    test_environment_reset_step()
    test_pipeline_stages_and_latency()
    test_compute_latency_estimate()
    test_collision_resolution()
    test_batched_actions()
    test_cnn_workload_partitioning()
    test_sa_improves_random()
    print("All functional smoke tests passed.\n")


def run_unit_tests():
    print("--- Running Reconciliation & Regression Unit Tests ---")
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TimingTests))
    suite.addTests(loader.loadTestsFromTestCase(WorkloadTests))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    run_smoke_tests()
    success = run_unit_tests()
    sys.exit(0 if success else 1)
