"""Run with python -m unittest discover -s src -p test_reconciliation.py."""
import contextlib
import io
import random
import unittest
from unittest.mock import patch
import numpy as np
from compute_model import channel_ranges, tile_work, compute_seconds, edge_bytes
from multi_chip_environment import MultiChipEnvironment
from multi_chip_topology import MultiChipTopology
import run_multi_chip as rm


class TimingTests(unittest.TestCase):
    def test_remainder_conserves_work(self):
        ops, kinds = tile_work(5, 7, 2, 3, 9, 2, 3)
        self.assertEqual(sum(o for o, k in zip(ops, kinds) if k == "vmm"), 5*7*2*3*9)
        self.assertEqual(sum(o for o, k in zip(ops, kinds) if k == "vva"), 7*2*3)
        self.assertEqual(channel_ranges(7, 3), [(0, 2), (2, 4), (4, 7)])

    def test_seconds_and_bytes(self):
        times = compute_seconds([128, 2], ["vmm", "vva"], vva_ops_per_cycle=1)
        np.testing.assert_allclose(times, [2.5e-9, 5e-9])
        graph = edge_bytes(np.array([[0, 16], [0, 0]]), ["vmm", "vva"])
        self.assertEqual(graph[0, 1], 64)
        env = MultiChipEnvironment(1, 1, 1, 2, on_chip_latency=1/64e9,
                                  task_graph=graph, num_tasks=2,
                                  compute_latency=times, timing_units="seconds")
        env.place(np.array([0, 1]))
        self.assertAlmostEqual(env.evaluate(), 5e-9, places=16)
        # A slower link makes the producer the bottleneck: 2.5ns + 64ns.
        env.topo.on_chip_latency = 1/1e9
        self.assertAlmostEqual(env.evaluate(), 66.5e-9, places=16)
        with self.assertRaisesRegex(ValueError, "Nonzero compute"):
            MultiChipEnvironment(1, 1, 1, 2, num_tasks=2,
                                 task_graph=graph, compute_latency=times)
        with self.assertRaisesRegex(ValueError, "capacity"):
            MultiChipEnvironment(1, 1, 1, 2, num_tasks=3)

    def test_torus_and_grid_ids(self):
        topo = MultiChipTopology(1, 1, 1, 4, topology="torus")
        self.assertEqual(topo.comm_cost(0, 3), 1)
        env = MultiChipEnvironment(2, 2, 2, 2, num_tasks=2,
                                  task_graph=np.zeros((2, 2)))
        mapper = rm.MultiChipCoreMapper(env)
        mapper._placement[:] = [2, 4]
        np.testing.assert_array_equal(mapper.get_placement(), [4, 2])

    def test_sa_exact_budget_and_unused_core(self):
        class Objective:
            num_tasks, total_cores = 1, 3
            def __init__(self): self.calls = 0
            def place(self, p): self.placement = p.copy()
            def evaluate(self):
                self.calls += 1
                return float(self.placement[0])
        obj = Objective()
        random.seed(3)
        with contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_sa(obj, n_iter=1000)
        self.assertEqual(obj.calls, 1001)  # initial plus budgeted neighbors
        self.assertEqual(result, 0)


@unittest.skipUnless(rm.HAS_TORCH, "PyTorch required")
class WorkloadTests(unittest.TestCase):
    def extract(self, model, shape, timing=True):
        import torch
        with patch.object(rm, "_build_model_and_input", return_value=(model.eval(), torch.zeros(shape), None)):
            with contextlib.redirect_stdout(io.StringIO()):
                return rm.extract_model_task_graph(channels_per_partition=3, return_work=timing)

    def test_pool_flatten_and_mac_conservation(self):
        from torch import nn
        model = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Linear(5*2*2, 7))
        g, n, labels, ops, kinds = self.extract(model, (1, 3, 6, 6))
        self.assertEqual(sum(o for o, k in zip(ops, kinds) if k == "vmm"),
                         3*5*3*3*4*4 + 20*7)
        src = [i for i, label in enumerate(labels) if label.startswith("0_conv_VVA")]
        dst = [i for i, label in enumerate(labels) if label.startswith("4_linear_VMM")]
        # Post-pool 20 elements broadcast to each of 3 output groups.
        self.assertEqual(g[np.ix_(src, dst)].sum(), 60)

    def test_residual_graph_and_timing_guard(self):
        import torch
        from torch import nn
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Conv2d(3, 3, 1)
                self.branch = nn.Conv2d(3, 3, 1)
                self.last = nn.Conv2d(3, 3, 1)
            def forward(self, x):
                y = self.first(x)
                return self.last(self.branch(y) + y)
        graph, _, labels = self.extract(Residual(), (1, 3, 4, 4), timing=False)
        first = labels.index("first_conv_VVA_m0")
        last = labels.index("last_conv_VMM_m0_n0")
        self.assertGreater(graph[first, last], 0)
        with self.assertRaisesRegex(ValueError, "residual"):
            self.extract(Residual(), (1, 3, 4, 4))

    def test_training_update_and_device(self):
        import torch
        torch.set_num_threads(2)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            agent = rm.DDPGAgent(8, 2, device=device)
            replay = rm.ReplayBuffer(64)
            for i in range(64):
                replay.add(np.ones(8, dtype=np.float32), np.zeros(2, dtype=np.float32),
                           0.2, np.zeros(8, dtype=np.float32), True)
            before = next(agent.actor.parameters()).detach().clone()
            agent.train(replay)
            after = next(agent.actor.parameters()).detach()
            self.assertFalse(torch.equal(before, after))
            self.assertTrue(torch.isfinite(after).all())
            self.assertEqual(after.device.type, device)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                rm.DDPGAgent(8, device="cuda")


if __name__ == "__main__":
    unittest.main()
