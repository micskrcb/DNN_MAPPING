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

    def test_sequential_baseline_uses_chip_major_core_order(self):
        env = MultiChipEnvironment(num_chips_x=2, num_chips_y=1,
                                   rows_per_chip=2, cols_per_chip=2,
                                   num_tasks=5)
        cost = rm.run_sequential(env)
        self.assertTrue(np.isfinite(cost))
        np.testing.assert_array_equal(env.placement[:5], np.arange(5, dtype=np.int32))

    def test_potential_shaping_preserves_discounted_return(self):
        gamma = 0.98
        graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
        actions = [np.array([-1.0, -1.0]), np.array([1.0, -1.0]),
                   np.array([1.0, 1.0])]
        returns = []
        for mode in ("sparse", "potential"):
            env = MultiChipEnvironment(1, 1, 2, 2, task_graph=graph, num_tasks=3)
            mapper = rm.MultiChipCoreMapper(env, baseline_latency=10.0, batch_z=1,
                                             reward_mode=mode, shaping_gamma=gamma)
            mapper.reset()
            rewards = [mapper.step(action)[0] for action in actions]
            returns.append(sum((gamma ** index) * reward
                               for index, reward in enumerate(rewards)))
        self.assertAlmostEqual(returns[0], returns[1], places=10)


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

    def test_cnn_agent_spatial_state_and_update(self):
        import torch
        torch.set_num_threads(2)
        rows, cols, tasks, action_dim = 8, 8, 5, 4
        state_dim = rows * cols + tasks
        state = np.zeros(state_dim, dtype=np.float32)
        state[0] = 0.2
        for architecture in ("cnn", "paper_cnn"):
            agent = rm.DDPGAgent(state_dim, action_dim, device="cpu", agent_arch=architecture,
                                 rows=rows, cols=cols, num_tasks=tasks)
            action = agent.select_action(state, explore=False)
            self.assertEqual(action.shape, (action_dim,))
            replay = rm.ReplayBuffer(64)
            for _ in range(64):
                replay.add(state, action, 0.2, state, True)
            before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
            loss = agent.train(replay)
            self.assertIsNotNone(loss)
            self.assertIn("actor_loss", loss)
            self.assertTrue(any(not torch.equal(old, new.detach())
                                for old, new in zip(before, agent.actor.parameters())))
        paper = rm.DDPGAgent(state_dim, action_dim, device="cpu", agent_arch="paper_cnn",
                             rows=rows, cols=cols, num_tasks=tasks)
        self.assertEqual(paper.actor.encoder.net[0].out_channels, 32)
        self.assertEqual(paper.actor.encoder.net[4].out_channels, 64)
        self.assertEqual(paper.actor.fc1.out_features, 600)
        self.assertEqual(paper.actor.fc2.out_features, 300)
        self.assertEqual(paper.critic.fc2.in_features, 600 + action_dim)
        with self.assertRaisesRegex(ValueError, "at least 4x4"):
            rm.DDPGAgent(9, 2, device="cpu", agent_arch="cnn", rows=3, cols=3, num_tasks=0)


if __name__ == "__main__":
    unittest.main()
