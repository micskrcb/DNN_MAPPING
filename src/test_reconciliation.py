"""Run with python -m unittest discover -s src -p test_reconciliation.py."""
import contextlib
import io
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from compute_model import (channel_ranges, tile_work, compute_seconds, edge_bytes,
                           paper_partition_grids)
from multi_chip_environment import MultiChipEnvironment
from multi_chip_topology import MultiChipTopology
import run_multi_chip as rm


class TimingTests(unittest.TestCase):
    def test_masked_mapper_exposes_only_unused_allowed_cores(self):
        graph = np.zeros((3, 3), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = 1.0
        env = MultiChipEnvironment(
            1, 1, 1, 5, task_graph=graph, num_tasks=3,
            allowed_cores=np.array([0, 2, 4], dtype=np.int32))
        mapper = rm.MultiChipCoreMapper(env, batch_z=1, compact_state=True)
        mapper.reset()
        np.testing.assert_array_equal(mapper.legal_action_mask(), [True, True, True])
        _, done, _, _ = mapper.step_masked(1)
        self.assertFalse(done)
        np.testing.assert_array_equal(mapper.legal_action_mask(), [True, False, True])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            mapper.step_masked(1)
        mapper.step_masked(0)
        _, done, _, _ = mapper.step_masked(2)
        self.assertTrue(done)
        self.assertEqual(mapper.collision_repairs, 0)
        self.assertEqual(len(np.unique(mapper.get_placement())), 3)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_masked_ppo_has_zero_illegal_probability_and_trains(self):
        import torch
        torch.set_num_threads(2)
        model = rm.MaskedPPOActorCritic(6, 4, hidden_dim=16)
        states = torch.zeros((2, 6))
        masks = torch.tensor([[True, False, True, False],
                              [False, True, False, True]])
        distribution, values = model.distribution_and_value(states, masks)
        self.assertTrue(torch.equal(distribution.probs[~masks], torch.zeros(4)))
        self.assertTrue(torch.isfinite(values).all())
        for _ in range(20):
            actions, _, _, _ = model.action_and_value(states, masks)
            self.assertTrue(bool(masks.gather(1, actions[:, None]).all()))

        graph = np.zeros((3, 3), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = 1.0
        env = MultiChipEnvironment(1, 1, 1, 4, task_graph=graph, num_tasks=3)
        metadata = {}
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_masked_ppo(
                env, n_episodes=4, baseline_trials=2, device="cpu",
                rollout_episodes=2, update_epochs=1, minibatch_size=8,
                hidden_dim=16, diagnostics_every=2, checkpoint_every=2,
                diagnostics_path=f"{directory}/diagnostics.jsonl",
                save_checkpoint=f"{directory}/checkpoint.pt", seed=11,
                run_metadata=metadata)
        self.assertTrue(np.isfinite(result))
        self.assertEqual(metadata["method"], "masked_categorical_ppo")
        self.assertEqual(metadata["collision_repairs"], 0)
        self.assertGreater(metadata["update_count"], 0)
        self.assertTrue(np.isfinite(metadata["initial_deterministic_cost"]))

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_ppo_asa_starts_uniform_and_respects_objective_budget(self):
        import torch
        torch.set_num_threads(2)
        model = rm.ProposalPPOActorCritic(8, 8, hidden_dim=16)
        distribution, values = model.distribution_and_value(
            torch.zeros((2, 8)), torch.randn((2, 4, 8)))
        torch.testing.assert_close(
            distribution.probs, torch.full((2, 4), 0.25))
        self.assertTrue(torch.isfinite(values).all())

        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        environments = [
            MultiChipEnvironment(1, 1, 2, 3, task_graph=graph, num_tasks=4)
            for _ in range(2)
        ]
        trained, control = {}, {}
        with tempfile.TemporaryDirectory() as directory:
            trained_cost = rm.run_ppo_asa(
                environments[0], n_iter=40, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=7,
                focus_fraction=0.5, restart_interval=10,
                diagnostics_path=f"{directory}/trained.jsonl",
                save_checkpoint=f"{directory}/trained.pt", metadata=trained)
            control_cost = rm.run_ppo_asa(
                environments[1], n_iter=40, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=7,
                focus_fraction=0.5, restart_interval=10,
                disable_learning=True, metadata=control)
        self.assertTrue(np.isfinite(trained_cost))
        self.assertTrue(np.isfinite(control_cost))
        self.assertEqual(trained["initial_cost"], control["initial_cost"])
        self.assertEqual(trained["candidate_evaluations"], 40)
        self.assertEqual(trained["initial_placement_evaluations"], 4)
        self.assertEqual(trained["total_objective_evaluations"], 44)
        self.assertEqual(trained["independent_chains"], 4)
        self.assertEqual(trained["proposal_focus"], "mixed_bottleneck_0.500")
        self.assertEqual(trained["policy_steps"], 36)
        self.assertGreater(trained["update_count"], 0)
        self.assertEqual(control["update_count"], 0)
        self.assertFalse(control["learning_enabled"])

    def test_focused_neighbor_always_moves_an_anchor_task(self):
        placement = np.arange(8, dtype=np.int32)
        free_pool = rm._FreeCorePool(placement, placement)
        rng = random.Random(19)
        anchors = np.asarray([1, 4], dtype=np.int64)
        for _ in range(50):
            candidate, changed, relocation = rm._placement_neighbor(
                placement, 3, free_pool, rng=rng, anchor_tasks=anchors)
            self.assertIsNone(relocation)
            self.assertTrue(set(changed.tolist()) & set(anchors.tolist()))
            self.assertEqual(len(np.unique(candidate)), len(candidate))

    def test_remainder_conserves_work(self):
        ops, kinds = tile_work(5, 7, 2, 3, 9, 2, 3)
        self.assertEqual(sum(o for o, k in zip(ops, kinds) if k == "vmm"), 5*7*2*3*9)
        self.assertEqual(sum(o for o, k in zip(ops, kinds) if k == "vva"), 7*2*3)
        self.assertEqual(channel_ranges(7, 3), [(0, 2), (2, 4), (4, 7)])
        one_group_ops, one_group_kinds = tile_work(4, 8, 2, 2, 1, 1, 1)
        self.assertEqual(one_group_ops[one_group_kinds.index("vva")], 8 * 2 * 2)

    def test_paper_partitions_balance_vmm_and_vva_without_changing_count(self):
        layers = [
            {"name": "conv1", "cin": 3, "cout": 64, "height": 55,
             "width": 55, "kernel_area": 121},
            {"name": "conv2", "cin": 64, "cout": 192, "height": 27,
             "width": 27, "kernel_area": 25},
            {"name": "conv3", "cin": 192, "cout": 384, "height": 13,
             "width": 13, "kernel_area": 9},
            {"name": "conv4", "cin": 384, "cout": 256, "height": 13,
             "width": 13, "kernel_area": 9},
            {"name": "conv5", "cin": 256, "cout": 256, "height": 13,
             "width": 13, "kernel_area": 9},
        ]
        grids = paper_partition_grids(layers, 183)
        self.assertEqual(sum(m * (n + 1) for m, n in grids), 183)
        # The old count-only tie-break produced (M=1,N=61), leaving one VVA
        # task as a placement-independent 5.2488-ms bottleneck.
        self.assertNotEqual(grids[1], (1, 61))
        self.assertGreater(grids[1][0], 1)

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

    def test_paper_xy_gateway_route_and_contention(self):
        topo = MultiChipTopology(2, 1, 2, 2, on_chip_latency=1.0,
                                 off_chip_latency=5.0,
                                 routing_model="paper_xy")
        route = topo.xy_route(0, 4)
        self.assertEqual([kind for kind, _ in route], ["on", "off", "on"])
        self.assertEqual(topo.comm_cost(0, 4), 7.0)

        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 3] = graph[1, 3] = graph[2, 3] = 10.0
        env = MultiChipEnvironment(1, 1, 1, 4, on_chip_latency=1.0,
                                   task_graph=graph, num_tasks=4,
                                   routing_model="paper_xy",
                                   pipeline_model="xy_contention")
        env.place(np.array([0, 1, 2, 3], dtype=np.int32))
        self.assertEqual(env.evaluate(), 30.0)
        diagnostics = env.routing_diagnostics()
        self.assertEqual(diagnostics["communicating_edges"], 3)
        self.assertEqual(diagnostics["mean_hops_per_edge"], 2.0)
        self.assertEqual(diagnostics["traffic_weighted_mean_hops"], 2.0)
        self.assertEqual(diagnostics["on_chip_links"]["max_load"], 30.0)
        self.assertEqual(diagnostics["off_chip_links"]["used_links"], 0)
        breakdown = env.pipeline_breakdown()
        self.assertEqual(breakdown["total"], env.evaluate())
        self.assertEqual(breakdown["compute"], 0.0)
        self.assertEqual(breakdown["communication"], 30.0)

    def test_objective_sensitivity_reports_range_without_changing_rng(self):
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 3] = graph[1, 3] = graph[2, 3] = 10.0
        env = MultiChipEnvironment(1, 1, 1, 5, on_chip_latency=1.0,
                                   task_graph=graph, num_tasks=4)
        random.seed(19)
        state = random.getstate()
        result = rm.measure_objective_sensitivity(env, n_trials=20)
        after = random.random()
        random.setstate(state)
        expected = random.random()
        self.assertEqual(after, expected)
        self.assertEqual(result["trials"], 20)
        self.assertGreater(result["relative_span"], 0)
        self.assertIn("compute_fraction", result["best_sample_breakdown"])

    def test_sa_exact_budget_and_unused_core(self):
        class Objective:
            num_tasks, total_cores = 1, 3
            allowed_cores = np.arange(3, dtype=np.int32)
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

    def test_adaptive_sa_exact_budget_and_warm_start(self):
        class Objective:
            num_tasks, total_cores = 1, 4
            allowed_cores = np.arange(4, dtype=np.int32)
            def __init__(self): self.calls = 0
            def place(self, placement): self.placement = placement.copy()
            def evaluate(self):
                self.calls += 1
                return float(self.placement[0])
        obj = Objective()
        metadata = {}
        random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_adaptive_sa(
                obj, n_iter=250, initial_placement=np.array([3]),
                adapt_window=20, stall_windows=2, calibration_trials=8,
                metadata=metadata)
        self.assertEqual(obj.calls, 251)  # warm-start value plus candidates
        self.assertEqual(metadata["candidate_evaluations"], 250)
        self.assertEqual(metadata["initialization"], "supplied")
        self.assertLessEqual(result, metadata["initial_cost"])
        self.assertEqual(result, 0)

    def test_incremental_pipeline_evaluator_matches_full_objective(self):
        graph = np.zeros((6, 6), dtype=np.float32)
        graph[0, 2], graph[1, 2] = 2.0, 3.0
        graph[2, 3], graph[2, 4] = 4.0, 5.0
        graph[3, 5], graph[4, 5] = 6.0, 7.0
        for pipeline_model, routing_model in (("task_sum", "legacy_distance"),
                                               ("xy_contention", "paper_xy")):
            env = MultiChipEnvironment(
                2, 1, 2, 2, on_chip_latency=1.0, off_chip_latency=5.0,
                task_graph=graph, num_tasks=6, routing_model=routing_model,
                pipeline_model=pipeline_model)
            placement = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            env.place(placement)
            evaluator = rm.IncrementalPipelineEvaluator(env)
            for first, second in ((0, 5), (1, 3), (2, 4)):
                candidate = placement.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                env.place(candidate)
                incremental = evaluator.candidate_cost([first, second])
                full = env.evaluate()
                self.assertAlmostEqual(incremental, full, places=12)
                evaluator.accept()
                placement = candidate

    def test_hybrid_warm_starts_asa_from_ddpg_best(self):
        class Objective:
            placement = np.array([9, 9], dtype=np.int32)
        objective = Objective()
        seen = {}
        def fake_ddpg(env, n_episodes, **options):
            self.assertEqual(n_episodes, 12)
            env.placement = np.array([2, 3], dtype=np.int32)
            options["run_metadata"]["phase"] = "ddpg"
            options["run_metadata"]["total_candidate_evaluations"] = 18
            return 8.0
        def fake_asa(env, n_iter, initial_placement, **options):
            self.assertEqual(n_iter, 4)
            seen["initial"] = initial_placement.copy()
            options["metadata"]["phase"] = "asa"
            env.placement = np.array([1, 3], dtype=np.int32)
            return 7.0
        metadata = {}
        with patch.object(rm, "run_ddpg", side_effect=fake_ddpg), \
             patch.object(rm, "run_adaptive_sa", side_effect=fake_asa):
            result = rm.run_ddpg_asa(objective, 12, 4, metadata=metadata)
        self.assertEqual(result, 7.0)
        np.testing.assert_array_equal(seen["initial"], [2, 3])
        self.assertEqual(metadata["ddpg_complete_placement_evaluations"], 12)
        self.assertEqual(metadata["ddpg_total_candidate_evaluations"], 18)
        self.assertEqual(metadata["combined_candidate_evaluations"], 22)

    def test_sequential_baseline_uses_chip_major_core_order(self):
        env = MultiChipEnvironment(num_chips_x=2, num_chips_y=1,
                                   rows_per_chip=2, cols_per_chip=2,
                                   num_tasks=5)
        cost = rm.run_sequential(env)
        self.assertTrue(np.isfinite(cost))
        np.testing.assert_array_equal(env.placement[:5], np.arange(5, dtype=np.int32))

    def test_masked_region_restricts_baselines_and_mapper(self):
        allowed = np.array([4, 5, 6], dtype=np.int32)
        env = MultiChipEnvironment(num_chips_x=2, num_chips_y=1,
                                   rows_per_chip=2, cols_per_chip=2,
                                   num_tasks=2, allowed_cores=allowed)
        rm.run_sequential(env)
        np.testing.assert_array_equal(env.placement, [4, 5])
        mapper = rm.MultiChipCoreMapper(env, baseline_latency=10.0, batch_z=1)
        state = mapper.reset()
        self.assertEqual(np.count_nonzero(state[:8] == -1), 5)
        mapper.step(np.array([-1.0, -1.0]))
        self.assertEqual(mapper.get_placement()[0], 4)
        self.assertEqual(mapper.collision_repairs, 0)

    def test_guided_legal_actions_are_executed_without_repairs(self):
        allowed = np.array([4, 5, 6, 7], dtype=np.int32)
        env = MultiChipEnvironment(num_chips_x=2, num_chips_y=1,
                                   rows_per_chip=2, cols_per_chip=2,
                                   num_tasks=3, allowed_cores=allowed)
        mapper = rm.MultiChipCoreMapper(env, baseline_latency=10.0, batch_z=2,
                                         compact_state=True)
        state = mapper.reset()
        self.assertEqual((mapper.state_rows, mapper.state_cols), (2, 2))
        self.assertEqual(len(state), 4 + env.num_tasks)
        candidates, core_batches = mapper.legal_action_candidates(
            np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32), top_k=4)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(len(set(core_batches[0].tolist())), 2)
        _, done, _, _ = mapper.step_legal(core_batches[0])
        self.assertFalse(done)
        self.assertEqual(mapper.collision_repairs, 0)
        candidates, core_batches = mapper.legal_action_candidates(
            np.zeros(4, dtype=np.float32), top_k=4)
        _, done, _, _ = mapper.step_legal(core_batches[0])
        self.assertTrue(done)
        self.assertEqual(mapper.collision_repairs, 0)
        self.assertEqual(len(set(mapper.get_placement().tolist())), 3)

    def test_potential_shaping_preserves_discounted_return(self):
        gamma = 0.98
        reward_scale = 400.0
        graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
        actions = [np.array([-1.0, -1.0]), np.array([1.0, -1.0]),
                   np.array([1.0, 1.0])]
        returns = []
        potential_rewards = None
        for mode in ("sparse", "potential"):
            env = MultiChipEnvironment(1, 1, 2, 2, task_graph=graph, num_tasks=3)
            mapper = rm.MultiChipCoreMapper(env, baseline_latency=10.0, batch_z=1,
                                             reward_mode=mode, shaping_gamma=gamma,
                                             reward_scale=reward_scale)
            mapper.reset()
            rewards = [mapper.step(action)[0] for action in actions]
            if mode == "potential":
                potential_rewards = rewards
            returns.append(sum((gamma ** index) * reward
                               for index, reward in enumerate(rewards)))
        self.assertAlmostEqual(returns[0], returns[1], places=10)
        # Shaping is normalized by the fixed baseline. Without this guard,
        # cycle-scaled partial rewards are tens to hundreds of times larger
        # than the normalized terminal learning signal.
        self.assertLess(max(abs(value) for value in potential_rewards[:-1]), 2.0)

    def test_reward_scale_changes_magnitude_not_placement_objective(self):
        graph = np.array([[0, 1], [0, 0]], dtype=np.float32)
        action_sequence = [np.array([-1.0, -1.0]), np.array([1.0, 1.0])]
        rewards = []
        for scale in (1.0, 400.0):
            env = MultiChipEnvironment(1, 1, 1, 2, task_graph=graph, num_tasks=2)
            mapper = rm.MultiChipCoreMapper(env, baseline_latency=10.0, batch_z=1,
                                             reward_scale=scale)
            mapper.reset()
            for action in action_sequence:
                reward, done, _, cost = mapper.step(action)
            self.assertTrue(done)
            self.assertEqual(cost, env.evaluate())
            rewards.append(reward)
        self.assertAlmostEqual(rewards[1], rewards[0] * 20.0, places=10)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch required")
    def test_deterministic_candidate_retention_is_explicit_and_counted(self):
        import torch

        graph = np.array([[0, 10, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        env = MultiChipEnvironment(1, 1, 2, 2, task_graph=graph, num_tasks=3)

        class FakeAgent:
            gamma = 0.98

            def __init__(self, *args, **kwargs):
                self.device = torch.device("cpu")
                self.ou_state = np.zeros(2, dtype=np.float32)
                self._explore_index = 0
                self._deterministic_index = 0

            def select_action(self, state, noise_scale=0.1, explore=True):
                # Noisy rollout puts communicating tasks on opposite corners;
                # deterministic rollout puts them next to each other.
                noisy = [(-1, -1), (1, 1), (1, -1)]
                deterministic = [(-1, -1), (1, -1), (1, 1)]
                if explore:
                    action = noisy[self._explore_index]
                    self._explore_index += 1
                else:
                    action = deterministic[self._deterministic_index]
                    self._deterministic_index += 1
                return np.asarray(action, dtype=np.float32)

            def train(self, replay_buffer, batch_size=64):
                return None

        def fake_baseline(target, n_trials):
            target.place(np.array([0, 3, 1], dtype=np.int32))
            return target.evaluate()

        metadata = {}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(rm, "DDPGAgent", FakeAgent), \
             patch.object(rm, "run_random", side_effect=fake_baseline):
            cost = rm.run_ddpg(
                env, n_episodes=1, baseline_trials=1, batch_z=1,
                device="cpu", diagnostics_path=f"{directory}/diagnostics.jsonl",
                diagnostics_every=1, retain_deterministic_candidates=True,
                run_metadata=metadata)

        self.assertEqual(metadata["training_candidate_evaluations"], 1)
        self.assertEqual(metadata["deterministic_candidate_evaluations"], 1)
        self.assertEqual(metadata["total_candidate_evaluations"], 2)
        self.assertEqual(metadata["best_candidate_source"], "deterministic_diagnostic")
        self.assertEqual(cost, env.evaluate())
        np.testing.assert_array_equal(env.placement, [0, 1, 3])


@unittest.skipUnless(rm.HAS_TORCH, "PyTorch required")
class WorkloadTests(unittest.TestCase):
    def extract(self, model, shape, timing=True):
        import torch
        with patch.object(rm, "_build_model_and_input", return_value=(model.eval(), torch.zeros(shape), None)):
            with contextlib.redirect_stdout(io.StringIO()):
                return rm.extract_model_task_graph(channels_per_partition=3, return_work=timing)

    @unittest.skipUnless(rm.HAS_TORCHVISION, "torchvision required")
    def test_paper_target_counts_alexnet(self):
        with contextlib.redirect_stdout(io.StringIO()):
            graph, tasks, labels = rm.extract_model_task_graph(
                "alexnet", partition_mode="paper_targets")
        self.assertEqual(tasks, 1115)
        self.assertEqual(sum("_conv_" in label for label in labels), 183)
        self.assertEqual(sum("_linear_" in label for label in labels), 932)
        self.assertEqual(graph.shape, (1115, 1115))

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

    def test_residual_graph_and_timing_assumption(self):
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
        timed_graph, tasks, timed_labels, operations, kinds = self.extract(
            Residual(), (1, 3, 4, 4))
        self.assertEqual(timed_graph.shape, (tasks, tasks))
        self.assertEqual(len(timed_labels), len(operations))
        self.assertEqual(len(operations), len(kinds))

    def test_target_batchnorm_is_stable_and_synchronized(self):
        import torch
        torch.set_num_threads(2)
        agent = rm.DDPGAgent(19, 2, device="cpu", agent_arch="paper_cnn",
                             rows=4, cols=4, num_tasks=3)
        replay = rm.ReplayBuffer(64)
        rng = np.random.default_rng(42)
        for _ in range(64):
            replay.add(rng.normal(size=19).astype(np.float32),
                       np.zeros(2, dtype=np.float32), 0.2,
                       rng.normal(size=19).astype(np.float32), False)
        agent.train(replay)
        self.assertFalse(agent.actor_target.training)
        self.assertFalse(agent.critic_target.training)
        # The critic sees one statistics update per replay batch, not another
        # update during actor optimization.
        self.assertEqual(agent.critic.bn1.num_batches_tracked.item(), 1)
        for online, target in ((agent.actor, agent.actor_target),
                               (agent.critic, agent.critic_target)):
            for source, destination in zip(online.buffers(), target.buffers()):
                torch.testing.assert_close(source, destination)
        sample = torch.randn(1, 19)
        with torch.no_grad():
            alone = agent.actor_target(sample)
            batched = agent.actor_target(torch.cat([sample, torch.randn(7, 19)]))[:1]
        torch.testing.assert_close(alone, batched, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(p.requires_grad for p in agent.critic.parameters()))

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

    def test_guided_replay_and_twin_critic_update_are_finite(self):
        import torch
        torch.set_num_threads(2)
        replay = rm.GuidedReplayBuffer(capacity=128)
        rng = np.random.default_rng(7)
        for index in range(80):
            replay.add(rng.normal(size=8).astype(np.float32),
                       np.tanh(rng.normal(size=2)).astype(np.float32),
                       return_target=float(rng.uniform(-0.2, 0.1)),
                       is_demo=index < 16)
        agent = rm.DDPGAgent(8, 2, device="cpu", stable=True)
        before = next(agent.actor.parameters()).detach().clone()
        first = agent.train_guided(replay, batch_size=64, bc_weight=1.0)
        second = agent.train_guided(replay, batch_size=64, bc_weight=1.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second["actor_loss"])
        self.assertTrue(np.isfinite(second["critic_loss"]))
        self.assertTrue(np.isfinite(second["q_abs_max"]))
        self.assertTrue(np.isfinite(second["critic_grad_norm"]))
        self.assertFalse(torch.equal(before, next(agent.actor.parameters()).detach()))

    def test_guided_replay_keeps_demonstrations_when_capacity_wraps(self):
        replay = rm.GuidedReplayBuffer(capacity=5, demo_fraction=0.4)
        state = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        replay.add(state, action, 0.1, is_demo=True)
        replay.add(state + 1, action + 1, 0.2, is_demo=True)
        for index in range(20):
            replay.add(state + index + 2, action, -0.1, is_demo=False)
        demonstrations = [entry for entry in replay.buffer if entry[3]]
        self.assertEqual(len(replay), 5)
        self.assertEqual(len(demonstrations), 2)
        self.assertTrue(np.array_equal(demonstrations[0][0], state))
        self.assertTrue(np.array_equal(demonstrations[1][0], state + 1))

    def test_guided_replay_has_an_independent_random_stream(self):
        np.random.seed(193)
        expected = np.random.random(4)
        np.random.seed(193)
        replay = rm.GuidedReplayBuffer(capacity=8, seed=7)
        for index in range(4):
            replay.add(np.array([index], dtype=np.float32),
                       np.array([index], dtype=np.float32), float(index),
                       is_demo=index == 0)
        for _ in range(10):
            replay.sample(4)
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)

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
