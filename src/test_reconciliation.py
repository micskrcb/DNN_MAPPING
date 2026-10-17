"""Run with python -m unittest discover -s src -p test_reconciliation.py."""
import contextlib
import io
import json
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
    def test_operator_bandit_sa_has_exact_budget_and_uniform_control(self):
        graph = np.zeros((5, 5), dtype=np.float32)
        graph[0, 1] = graph[0, 2] = graph[1, 3] = graph[2, 3] = 2.0
        graph[3, 4] = 3.0
        cases = [(False, "learned", 16), (True, "learned", 0),
                 (False, "proxy", 0)]
        for disabled, control, expected_updates in cases:
            env = MultiChipEnvironment(
                2, 1, 2, 2, task_graph=graph, num_tasks=5)
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_operator_bandit_sa(
                    env, n_iter=48, proxy_candidates=4,
                    disable_learning=disabled, control=control,
                    seed=23, metadata=metadata)
            self.assertTrue(np.isfinite(result))
            self.assertEqual(metadata["total_objective_evaluations"], 49)
            self.assertEqual(sum(metadata["operator_selection_counts"].values()),
                             16)
            self.assertEqual(set(metadata["operator_selection_counts"]),
                             {"proxy", "bottleneck", "swap", "relocation"})
            expected_control = "uniform" if disabled else control
            self.assertEqual(metadata["control"], expected_control)
            self.assertEqual(metadata["learning_enabled"],
                             expected_control == "learned")
            self.assertEqual(metadata["update_count"], expected_updates)
            if expected_control == "proxy":
                self.assertEqual(metadata["operator_selection_counts"]["proxy"], 16)

    def test_full_information_operator_gate_matches_proxy_budget(self):
        graph = np.zeros((5, 5), dtype=np.float32)
        graph[0, 1] = graph[0, 2] = graph[1, 3] = graph[2, 3] = 2.0
        graph[3, 4] = 3.0
        for control in ("learned", "proxy"):
            env = MultiChipEnvironment(
                2, 1, 2, 2, task_graph=graph, num_tasks=5)
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_operator_bandit_sa(
                    env, n_iter=64, proxy_candidates=4, control=control,
                    full_information_stride=4, minimum_exploration=0,
                    seed=29, metadata=metadata)
            self.assertTrue(np.isfinite(result))
            self.assertEqual(metadata["total_objective_evaluations"], 65)
            self.assertGreater(metadata["supervised_pools"], 0)
            self.assertEqual(metadata["supervised_operator_labels"],
                             4 * metadata["supervised_pools"])
            self.assertEqual(sum(metadata["operator_selection_counts"].values()),
                             metadata["decision_steps"])
            self.assertEqual(metadata["update_count"],
                             metadata["supervised_operator_labels"]
                             if control == "learned" else 0)

    def test_proxy_gated_lints_uses_one_true_call_and_is_reproducible(self):
        graph = np.zeros((5, 5), dtype=np.float32)
        graph[0, 1] = graph[0, 2] = graph[1, 3] = graph[2, 3] = 2.0
        graph[3, 4] = 3.0
        rows = []
        for _ in range(2):
            env = MultiChipEnvironment(
                2, 1, 2, 2, task_graph=graph, num_tasks=5)
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_operator_bandit_sa(
                    env, n_iter=64, proxy_candidates=4,
                    selection_policy="lints", proxy_gate_slate=2,
                    thompson_scale=0.25, discount=0.98,
                    minimum_exploration=0.03, reward_mode="outcome",
                    reward_window=20, seed=31, metadata=metadata)
            rows.append((result, metadata))
        result, metadata = rows[0]
        self.assertTrue(np.isfinite(result))
        self.assertEqual(metadata["total_objective_evaluations"], 65)
        self.assertEqual(metadata["decision_steps"], 32)
        self.assertEqual(metadata["proxy_gate_candidate_proposals"], 64)
        self.assertEqual(sum(metadata["operator_selection_counts"].values()),
                         metadata["decision_steps"])
        self.assertEqual(sum(metadata["bandit_primary_selection_counts"].values()),
                         metadata["decision_steps"])
        self.assertEqual(metadata["update_count"], metadata["decision_steps"])
        self.assertTrue(all(value >= 0 for value in
                            metadata["operator_reward_sums"].values()))
        self.assertEqual(rows[0][0], rows[1][0])
        self.assertEqual(rows[0][1]["operator_selection_counts"],
                         rows[1][1]["operator_selection_counts"])

    def test_delayed_macro_operator_credit_is_budgeted_and_reproducible(self):
        graph = np.zeros((5, 5), dtype=np.float32)
        graph[0, 1] = graph[0, 2] = graph[1, 3] = graph[2, 3] = 2.0
        graph[3, 4] = 3.0
        rows = []
        for _ in range(2):
            env = MultiChipEnvironment(
                2, 1, 2, 2, task_graph=graph, num_tasks=5)
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_operator_bandit_sa(
                    env, n_iter=96, proxy_candidates=4,
                    selection_policy="lints", proxy_gate_slate=2,
                    macro_horizon=8, thompson_scale=0.25,
                    discount=0.98, minimum_exploration=0.03,
                    seed=37, metadata=metadata)
            rows.append((result, metadata))
        result, metadata = rows[0]
        self.assertTrue(np.isfinite(result))
        self.assertEqual(metadata["method"],
                         "delayed_macro_operator_lints_sa_v5")
        self.assertEqual(metadata["total_objective_evaluations"], 97)
        self.assertEqual(metadata["decision_steps"], 64)
        self.assertEqual(metadata["macro_updates"], 8)
        self.assertEqual(metadata["update_count"], 8)
        self.assertEqual(sum(metadata["macro_action_counts"].values()), 8)
        self.assertGreaterEqual(metadata["proxy_gate_candidate_proposals"], 64)
        self.assertLessEqual(metadata["proxy_gate_candidate_proposals"], 128)
        self.assertTrue(all(value >= 0 for value in
                            metadata["macro_reward_sums"].values()))
        self.assertEqual(rows[0][0], rows[1][0])
        self.assertEqual(rows[0][1]["macro_action_counts"],
                         rows[1][1]["macro_action_counts"])
        control_env = MultiChipEnvironment(
            2, 1, 2, 2, task_graph=graph, num_tasks=5)
        control_metadata = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rm.run_operator_bandit_sa(
                control_env, n_iter=96, proxy_candidates=4,
                control="uniform", macro_horizon=8,
                seed=37, metadata=control_metadata)
        self.assertEqual(control_metadata["total_objective_evaluations"], 97)
        self.assertEqual(control_metadata["update_count"], 0)
        self.assertEqual(sum(control_metadata["macro_action_counts"].values()),
                         8)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_proxy_guided_sa_is_training_free_across_supported_topologies(self):
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[0, 2] = graph[1, 3] = graph[2, 3] = 1.0
        for topology in ("mesh", "torus"):
            env = MultiChipEnvironment(
                2, 1, 2, 2, task_graph=graph, num_tasks=4,
                topology=topology)
            metadata = {}
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_proxy_guided_sa(
                    env, n_iter=40, candidate_count=4,
                    focus_fraction=0.5, seed=17, metadata=metadata)
            self.assertTrue(np.isfinite(result))
            self.assertEqual(metadata["method"],
                             "training_free_proxy_guided_fixed_sa")
            self.assertTrue(metadata["unsupervised"])
            self.assertFalse(metadata["requires_training_data"])
            self.assertEqual(metadata["update_count"], 0)
            self.assertEqual(metadata["total_objective_evaluations"], 41)

    def test_relative_cost_state_keeps_late_progress_distinct(self):
        self.assertEqual(rm.proposal_cost_state(100, 80, 70, "legacy_clipped"),
                         rm.proposal_cost_state(100, 40, 30, "legacy_clipped"))
        self.assertEqual(rm.proposal_cost_state(100, 80, 70, "relative_cost"),
                         (0.8, 0.7))
        self.assertEqual(rm.proposal_cost_state(100, 40, 30, "relative_cost"),
                         (0.4, 0.3))

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_residual_bandit_starts_at_proxy_and_learns_planted_correction(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(81)
        model = rm.ProposalBanditRegressor(
            8, 9, hidden_dim=16, scoring="proxy_residual", residual_limit=0.25)
        state = torch.zeros((64, 8))
        features = torch.zeros((64, 4, 9))
        features[..., 0] = torch.rand((64, 4)) * 2 - 1
        features[..., 7] = torch.rand((64, 4)) * 0.1
        proxy = features[..., 7]
        torch.testing.assert_close(model(state, features), proxy, rtol=0, atol=0)
        target = proxy + 0.15 * features[..., 0]
        initial_loss = float(((proxy - target) ** 2).mean())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(80):
            loss = ((model(state, features) - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            fresh = features.clone()
            fresh[..., 0] = torch.rand((64, 4)) * 2 - 1
            prediction = model(state, fresh)
            error = ((prediction - (proxy + 0.15 * fresh[..., 0])) ** 2).mean()
            self.assertLess(float(error), initial_loss * 0.2)
            self.assertLessEqual(float((prediction - proxy).abs().max()), 0.25)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_bandit_uses_selected_reward_and_checks_checkpoint_semantics(self):
        import torch
        torch.set_num_threads(2)
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        def environment():
            return MultiChipEnvironment(1, 1, 2, 3, task_graph=graph, num_tasks=4)
        options = dict(
            n_iter=24, candidate_count=4, rollout_steps=4, device="cpu",
            update_epochs=1, minibatch_size=4, hidden_dim=16,
            calibration_trials=4, adapt_window=4, seed=13,
            proposal_learner="bandit", bandit_warmup=4,
            annealing_schedule="fixed", reward_mode="positive_delta")
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(rm, "proposal_training_reward", return_value=0.125):
            for scoring, expected in [("direct", 0.125),
                                      ("proxy_residual", np.tanh(0.125))]:
                metadata = {}
                checkpoint = f"{directory}/{scoring}.pt"
                rm.run_ppo_asa(environment(), **options, metadata=metadata,
                              bandit_scoring=scoring, state_encoding="relative_cost",
                              save_checkpoint=checkpoint)
                self.assertEqual(metadata["total_objective_evaluations"], 25)
                self.assertGreater(metadata["update_count"], 0)
                self.assertAlmostEqual(
                    metadata["final_update_metrics"]["mean_target"], expected, places=6)
                with self.assertRaisesRegex(ValueError, "state encoding"):
                    rm.run_ppo_asa(environment(), **options, disable_learning=True,
                                  bandit_scoring=scoring, load_checkpoint=checkpoint)
                loaded = {}
                rm.run_ppo_asa(environment(), **options, disable_learning=True,
                              state_encoding="relative_cost", bandit_scoring=scoring,
                              load_checkpoint=checkpoint, metadata=loaded)
                self.assertEqual(loaded["model_initialization"], "loaded_checkpoint")
            with self.assertRaisesRegex(ValueError, "bandit scoring"):
                rm.run_ppo_asa(environment(), **options, disable_learning=True,
                              state_encoding="relative_cost", load_checkpoint=checkpoint)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_zero_residual_matches_proxy_search_before_updates(self):
        import torch
        torch.set_num_threads(2)
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        results, placements, diagnostics = [], [], []
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            for control in (False, True):
                env = MultiChipEnvironment(1, 1, 2, 3, task_graph=graph, num_tasks=4)
                path = f"{directory}/{control}.jsonl"
                results.append(rm.run_ppo_asa(
                    env, n_iter=40, calibration_trials=4, candidate_count=4,
                    bandit_warmup=4, rollout_steps=100, device="cpu", seed=17,
                    proposal_learner="bandit", bandit_scoring="proxy_residual",
                    state_encoding="relative_cost", bandit_exploration="epsilon_top_m",
                    bandit_top_m=2, bandit_epsilon_final=0.25,
                    disable_learning=control,
                    bandit_control="proxy_greedy" if control else "uniform",
                    annealing_schedule="fixed", diagnostics_path=path))
                placements.append(env.placement.copy())
                with open(path) as stream:
                    diagnostics.append([json.loads(line) for line in stream])
        self.assertEqual(results[0], results[1])
        np.testing.assert_array_equal(*placements)
        self.assertEqual([d["current_cost"] for d in diagnostics[0]],
                         [d["current_cost"] for d in diagnostics[1]])

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_pool_supervision_counts_every_candidate_in_fixed_budget(self):
        import torch
        torch.set_num_threads(2)
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        env = MultiChipEnvironment(1, 1, 2, 3, task_graph=graph, num_tasks=4)
        metadata = {}
        with contextlib.redirect_stdout(io.StringIO()):
            rm.run_ppo_asa(
                env, n_iter=40, calibration_trials=4, candidate_count=4,
                bandit_warmup=4, rollout_steps=4, update_epochs=1,
                minibatch_size=4, device="cpu", seed=31,
                proposal_learner="bandit", bandit_scoring="proxy_residual",
                state_encoding="relative_cost", reward_mode="proposed_delta",
                bandit_pool_supervision_stride=2,
                annealing_schedule="fixed", metadata=metadata)
        self.assertEqual(metadata["total_objective_evaluations"], 41)
        self.assertGreater(metadata["supervised_pools"], 0)
        self.assertEqual(metadata["supervised_candidate_evaluations"],
                         metadata["supervised_pools"] * 4)
        self.assertGreater(metadata["bandit_observations"],
                           metadata["policy_steps"])

    def test_zero_discount_is_only_valid_for_proposal_ranking(self):
        self.assertTrue(rm.valid_ppo_gamma("ppo_asa", 0.0))
        self.assertTrue(rm.valid_ppo_gamma("ppo_asa", 1.0))
        self.assertFalse(rm.valid_ppo_gamma("ppo_masked", 0.0))
        self.assertFalse(rm.valid_ppo_gamma("ppo_asa", -0.01))
        self.assertFalse(rm.valid_ppo_gamma("ppo_asa", 1.01))

    def test_improvement_only_proposal_reward(self):
        self.assertEqual(
            rm.proposal_training_reward("positive_delta", 2.5, -1.0, 9.0),
            2.5)
        self.assertEqual(
            rm.proposal_training_reward("positive_delta", -2.5, 1.0, 9.0),
            0.0)
        self.assertEqual(
            rm.proposal_training_reward("proposed_delta", -2.5, 1.0, 9.0),
            -2.5)

    def test_proposal_focus_labels_preserve_legacy_zero_slack(self):
        self.assertEqual(
            rm.proposal_focus_label(0.5, 0.0),
            "mixed_bottleneck_0.500")
        self.assertEqual(
            rm.proposal_focus_label(1.0, 0.0),
            "current_bottleneck_stage")
        self.assertEqual(
            rm.proposal_focus_label(0.5, 0.2),
            "mixed_near_bottleneck_0.500_slack_0.200")

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
            for _ in range(4)
        ]
        trained, control, loaded, fixed = {}, {}, {}, {}
        with tempfile.TemporaryDirectory() as directory:
            trained_cost = rm.run_ppo_asa(
                environments[0], n_iter=40, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=7,
                focus_fraction=0.5, restart_interval=10,
                update_at_chain_end=True,
                diagnostics_path=f"{directory}/trained.jsonl",
                save_checkpoint=f"{directory}/trained.pt", metadata=trained)
            control_cost = rm.run_ppo_asa(
                environments[1], n_iter=40, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=7,
                focus_fraction=0.5, restart_interval=10,
                disable_learning=True, metadata=control)
            loaded_cost = rm.run_ppo_asa(
                environments[2], n_iter=16, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=9,
                focus_fraction=0.5, disable_learning=True,
                load_checkpoint=f"{directory}/trained.pt", metadata=loaded)
            fixed_cost = rm.run_ppo_asa(
                environments[3], n_iter=16, candidate_count=4,
                rollout_steps=4, device="cpu", update_epochs=1,
                minibatch_size=4, hidden_dim=16, calibration_trials=4,
                adapt_window=4, checkpoint_every=16, seed=23,
                gamma=0.0, gae_lambda=0.0,
                reward_mode="proposed_delta", annealing_schedule="fixed",
                fixed_cooldown=0.99, fixed_perturb_fraction=0.01,
                final_temperature_ratio=1e-6, metadata=fixed)
        self.assertTrue(np.isfinite(trained_cost))
        self.assertTrue(np.isfinite(control_cost))
        self.assertTrue(np.isfinite(loaded_cost))
        self.assertTrue(np.isfinite(fixed_cost))
        self.assertEqual(trained["initial_cost"], control["initial_cost"])
        self.assertEqual(trained["candidate_evaluations"], 40)
        self.assertEqual(trained["initial_placement_evaluations"], 4)
        self.assertEqual(trained["total_objective_evaluations"], 44)
        self.assertEqual(trained["independent_chains"], 4)
        self.assertEqual(trained["proposal_focus"], "mixed_bottleneck_0.500")
        self.assertEqual(trained["progress_normalization"],
                         "within_restart_chain")
        self.assertTrue(trained["update_at_chain_end"])
        self.assertEqual(trained["weight_decay"], 0.0)
        self.assertEqual(trained["policy_steps"], 36)
        self.assertGreater(trained["update_count"], 0)
        self.assertEqual(control["update_count"], 0)
        self.assertFalse(control["learning_enabled"])
        self.assertEqual(loaded["model_initialization"], "loaded_checkpoint")
        self.assertEqual(loaded["source_training_evaluations"], 40)
        self.assertEqual(loaded["source_training_updates"], 7)
        self.assertEqual(fixed["method"],
                         "ppo_guided_fixed_simulated_annealing")
        self.assertEqual(fixed["annealing_schedule"], "fixed")
        self.assertEqual(fixed["temperature_controller"],
                         "paper_style_fixed_schedule")
        self.assertEqual(fixed["fixed_cooldown"], 0.99)
        self.assertEqual(fixed["final_perturb_fraction"], 0.01)
        self.assertEqual(fixed["reheats"], 0)
        self.assertGreater(fixed["update_count"], 0)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_contextual_bandit_starts_tied_and_updates_with_fixed_sa(self):
        import torch
        torch.set_num_threads(2)
        model = rm.ProposalBanditRegressor(8, 9, hidden_dim=16)
        scores = model(torch.zeros((2, 8)), torch.randn((2, 4, 9)))
        torch.testing.assert_close(scores, torch.zeros((2, 4)))

        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        env = MultiChipEnvironment(
            1, 1, 2, 3, task_graph=graph, num_tasks=4)
        metadata = {}
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_ppo_asa(
                env, n_iter=24, candidate_count=4, rollout_steps=4,
                device="cpu", update_epochs=1, minibatch_size=4,
                hidden_dim=16, calibration_trials=4, adapt_window=4,
                checkpoint_every=24, seed=13, proposal_learner="bandit",
                bandit_replay_capacity=32, bandit_warmup=4,
                annealing_schedule="fixed",
                diagnostics_path=f"{directory}/bandit.jsonl",
                save_checkpoint=f"{directory}/bandit.pt", metadata=metadata)
            checkpoint = torch.load(
                f"{directory}/bandit.pt", map_location="cpu",
                weights_only=False)
        self.assertTrue(np.isfinite(result))
        self.assertEqual(metadata["proposal_learner"], "bandit")
        self.assertEqual(metadata["proposal_policy"],
                         "epsilon_greedy_contextual_bandit")
        self.assertGreater(metadata["update_count"], 0)
        self.assertEqual(checkpoint["algorithm"], "bandit_sa_proposal_v1")

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_proxy_greedy_control_uses_proxy_without_learning(self):
        import torch
        torch.set_num_threads(2)
        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        env = MultiChipEnvironment(
            1, 1, 2, 3, task_graph=graph, num_tasks=4)
        metadata = {}
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_ppo_asa(
                env, n_iter=24, candidate_count=4, rollout_steps=4,
                device="cpu", update_epochs=1, minibatch_size=4,
                hidden_dim=16, calibration_trials=4, adapt_window=4,
                checkpoint_every=24, seed=29, proposal_learner="bandit",
                bandit_control="proxy_greedy", disable_learning=True,
                bandit_replay_capacity=32, bandit_warmup=4,
                bandit_epsilon_final=0.5,
                bandit_epsilon_decay_fraction=1.0,
                annealing_schedule="fixed",
                diagnostics_path=f"{directory}/proxy.jsonl",
                metadata=metadata)
            with open(f"{directory}/proxy.jsonl") as stream:
                records = [json.loads(line) for line in stream]
        self.assertTrue(np.isfinite(result))
        self.assertFalse(metadata["learning_enabled"])
        self.assertEqual(metadata["update_count"], 0)
        self.assertEqual(metadata["bandit_control"], "proxy_greedy")
        self.assertEqual(metadata["proposal_policy"],
                         "epsilon_greedy_proxy_gain")
        self.assertLess(metadata["final_exploration_epsilon"], 1.0)
        self.assertTrue(all(record["update_count"] == 0
                            for record in records))
        self.assertTrue(all(record["bandit_loss"] is None
                            for record in records))

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_reference_ppo_asa_separates_actor_and_critic(self):
        import torch
        torch.set_num_threads(2)
        actor = rm.ProposalPPOActor(8, 9, hidden_dim=16)
        critic = rm.ProposalPPOCritic(8, 9, hidden_dim=16)
        self.assertTrue(
            {id(parameter) for parameter in actor.parameters()}.isdisjoint(
                {id(parameter) for parameter in critic.parameters()}))
        states = torch.zeros((2, 8))
        candidates = torch.randn((2, 4, 9))
        distribution = actor.distribution(states, candidates)
        torch.testing.assert_close(
            distribution.probs, torch.full((2, 4), 0.25))
        self.assertTrue(torch.isfinite(critic(states, candidates)).all())

        graph = np.zeros((4, 4), dtype=np.float32)
        graph[0, 1] = graph[1, 2] = graph[2, 3] = 1.0
        environments = [
            MultiChipEnvironment(
                1, 1, 2, 3, task_graph=graph, num_tasks=4)
            for _ in range(3)
        ]
        trained, loaded, fine_tuned = {}, {}, {}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f"{directory}/reference.pt"
            trained_cost = rm.run_ppo_asa(
                environments[0], n_iter=40, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=13,
                update_style="reference_separate",
                reward_mode="planted_proxy",
                diagnostics_path=f"{directory}/reference.jsonl",
                oracle_diagnostics_path=f"{directory}/oracle.json",
                oracle_dataset_path=f"{directory}/oracle.npz",
                oracle_pools=4, oracle_stride=2,
                save_checkpoint=checkpoint, metadata=trained)
            with open(f"{directory}/oracle.json") as stream:
                oracle = json.load(stream)
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=False)
            with np.load(f"{directory}/oracle.npz", allow_pickle=False) as dataset:
                self.assertEqual(dataset["global_states"].shape, (4, 8))
                self.assertEqual(dataset["candidate_features"].shape, (4, 4, 9))
                self.assertEqual(dataset["true_gains"].shape, (4, 4))
                self.assertEqual(dataset["chain_ids"].shape, (4,))
            loaded_cost = rm.run_ppo_asa(
                environments[1], n_iter=16, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=17,
                update_style="reference_separate", disable_learning=True,
                load_checkpoint=checkpoint, metadata=loaded)
            fine_tuned_cost = rm.run_ppo_asa(
                environments[2], n_iter=16, candidate_count=4,
                rollout_steps=8, device="cpu", update_epochs=1,
                minibatch_size=8, hidden_dim=16, calibration_trials=4,
                adapt_window=8, checkpoint_every=20, seed=19,
                update_style="reference_separate",
                load_checkpoint=checkpoint, allow_finetune_loaded=True,
                metadata=fine_tuned)
        self.assertTrue(np.isfinite(trained_cost))
        self.assertTrue(np.isfinite(loaded_cost))
        self.assertTrue(np.isfinite(fine_tuned_cost))
        self.assertEqual(saved["algorithm"],
                         "ppo_asa_reference_separate_v2")
        self.assertIn("actor", saved)
        self.assertIn("critic", saved)
        self.assertEqual(trained["update_style"], "reference_separate")
        self.assertEqual(trained["reward_mode"], "planted_proxy")
        self.assertFalse(trained["policy_value_parameter_sharing"])
        self.assertEqual(
            trained["reference_implementation_commit"],
            "6e64bec12b9a7c6db601a4a4ae254b942e4aed79")
        self.assertGreater(trained["update_count"], 0)
        self.assertEqual(oracle["oracle_pools"], 4)
        self.assertEqual(oracle["oracle_candidate_evaluations"], 16)
        self.assertEqual(trained["oracle_additional_objective_evaluations"], 12)
        self.assertEqual(trained["optimization_objective_evaluations"], 41)
        self.assertEqual(trained["total_objective_evaluations"], 53)
        self.assertEqual(loaded["model_initialization"], "loaded_checkpoint")
        self.assertEqual(fine_tuned["model_initialization"], "loaded_checkpoint")
        self.assertGreater(fine_tuned["update_count"], 0)

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_oracle_supervised_ranker_generalizes_planted_ordering(self):
        import torch
        import train_oracle_candidate_ranker as oracle_ranker
        torch.set_num_threads(2)
        rng = np.random.default_rng(23)
        pools, candidates = 240, 6
        global_states = rng.normal(size=(pools, 8)).astype(np.float32)
        features = rng.normal(size=(pools, candidates, 9)).astype(np.float32)
        gains = (features[:, :, 0].astype(np.float64) -
                 0.4 * features[:, :, 3].astype(np.float64))
        group_ids = np.repeat(np.arange(12), pools // 12)
        result = oracle_ranker.train_ranker(
            global_states, features, gains, group_ids=group_ids,
            hidden_dim=32, epochs=100,
            batch_size=64, learning_rate=2e-3, weight_decay=0.0,
            target_temperature=0.4, validation_fraction=0.25,
            patience=20, seed=29, device="cpu")
        validation = result["validation_metrics"]
        self.assertGreater(
            validation["selected_advantage_95_percent_interval_seconds"][0], 0)
        self.assertGreater(
            validation["top1_oracle_match_fraction_on_informative_pools"],
            validation[
                "random_expected_oracle_match_fraction_on_informative_pools"])
        self.assertTrue(set(result["training_groups"]).isdisjoint(
            set(result["validation_groups"])))

    @unittest.skipUnless(rm.HAS_TORCH, "PyTorch is not installed")
    def test_chain_metrics_use_trajectories_as_independent_units(self):
        import train_oracle_candidate_ranker as oracle_ranker
        groups, pools, candidates = 10, 50, 4
        raw = np.zeros((groups * pools, candidates), dtype=np.float64)
        raw[:, 0] = 2.0
        raw[:, 1] = 1.0
        logits = raw.copy()
        group_ids = np.repeat(np.arange(groups), pools)
        progress = np.tile(np.linspace(0, 1, pools), groups)
        metrics = oracle_ranker.chain_bootstrap_metrics(
            logits, raw, np.maximum(raw, 0), group_ids,
            progress=progress, minimum_progress=0.5,
            n_bootstrap=200, seed=31)
        self.assertEqual(metrics["groups"], groups)
        self.assertEqual(metrics["positive_capture_groups"], groups)
        self.assertAlmostEqual(metrics["mean_capture_fraction"], 1.0)
        self.assertGreater(
            metrics["selected_to_candidate_improvement_ratio"], 1.5)

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

    def test_logic_task_order_reindexes_graph_and_work_together(self):
        graph = np.arange(16, dtype=np.float64).reshape(4, 4)
        labels = ["layer_VMM_m0_n0", "layer_VVA_m0",
                  "layer_VMM_m1_n0", "layer_VVA_m1"]
        operations = np.array([10, 20, 30, 40], dtype=np.float64)
        kinds = ["vmm", "vva", "vmm", "vva"]
        remapped = rm.reorder_extracted_tasks(
            graph, labels, operations, kinds, "all_vmm_then_vva")
        remapped_graph, remapped_labels, remapped_operations, remapped_kinds = remapped
        expected = np.array([0, 2, 1, 3], dtype=np.int64)
        np.testing.assert_array_equal(remapped_graph, graph[np.ix_(expected, expected)])
        self.assertEqual(remapped_labels,
                         [labels[index] for index in expected])
        np.testing.assert_array_equal(remapped_operations, operations[expected])
        self.assertEqual(remapped_kinds, [kinds[index] for index in expected])

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
        metadata = {}
        random.seed(3)
        with contextlib.redirect_stdout(io.StringIO()):
            result = rm.run_sa(obj, n_iter=1000, metadata=metadata)
        self.assertEqual(obj.calls, 1001)  # initial plus budgeted neighbors
        self.assertEqual(result, 0)
        self.assertEqual(metadata["candidate_evaluations"], 1000)
        self.assertEqual(metadata["total_objective_evaluations"], 1001)
        self.assertEqual(metadata["temperature_mode"],
                         "measured_objective_scale")
        self.assertGreater(metadata["initial_temperature"],
                           metadata["ending_temperature"])

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
        self.assertEqual(metadata["initial_placement_evaluations"], 1)
        self.assertEqual(metadata["total_objective_evaluations"], 251)
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

    def test_contention_features_match_exact_affected_stage_components(self):
        graph = np.zeros((6, 6), dtype=np.float32)
        graph[0, 2], graph[1, 2] = 2.0, 3.0
        graph[2, 3], graph[2, 4] = 4.0, 5.0
        graph[3, 5], graph[4, 5] = 6.0, 7.0
        env = MultiChipEnvironment(
            2, 1, 2, 2, on_chip_latency=1.0, off_chip_latency=5.0,
            task_graph=graph, num_tasks=6, routing_model="paper_xy",
            pipeline_model="xy_contention")
        placement = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        env.place(placement)
        evaluator = rm.IncrementalPipelineEvaluator(env)
        candidate = placement.copy()
        candidate[0], candidate[5] = candidate[5], candidate[0]
        changed = [0, 5]

        actual = evaluator.candidate_contention_features(candidate, changed)
        affected = evaluator._affected_stages(changed)
        scale = max(float(evaluator.stage_costs.max()), np.finfo(float).eps)
        old_link = max(evaluator.stage_details[s][1] for s in affected)
        old_source = max(evaluator.stage_details[s][2] for s in affected)
        env.place(candidate)
        new_details = [evaluator._stage_details(s) for s in affected]
        env.place(placement)
        expected = (
            np.clip((old_link - max(row[1] for row in new_details)) / scale,
                    -1.0, 1.0),
            max(evaluator.stage_costs[s] for s in affected) / scale,
            np.clip((old_source - max(row[2] for row in new_details)) / scale,
                    -1.0, 1.0),
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(env.placement, placement)

    def test_fixed_sa_incremental_path_matches_full_evaluation(self):
        graph = np.zeros((6, 6), dtype=np.float32)
        graph[0, 2], graph[1, 2] = 2.0, 3.0
        graph[2, 3], graph[2, 4] = 4.0, 5.0
        graph[3, 5], graph[4, 5] = 6.0, 7.0

        def run(incremental):
            env = MultiChipEnvironment(
                2, 1, 2, 2, on_chip_latency=1.0,
                off_chip_latency=5.0, task_graph=graph, num_tasks=6,
                routing_model="paper_xy", pipeline_model="xy_contention")
            metadata = {}
            random.seed(41)
            with contextlib.redirect_stdout(io.StringIO()):
                result = rm.run_sa(
                    env, n_iter=240, calibration_trials=16,
                    metadata=metadata, incremental=incremental)
            return result, env.placement.copy(), metadata

        incremental_result = run(True)
        full_result = run(False)
        self.assertAlmostEqual(incremental_result[0], full_result[0], places=12)
        np.testing.assert_array_equal(
            incremental_result[1], full_result[1])
        self.assertTrue(incremental_result[2]["incremental_evaluation"])
        self.assertFalse(full_result[2]["incremental_evaluation"])

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
