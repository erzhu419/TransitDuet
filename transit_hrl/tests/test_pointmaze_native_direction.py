import copy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from freq_hrl.experiments import pointmaze_native_direction as experiment
from scripts import pointmaze_native_direction_stage73_spec as spec
from scripts.submit_pointmaze_native_direction_stage73_scheduleurm import task_specification, qualification_task


class NativeDirectionTest(unittest.TestCase):
    def test_fisher_radius_matches_exact_linear_gaussian_kl_and_keeps_source(self):
        torch.set_num_threads(1)
        actor = GaussianActor(1, 1, 0, 0.)
        with torch.no_grad():
            for p in actor.parameters():p.zero_()
        before = copy.deepcopy(actor.state_dict())
        states = np.linspace(1., 2., 12, dtype=np.float32).reshape(-1, 1)
        gradient = np.array([0., -1., -1.])
        actors, row, cost = experiment.matched_perturbations(actor, states, gradient, delta=.001, chunk_size=5)
        self.assertAlmostEqual(row["fisher_quadratic"], float(np.mean((states[:, 0].astype(float) + 1.) ** 2 / 2)), places=6)
        for kl in row["exact_kl"].values():self.assertAlmostEqual(kl, .001, places=8)
        self.assertEqual(cost, {"fisher_jvp_batches": 3, "exact_kl_forward_batches": 6})
        plus, minus = actors["plus"].net[0].weight, actors["minus"].net[0].weight
        torch.testing.assert_close(plus, -minus, atol=0, rtol=0)
        self.assertGreater(float(plus.item()), 0.)
        torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_fisher_uses_actor_std_clamp_and_rejects_undefined_direction(self):
        actor = GaussianActor(1, 1, 0, 2.)
        states = np.ones((8, 1), dtype=np.float32)
        gradient = np.array([3., -1., -1.])
        _, row, _ = experiment.matched_perturbations(actor, states, gradient, delta=.001, chunk_size=5)
        expected = 4. / (11. * 9.)
        self.assertAlmostEqual(row["fisher_quadratic"], expected, places=7)
        with self.assertRaises(ValueError):
            experiment.matched_perturbations(actor, states, np.zeros(3), delta=.001, chunk_size=5)

    def test_pairs_use_all_seeds_and_variants_with_actual_common_noise(self):
        seeds = [11, 12, 13, 14]
        evaluation = {v: [{"seed": s, "episode_return": float(i), "policy_seed": 1, "lower_seed": 2,
            "decision_steps": [0], "upper_proposed_actions": [[0., 0., 0., 0.]]} for i, s in enumerate(seeds)] for v in spec.VARIANTS}
        for d in spec.DIRECTIONS:
            for r in evaluation[d + "_plus"]:r["episode_return"] += 1.
            for r in evaluation[d + "_minus"]:r["episode_return"] -= .5
        effects = experiment.paired_effects(evaluation, seeds)
        for row in effects.values():self.assertEqual(row, {"plus_minus": 1.5, "plus_base": 1., "minus_base": -.5})
        evaluation["gae_control_plus"][0]["upper_proposed_actions"] = [[1., 0., 0., 0.]]
        with self.assertRaises(ValueError):experiment.paired_effects(evaluation, seeds)

    def test_forward_worker_samples_without_materializing_traces_or_batches(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=1, lower_state_dim=1,
            upper_action_dim=4, lower_action_dim=1, lower_cost_critic=False))
        args = SimpleNamespace(optimizer_seed=310001, maximum_subgoal_delta=.2)
        weights = experiment.joint.inference_weights(model)
        plan = MagicMock(ols_fits=1, ridge_predictions=1, calls=3, context_calls=3,
            proposed_actions=[np.zeros(4)])
        row = {"episode_return": 2., "episode_length": 3, "upper_inference_calls": 1,
            "lower_inference_calls": 3, "decision_steps": [0], "lower_seed": 2}
        with patch.object(experiment, "_WORKER", (model, args)), patch.object(experiment.execution, "ExecutedPlan", return_value=plan), \
                patch.object(experiment.joint, "rollout", return_value=(None, row, None)) as rollout:
            observed = experiment.worker_native((weights, 73095001, "zero_train", 50, {}))
            kwargs = rollout.call_args.kwargs
            self.assertFalse(kwargs["sample"])
            self.assertFalse(kwargs["capture"])
            self.assertTrue(kwargs["upper_sample"] and kwargs["lower_sample"])
            self.assertIsNotNone(kwargs["lower_seed"])
            self.assertEqual(observed["network_check"], "passed")

    def test_complete_budget_fresh_seeds_and_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        for key, expected in {"calibration_archive_episodes": 4096, "reconstructed_lower_calls": 4915200,
                "native_episodes": 7168, "native_steps": 8601600, "actor_score_forward_batches": 8192,
                "actor_score_backward_batches": 40960, "fisher_jvp_batches": 14400,
                "exact_kl_forward_batches": 28800, "actor_parameter_perturbations": 192}.items():
            self.assertEqual(b[key] * 8, expected)
        self.assertEqual(len(spec.ENDPOINTS), 36)
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertFalse(set(roles["calibration"]).intersection(roles["native_evaluation"]))
                task = task_specification("unit_stage73", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage73", preflight=preflight)
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q["result_dir"])
            self.assertEqual(q["ram_mb"], 2048)


if __name__ == "__main__":
    unittest.main()
