import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_forecast_tracking as experiment
from freq_hrl.experiments import pointmaze_plan_alignment as old
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_forecast_tracking_stage54_spec as spec
from scripts.submit_pointmaze_forecast_tracking_stage54_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class ForecastTrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.predictor, cls.fitting = experiment.fit_forecaster(spec.arguments(310001, preflight=True),
            spec.seed_roles(310001, preflight=True)["fitting"])

    def controller(self, *, cost=True):
        torch.manual_seed(54)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=cost,
            lower_value_state_dim=392, promotion_state_dim=394))

    def rollout(self, policy, *, shift=0., period=100):
        model = self.controller()
        original = model.lower_actor
        if policy != "frozen":
            actor = experiment.VelocityFeedbackActor if policy.endswith("_velocity") else experiment.FeedbackActor
            model.lower_actor = actor(original, experiment.feedback_gain(), "waypoint")
        reference = experiment.PlanReference(policy, self.predictor, period)
        args = spec.arguments(310001, preflight=True)
        args.horizon = 200
        with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask(shift)), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))), \
                patch.object(model, "act_lower", wraps=model.act_lower) as call:
            _, row, raw = joint.rollout(model, args, f"fixed{period}", seed=1, sample=False, capture=True,
                lower_credit="task_option", lower_value_context_builder=clocks.context_builder("task_clock"),
                lower_reference_builder=reference,
                lower_actor_context_builder=reference.actor_context if policy.endswith("_velocity") else None)
        if policy.endswith("_velocity"):
            for item in call.call_args_list:
                self.assertEqual(item.args[0].shape, (392,))
                self.assertEqual(item.kwargs["value_state"].shape, (392,))
                self.assertEqual(item.kwargs["cost_state"].shape, (390,))
                np.testing.assert_array_equal(item.args[0][:-2], item.kwargs["value_state"][:-2])
        return row, raw, reference

    def test_features_observed_motion_age_valid_history_and_fit_budget(self):
        target = .01 * np.arange(64)[:, None] * [.5, -.2]
        x = experiment.forecast_features(target)
        self.assertEqual(x.shape, (17,))
        np.testing.assert_allclose(x[2:10], [.5, -.2] * 4, atol=1e-12, rtol=0)
        self.assertEqual(x[10], 1.)
        target[-1] += [.01, 0]
        self.assertEqual(experiment.forecast_features(target)[10], 1 / 63)
        self.assertEqual(self.fitting["rows"], 400)
        self.assertEqual(self.fitting["driver_paths"], 2)
        self.assertEqual(self.fitting["native_steps"], 0)
        self.assertEqual(self.predictor["weights"].shape, (18, 200))
        points = experiment.plan_points(np.array([[.2, -.4]]), "ridge_velocity", self.predictor, 100, (-2, 2))
        np.testing.assert_array_equal(points, np.tile(np.array([.2, -.4], dtype=np.float32), (101, 1)))

    def test_linear_position_exactly_preserves_stage52_reference(self):
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.003, -.002]
        history = SimpleNamespace(history=frames.reshape(-1))
        obs = SimpleNamespace(task_measurement=frames[-1])
        a, b = old.CausalReference("target_curve"), experiment.PlanReference("linear_position", self.predictor, 100)
        for age in range(100):
            kwargs = dict(observation=obs, history=history, subgoal=np.zeros(2), age=age, step=100 + age,
                          world_low=-2 * np.ones(2), world_high=2 * np.ones(2))
            np.testing.assert_array_equal(a(**kwargs), b(**kwargs))

    def test_zero_velocity_preserves_actor_and_nonzero_input_changes_it(self):
        source, gain = self.controller().lower_actor, experiment.feedback_gain()
        a, b = experiment.FeedbackActor(source, gain, "waypoint"), experiment.VelocityFeedbackActor(source, gain, "waypoint")
        state = torch.zeros(3, 390)
        state[:, 4:6] = .01
        context = torch.zeros(3, 2)
        torch.testing.assert_close(a.distribution(state).loc, b.distribution(torch.cat((state, context), dim=1)).loc, atol=0, rtol=0)
        context[:, 0] = .02
        self.assertFalse(torch.equal(a.distribution(state).loc, b.distribution(torch.cat((state, context), dim=1)).loc))
        torch.testing.assert_close(a.log_std, b.log_std, atol=0, rtol=0)

    def test_future_changes_do_not_update_plan_or_velocity_within_option(self):
        for policy in spec.POLICIES[1:]:
            _, left, _ = self.rollout(policy)
            _, right, _ = self.rollout(policy, shift=2.)
            np.testing.assert_array_equal(left["lower_reference"][:100], right["lower_reference"][:100])
            np.testing.assert_array_equal(left["action"][:100], right["action"][:100])
            if policy.endswith("_velocity"):
                np.testing.assert_array_equal(left["lower_actor_context"][:100], right["lower_actor_context"][:100])
            self.assertFalse(np.array_equal(left["lower_reference"][100:], right["lower_reference"][100:]))

    def test_plan_velocity_audits_and_critic_clock_unchanged(self):
        for policy in spec.POLICIES:
            row, raw, ref = self.rollout(policy, period=50)
            result = experiment.audit_plan(raw, row, policy=policy, period=50, predictor=self.predictor, bounds=ref.bounds)
            self.assertEqual(result["audit_ols_fits"], ref.ols_fits)
            clocks.audit_context(None, row, raw["lower_value_context"], clock=True)
            bad = copy.deepcopy(raw)
            bad["lower_reference"][71, 0] = np.nextafter(bad["lower_reference"][71, 0], np.float32(np.inf))
            with self.assertRaisesRegex(AssertionError, "causal frozen plan"):
                experiment.audit_plan(bad, row, policy=policy, period=50, predictor=self.predictor, bounds=ref.bounds)
            if policy.endswith("_velocity"):
                self.assertTrue(np.any(raw["lower_actor_context"][50:] != 0))
                bad = copy.deepcopy(raw)
                bad["lower_actor_context"][75, 0] += .1
                with self.assertRaisesRegex(AssertionError, "planned velocity"):
                    experiment.audit_plan(bad, row, policy=policy, period=50, predictor=self.predictor, bounds=ref.bounds)

    def test_complete_pipeline_and_changed_roster_rejection(self):
        model = self.controller(cost=False)
        original = copy.deepcopy(joint.inference_weights(model))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file = directory / "source.json"
            source_file.write_text(json.dumps({"snapshots": {"2": {"checkpoint": "fixture.pt"}}}))
            with patch.object(spec, "source_result", return_value=source_file), \
                    patch.object(experiment, "load_pair", return_value=[model]), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                result = experiment.train(310001, preflight=True, output=directory / "run/result.json")
                summary = experiment.aggregate([result], preflight=True)
            self.assertEqual(summary["status"], "preflight_passed")
            self.assertEqual(summary["method_cost"]["primitive_steps"], 14400)
            self.assertEqual(summary["method_cost"]["upper_inference_calls"], 216)
            self.assertEqual(summary["method_cost"]["plan_ols_fits"], 112)
            self.assertEqual(summary["method_cost"]["plan_ridge_predictions"], 56)
            self.assertEqual(summary["method_cost"]["actor_context_evaluations"], 4800)
            self.assertNotIn("primary_endpoints", summary)
            torch.testing.assert_close(joint.inference_weights(model), original, atol=0, rtol=0)
            for key in ("actor_context_evaluations", "plan_ridge_predictions", "lower_seed"):
                bad = copy.deepcopy(result)
                bad["evaluation_rows"]["50"]["ridge_velocity"]["lower_sampled"][0][key] += 1
                with self.assertRaisesRegex(ValueError, "matched-call"):
                    experiment.aggregate([bad], preflight=True)
            bad = copy.deepcopy(result)
            bad["fitting"]["rows"] += 1
            with self.assertRaisesRegex(ValueError, "fitting budget"):
                experiment.aggregate([bad], preflight=True)

    def test_fresh_seeds_factorial_contrasts_budgets_and_dynamic_placement(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                for seeds in roles.values():
                    self.assertFalse(set(seeds) & seen)
                    self.assertFalse(set(seeds) & set(spec.previous.seed_roles(root, preflight=preflight)["evaluation"]))
                    seen.update(seeds)
            task = task_specification("unit_stage54", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        budget = spec.budget(preflight=False)
        self.assertEqual(8 * budget["total_primitive_steps"], 3686400)
        self.assertEqual(8 * budget["fitting_rows"], 281600)
        means = {str(p): {"deterministic": {k: {"episode_return": v} for k, v in
                 zip(spec.POLICIES, (0, 1, 2, 3, 4, 5))}} for p in spec.PERIODS}
        self.assertEqual(list(spec.contrasts(means).values()), [2, 1, 0, 4, 5, 3] * 2)
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, [-1., 1., 0., 2., -2., 3.] * 2))} for _ in range(8)]
        result = experiment.bootstrap(rows)
        self.assertEqual(result[spec.ENDPOINTS[0]]["effect"], "negative")
        self.assertEqual(result[spec.ENDPOINTS[1]]["effect"], "positive")
        self.assertEqual(result[spec.ENDPOINTS[2]]["effect"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
