import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_learned_plan as experiment
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_forecast_tracking as forecast
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_learned_plan_stage55_spec as spec
from scripts.submit_pointmaze_learned_plan_stage55_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class LearnedPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.predictor, _ = forecast.fit_forecaster(spec.arguments(310001, preflight=True),
            spec.seed_roles(310001, preflight=True)["fitting"])

    def controller(self):
        torch.manual_seed(55)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False,
            lower_value_state_dim=392, epochs=1, minibatch_size=128))

    def test_network_dimensions_zero_head_and_velocity_critic_columns(self):
        source = self.controller()
        new = experiment.make_model(source)
        self.assertEqual((new.config.lower_state_dim, new.config.lower_value_state_dim, new.config.upper_action_dim), (392, 394, 4))
        base = torch.randn(3, 390)
        context, velocity = torch.randn(3, 2), torch.randn(3, 2)
        torch.testing.assert_close(new.upper_actor.net(base), torch.zeros(3, 4), atol=0, rtol=0)
        torch.testing.assert_close(source.lower_actor.net(base), new.lower_actor.net(torch.cat((base, velocity), dim=1)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(source.lower_value(torch.cat((base, context), dim=1)),
            new.lower_value(torch.cat((base, velocity, context), dim=1)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(source.lower_actor.log_std, new.lower_actor.log_std, atol=0, rtol=0)
        torch.testing.assert_close(new.upper_actor.log_std, source.upper_actor.log_std.repeat_interleave(2), atol=0, rtol=0)

    def test_executed_upper_action_is_nonnull_anchored_and_frozen_inside_option(self):
        args = spec.arguments(310001, preflight=True)
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.001, .002]
        history, obs = SimpleNamespace(history=frames.reshape(-1)), SimpleNamespace(task_measurement=frames[-1])
        kwargs = dict(observation=obs, history=history, step=63, world_low=-2 * np.ones(2), world_high=2 * np.ones(2))
        a, b = [experiment.ResidualPlan(self.predictor, 100, args.maximum_subgoal_delta) for _ in range(2)]
        a.decode(action=np.zeros(4), **kwargs)
        b.decode(action=np.array([.2, -.1, .1, -.2]), **kwargs)
        np.testing.assert_array_equal(a.points, forecast.plan_points(frames[:, :2], "ridge_velocity", self.predictor, 100, a.bounds))
        np.testing.assert_array_equal(a.points[0], b.points[0])
        self.assertFalse(np.array_equal(a.points[1:], b.points[1:]))
        saved = b.points.copy()
        history.history[:] = 999
        obs.task_measurement[:] = -999
        np.testing.assert_array_equal(b(age=35, **kwargs), saved[35])
        np.testing.assert_array_equal(b.actor_context(age=35, step=98, horizon=300), (saved[36] - saved[35]) / .01)

    def test_native_training_batches_plan_credit_and_mutation_rejection(self):
        source = self.controller()
        model = experiment.make_model(source)
        args = spec.arguments(310001, preflight=True)
        experiment.init_worker(source.config, model.config, args)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                batch, row = experiment.worker_rollout((joint.inference_weights(model), 123, "joint_ppo", 50,
                    "train", "training", experiment.feedback_gain(), self.predictor, str(Path(directory) / "episode_123.npz")))
            self.assertEqual((batch.lower.state.shape, batch.lower.value_state.shape, batch.upper.action.shape), ((300, 392), (300, 394), (6, 4)))
            self.assertIsNone(batch.lower.cost_state)
            self.assertEqual(row["lower_actor_type"], "GaussianActor")
            with np.load(Path(directory) / "episode_123.npz") as file:
                raw = {k: file[k] for k in file.files}
            for index, start in enumerate(row["decision_steps"]):
                rewards = raw["reward"][start:start + 50].copy()
                rewards[0] -= joint.spec.CALL_COST
                self.assertAlmostEqual(batch.upper.reward[index], float(rewards @ (model.config.gamma ** np.arange(50))), places=4)
            experiment.audit_plan(raw, row, predictor=self.predictor, period=50, scale=args.maximum_subgoal_delta,
                bounds=(-2, 2), batch=batch)
            bad = copy.deepcopy(raw)
            bad["upper_plan_action"][1, 0] += .2
            with self.assertRaises(AssertionError):
                experiment.audit_plan(bad, row, predictor=self.predictor, period=50, scale=args.maximum_subgoal_delta, bounds=(-2, 2))
            bad = copy.deepcopy(raw)
            bad["lower_value_context"][25, 0] += .1
            with self.assertRaises(AssertionError):
                experiment.audit_plan(bad, row, predictor=self.predictor, period=50, scale=args.maximum_subgoal_delta, bounds=(-2, 2))

    def test_full_pipeline_paired_first_update_frozen_calibration_and_accounting(self):
        model = self.controller()
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
            self.assertEqual(sum(x["primitive_steps"] for x in summary["method_cost"].values()), 22800)
            self.assertEqual(summary["native_trace_audits"], 76)
            self.assertEqual(summary["optimizer_steps"]["supervised"], 16)
            self.assertNotIn("primary_endpoints", summary)
            torch.testing.assert_close(joint.inference_weights(model), original, atol=0, rtol=0)
            for p in ("50", "100"):
                self.assertEqual(result["first_batch_pairs"][p]["first_lower_update"], "passed")
                self.assertEqual(result["calibration"][p]["parameter_changes"]["lower_actor"], 0)
                self.assertEqual(result["training"][p]["lower_ppo"]["parameter_changes"]["upper_actor"], 0)
                self.assertGreater(result["training"][p]["joint_ppo"]["parameter_changes"]["upper_actor"], 0)
                self.assertGreater(sum(r["executed_plan_delta_squared_sum"] for r in
                    result["evaluation_rows"][p]["joint_ppo"]["deterministic"]), 0)
            mutations = (("optimizer", "PPO optimizer"), ("network", "network"), ("actor_type", "executed learned policy"), ("config", "hyperparameters"))
            for mutation, message in mutations:
                bad = copy.deepcopy(result)
                if mutation == "optimizer":
                    bad["training"]["50"]["joint_ppo"]["history"][0]["optimizer_steps"]["upper_actor_optimizer_steps"] += 1
                elif mutation == "network":
                    bad["training"]["100"]["joint_ppo"]["parameter_changes"]["upper_actor"] = 0
                elif mutation == "actor_type":
                    bad["evaluation_rows"]["50"]["clone"]["deterministic"][0]["lower_actor_type"] = "VelocityFeedbackActor"
                else:
                    bad["config"]["lower_learning_rate"] *= 2
                with self.assertRaisesRegex(ValueError, message):
                    experiment.aggregate([bad], preflight=True)

    def test_fresh_paths_full_budget_placement_and_signed_endpoints(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                old = {s for values in spec.previous.seed_roles(root, preflight=preflight).values() for s in values}
                for seeds in spec.seed_roles(root, preflight=preflight).values():
                    self.assertFalse(set(seeds) & seen)
                    self.assertFalse(set(seeds) & old)
                    seen.update(seeds)
            task = task_specification("unit_stage55", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 16128000)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 13440)
        means = {str(p): {"deterministic": {k: {"episode_return": v} for k, v in
                 zip(spec.POLICIES, (0, 1, 2, 3, 4, 5))}} for p in spec.PERIODS}
        self.assertEqual(list(spec.contrasts(means).values()), [1, 2, -1, 2, 1, 5, 3] * 2)
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, [-1., 1., 0., 2., -2., 3., 4.] * 2))} for _ in range(8)]
        result = experiment.bootstrap(rows)
        self.assertEqual(result[spec.ENDPOINTS[0]]["effect"], "negative")
        self.assertEqual(result[spec.ENDPOINTS[1]]["effect"], "positive")
        self.assertEqual(result[spec.ENDPOINTS[2]]["effect"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
