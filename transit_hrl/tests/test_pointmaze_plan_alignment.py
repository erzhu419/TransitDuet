import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_plan_alignment as experiment
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeDriver
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_plan_alignment_stage52_spec as spec
from scripts.submit_pointmaze_plan_alignment_stage52_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class PlanAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(52)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False,
            lower_value_state_dim=392, promotion_state_dim=394))

    def rollout(self, policy, *, shift=0., period=100, hook=True, controller=None):
        model = self.controller() if controller is None else controller
        original = model.lower_actor
        if policy != "frozen":
            model.lower_actor = experiment.FeedbackActor(original, experiment.feedback_gain(), "waypoint")
        reference = experiment.CausalReference(policy)
        args = spec.arguments(310001, preflight=True)
        args.horizon = 200
        try:
            with patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask(shift)), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                _, row, raw = joint.rollout(model, args, f"fixed{period}", seed=1, sample=False, capture=True,
                    lower_credit="task_option", lower_value_context_builder=clocks.context_builder("task_clock"),
                    lower_reference_builder=reference if hook else None)
        finally:
            model.lower_actor = original
        return row, raw, reference

    def test_velocity_fit_exact_and_valid_history_not_padding(self):
        t = np.arange(64) * .01
        samples = np.array([.2, -.4]) + t[:, None] * [.3, -.2]
        np.testing.assert_allclose(experiment.fit_velocity(samples), [.3, -.2], atol=1e-12, rtol=0)
        reference = experiment.CausalReference("target_curve")
        zero = SimpleNamespace(task_measurement=np.zeros(6))
        history = SimpleNamespace(history=np.zeros(384))
        kwargs = dict(subgoal=np.zeros(2), world_low=-2 * np.ones(2), world_high=2 * np.ones(2))
        np.testing.assert_array_equal(reference(observation=zero, history=history, age=0, step=0, **kwargs), [0., 0.])
        self.assertEqual(reference.fits, 0)
        frames = np.full((64, 6), 1000., dtype=np.float32)
        frames[-11:, :2] = np.arange(11)[:, None] * [.003, -.002]
        history.history = frames.reshape(-1)
        current = SimpleNamespace(task_measurement=np.r_[frames[-1, :2], np.zeros(4)])
        reference(observation=current, history=history, age=0, step=10, **kwargs)
        np.testing.assert_allclose(reference.velocity, [.3, -.2], atol=1e-7, rtol=0)
        self.assertEqual(reference.fits, 1)

    def test_between_renewals_only_age_advances_and_bounds_apply(self):
        reference = experiment.CausalReference("target_curve")
        frames = np.zeros((64, 6), dtype=np.float32)
        frames[:, :2] = np.arange(64)[:, None] * [.003, -.002]
        history = SimpleNamespace(history=frames.reshape(-1))
        current = SimpleNamespace(task_measurement=np.r_[frames[-1, :2], np.zeros(4)])
        kwargs = dict(subgoal=np.zeros(2), world_low=-np.ones(2), world_high=np.ones(2))
        reference(observation=current, history=history, age=0, step=63, **kwargs)
        anchor, velocity = reference.anchor.copy(), reference.velocity.copy()
        current.task_measurement[:] = 999.
        history.history[:] = -999.
        actual = reference(observation=current, history=history, age=20, step=83, **kwargs)
        np.testing.assert_array_equal(actual, np.clip(anchor + .2 * velocity, -1, 1).astype(np.float32))
        self.assertEqual(reference.fits, 1)
        actual = reference(observation=current, history=history, age=10000, step=10063, **kwargs)
        np.testing.assert_array_equal(actual, [1., -1.])

    def test_default_frozen_rollout_is_unchanged_with_reference_hook(self):
        before, raw_before, _ = self.rollout("frozen", hook=False)
        after, raw_after, _ = self.rollout("frozen", hook=True)
        for key in ("episode_return", "tracking_squared_error_integral", "charged_utility", "decision_steps",
                    "upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
            self.assertEqual(before[key], after[key])
        for key in raw_before:
            np.testing.assert_array_equal(raw_after[key], raw_before[key])
        np.testing.assert_array_equal(raw_after["lower_reference"], raw_after["subgoal"])

    def test_future_target_does_not_change_plan_before_next_renewal(self):
        for policy in ("target_hold", "target_curve", "reverse_curve", "current_target"):
            _, left, _ = self.rollout(policy)
            _, right, _ = self.rollout(policy, shift=2.)
            stop = 80 if policy == "current_target" else 100
            np.testing.assert_array_equal(left["lower_reference"][:stop], right["lower_reference"][:stop])
            np.testing.assert_array_equal(left["action"][:stop], right["action"][:stop])
            self.assertFalse(np.array_equal(left["lower_reference"][stop:], right["lower_reference"][stop:]))

    def test_reference_audit_and_non_null_curve_direction(self):
        row, forward, ref = self.rollout("target_curve", period=50)
        audit = experiment.audit_reference(forward, row, policy="target_curve", period=50, bounds=ref.bounds)
        self.assertEqual(ref.fits, 3)
        self.assertEqual(audit["audit_regression_fits"], 3)
        _, backward, _ = self.rollout("reverse_curve", period=50)
        self.assertFalse(np.array_equal(forward["lower_reference"][50:], backward["lower_reference"][50:]))
        bad = copy.deepcopy(forward)
        bad["lower_reference"][75, 0] += .1
        with self.assertRaisesRegex(AssertionError, "causal renewal-only"):
            experiment.audit_reference(bad, row, policy="target_curve", period=50, bounds=ref.bounds)

    def test_native_target_cancellation_audit_is_bitwise_exact(self):
        root = spec.roots(preflight=False)[1]
        args = spec.arguments(root, preflight=False)
        reassociated_mismatches = 0
        bounds = (-2 * np.ones(2), 2 * np.ones(2))
        for seed in spec.seed_roles(root, preflight=False)["evaluation"]:
            driver = PointMazeRegimeDriver(seed=seed, horizon=args.horizon,
                dt_seconds=spec.DT_SECONDS, **joint._task_options(args))
            measurements = np.array([np.concatenate(driver.sample(t)) for t in range(args.horizon)])
            for period in spec.PERIODS:
                for policy in ("target_curve", "reverse_curve"):
                    reference, actual = experiment.CausalReference(policy), []
                    for step, measurement in enumerate(measurements):
                        start = max(0, step + 1 - spec.LOOKBACK_STEPS)
                        history = SimpleNamespace(history=measurements[start:step + 1].reshape(-1))
                        actual.append(reference(observation=SimpleNamespace(task_measurement=measurement),
                            history=history, subgoal=np.zeros(2), age=step % period, step=step,
                            world_low=bounds[0], world_high=bounds[1]))
                    raw = {"measurement": measurements, "lower_reference": np.array(actual)}
                    row = {"episode_length": args.horizon}
                    audit = experiment.audit_reference(raw, row, policy=policy, period=period, bounds=bounds)
                    self.assertEqual(audit["audit_regression_fits"], reference.fits)
                    for start in range(period, args.horizon, period):
                        velocity = experiment.fit_velocity(measurements[max(0, start + 1 - 64):start + 1, :2])
                        velocity *= -1 if policy == "reverse_curve" else 1
                        age_seconds = np.arange(period, dtype=np.float64) * spec.DT_SECONDS
                        old = np.clip(measurements[start, :2].astype(np.float64)
                            + age_seconds[:, None] * velocity, *bounds).astype(np.float32)
                        reassociated_mismatches += np.count_nonzero(old != raw["lower_reference"][start:start + period])
                    raw["lower_reference"][1, 0] = np.nextafter(raw["lower_reference"][1, 0], np.float32(np.inf))
                    with self.assertRaisesRegex(AssertionError, "causal renewal-only"):
                        experiment.audit_reference(raw, row, policy=policy, period=period, bounds=bounds)
        self.assertGreater(reassociated_mismatches, 0)

    def test_native_style_pipeline_equal_costs_and_roster_failures(self):
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
            self.assertEqual(summary["method_cost"]["primitive_steps"], 14400)
            self.assertEqual(summary["method_cost"]["upper_inference_calls"], 216)
            self.assertEqual(summary["method_cost"]["gate_inference_calls"], 0)
            self.assertEqual(summary["method_cost"]["plan_regression_fits"], 56)
            self.assertEqual(summary["method_cost"]["audit_regression_fits"], 56)
            self.assertEqual(summary["native_trace_audits"], 48)
            self.assertNotIn("primary_endpoints", summary)
            torch.testing.assert_close(joint.inference_weights(model), original, atol=0, rtol=0)
            self.assertIsInstance(joint._WORKER[0].lower_actor, type(model.lower_actor))
            bad = copy.deepcopy(result)
            bad["evaluation_rows"]["50"]["target_curve"]["deterministic"][0]["upper_inference_calls"] += 1
            with self.assertRaisesRegex(ValueError, "equal-call accounting"):
                experiment.aggregate([bad], preflight=True)
            bad = copy.deepcopy(result)
            bad["evaluation_rows"]["100"].pop("reverse_curve")
            with self.assertRaisesRegex(ValueError, "policy roster"):
                experiment.aggregate([bad], preflight=True)

    def test_fresh_paths_full_budgets_dynamic_placement_and_signed_intervals(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = spec.seed_roles(root, preflight=preflight)["evaluation"]
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & {s for v in spec.previous.seed_roles(root, preflight=preflight).values() for s in v})
                seen.update(seeds)
            task = task_specification("unit_stage52", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertNotIn("result_dir", task)
        budget = spec.budget(preflight=False)
        self.assertEqual(8 * budget["total_primitive_steps"], 3686400)
        self.assertEqual(8 * budget["upper_inference_calls"], 55296)
        self.assertEqual(8 * budget["plan_regression_fits"], 17408)
        means = {str(p): {"deterministic": {k: {"episode_return": v} for k, v in
                 zip(spec.POLICIES, (0, 1, 2, 3, 4, 5))}} for p in spec.PERIODS}
        self.assertEqual(list(spec.contrasts(means).values()), [1, 1, -1, 3, 1, 1, -1, 3])
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, [-1., 1., 0., 2., -2., 3., 0., 4.]))} for _ in range(8)]
        result = experiment.bootstrap_endpoints(rows)
        self.assertEqual(result[spec.ENDPOINTS[0]]["effect"], "negative")
        self.assertEqual(result[spec.ENDPOINTS[1]]["effect"], "positive")
        self.assertEqual(result[spec.ENDPOINTS[2]]["effect"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
