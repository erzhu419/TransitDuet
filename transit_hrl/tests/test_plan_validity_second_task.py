import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.mujoco import PointMazeRegimeDriver, PointMazeRegimeObservation
from freq_hrl.domains.mujoco.reacher_tracking import (
    REACHER_POSITION_SCALE_METRES, ReacherSubgoalAdapter, ReacherTrackingDriver, ReacherTrackingTask,
)
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments import pointmaze_budgeted_trigger as legacy
from freq_hrl.experiments import pointmaze_plan_validity_branching as branching
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule, collect_timing_pairs
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    PointMazePlanValueDimensions, build_pointmaze_plan_value_model,
)
from scripts import run_plan_validity_second_task as runner
from scripts import run_pointmaze_timing_baselines as timing_runner
from test_pointmaze_timing_baselines import fitting_rows


def options():
    parser = timing_runner.build_timing_parser()
    return parser.parse_args([
        "--optimizer-seed", "305001", "--horizon", "300", "--pairs-per-path", "2",
        "--train-seeds", "1", "2", "3", "--selection-seeds", "4", "5",
        "--branch-fit-seeds", "6", "7", "8", "--timing-selection-seeds", "9", "10",
        "--trigger-eval-seeds", "11", "12", "--output", "/tmp/unused_second_task_plan.json"])


class DenseTask:
    action_low, action_high = np.full(2, -1.0), np.ones(2)
    dt_seconds = 0.01

    def __init__(self, seed, horizon, **kwargs):
        self.environment = SimpleNamespace(close=lambda: None)
        self.horizon = horizon
        self.driver = PointMazeRegimeDriver(seed=seed, horizon=horizon, dt_seconds=0.01, **kwargs)
        self.position, self.velocity = np.zeros(2), np.zeros(2)
        self.index = 0

    def observation(self):
        target, force, distractor = self.driver.sample(self.index)
        return PointMazeRegimeObservation(np.r_[self.position, self.velocity].astype(np.float32),
            self.position.astype(np.float32), np.r_[target, force, distractor], target, force, distractor)

    def reset(self):
        return self.observation()

    def step(self, action):
        target, force, _ = self.driver.sample(self.index)
        self.velocity = 0.98 * self.velocity + 0.02 * np.clip(action + force, -1, 1)
        self.position += 0.01 * self.velocity
        self.index += 1
        distance = float(np.linalg.norm(self.position - target))
        return self.observation(), float(np.exp(-distance)), False, self.index == self.horizon, {
            "tracking_distance": distance, "tracking_success": float(distance < 0.45)}

    def privileged_context(self):
        return np.eye(4)[self.driver.regime_id(self.index)]

    def diagnostics(self):
        return self.driver.diagnostics()


class Controller:
    config = SimpleNamespace(gamma=0.995, state_encoder="mlp")

    def reset_recurrent_inference(self):
        pass

    def plan_goal(self, state, **kwargs):
        return {"action": state[4:6] * 0.5, "logp": 0.0, "value": 0.0}

    def act_conditioned(self, state, **kwargs):
        return {"action": state[4:6] * 0.2 - state[2:4] * 0.1, "logp": 0.0, "value": 0.0}


class SecondTaskTest(unittest.TestCase):
    def test_reacher_retains_the_existing_exogenous_clocks_but_changes_target_geometry(self):
        old = PointMazeRegimeDriver(seed=71, horizon=1200, dt_seconds=0.01)
        new = ReacherTrackingDriver(seed=71, horizon=1200, dt_seconds=0.01)
        self.assertEqual(old.regime_change_steps, new.regime_change_steps)
        self.assertEqual(old.pulse_start_steps, new.pulse_start_steps)
        self.assertEqual(old.distractor_change_steps, new.distractor_change_steps)
        for step in [0, 100, 700, 1200]:
            target, force, distractor = new.sample(step)
            self.assertLessEqual(np.linalg.norm(target), 0.65 + 1e-7)
            np.testing.assert_array_equal(force, old.sample(step)[1])
            np.testing.assert_array_equal(distractor, old.sample(step)[2])
        self.assertNotIn("pointmaze", json.dumps(new.diagnostics()).lower())

    def test_cartesian_goals_respect_the_real_joint_limit_workspace(self):
        adapter = ReacherSubgoalAdapter(maximum_delta=np.full(2, 0.75))
        rng = np.random.default_rng(17)
        for _ in range(100):
            goal = adapter.decode(rng.normal(size=2) * 5, np.array([0.6, 0.2]))
            radius = float(np.linalg.norm(goal))
            self.assertGreaterEqual(radius, 0.1 - 1e-7)
            self.assertLessEqual(radius, 0.95 + 1e-6)
            elbow_cosine = ((radius * 0.21) ** 2 - 0.1 ** 2 - 0.11 ** 2) / (2 * 0.1 * 0.11)
            self.assertGreaterEqual(elbow_cosine, np.cos(3.0))
            self.assertLessEqual(elbow_cosine, 1)

    def test_native_reacher_adapter_dt_score_alignment_and_visual_target(self):
        # Six API steps, not a training or performance experiment.
        task = ReacherTrackingTask(seed=71, horizon=6)
        try:
            observation = task.reset()
            self.assertAlmostEqual(task.dt_seconds, 0.01)
            self.assertEqual(observation.physical.shape, (4,))
            self.assertEqual(observation.task_measurement.shape, (6,))
            self.assertFalse(hasattr(observation, "regime_id"))
            for step in range(6):
                old_target = observation.target.copy()
                observation, reward, terminated, truncated, info = task.step(np.array([0.8, -0.6]))
                expected = np.linalg.norm(observation.achieved_goal - old_target)
                self.assertAlmostEqual(info["tracking_distance"], expected, places=6)
                self.assertAlmostEqual(reward, np.exp(-expected), places=6)
                np.testing.assert_allclose(task.environment.unwrapped.get_body_com("target")[:2],
                                           observation.target * REACHER_POSITION_SCALE_METRES, atol=1e-7)
                self.assertFalse(terminated)
                self.assertEqual(truncated, step == 5)
        finally:
            task.environment.close()

    def test_shared_kernel_replays_legacy_stage9_decisions_and_episode_scores(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        time_scale = shared.time_contract(args)
        fitted = legacy.fit_budgeted_trigger_predictors(fitting_rows(), alpha_grid=[1.0])
        name = "causal_validity_interactions"
        rule = TimingRule(name, "legacy", view=name, threshold=fitted[name]["threshold"], model=fitted[name])
        bounds = np.full(2, -2.0), np.full(2, 2.0)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(legacy, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=bounds),
            patch.object(legacy, "pointmaze_goal_bounds", return_value=bounds),
        ):
            for policy in [rule, TimingRule("balanced_jitter", "jitter")]:
                old = legacy.rollout_budgeted_trigger(Controller(), seed=71, mode=policy.name, fitted_predictors=fitted,
                    env_id=args.env_id, horizon=300, time_scale=time_scale, maximum_subgoal_delta=0.75,
                    max_offset_steps=25, check_stride_steps=5, task_options=spec.options)
                _, new = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                                             time_scale=time_scale, rule=policy)
                self.assertEqual(old["decision_steps"], new["decision_steps"])
                self.assertEqual(old["trigger_score_checks"], new["trigger_score_checks"])
                for field in ("episode_return", "tracking_squared_error_integral"):
                    self.assertAlmostEqual(old[field], new[field], places=12)

    def test_keep_renew_target_and_features_replay_the_legacy_branch(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        time_scale = shared.time_contract(args)
        schedule = (0, 50, 100, 150, 200, 250)
        query = branching.PlanRenewalOpportunity(category="neutral_matched", step=75, source_step=None, lag_steps=None)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(branching, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            patch.object(branching, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
        ):
            for renew in [False, True]:
                old = branching._run_branch(Controller(), seed=71, opportunity=query, renew=renew,
                    branch_window_steps=50, env_id=args.env_id, horizon=300, time_scale=time_scale,
                    maximum_subgoal_delta=0.75, task_options=spec.options, prefix_decision_steps=schedule)
                _, new = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                    time_scale=time_scale, schedule=schedule, branch_step=75, renew=renew)
                self.assertAlmostEqual(old["branch_tracking_squared_error_integral"], new["branch_ise"], places=12)
                np.testing.assert_array_equal(old["feature_vector"], new["features"])
                self.assertTrue(all(step <= 75 for step in new["decision_steps"]))

    def test_shared_kernel_training_uses_actual_durations_and_lower_boundaries(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        schedule = (0, 55, 110, 165, 220, 275)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
        ):
            batch, row = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                time_scale=shared.time_contract(args), schedule=schedule, sample=True)
        np.testing.assert_array_equal(batch.upper.duration, [55, 55, 55, 55, 55, 25])
        self.assertEqual(batch.lower.size, 300)
        np.testing.assert_array_equal(np.flatnonzero(batch.lower.done), [54, 109, 164, 219, 274, 299])
        self.assertEqual(row["upper_decision_count"], 6)

    def test_existing_timing_pair_collector_uses_the_shared_kernel(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        time_scale = shared.time_contract(args)
        rule = TimingRule("clock_10", "clock", threshold=0.4)

        def rollout(controller, policy, *, seed, args, time_scale, **kwargs):
            return shared.run_episode(controller, task_spec=spec, seed=seed, args=args,
                                       time_scale=time_scale, rule=policy, **kwargs)[1]

        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
        ):
            rows, steps = collect_timing_pairs(Controller(), [rule], seeds=[71], args=args,
                time_scale=time_scale, round_index=0, rollout_fn=rollout)
        self.assertEqual(len(rows), 2)
        self.assertEqual(steps, 1500)
        self.assertTrue(all(len(row["causal_features"]) == 37 for row in rows))

    def test_shared_renewal_collector_retains_query_provenance_and_window_cost(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        query = branching.PlanRenewalOpportunity("regime_lag_100ms", 80, 70, 10)
        schedule = (0, 50, 100, 150, 200, 250)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            patch.object(shared, "jitter_schedule", return_value=schedule),
            patch.object(shared, "plan_renewal_opportunities", return_value=[query]),
        ):
            rows = shared.collect_renewal_labels(Controller(), task_spec=spec, args=args,
                                                 time_scale=shared.time_contract(args))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["source_step"], 70)
            self.assertEqual(row["lag_steps"], 10)
            self.assertEqual(row["keep_primitive_steps_replayed"], 130)
            self.assertEqual(row["renew_ise_advantage"], row["keep_tracking_squared_error_integral"]
                             - row["renew_tracking_squared_error_integral"])
            self.assertEqual(row["downstream_upper_call_count_per_branch"], 0)

    def test_shared_controller_initialization_matches_the_stage9_ppo_backend(self):
        args = options()
        args.reference_hidden_dim = 16
        time_scale = shared.time_contract(args)
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        dimensions = PointMazePlanValueDimensions(physical=4, goal=2, task=6, history=384,
            oracle_context=4, base_upper=390, oracle_upper=394, lower=390, action=2)
        expected, _ = build_pointmaze_plan_value_model(method="hrl_regime_history", dimensions=dimensions,
            reference_hidden_dim=16, learning_rate=args.learning_rate, optimizer_seed=args.optimizer_seed)

        def train(**kwargs):
            model = kwargs["model"]
            self.assertEqual(model.config, expected.config)
            for network in ["upper_actor", "lower_actor", "upper_value", "lower_value"]:
                for old, new in zip(getattr(expected, network).parameters(), getattr(model, network).parameters()):
                    self.assertTrue(torch.equal(old, new))
            self.assertEqual(kwargs["eval_seeds"], args.timing_selection_seeds)
            self.assertNotIn(args.trigger_eval_seeds[0], kwargs["eval_seeds"])
            return kwargs["metadata"], [], model

        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "train_frequency_separated_ppo", side_effect=train),
        ):
            shared.train_shared_controller(spec, args, time_scale)

    def test_second_task_defaults_to_preparation_and_reuses_the_comparison_function(self):
        self.assertIs(runner.fit_timing_comparison, timing_runner.fit_timing_comparison)
        command = ["--optimizer-seed", "305001", "--train-seeds", "1", "2", "3",
            "--selection-seeds", "4", "5", "--branch-fit-seeds", "6", "7", "8",
            "--timing-selection-seeds", "9", "10", "--trigger-eval-seeds", "11", "12"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan.json"
            with patch.object(runner, "run_task", side_effect=AssertionError("must not run")):
                runner.main([*command, "--output", str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(data["status"], "prepared")
            self.assertEqual(data["protocol"]["tasks"], ["pointmaze", "reacher_tracking"])
            self.assertFalse(data["protocol"]["scheduler_submission"])
            self.assertEqual(data["cells"], [])
            self.assertEqual(list(Path(folder).iterdir()), [output])
            with self.assertRaisesRegex(ValueError, "q75"):
                runner.main([*command, "--threshold-quantile", "0.5", "--output", str(output)])

    def test_both_tasks_delegate_to_one_fit_comparison_and_keep_compact_learning_logs(self):
        args = options()
        payload = {"task_spec": {}, "history": [
            {"iteration": -1, "tracking_squared_error_integral_mean": 2},
            {"iteration": 0},
            {"iteration": 1, "checkpoint_evaluation_performed": True,
             "tracking_squared_error_integral_mean": 1}]}
        labels = [{"renew_waypoint_delta": 0.2, "renew_ise_advantage": 0.03}]
        with (
            patch.object(runner, "train_shared_controller", return_value=(payload, object())),
            patch.object(runner, "collect_renewal_labels", return_value=labels),
            patch.object(runner, "fit_timing_comparison", side_effect=lambda *a, **kw: {}) as compare,
        ):
            for name in shared.TASKS:
                result = runner.run_task(args, name)
                self.assertEqual(result["task"], name)
                self.assertEqual(len(result["controller_selection_history"]), 2)
                self.assertEqual(result["fit_branch_summary"]["mean_waypoint_change"], 0.2)
            self.assertEqual(compare.call_count, 2)
            for call in compare.call_args_list:
                self.assertIs(call.args[0], args)
                self.assertIs(call.args[3], labels)
                self.assertTrue(callable(call.kwargs["rollout_fn"]))
                self.assertTrue(callable(call.kwargs["pair_collector"]))


if __name__ == "__main__":
    unittest.main()
