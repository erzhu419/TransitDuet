from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from freq_hrl.experiments import pointmaze_timing_pair as timing
from scripts import pointmaze_temporal_plan_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options, seed_roles
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class FakeTask:
    def __init__(self, future_shift=0.):
        self.environment = self
        self.action_low, self.action_high = -np.ones(2), np.ones(2)
        self.future_shift = future_shift
        self.closed = False

    def observation(self):
        target = np.array([.8 + self.t * .001 + (self.future_shift if self.t >= 80 else 0.), .5])
        achieved = self.position.copy()
        return SimpleNamespace(physical=np.r_[achieved, self.velocity], achieved_goal=achieved,
                               target=target, target_error=target - achieved,
                               task_measurement=np.r_[target, [.1, .2, .3, .4]])

    def reset(self):
        self.t = 0
        self.position, self.velocity = np.zeros(2), np.zeros(2)
        return self.observation()

    def step(self, action):
        self.t += 1
        self.velocity = np.asarray(action).copy()
        self.position += self.velocity * .01
        obs = self.observation()
        return obs, -1., False, False, {"tracking_distance": np.linalg.norm(obs.target_error)}

    def close(self):
        self.closed = True


class FakeController:
    def reset_recurrent_inference(self):
        pass

    def plan_goal(self, state, sample):
        return {"action": np.array([.2, -.1])}

    def act_conditioned(self, state, sample):
        return {"action": np.tanh(state[4:6])}


class TemporalPlanTest(unittest.TestCase):
    def test_new_roles_and_balanced_coverage_are_disjoint_from_inherited_paths(self):
        all_new = set()
        for root in (208001, 209011, 209061):
            preflight = root == 208001
            roles = temporal.path_roles(root, preflight=preflight)
            flat = roles["fit"] + roles["evaluation"]
            self.assertFalse(all_new.intersection(flat))
            all_new.update(flat)
            self.assertFalse(set(flat).intersection(s for values in seed_roles(root).values() for s in values))
            self.assertEqual(len(flat), 4 if preflight else 24)
            cases = temporal.cases_for_paths(root, flat, horizon=300 if preflight else 1200,
                                             pairs_per_path=2 if preflight else 20)
            self.assertEqual(len(cases), 8 if preflight else 480)
            for s in flat:
                checks = [c["check_step"] for c in cases if c["seed"] == s]
                self.assertEqual(len(checks), len(set(checks)))
                if not preflight:
                    self.assertEqual([sum(c % 50 == o for c in checks) for o in (0, 5, 10, 15, 20)], [4] * 5)
                for c in checks:
                    now, wait = temporal.pair_schedules(s, c, 300 if preflight else 1200)
                    self.assertEqual([v for v in now if v < c], [v for v in wait if v < c])
                    self.assertEqual([i for i,(a,b) in enumerate(zip(now,wait)) if a != b], [c // 50])
                    self.assertEqual(wait[c // 50] - now[c // 50], 5)

    def test_prefix_capture_matches_frozen_rollout_and_excludes_future(self):
        opts = cell_options(208001, preflight=True)
        args = SimpleNamespace(**opts)
        scale = PhysicalTimeScaleContract(dt_seconds=.01, upper_period_seconds=.5,
                                          history_seconds=.64, fast_period_seconds=.04)
        schedule = temporal.pair_schedules(3259001, 50, 300)[0]
        bounds = (np.array([-2., -2.]), np.array([2., 2.]))
        task = FakeTask()
        with patch.object(temporal, "_make_task", return_value=task), patch.object(temporal, "pointmaze_goal_bounds", return_value=bounds):
            row = temporal.rollout_window(FakeController(), seed=1, check=50, schedule=schedule, args=args, time_scale=scale)
        self.assertTrue(task.closed)
        self.assertEqual(row["primitive_steps"], 100)
        self.assertEqual(row["sequence"].shape, (64, len(row["feature_names"])))
        np.testing.assert_equal(row["sequence"][:13, -1], np.zeros(13))
        np.testing.assert_equal(row["sequence"][13:, -1], np.ones(51))
        with patch.object(timing, "_make_task", return_value=FakeTask()), patch.object(timing, "pointmaze_goal_bounds", return_value=bounds):
            original = timing.rollout_timing_schedule(FakeController(), seed=1, decision_steps=schedule,
                capture_step=50, env_id=args.env_id, horizon=300, time_scale=scale,
                maximum_subgoal_delta=args.maximum_subgoal_delta, task_options=temporal._task_options(args), credit_window_steps=50)
        self.assertAlmostEqual(row["step_ise"].sum(), original["credit_window_squared_error_integral"])
        with patch.object(temporal, "_make_task", return_value=FakeTask(2.)), patch.object(temporal, "pointmaze_goal_bounds", return_value=bounds):
            changed = temporal.rollout_window(FakeController(), seed=1, check=50, schedule=schedule, args=args, time_scale=scale)
        np.testing.assert_equal(row["sequence"], changed["sequence"])
        np.testing.assert_equal(row["policy_prefix"], changed["policy_prefix"])
        self.assertFalse(np.array_equal(row["step_ise"], changed["step_ise"]))

    def test_curve_sign_credit_length_and_equal_budget(self):
        common = {"sequence": np.zeros((64, 3)), "policy_prefix": np.zeros(5),
                  "feature_names": ["a", "b", "c"], "primitive_steps": 100}
        now = {**common, "step_ise": np.full(50, .01), "calls": [0, 50]}
        wait = {**common, "step_ise": np.full(50, .02), "calls": [0, 55]}
        row = temporal.combine_pair({"seed": 1, "check_step": 50}, now, wait)
        np.testing.assert_allclose(row["curve"], [.1, .25, .5])
        self.assertEqual(row["primitive_steps"], 200)
        with self.assertRaisesRegex(RuntimeError, "budgets"):
            temporal.combine_pair({"seed": 1, "check_step": 50}, now, {**wait, "calls": [0]})
        with self.assertRaisesRegex(RuntimeError, "before intervention"):
            temporal.combine_pair({"seed": 1, "check_step": 50}, now, {**wait, "sequence": np.ones((64, 3))})

    def test_representation_controls_keep_current_and_past_multiset(self):
        x = np.arange(2 * 64 * 3).reshape(2, 64, 3).astype(float)
        rows = [{"seed": s, "check_step": 50} for s in (1, 2)]
        shuffled = temporal.history_view(x, rows, "shuffled_history", root=3)
        np.testing.assert_equal(shuffled[:, -1], x[:, -1])
        for a,b in zip(x,shuffled):
            self.assertEqual(sorted(map(tuple,a[:-1])), sorted(map(tuple,b[:-1])))
        self.assertFalse(np.array_equal(x, shuffled))
        repeated = temporal.history_view(x, rows, "current_repeat", root=3)
        np.testing.assert_equal(repeated, np.repeat(x[:, -1:], 64, axis=1))

    def test_equal_capacity_and_evaluation_labels_do_not_affect_training(self):
        rng = np.random.default_rng(26)
        rows = [{"seed": s, "check_step": 50, "sequence": rng.normal(size=(64,3)).astype(np.float32),
                 "curve": rng.normal(size=3), "feature_names": ["a", "b", "c"]} for s in range(6)]
        train, query = rows[:4], rows[4:]
        predictions, diag, states = temporal.fit_temporal(train, query, root=26, epochs=2)
        changed = deepcopy(query)
        for r in changed:
            r["curve"] *= 1000
        other, other_diag, other_states = temporal.fit_temporal(train, changed, root=26, epochs=2)
        self.assertEqual(diag, other_diag)
        self.assertEqual(len({d["parameter_count"] for d in diag["fits"].values()}), 1)
        for m in temporal.METHODS:
            self.assertEqual(diag["fits"][m]["optimizer_steps"], 2)
            np.testing.assert_equal(predictions[m], other[m])
            for k in states[m]:
                np.testing.assert_equal(states[m][k].numpy(), other_states[m][k].numpy())
        with self.assertRaisesRegex(ValueError, "paths overlap"):
            temporal.fit_temporal(train, train[:1], root=26, epochs=2)

    def test_prediction_and_decision_gates_require_all_controls(self):
        rows = [{"curve": np.asarray(temporal.HORIZONS) * .01 * v,
                 "predicted_rates": {"history": np.full(3,v), "current_repeat": np.zeros(3),
                                     "shuffled_history": np.full(3,-v)}} for v in (1., -1.)]
        metrics = temporal.summarize(rows)
        self.assertTrue(metrics["prediction_gate_passed"])
        self.assertTrue(metrics["decision_gate_passed"])
        self.assertEqual(metrics["history_ise_benefit_vs_control"]["always_now"], .25)
        rows[0]["predicted_rates"]["current_repeat"] = np.ones(3)
        self.assertFalse(temporal.summarize(rows)["decision_gate_passed"])

    def test_scheduler_stages_only_source_not_raw_cache(self):
        for preflight in (True, False):
            root = spec.roots(preflight=preflight)[0]
            task = task_specification("unit_temporal", root, preflight=preflight, protocol_spec=spec)
            self.assertEqual(task["cpu"], 2 if preflight else 17)
            self.assertEqual(task["ram_mb"], 3072 if preflight else 24576)
            self.assertIsNone(task["require_node"])
            self.assertIn(str(spec.source_result(root, preflight=preflight).parent), task["stage_input_paths"])
            self.assertFalse(any(p.endswith("_raw") for p in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
