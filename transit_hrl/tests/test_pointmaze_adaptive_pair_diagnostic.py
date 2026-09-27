from argparse import Namespace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments import pointmaze_adaptive_pair_diagnostic as diagnostic
from scripts import pointmaze_adaptive_pair_diagnostic_spec as spec
from scripts.pointmaze_timing_pair_stage12_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class ToyTask:
    action_low, action_high = np.full(2, -1.0), np.ones(2)

    def __init__(self):
        self.environment = SimpleNamespace(close=lambda: None)
        self.step_index = 0
        self.position = np.zeros(2)

    def observation(self):
        return SimpleNamespace(
            physical=self.position.copy(), achieved_goal=self.position.copy(),
            target=np.array([0.01 * self.step_index, 0.0]),
            task_measurement=np.array([float(self.step_index)]),
        )

    def reset(self):
        return self.observation()

    def step(self, action):
        self.position += action * 0.01
        self.step_index += 1
        observation = self.observation()
        distance = float(np.linalg.norm(observation.target - self.position))
        return observation, -distance, False, self.step_index == 300, {"tracking_distance": distance}


class ToyHistory:
    def __init__(self, **kwargs):
        self.history = np.zeros(1)

    def reset(self, observation):
        self.history = observation.task_measurement.copy()

    update = reset

    def upper_state(self, observation, **kwargs):
        return observation.target - observation.achieved_goal

    def lower_state(self, observation, *, subgoal):
        return subgoal - observation.achieved_goal


class ToyController:
    def reset_recurrent_inference(self):
        pass

    def plan_goal(self, state, **kwargs):
        return {"action": state}

    act_conditioned = plan_goal


class AdaptivePairDiagnosticTest(unittest.TestCase):
    def test_wait_one_check_is_not_deadline_and_intervention_does_not_persist(self):
        args = Namespace(**cell_options(208001, preflight=True))
        time_scale = PhysicalTimeScaleContract(dt_seconds=0.01, upper_period_seconds=0.5,
                                               history_seconds=0.64, fast_period_seconds=0.04)
        with (
            patch.object(diagnostic, "_make_task", side_effect=lambda **kw: ToyTask()),
            patch.object(diagnostic, "pointmaze_goal_bounds", return_value=(np.full(2, -10), np.full(2, 10))),
            patch.object(diagnostic, "PointMazeRegimeFeatureBuilder", ToyHistory),
            patch.object(diagnostic, "_causal_plan_features", side_effect=lambda **kw: (
                ["step"], kw["observation"].task_measurement.copy(), None)),
            patch.object(diagnostic, "_predict_renewal_advantage", side_effect=lambda **kw:
                         float(kw["features"][0] % 50 >= 5)),
        ):
            for check in (50, 55):
                results = {arm: diagnostic.rollout_intervention(
                    ToyController(), seed=1, intervention_step=check, arm=arm,
                    predictor={"threshold": 0.5}, args=args, time_scale=time_scale,
                ) for arm in diagnostic.ARMS}
                self.assertEqual(results["now"]["decision_steps"][1], check)
                self.assertEqual(results["wait_one_check"]["decision_steps"][1], check + 5)
                self.assertEqual(results["wait_deadline"]["decision_steps"][1], 75)
                for result in results.values():
                    self.assertEqual(result["decision_steps"][2:], [105, 155, 205, 255])
                    np.testing.assert_array_equal(result["prefix"], results["now"]["prefix"])
            factual = diagnostic.rollout_intervention(
                ToyController(), seed=1, intervention_step=None, arm=None,
                predictor={"threshold": 0.5}, args=args, time_scale=time_scale,
            )
            self.assertEqual(factual["decision_steps"], [0, 55, 105, 155, 205, 255])
            all_wait = diagnostic.rollout_intervention(
                ToyController(), seed=1, intervention_step=None, arm=None,
                predictor={"threshold": 0.0}, args=args, time_scale=time_scale,
                score_fn=lambda *a: -1.0,
            )
            self.assertEqual(all_wait["decision_steps"], [0, 75, 125, 175, 225, 275])

    def test_submission_keeps_original_roots_and_compact_inputs(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_adaptive_pair", 209011, preflight=False,
                                  protocol_spec=spec)
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertEqual(task["cpu"], 1)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)

    def test_suffix_starts_before_action_and_reconstructs_pair_credit(self):
        args = Namespace(**cell_options(208001, preflight=True))
        time_scale = PhysicalTimeScaleContract(dt_seconds=0.01, upper_period_seconds=0.5,
                                               history_seconds=0.64, fast_period_seconds=0.04)
        with (
            patch.object(diagnostic, "_make_task", side_effect=lambda **kw: ToyTask()),
            patch.object(diagnostic, "pointmaze_goal_bounds", return_value=(np.full(2, -10), np.full(2, 10))),
            patch.object(diagnostic, "PointMazeRegimeFeatureBuilder", ToyHistory),
            patch.object(diagnostic, "_causal_plan_features", side_effect=lambda **kw: (
                ["step"], kw["observation"].task_measurement.copy(), None)),
            patch.object(diagnostic, "_predict_renewal_advantage", side_effect=lambda **kw:
                         float(kw["features"][0] // 50 != 1 or kw["features"][0] % 50 >= 5)),
        ):
            arms = [diagnostic.rollout_intervention(
                ToyController(), seed=1, intervention_step=55, arm=arm,
                predictor={"threshold": 0.5}, args=args, time_scale=time_scale,
                collect_value_trace=True,
            ) for arm in ("now", "wait_one_check")]
            factual = diagnostic.rollout_intervention(
                ToyController(), seed=1, intervention_step=None, arm=None,
                predictor={"threshold": 0.5}, args=args, time_scale=time_scale,
                collect_value_trace=True,
            )
        for row in arms:
            self.assertEqual([r["step"] for r in row["value_trace"]], [105, 155, 205, 255])
            first = row["value_trace"][0]
            self.assertEqual(first["features"][0], 105)
            self.assertEqual(first["features"][-1], 1.0)
            self.assertEqual(first["features"][-3], 0.1)
            self.assertEqual(first["features"][-2], 195 / 300)
        self.assertTrue(all(r["features"][-1] == 0.0 for r in factual["value_trace"]))
        self.assertAlmostEqual(factual["value_trace"][0]["cost_to_go"],
                               factual["tracking_squared_error_integral"])
        now, wait = arms
        full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
        short = wait["window_ise"] - now["window_ise"]
        tail = wait["value_trace"][0]["cost_to_go"] - now["value_trace"][0]["cost_to_go"]
        self.assertAlmostEqual(full, short + tail)


if __name__ == "__main__":
    unittest.main()
