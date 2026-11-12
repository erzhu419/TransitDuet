from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts.run_reacher_timing_applicability import clock_schedule, event_schedules, contrast
from test_plan_validity_second_task import Controller, DenseTask, options


class ApplicabilityTest(unittest.TestCase):
    def test_event_and_rotated_schedules_keep_identical_offset_counts(self):
        events = [1, 54, 62, 125, 226, 260, 1190]
        event, rotated = event_schedules(1200, 15, events)
        self.assertEqual(event[1:6], (55, 125, 165, 215, 260))
        self.assertEqual(Counter(s % 50 for s in event), Counter(s % 50 for s in rotated))
        for schedule in (event, rotated):
            self.assertEqual([s // 50 for s in schedule], list(range(24)))
            self.assertEqual(schedule[0], 0)
            self.assertTrue(all(s % 50 in (0, 5, 10, 15, 20, 25) for s in schedule))

    def test_lag_changes_applied_waypoint_but_not_clock_or_first_option(self):
        class TracedController(Controller):
            def __init__(self):
                self.calls = 0
                self.goals = []

            def plan_goal(self, state, **kwargs):
                self.calls += 1
                return {"action": np.array([self.calls * .05, 0.], dtype=np.float32)}

            def act_conditioned(self, state, **kwargs):
                self.goals.append(state[4:6] + state[:2])
                return {"action": np.zeros(2)}

        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        schedule = clock_schedule(args.horizon, 10)
        same, lag = TracedController(), TracedController()
        with (patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(
                  **{k: v for k, v in kw.items() if k != "env_id"})),
              patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2)))):
            records = [shared.run_episode(c, task_spec=spec, seed=71, args=args, time_scale=shared.time_contract(args),
                schedule=schedule, waypoint_audit=audit)[1] for c, audit in ((same, "current"), (lag, "lag_one_bin"))]
        self.assertEqual(same.calls, lag.calls)
        self.assertEqual(records[0]["decision_steps"], records[1]["decision_steps"])
        np.testing.assert_array_equal(same.goals[:60], lag.goals[:60])
        np.testing.assert_allclose(lag.goals[60], lag.goals[0], atol=1e-7)
        self.assertGreater(np.linalg.norm(same.goals[60] - lag.goals[60]), .01)
        self.assertEqual(records[0]["late_option_ise"], records[1]["late_option_ise"])

    def test_precheck_requires_both_halves(self):
        good = [{"ise": {"event": .9, "clock": 1.}} for _ in range(24)]
        self.assertTrue(contrast(good, "ise", "event", "clock", .02)["passed"])
        for r in good[12:]:
            r["ise"]["event"] = 1.01
        self.assertFalse(contrast(good, "ise", "event", "clock", .02)["passed"])
