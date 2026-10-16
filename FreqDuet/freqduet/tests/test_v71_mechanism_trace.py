import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v71_mechanism_trace as audit


def trace(candidate=False):
    timeline = np.array([[0, 0, 0, 0, 0, 0], [1000, 100, 200, 0, 0, 0],
                         [1900, 200, 400, 0, 0, 0], [2800, 300, 600, 0, 0, 0],
                         [3000, 350, 700, 0, 0, 0], [4000, 500, 1000, 0, 0, 0]], dtype=float)
    if candidate:
        timeline[2:, 1] += [100, 60, 20, -100]
    holding = np.array([[0, 0, 0], [1000, 0, 0], [1900, 5, 10], [4000, 15, 30]], dtype=float)
    if candidate:
        holding[2:, 1:] = [[45, 90], [45, 90]]
    return dict(identity=(1000, 7, 1, 2, True), native_action_s=5, timeline=timeline, holding=holding,
                holds=[dict(time_s=1000, action_s=45 if candidate else 5),
                       dict(time_s=1900, action_s=0 if candidate else 10)],
                returns=[dict(trip_id=1, bus_id=7, physical_time_s=3500 if candidate else 3000)],
                launches=[dict(time_s=500, trip_id=1, bus_id=7, readiness_s=0),
                          dict(time_s=3100, trip_id=2, bus_id=7, readiness_s=0)],
                headways=[dict(arrival_time_s=900, headway_s=100), dict(arrival_time_s=950, headway_s=200),
                          dict(arrival_time_s=1900, headway_s=300)],
                summary=dict(passenger_person_s=float(timeline[-1, 1:3].sum()), holding_total_s=float(holding[-1, 1])))


class MechanismTraceTest(unittest.TestCase):
    def test_fixed_interventions_keep_all_32_consumed_futures(self):
        self.assertEqual(list(audit.PLAN), [15, 123, 51, 87, 125])
        self.assertEqual(len(audit.ITEMS), 160)
        self.assertEqual(32 * sum(1 + len(a) for _, a in audit.PLAN.values()), 384)
        self.assertEqual([audit.ITEMS[i][0] for i in audit.SMOKE_ITEMS], [15, 123, 125])
        for native, actions in audit.PLAN.values():
            self.assertIn(native, audit.joint.ACTIONS)
            self.assertTrue(all(a in audit.joint.ACTIONS and a != native for a in actions))

    def test_timeline_keeps_exact_base_exposure_and_service_cleared_waiting_tail(self):
        base, timed = audit.SystemCreditStream(), audit.TimelineStream()
        for stream in (base, timed):
            stream.advance(0, 2, 3, 2, 1, 1, 0)
            stream.finish(10, service_cleared=True, unserved=4, lf_unserved=1)
        np.testing.assert_array_equal(base.total, timed.total)
        np.testing.assert_array_equal(np.asarray(timed.timeline)[-1, 1:], timed.total)
        np.testing.assert_array_equal(audit.cumulative(np.asarray(timed.timeline), 6), [22, 4, 6, 2, 0])

    def test_fractional_cuts_interpolate_the_recorded_constant_rate_interval(self):
        timeline = np.array([[0, 0, 0], [2, 10, 4]], dtype=float)
        np.testing.assert_array_equal(audit.cumulative(timeline, .5), [2.5, 1.])

    def test_reference_return_cut_does_not_move_with_candidate_return(self):
        phases = audit.timing_effects(trace(), trace(True), 45)
        self.assertEqual(phases["reference_return"]["time_s"], 3000)
        self.assertEqual(phases["reference_return"]["passenger_person_s"], 20)
        self.assertEqual(phases["min15"]["passenger_person_s"], 100)
        self.assertEqual(phases["day_end"]["passenger_person_s"], -100)

    def test_target_command_boundary_and_later_commands_are_separate(self):
        phases = audit.timing_effects(trace(), trace(True), 45)
        self.assertEqual(phases["before"]["commanded_holding_s"], 0)
        self.assertEqual(phases["min15"]["commanded_holding_s"], 40)
        self.assertEqual(phases["min15"]["other_command_holding_s"], 0)
        self.assertEqual(phases["day_end"]["commanded_holding_s"], 30)
        self.assertEqual(phases["day_end"]["other_command_holding_s"], -10)
        self.assertEqual(phases["day_end"]["executed_holding_s"], 30)

    def test_last_trip_without_a_next_launch_is_explicitly_absent(self):
        data = trace()
        data["launches"] = data["launches"][:1]
        self.assertNotIn("reference_next_launch", audit.common_cuts(data))
        self.assertEqual(audit.common_cuts(data)["day_end"], 4000)

    def test_prefix_exposure_mismatch_is_not_hidden_by_endpoint_agreement(self):
        data = trace(True)
        data["timeline"][1, 1] += 1
        with self.assertRaises(AssertionError):
            audit.timing_effects(trace(), data, 45)

    def test_cv_cut_excludes_events_at_or_after_its_clock(self):
        expected = np.var([100., 200.]) / np.mean([100., 200.]) ** 2
        self.assertAlmostEqual(audit.cv_squared_at(trace(), 1900), expected)

    def test_future_interval_is_conditional_not_a_training_seed_interval(self):
        result = audit.future_interval(np.full(32, -2.))
        self.assertEqual(result, dict(n_futures=32, mean=-2., ci95=[-2., -2.]))

    def test_optional_cut_uses_available_futures_without_zero_padding(self):
        phases = audit.timing_effects(trace(), trace(True), 45)
        first, second = dict(phases=phases), dict(phases=copy.deepcopy(phases))
        second["phases"].pop("reference_next_launch")
        result = audit.phase_summary([first, second], "reference_next_launch")
        self.assertEqual(result["passenger_person_s"]["n_futures"], 1)
        self.assertEqual(result["passenger_person_s"]["mean"], phases["reference_next_launch"]["passenger_person_s"])


if __name__ == "__main__":
    unittest.main()
