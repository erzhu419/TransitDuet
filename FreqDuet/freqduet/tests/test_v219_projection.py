import unittest

import numpy as np
from upper.timetable_planner import TimetableCurvePlanner

import freqduet_v219_projection as preview


def query(stock=(3, 3)):
    return dict(time_s=7200., event_seq=10, trip_id=40,
        plans=[dict(trip_id=40+2*i+int(not d), direction=d, public_s=7200.+360*i+180*int(not d),
                    planned_s=7200.+360*i+180*int(not d), target_s=360.)
               for d in (True, False) for i in range(4)],
        ready=[dict(bus_id=10*int(d)+i, direction=d, return_observed=True, last_observed_return_s=7000.+i)
               for d, n in zip((False, True), stock) for i in range(n)])


class ProjectionTest(unittest.TestCase):
    def test_ema_keeps_native_previous_state(self):
        prev = np.array([4.], np.float32)
        before = prev.copy()
        np.testing.assert_array_equal(preview.ema_action([0.], prev, .35), np.array([2.6], np.float32))
        np.testing.assert_array_equal(prev, before)

    def test_native_preview_is_nonmutating_and_preserves_both_anchors(self):
        q = query()
        trips = preview.query_timetable(q, [])
        original = [vars(t).copy() for t in trips]
        p = TimetableCurvePlanner(basis_per_direction=2, shared_directions=True, plan_all_directions=True,
            coefficient_parameterization='antisymmetric_linear_v5', terminal_schedule_mode='exact_headway_curve',
            headway_budget_mode='rolling_zero_sum_delta_v6', delta_min_s=-60, delta_max_s=60)
        result = preview.preview(p, trips, q, np.array([10.5], np.float32))
        self.assertEqual(original, [vars(t) for t in trips])
        self.assertEqual(result[40]['planned_s'], 7200.)
        self.assertEqual(result[41]['planned_s'], 7380.)

    def test_unfunded_other_direction_rejects_shared_action(self):
        q = query((1, 3))
        native = {p['trip_id']: dict(planned_s=p['planned_s'], target_s=360.) for p in q['plans']}
        candidate = {t: dict(r) for t, r in native.items()}
        candidate[42]['planned_s'] -= 5
        candidate[43]['planned_s'] -= 5
        result = preview.inspect_proposal(q, native, candidate, 1.)
        self.assertFalse(result['supported'])
        self.assertEqual(result['unfunded_within_cycle_trip_ids'], [43])

    def test_shared_coefficient_does_not_imply_global_plan_writes(self):
        q = query()
        trips = preview.query_timetable(q, [])
        p = TimetableCurvePlanner(basis_per_direction=2, shared_directions=True, plan_all_directions=False,
            coefficient_parameterization='antisymmetric_linear_v5', terminal_schedule_mode='exact_headway_curve',
            headway_budget_mode='rolling_zero_sum_delta_v6', delta_min_s=-60, delta_max_s=60)
        result = preview.preview(p, trips, q, np.array([10.5], np.float32))
        for row in q['plans']:
            if not row['direction']:
                self.assertEqual(result[row['trip_id']]['planned_s'], row['planned_s'])
        self.assertLess(result[42]['planned_s'], 7560.)

    def test_subtick_continuous_delay_is_not_silently_admitted(self):
        q = query()
        native = {p['trip_id']: dict(planned_s=p['planned_s']+.5, target_s=360.) for p in q['plans']}
        candidate = {t: dict(r) for t, r in native.items()}
        candidate[42]['planned_s'] -= 5
        candidate[46]['planned_s'] += .1
        result = preview.inspect_proposal(q, native, candidate, 1.)
        self.assertEqual(result['continuous_later_trip_ids'], [46])
        self.assertFalse(result['supported'])

    def test_tick_mismatch_fails_even_below_numeric_tolerance(self):
        a = {1: dict(planned_s=360., target_s=360.)}
        b = {1: dict(planned_s=360.+1e-12, target_s=360.)}
        with self.assertRaises(AssertionError):
            preview.verify_native(a, b, 1.)


if __name__ == '__main__':
    unittest.main()
