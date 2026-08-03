import unittest
from collections import defaultdict
from types import SimpleNamespace

from runner_v3 import TransitDuetV2Runner
from upper.timetable_planner import TimetableCurvePlanner


class TimetablePlanCreditTest(unittest.TestCase):
    @staticmethod
    def _trips():
        return [
            SimpleNamespace(
                launch_time=i * 300,
                launch_turn=i,
                direction=True,
                launched=False,
                target_headway=300.0,
            )
            for i in range(5)
        ]

    def test_reused_plan_keeps_original_decision_owner(self):
        trips = self._trips()
        planner = TimetableCurvePlanner(
            horizon_s=1200.0,
            basis_per_direction=2,
            shared_directions=True,
        )
        action = [0.0, 0.0]

        planner.apply(trips, trips[0], action, plan_id=0)
        trips[0].launched = True
        planner.apply(
            trips,
            trips[1],
            action,
            origin_launch_s=0.0,
            plan_id=0,
        )

        self.assertEqual(trips[1]._freqduet_planned_by, 0)
        self.assertEqual(trips[3]._freqduet_planned_by, 0)

        trips[1].launched = True
        planner.apply(trips, trips[2], action, plan_id=2)
        self.assertEqual(trips[2]._freqduet_planned_by, 2)
        self.assertEqual(trips[4]._freqduet_planned_by, 2)

    def test_wait_credit_aggregates_all_trips_owned_by_plan(self):
        runner = object.__new__(TransitDuetV2Runner)
        runner.freq_wait_enable = True
        runner.freq_wait_upper_weight = 1.0
        runner.freq_wait_normalize_upper = False
        runner.env = SimpleNamespace(timetables=self._trips())
        for trip in runner.env.timetables:
            trip._freqduet_planned_by = 0 if trip.launch_turn < 3 else 3
        runner._ep_trip_wait_stats = defaultdict(dict, {
            0: {'pax': 10, 'upper_wait_norm_sum': 10.0},
            1: {'pax': 20, 'upper_wait_norm_sum': 40.0},
            2: {'pax': 10, 'upper_wait_norm_sum': 30.0},
            3: {'pax': 10, 'upper_wait_norm_sum': 5.0},
            4: {'pax': 10, 'upper_wait_norm_sum': 15.0},
        })
        transitions = [{'tid': 0}, {'tid': 3}]

        credits = runner._upper_frequency_wait_credits(transitions)

        self.assertAlmostEqual(credits[0], -2.0)
        self.assertAlmostEqual(credits[3], -1.0)

    def test_exact_curve_matches_every_projected_schedule_gap(self):
        trips = self._trips()
        planner = TimetableCurvePlanner(
            horizon_s=1200.0,
            basis_per_direction=2,
            shared_directions=True,
            min_headway_s=120.0,
            max_headway_s=600.0,
            terminal_schedule_mode="exact_headway_curve",
        )

        summary = planner.apply(
            trips,
            trips[0],
            [-30.0, -30.0],
            write_scheduled_launch=True,
            plan_id=17,
        )

        scheduled = [trip._freqduet_scheduled_launch for trip in trips]
        self.assertEqual(scheduled, [0.0, 270.0, 540.0, 810.0, 1080.0])
        self.assertEqual(summary["projection_mode"], "exact_headway_curve")
        self.assertEqual(trips[0]._freqduet_planned_by, 0)
        for index, trip in enumerate(trips[1:], start=1):
            self.assertAlmostEqual(
                trip.target_headway,
                scheduled[index] - scheduled[index - 1],
            )
            self.assertAlmostEqual(trip.target_headway, 270.0)
            self.assertEqual(trip._freqduet_planned_by, 17)
            self.assertAlmostEqual(
                trip._freqduet_phase_displacement_s,
                -30.0 * index,
            )

    def test_exact_curve_keeps_existing_anchor_and_replans_only_future_trips(self):
        trips = self._trips()
        trips[0].launched = True
        trips[0]._freqduet_actual_launch = 10.0
        trips[1]._freqduet_scheduled_launch = 280.0
        trips[1]._freqduet_planned_by = 3
        planner = TimetableCurvePlanner(
            horizon_s=900.0,
            basis_per_direction=1,
            shared_directions=True,
            min_headway_s=120.0,
            max_headway_s=600.0,
            terminal_schedule_mode="exact_headway_curve",
        )

        planner.apply(
            trips,
            trips[1],
            [60.0],
            origin_launch_s=300.0,
            write_scheduled_launch=True,
            plan_id=9,
        )

        self.assertEqual(trips[1]._freqduet_scheduled_launch, 280.0)
        self.assertEqual(trips[1].target_headway, 270.0)
        self.assertEqual(trips[1]._freqduet_planned_by, 3)
        self.assertEqual(trips[2]._freqduet_scheduled_launch, 640.0)
        self.assertEqual(trips[3]._freqduet_scheduled_launch, 1000.0)
        self.assertEqual(trips[4]._freqduet_scheduled_launch, 1360.0)
        self.assertEqual(trips[2]._freqduet_planned_by, 9)

    def test_exact_curve_rejects_independent_terminal_bias(self):
        trips = self._trips()
        planner = TimetableCurvePlanner(
            basis_per_direction=1,
            shared_directions=True,
            terminal_schedule_mode="exact_headway_curve",
        )
        with self.assertRaisesRegex(ValueError, "independent terminal shift"):
            planner.apply(
                trips,
                trips[0],
                [0.0],
                write_scheduled_launch=True,
                terminal_shift_bias_s=5.0,
            )


if __name__ == '__main__':
    unittest.main()
