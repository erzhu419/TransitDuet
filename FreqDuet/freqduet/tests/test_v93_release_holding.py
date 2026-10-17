from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v93_release_holding as experiment


class ReleaseHoldingTest(unittest.TestCase):
    def law(self, index):
        return np.eye(7)[index]

    def test_short_release_gap_can_need_holding_but_long_gap_does_not(self):
        late, early = self.law(6), self.law(0)
        self.assertGreater(experiment.compensation_delta(late, early, 300., 10., 360.), 0.)
        self.assertLess(experiment.compensation_delta(late, early, 400., 10., 360.), 0.)

    def test_actual_cumulative_holding_changes_the_compensation_direction(self):
        late, early = self.law(6), self.law(0)
        self.assertGreater(experiment.compensation_delta(late, early, 300., 0., 360.), 0.)
        self.assertLess(experiment.compensation_delta(late, early, 300., 80., 360.), 0.)

    def test_full_distribution_second_moment_not_square_of_expected_action(self):
        late = .5 * self.law(0) + .5 * self.law(6)
        early = self.law(3)
        delta = experiment.compensation_delta(late, early, 330., 0., 360.)
        self.assertLess(delta, 0.)
        means_only = (330. + 15. - 360.) ** 2 - (330. + 22.5 - 360.) ** 2
        self.assertGreater(means_only, 0.)

    def test_probability_normalization_and_negative_release_coordinates_are_retained(self):
        late, early = self.law(6), self.law(0)
        a = experiment.compensation_delta(late, early, -20., 50., 360.)
        self.assertGreater(a, 0.)
        self.assertAlmostEqual(a, experiment.compensation_delta(late * .99999, early * 1.00001, -20., 50., 360.))

    def test_four_regimes_are_a_partition_not_a_posthoc_arm_choice(self):
        for g in experiment.GROUPS:
            self.assertFalse(experiment.selected("late", g))
            self.assertTrue(experiment.selected("no_gain_early", g))
            self.assertNotEqual(experiment.selected("preserve", g), experiment.selected("loss_only", g))
        e = experiment.contrasts(dict(late=10., preserve=8., loss_only=12., no_gain_early=9.))
        self.assertEqual(e["loss_added_to_preserve"], 1.)
        self.assertEqual(e["interaction"], -1.)

    def test_new_days_are_not_consumed_v89_days(self):
        self.assertFalse(set(experiment.DAYS) & set(experiment.source.DAYS))
        self.assertEqual(len(experiment.ITEMS) * len(experiment.MODES), 384)

    def test_timing_coordinates_preserve_cross_covariance(self):
        records = [dict(trip_id=i, station_id=2, direction=True, time_s=r+h,
            release=r, realized_holding=h, other_elapsed=0.) for i, (r, h) in enumerate(
                ((0., 0.), (100., 20.), (300., 10.), (600., 10.)))]
        moments = experiment.coordinate_moments(records)
        c = np.asarray(moments["covariance"])
        self.assertLess(c[0, 1], 0.)
        self.assertAlmostEqual(c.sum(), moments["variance"])

    def test_prefix_uses_immutable_predecessor_and_subtracts_current_launch_counter(self):
        rule = experiment.ReleaseHolding(experiment.pilot.SEEDS[0], "late")
        rule.departures = [dict(trip_id=1, station_id=2, direction=True,
            time_s=70., release=0., realized_holding=50.)]
        rule.launches = {3: dict(time_s=40., holding_counter_at_launch=100.)}
        rule.env = SimpleNamespace(current_time=80., headway_events=SimpleNamespace(
            previous_departure_event=lambda *args: dict(trip_id=1, time_s=70.)))
        bus = SimpleNamespace(trip_id=3, direction=True, episode_hold_vehicle_seconds=125.,
            forward_predecessor_trip_id=1, last_station=SimpleNamespace(station_id=2))
        context = SimpleNamespace(forward_headway_s=10.)
        self.assertEqual(rule.observable_coordinates(bus, context), (40., -25., -5.))
        rule.env.current_time = 60.
        with self.assertRaises(AssertionError):
            rule.observable_coordinates(bus, context)


if __name__ == "__main__":
    unittest.main()
