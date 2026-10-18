from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v110_observed_disturbance as run


class ObservedDisturbanceTest(unittest.TestCase):
    def context(self, **kw):
        return SimpleNamespace(**(dict(evidence_valid=True, follower_evidence_valid=True,
            forward_headway_s=300., follower_departure_gap_s=420., target_headway_s=360.) | kw))

    def test_zero_disturbance_recovers_unrounded_quarter_feedback(self):
        context = self.context(forward_headway_s=347.5, follower_departure_gap_s=372.5)
        mask = np.ones(7, bool)
        self.assertEqual(run.quantized_action(run.desired_holding(context, 0.), mask),
                         run.prior.rule_action(context, 'bilateral', .25, mask))

    def test_observed_fast_bus_gets_more_hold_and_slow_bus_less(self):
        context = self.context()
        self.assertEqual(run.desired_holding(context, 0.), 15.)
        self.assertEqual(run.desired_holding(context, -20.), 30.)
        self.assertEqual(run.desired_holding(context, 20.), 0.)

    def test_fixed_complement_cancels_additive_ready_time_disturbance(self):
        baseline = self.context()
        shifted = self.context(forward_headway_s=320., follower_departure_gap_s=400.)
        self.assertEqual(run.desired_holding(shifted, 20.) - run.desired_holding(baseline, 0.), -20.)

    def test_missing_evidence_uses_existing_fallback(self):
        self.assertEqual(run.desired_holding(self.context(evidence_valid=False), 50.), 0.)
        context = self.context(follower_evidence_valid=False, follower_departure_gap_s=None)
        self.assertEqual(run.desired_holding(context, 20.), 0.)

    def test_original_cap_feasibility_and_shorter_ties(self):
        mask = np.ones(7, bool)
        self.assertEqual(run.pilot.step.ACTIONS[run.quantized_action(100., mask)], 45.)
        self.assertEqual(run.pilot.step.ACTIONS[run.quantized_action(-10., mask)], 0.)
        self.assertEqual(run.pilot.step.ACTIONS[run.quantized_action(12.5, mask)], 10.)
        self.assertEqual(run.quantized_action(100., np.asarray([True]+[False]*6)), 0)

    def test_coordinates_use_matching_completed_predecessor_and_executed_counter(self):
        audit = run.DisturbanceAudit(run.SEEDS[0], run.CANDIDATE)
        audit.env = SimpleNamespace(current_time=200.)
        audit.departures = [dict(trip_id=1, station_id=2, direction=True, time_s=100.,
                                release=10., realized_holding=15.)]
        audit.launches = {2: dict(time_s=30., holding_counter_at_launch=100.)}
        bus = SimpleNamespace(trip_id=2, forward_predecessor_trip_id=1,
            last_station=SimpleNamespace(station_id=2), direction=True, episode_hold_vehicle_seconds=125.)
        self.assertEqual(audit.observable_coordinates(bus, self.context(forward_headway_s=80.)), (20., 10., 50.))
        with self.assertRaises(AssertionError):
            audit.observable_coordinates(bus, self.context(forward_headway_s=110.))

    def test_pilot_budget_has_new_days_and_shared_deterministic_paths(self):
        self.assertEqual(len(run.ITEMS), 72)
        self.assertEqual(len(set(run.ITEMS)), 72)
        self.assertFalse(set(run.DAYS) & set(run.prior.DAYS))
        self.assertFalse(set(run.DAYS) & set(run.confirmation.DAYS))
        self.assertEqual(run.SEEDS, run.prior.SEEDS)
        for m in run.METHODS:
            self.assertEqual(sum(x[2] == m for x in run.ITEMS), 24 if m in ('native', 'preserve') else 8)


if __name__ == '__main__':
    unittest.main()
