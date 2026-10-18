import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v111_disturbance_accounting as audit


class DisturbanceAccountingTests(unittest.TestCase):
    def test_balanced_gap_does_not_cancel_level_feedforward(self):
        negative = audit.balanced_response(-60.)
        self.assertEqual(negative['original_command_s'], 0.)
        self.assertEqual(negative['ff_command_s'], 45.)
        self.assertEqual(audit.balanced_response(0.)['ff_command_s'], 0.)
        self.assertEqual(audit.balanced_response(60.)['ff_command_s'], 0.)

    def fixture(self):
        arrivals = [dict(trip_id=i, station_id=1, direction=True, time_s=t,
            release=release, realized_holding=0., other_elapsed=t-release)
            for i, t, release in ((0, 20., 0.), (2, 380., 360.))]
        departures = [dict(r, time_s=r['time_s']+12+hold, realized_holding=float(hold),
            other_elapsed=r['other_elapsed']+12) for r, hold in zip(arrivals, (0, 10))]
        results = dict(action_counts={'0': 0, '10': 1}, holding_s_per_trip=5., cv_squared=0.)
        return dict(arrivals=arrivals, departures=departures), results

    def test_exact_executed_dose_and_unobserved_zero(self):
        records, results = self.fixture()
        row = audit.path_accounting(records, results, 2)
        self.assertEqual(row['dose']['executed_holding_s'], 10.)
        self.assertEqual(row['dose']['unobserved_zero_events'], 1)
        self.assertEqual(row['dose']['mean_command_s'], 10.)
        json.dumps(row)

    def test_dose_mismatch_stops_review(self):
        records, results = self.fixture()
        results['action_counts']['10'] = 2
        with self.assertRaisesRegex(AssertionError, 'positive-bin'):
            audit.path_accounting(records, results, 2)

    def test_arrival_persistence_requires_same_predecessor(self):
        arrivals = [dict(trip_id=t, station_id=s, direction=True, time_s=time,
            release=release, realized_holding=held, other_elapsed=time-release-held)
            for t, s, time, release, held in
            ((0, 1, 20., 0., 0.), (0, 2, 70., 0., 0.),
             (2, 1, 375., 360., 0.), (2, 2, 420., 360., 10.))]
        departures = [dict(r, time_s=r['time_s']+12+h,
            realized_holding=r['realized_holding']+h, other_elapsed=r['other_elapsed']+12)
            for r, h in zip(arrivals, (0, 0, 10, 20))]
        row = audit.path_accounting(dict(arrivals=arrivals, departures=departures),
            dict(action_counts={'10': 1, '20': 1}, holding_s_per_trip=15.,
                 cv_squared=6.25/352.5**2), 2)
        self.assertEqual(row['arrival_persistence']['adjacent_same_predecessor'], 1)
        self.assertEqual(row['arrival_persistence']['negative_then_negative_after_own_holding'], 1)
        self.assertEqual(row['arrival_persistence']['holding_at_repeated_negative_s'], 20.)


if __name__ == '__main__':
    unittest.main()
