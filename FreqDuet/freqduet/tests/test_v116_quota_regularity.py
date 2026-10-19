import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v116_quota_regularity as audit


class QuotaRegularityTests(unittest.TestCase):
    def row(self, raw, available, chosen, shadow=10., increment=-5.):
        return dict(shadow_s=shadow, chosen_s=chosen, raw_request_s=raw,
            feedback_error_s=40., available_s=available, increment_s=increment)

    def test_negative_and_quota_cuts_are_disjoint(self):
        parts = audit.action_parts(self.row(5., 0., 0.))
        self.assertEqual((parts['negative_cut_s'], parts['quota_cut_s']), (5., 5.))
        self.assertEqual(sum(parts[k] for k in audit.ARRIVAL_PARTS[1:-1]), 0.)

    def test_quota_of_positive_extra_is_not_baseline_cut(self):
        parts = audit.action_parts(self.row(20., 10., 10., increment=10.))
        self.assertEqual(parts['quota_cut_s'], 0.)
        self.assertEqual(parts['extra_quota_withheld_s'], 10.)
        parts = audit.action_parts(self.row(20., 5., 5., increment=10.))
        self.assertEqual(parts['quota_cut_s'], 5.)
        self.assertEqual(parts['extra_quota_withheld_s'], 10.)

    def test_current_action_not_in_current_arrival(self):
        arrivals = [dict(trip_id=1, station_id=i, direction=1, time_s=t,
            release=0., realized_holding=h, other_elapsed=t-h) for i, t, h in ((0, 10., 0.), (1, 30., 5.))]
        departures = [dict(r, time_s=r['time_s']+h, realized_holding=r['realized_holding']+h)
            for r, h in zip(arrivals, (5., 10.))]
        a, d, _ = audit.coordinates(arrivals, departures, None)
        self.assertEqual([r['quarter_shadow'] for r in a], [0., 5.])
        self.assertEqual([r['current_action'] for r in d], [5., 10.])

    def test_initial_uncontrolled_events_stay_in_metric(self):
        arrival = dict(trip_id=0, station_id=1, direction=True, time_s=10.,
            release=0., realized_holding=0., other_elapsed=10.)
        departure = dict(arrival, time_s=12., other_elapsed=12.)
        a, d, _ = audit.coordinates([arrival], [departure], {})
        self.assertEqual(len(a), 1)
        self.assertEqual(d[0]['service_dwell'], 2.)
        with self.assertRaisesRegex(AssertionError, 'untraced holding'):
            audit.coordinates([arrival], [dict(departure, realized_holding=5.)], {})

    def test_geometry_resorts_when_shift_crosses_completed_neighbour(self):
        times = np.array([0., 10., 20., 40.])
        h = np.diff(times)
        moment = dict(samples=3, mean_s=float(h.mean()), variance_s2=float(h.var()),
            cv_squared=float(h.var()/h.mean()**2))
        result = audit.geometry_change(times, np.array([0., 30., 20., 40.]), moment)
        changed = np.diff([0., 20., 30., 40.])
        self.assertAlmostEqual(result['squared_gap_change_s2'], float(changed@changed-h@h))
        self.assertAlmostEqual(result['cv_squared_change'], 0.)
        end = audit.geometry_change(times, np.array([0., 10., 20., 50.]), moment)
        self.assertAlmostEqual(end['mean_headway_change_s'], 10/3)
        self.assertAlmostEqual(end['cv_squared_change'], float(np.var([10., 10., 30.])/(50/3)**2)-moment['cv_squared'])

    def test_exact_covariance_and_boundary_normalization(self):
        def events(times, actions):
            return [dict(trip_id=i, station_id=0, direction=True, time_s=t,
                current_action=a, arrival_time=t-a-1., service_dwell=1.)
                for i, (t, a) in enumerate(zip(times, actions))]
        b = audit.series(events([0., 10., 30., 60.], [0., 5., 0., 10.]), audit.DEPARTURE_PARTS)
        c = audit.series(events([0., 12., 31., 66.], [0., 0., 5., 5.]), audit.DEPARTURE_PARTS)
        e = audit.effect(b, c, audit.DEPARTURE_PARTS)
        self.assertAlmostEqual(sum(e[n] for n in (*audit.DEPARTURE_PARTS, 'mean_normalization')), e['cv_squared'])
        self.assertAlmostEqual(e['last_boundary_mean_s'], 2.)
        self.assertAlmostEqual(e['first_boundary_mean_s'], 0.)


if __name__ == '__main__':
    unittest.main()
