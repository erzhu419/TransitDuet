import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v119_eta_clock as run


class EtaClockTests(unittest.TestCase):
    def prefix(self, last_departure=50.):
        a = {(1, 1, True): dict(time_s=10., realized_holding=0.),
             (1, 2, True): dict(time_s=40., realized_holding=5.)}
        d = [dict(trip_id=1, station_id=0, direction=True, time_s=0., realized_holding=0.),
             dict(trip_id=1, station_id=1, direction=True, time_s=20., realized_holding=5.),
             dict(trip_id=1, station_id=2, direction=True, time_s=last_departure, realized_holding=8.)]
        r = {(1, 1, True): dict(ready_time_s=14.), (1, 2, True): dict(ready_time_s=45.)}
        return run.CompletedPrefix(d, a, r)

    def test_strict_prefix_excludes_current_and_future_departures(self):
        p = self.prefix()
        np.testing.assert_array_equal(p.before(20.), [0., 0., 0.])
        np.testing.assert_array_equal(p.before(21.), [5., 4., 1.])
        np.testing.assert_array_equal(p.before(50.), [5., 4., 1.])
        np.testing.assert_array_equal(p.before(51.), [8., 9., 3.])
        np.testing.assert_array_equal(p.before(30.), self.prefix(60.).before(30.))

    def test_projection_obeys_clock_algebra_and_rejects_bad_clock(self):
        e = dict(ready_time_s=30., speed_mps=2., spatial_gap_m=120., eta_prediction_s=60.)
        np.testing.assert_array_equal(run.clock_projection(e, 0., self.prefix().before(30.)), [10., 8., 2.])
        with self.assertRaises(AssertionError):
            run.clock_projection(e, 25., self.prefix().before(30.))

    def test_future_decomposition_closes(self):
        p = self.prefix()
        future = p.before(70.)-p.before(30.)
        result = run.future_parts(60., 40., np.array([10., 8., 2.]), future)
        self.assertEqual(result['holding_future_s'], 3.)
        self.assertAlmostEqual(sum(result[k] for k in ('holding_mismatch_s', 'service_mismatch_s',
            'ticks_mismatch_s', 'other_error_s')), 20.)

    def test_future_phase_is_label_not_fabricated_zero(self):
        p = self.prefix()
        self.assertEqual(p.phase_label(30., 70.), 'in_transit')
        self.assertEqual(p.phase_label(42., 70.), 'pending_stop')
        self.assertEqual(p.phase_label(40., 70.), 'event_boundary')
        self.assertEqual(p.phase_label(20., 70.), 'event_boundary')
        self.assertEqual(p.phase_label(71., 70.), 'target_already_arrived')

    def test_sign_improvement_and_lost_opportunity_are_separate(self):
        gaps = dict(raw=450., minus_holding=250.)
        good = run.sign_metrics(gaps, 450., 300., 5., 10.)
        self.assertTrue(good['minus_holding_false_negative'])
        self.assertTrue(good['minus_holding_lost_benefit_vs_raw'])
        bad = run.sign_metrics(gaps, 250., 300., 5., 10.)
        self.assertTrue(bad['raw_false_positive'])
        self.assertTrue(bad['minus_holding_corrected_false_positive'])
        self.assertFalse(bad['minus_holding_lost_benefit_vs_raw'])


if __name__ == '__main__':
    unittest.main()
