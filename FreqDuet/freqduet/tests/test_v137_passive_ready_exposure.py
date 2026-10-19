import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_freqduet_v137_passive_ready_exposure import (
    ExposureTimeline, holding_exposure, ready_label, pair_loss_change, label_summary)


class PassiveReadyExposureTests(unittest.TestCase):
    def row(self):
        return dict(station_id=4, direction=True, evidence=dict(predicted_follower_trip_id=2,
            predicted_follower_bus_id=9, ready_time_s=100.), original_prediction=dict(gap_s=500.),
            online=dict(gap_s=350.))

    def test_ready_truth_is_not_arrival_truth(self):
        result = ready_label(self.row(), {(2, 4, True): dict(bus_id=9, ready_time_s=430.)})
        self.assertEqual(result['same_ready_gap_s'], 330.)
        self.assertEqual(result['original_error_s'], 170.)
        self.assertEqual(result['handoff_error_s'], 20.)

    def test_bus_identity_mismatch_is_not_a_zero_error(self):
        result = ready_label(self.row(), {(2, 4, True): dict(bus_id=8, ready_time_s=430.)})
        self.assertFalse(result['resolved'])
        self.assertEqual(result['reason'], 'same_trip_bus_mismatch')
        self.assertNotIn('handoff_error_s', result)

    def test_missing_or_already_ready_truth_remains_explicit(self):
        self.assertEqual(ready_label(self.row(), {})['reason'], 'same_trip_ready_missing')
        result = ready_label(self.row(), {(2, 4, True): dict(bus_id=9, ready_time_s=90.)})
        self.assertFalse(result['resolved'])
        self.assertEqual(result['same_ready_gap_s'], -10.)

    def test_exposure_uses_counter_difference_and_catches_wrong_load(self):
        row = dict(person_counter_before_s=100., executed_s=10., query_load=3)
        self.assertEqual(holding_exposure(row, dict(person_counter_s=130.)), 30.)
        with self.assertRaises(AssertionError):
            holding_exposure(row, dict(person_counter_s=140.))

    def test_timeline_splits_boundary_without_changing_original_total(self):
        stream = ExposureTimeline()
        stream.advance(0., 950., 2., 3., 1., 1., 0.)
        np.testing.assert_array_equal(stream.bands[0], np.array([2., 3., 1., 1., 0.])*900)
        np.testing.assert_array_equal(stream.bands[1], np.array([2., 3., 1., 1., 0.])*50)
        np.testing.assert_array_equal(stream.total, sum(stream.bands.values()))

    def test_holding_cut_can_remove_useful_pair_compensation(self):
        self.assertLess(pair_loss_change(300., 300., 30., 0.), 0.)
        self.assertGreater(pair_loss_change(300., 420., 30., 0.), 0.)

    def test_fractional_lf_bands_difference_the_native_ledger(self):
        stream = ExposureTimeline()
        for i in range(12000):
            stream.advance(float(i), 1., 2., 3., .1234567, .2345678, 0.)
        np.testing.assert_allclose(sum(stream.bands.values()), stream.total, rtol=0, atol=1e-10)

    def test_empty_forecast_support_is_unavailable_not_zero_mae(self):
        self.assertIsNone(label_summary([])['prediction_means'])


if __name__ == '__main__':
    unittest.main()
