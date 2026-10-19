import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from audit_freqduet_v123_request_mechanism import holding_parts, forecast_and_expiry


class MechanismTests(unittest.TestCase):
    def test_request_and_ledger_parts_conserve_actual(self):
        row = dict(shadow_s=30., request_shadow_s=15., chosen_s=10., executed_s=10.)
        self.assertEqual(holding_parts(row, 10.), dict(raw_shadow=30., request_shift=-15., innovation_quota=-5.))
        self.assertEqual(sum(holding_parts(row, 10.).values()), 10.)
        self.assertEqual(sum(holding_parts(None, 0.).values()), 0.)
        with self.assertRaises(AssertionError):
            holding_parts(row, 15.)

    def test_changed_path_labels_and_expiry_are_observed_not_imputed(self):
        evidence = dict(ready_time_s=100., predicted_follower_trip_id=2,
            predicted_follower_bus_id=4, follower_gap_prediction_s=80.)
        row = dict(expired_s=20., reset_reason='continued', raw_feedback_error_s=10.,
            feedback_error_s=-10., component_used=True, request_gap_s=60., evidence=evidence,
            trip_id=1, station_id=3, direction=True)
        unresolved = forecast_and_expiry({(1, 3, True): row})
        self.assertEqual(unresolved['missing_same_follower_ready'], 1)
        self.assertNotIn('split_mae_s', unresolved)
        follower = dict(row, component_used=False, expired_s=0., raw_feedback_error_s=0.,
            evidence=dict(ready_time_s=150., bus_id=4))
        resolved = forecast_and_expiry({(1, 3, True): row, (2, 3, True): follower})
        self.assertEqual(resolved['balanced_expired_s'], 20.)
        self.assertEqual(resolved['balance_flip_expired_s'], 20.)
        self.assertEqual(resolved['split_mae_s'], 10.)
        self.assertEqual(resolved['raw_mae_s'], 30.)


if __name__ == '__main__':
    unittest.main()
