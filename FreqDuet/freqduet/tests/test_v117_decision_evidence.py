import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v117_decision_evidence as audit


class DecisionEvidenceTests(unittest.TestCase):
    def recover(self, *, residual=50., error=40., follower=True, increment=-15.):
        row = dict(feedback_error_s=error, increment_s=increment,
            raw_request_s=.25*error+(increment if error > 0 else 0), onboard_passengers=20,
            residual_s=residual, executed_before_s=20., follower_valid=follower,
            executed_s=5., policy_time_s=190.)
        departure = dict(release=100., arrival_time=170., time_s=195.)
        pred = dict(release=80., realized_holding=10., time_s=110.)
        return row, audit.recover(row, departure, pred)

    def test_recover_current_context_without_future_ready_input(self):
        _, c = self.recover()
        self.assertEqual(c['forward_gap_s'], 80.)
        self.assertEqual(c['ready_time_s'], 190.)
        self.assertEqual(c['predicted_follower_gap_s'], 160.)
        self.assertEqual(c['pre_action_lag_s'], 0.)

    def test_fallback_has_no_fabricated_follower_prediction(self):
        _, c = self.recover(follower=False)
        self.assertIsNone(c['predicted_follower_gap_s'])
        self.assertEqual(c['target_s'], 120.)

    def test_stable_bilateral_shared_residual_is_exact_not_a_causal_verdict(self):
        row, current = self.recover()
        old = dict(current, residual_s=30., D_s=50., forward_gap_s=60.,
            raw_request_s=12.5, applied_increment_s=0.)
        row.update(reset_reason='continued', innovation_s=20., predecessor_trip_id=2, station_id=4)
        prior_row = dict(predecessor_trip_id=2, station_id=3)
        result = audit.transition(row, current, prior_row, old)
        self.assertEqual(result['delta_base_s'], -2.5)
        self.assertEqual(result['residual_in_delta_base_s'], -2.5)
        self.assertEqual(result['other_in_delta_base_s'], 0.)
        self.assertTrue(result['base_and_increment_same_sign'])
        self.assertEqual(row['increment_s']/result['residual_in_delta_base_s'], 6.)

    def test_stable_fallback_and_regime_change_not_collapsed(self):
        row, current = self.recover(follower=False)
        old = dict(current, residual_s=30., D_s=60., forward_gap_s=60.,
            raw_request_s=15., applied_increment_s=0.)
        row.update(reset_reason='continued', innovation_s=20., predecessor_trip_id=2, station_id=4)
        prior_row = dict(predecessor_trip_id=2, station_id=3)
        result = audit.transition(row, current, prior_row, old)
        self.assertEqual(result['residual_in_delta_base_s'], -5.)
        self.assertEqual(row['increment_s']/result['residual_in_delta_base_s'], 3.)
        changed = audit.transition(row, current, prior_row, dict(old, regime='bilateral'))
        self.assertEqual(changed['transition'], 'regime_changed')
        self.assertNotIn('delta_base_s', changed)

    def test_follower_holding_can_change_ready_vs_departure_sign(self):
        self.assertGreater(audit.pair_restore_change(100., 100., 5., 5.), 0.)
        self.assertLess(audit.pair_restore_change(100., 200., 5., 5.), 0.)

    def test_missing_labels_not_reported_as_zero_error(self):
        s = audit.summarize([dict(D_s=1., regime='fallback', predicted_follower_gap_s=None),
            dict(D_s=3., regime='bilateral', predicted_follower_gap_s=100.)])
        self.assertEqual(s['metrics']['predicted_follower_gap_s']['n'], 1)
        self.assertEqual(s['metrics']['predicted_follower_gap_s']['mean'], 100.)


if __name__ == '__main__':
    unittest.main()
