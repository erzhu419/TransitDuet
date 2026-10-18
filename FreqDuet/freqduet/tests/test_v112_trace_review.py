from collections import defaultdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_freqduet_v112_decision_trace as review


class TraceReviewTests(unittest.TestCase):
    def row(self):
        return dict(valid_forward=True, valid_bilateral=True, previous_residual_s=-60.,
            residual_s=-55., own_executed_holding_s=90., prior_positive_excess_s=45.,
            feedback_error_s=-5., chosen_s=45., base_shadow_s=0., innovation_s=5.,
            actual_feedforward_s=41.25, clipping_delta_s=-1.25, quantization_delta_s=5.,
            completed_follower=dict(ready_label_temporal_valid=True, next_arrival_identity_matches=True,
                completed_feedback_error_s=-10., completed_shadow_command_s=45., prediction_error_s=10.))

    def test_repeated_response_persists_with_matched_completed_gap(self):
        groups = review.cohorts(self.row())
        self.assertIn('repeat_positive_hold_completed_feedback_nonpositive', groups)
        self.assertIn('repeat_no_new_negative_innovation_positive_hold', groups)
        self.assertIn('cap45_persists_with_completed_follower', groups)

    def test_missing_and_mismatched_labels_are_not_zero_error(self):
        row = self.row()
        row['completed_follower']['next_arrival_identity_matches'] = False
        self.assertNotIn('matched_completed_follower', review.cohorts(row))
        row['completed_follower'] = None
        row['valid_bilateral'] = False
        self.assertIn('repeat_positive_hold_without_follower_forecast', review.cohorts(row))
        totals = defaultdict(lambda: defaultdict(float))
        review.accumulate(totals, row)
        self.assertEqual(totals['valid_forward']['count'], 1)
        self.assertEqual(totals['valid_forward']['matched_labels'], 0)


if __name__ == '__main__':
    unittest.main()
