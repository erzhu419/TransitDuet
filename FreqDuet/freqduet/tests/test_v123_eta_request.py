import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from freqduet_funded_quota import FundedQuota
from run_freqduet_v123_eta_request_pilot import request_error


class RequestTests(unittest.TestCase):
    def context(self, **kw):
        return SimpleNamespace(**dict(dict(evidence_valid=True, follower_evidence_valid=True,
            target_headway_s=300., forward_headway_s=200., follower_departure_gap_s=400.), **kw))

    def test_split_changes_only_bilateral_feedback(self):
        self.assertEqual(request_error(self.context(), dict(component_used=True, gap_s=300.)), 50.)
        self.assertEqual(request_error(self.context(), dict(component_used=False)), 100.)

    def test_unavailable_forward_is_zero_not_prediction(self):
        self.assertEqual(request_error(self.context(evidence_valid=False), {}), 0.)

    def test_missing_follower_keeps_original_forward_fallback(self):
        self.assertEqual(request_error(self.context(follower_evidence_valid=False), {}), 100.)

    def test_all_ineligible_forecasts_keep_raw(self):
        for status in ('not_causally_moving', 'missing_completed_component', 'other_eta_source'):
            self.assertEqual(request_error(self.context(), dict(component_used=False, gap_s=None, status=status)), 100.)

    def test_corrected_balance_expires_old_credit(self):
        budget = FundedQuota()
        key = (1, 1, True)
        budget.request(key, 0, 0., 80., [True]*7, 30., 0.)
        budget.settle(key, 20.)
        key = (1, 2, True)
        budget.request(key, 0, 0., 80., [True]*7, 30., 20.)
        budget.settle(key, 40.)
        error = request_error(self.context(), dict(component_used=True, gap_s=100.))
        _, row = budget.request((1, 3, True), 0, -100., error, [True]*7, 30., 40.)
        self.assertEqual(row['expired_s'], 20.)
        self.assertEqual(row['chosen_s'], 0.)
        self.assertEqual(row['issued_total_s'], 90.)

    def test_corrected_request_cannot_borrow_more_authority(self):
        budget = FundedQuota()
        _, row = budget.request((1, 1, True), 0, 0., 1000., [True]*7, 15., 0.)
        self.assertLessEqual(row['chosen_s'], 15.)
        self.assertEqual(row['shadow_s'], 15.)
        budget.settle((1, 1, True), row['chosen_s'])
        self.assertEqual(budget.audit()['conservation_error'], 0.)


if __name__ == '__main__':
    unittest.main()
