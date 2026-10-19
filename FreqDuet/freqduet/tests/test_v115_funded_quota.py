from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freqduet_funded_quota import ACTIONS, FundedQuota


class FundedQuotaTests(unittest.TestCase):
    def setUp(self):
        self.rule, self.actual = FundedQuota(), 0.

    def request(self, station, residual, error=60., predecessor=4):
        shadow = ACTIONS[min(range(7), key=lambda i: abs(ACTIONS[i]-min(max(.25*error, 0.), 45.)))]
        key = (9, station, True)
        _, row = self.rule.request(key, predecessor, residual, error, [True]*7, shadow, self.actual)
        return key, row

    def step(self, station, residual, error=60., predecessor=4, actual=None):
        key, row = self.request(station, residual, error, predecessor)
        self.actual += row['chosen_s'] if actual is None else actual
        self.rule.settle(key, self.actual)
        self.assertEqual(self.rule.audit()['conservation_error'], 0.)
        self.assertLessEqual(self.actual, row['issued_total_s']-row['expired_total_s'])
        return row

    def test_no_unearned_credit_for_positive_shock(self):
        self.step(0, 0.)
        row = self.step(1, -60.)
        self.assertEqual((row['chosen_s'], row['available_s']), (15., 15.))
        self.assertLess(row['chosen_s'], row['raw_request_s'])

    def test_savings_can_fund_current_positive_innovation(self):
        self.step(0, 0.)
        saving = self.step(1, 20.)
        using = self.step(2, 0.)
        self.assertEqual((saving['chosen_s'], saving['credit_after_s']), (0., 15.))
        self.assertEqual((using['chosen_s'], using['credit_after_s']), (30., 0.))
        self.assertEqual(self.actual, using['issued_total_s'])

    def test_negative_new_innovation_not_overridden_by_old_credit(self):
        self.step(0, 0.)
        for station in range(1, 8):
            row = self.step(station, 20.*station)
            self.assertEqual(row['chosen_s'], 0.)
            self.assertLessEqual(row['chosen_s'], row['shadow_s'])
        self.assertEqual(row['credit_after_s'], 105.)

    def test_zero_innovation_does_not_spend_credit_above_feedback(self):
        self.step(0, 0.)
        self.step(1, 20.)
        row = self.step(2, 20.)
        self.assertEqual(row['chosen_s'], row['shadow_s'])
        self.assertEqual(row['credit_after_s'], 15.)

    def test_balanced_gap_expires_credit_and_request(self):
        self.step(0, 0.)
        self.step(1, 20.)
        row = self.step(2, -40., error=0.)
        self.assertEqual((row['chosen_s'], row['expired_s'], row['credit_after_s']), (0., 15., 0.))
        self.assertEqual(self.step(3, -40.)['chosen_s'], 15.)

    def test_reference_changes_expire_credit_without_old_shock(self):
        self.step(0, 0.)
        self.step(1, 20.)
        row = self.step(2, -400., predecessor=5)
        self.assertEqual((row['increment_s'], row['expired_s'], row['chosen_s']), (0., 15., 15.))
        self.step(3, -380., predecessor=5)
        self.assertEqual(self.step(5, -800., predecessor=5)['chosen_s'], 15.)
        self.step(6, None, error=0., predecessor=None)
        self.assertEqual(self.step(7, -1000.)['increment_s'], 0.)

    def test_actual_not_requested_holding_is_debited(self):
        row = self.step(0, 0., actual=5.)
        self.assertEqual((row['chosen_s'], row['executed_s'], row['credit_after_s']), (15., 5., 10.))
        row = self.step(1, -40.)
        self.assertEqual((row['available_s'], row['chosen_s']), (25., 20.))

    def test_all_inflight_prefixes_are_funded(self):
        key, row = self.request(0, 0.)
        self.assertEqual(self.rule.trips[9].spent, 0.)
        for actual in range(16):
            self.rule.check_actual(9, actual)
        with self.assertRaises(AssertionError):
            self.rule.check_actual(9, 16.)
        self.actual = row['chosen_s']
        self.rule.settle(key, self.actual)

    def test_cannot_borrow_next_stop_or_other_trip_authorization(self):
        key, _ = self.request(0, 0.)
        with self.assertRaises(AssertionError):
            self.request(1, -60.)
        self.rule.request((10, 0, True), 4, 0., 60., [True]*7, 15., 0.)
        with self.assertRaises(AssertionError):
            self.rule.check_actual(9, 20.)
        self.rule.settle(key, 15.)
        with self.assertRaises(KeyError):
            self.rule.settle(key, 15.)

    def test_quantization_never_rounds_above_credit_or_gap(self):
        self.step(0, 0., actual=5.)
        row = self.step(1, -100., error=12.)
        self.assertEqual(row['chosen_s'], 10.)
        self.assertLessEqual(row['chosen_s'], min(row['available_s'], 12.))

    def test_long_multistop_prefixes_under_alternating_shocks(self):
        for station in range(20):
            self.step(station, float((station % 7)*20), error=float((station % 11)*10))
        self.assertEqual(self.rule.audit()['conservation_error'], 0.)


if __name__ == '__main__':
    unittest.main()
