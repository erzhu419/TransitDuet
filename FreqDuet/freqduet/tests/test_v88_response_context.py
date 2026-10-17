from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v88_response_context import ContextSums, GROUPS, context_group, group_transport, ratio_stat


class ResponseContextTest(unittest.TestCase):
    def test_missing_evidence_is_not_classified_as_zero_regularity_benefit(self):
        self.assertEqual(context_group(False, 10., 0.), "missing_bilateral")
        self.assertEqual(context_group(True, 10., 0.), "more_hold_gain_not_up")
        self.assertEqual(context_group(True, 10., .1), "more_hold_gain_up")
        self.assertEqual(context_group(True, -1., .1), "no_more_hold")

    def test_contributions_use_all_decisions_not_each_groups_count(self):
        sums = ContextSums()
        early = np.eye(7)[0]
        sums.add(np.eye(7)[1], early, np.zeros(7), False, 2, 3., 4.)
        sums.add(early, early, np.zeros(7), True, 1, 1., 2.)
        r = sums.summary()
        self.assertEqual(r["groups"]["missing_bilateral"]["fraction"], .5)
        self.assertEqual(r["groups"]["missing_bilateral"]["contribution"]["holding_difference_s"], 2.5)
        self.assertEqual(r["groups"]["missing_bilateral"]["contribution"]["passenger_command_difference_s"], 5.)
        self.assertEqual(r["groups"]["more_hold_gain_up"]["contribution"]["holding_difference_s"], 0.)

    def test_group_transport_is_additive_when_visitation_changes(self):
        a, b = ContextSums(), ContextSums()
        zero = np.eye(7)[0]
        a.add(np.eye(7)[1], zero, np.zeros(7), False, 2, 0., 0.)
        b.add(np.eye(7)[2], np.eye(7)[1], np.zeros(7), True, 1, 0., 0.)
        result = group_transport(a.summary(), b.summary())
        self.assertEqual(sum(result[g]["expected_command_s"] for g in GROUPS), 0.)
        self.assertEqual(sum(result[g]["common_state_law_s"] for g in GROUPS), 5.)
        self.assertEqual(sum(result[g]["visitation_s"] for g in GROUPS), -5.)

    def test_holding_weighted_condition_is_ratio_of_totals_not_mean_of_daily_ratios(self):
        result = ratio_stat([1., 9.], [1., 3.], np.array([[0, 1], [0, 1]]))
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["ci95"], [2.5, 2.5])


if __name__ == "__main__":
    unittest.main()
