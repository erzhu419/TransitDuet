from collections import defaultdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_freqduet_v114_incremental_budget as audit


class IncrementalAuditTests(unittest.TestCase):
    def row(self, error=60., increment=0., budget=15., chosen=30., shadow=15.):
        return dict(feedback_error_s=error, increment_s=increment, unserved_s=budget,
            raw_request_s=.25*error+budget, chosen_s=chosen, shadow_s=shadow, innovation_s=-increment/.75)

    def test_carried_budget_and_pulse_are_distinguished(self):
        values = audit.decision_math(self.row())
        self.assertEqual(values['carry_command_delta_s'], 15.)
        self.assertEqual(values['effective_innovation_s'], 0.)
        self.assertIn('positive_carry_without_new_innovation', audit.cohorts(self.row(), values))

    def test_balanced_new_shock_is_expired(self):
        row = self.row(error=-20., increment=45., budget=0., chosen=0., shadow=0.)
        values = audit.decision_math(row)
        self.assertEqual((values['effective_innovation_s'], values['carry_s']), (0., 0.))
        self.assertEqual(sum(values[k] for k in audit.COMPONENTS), 0.)

    def test_cap_and_balance_bound_and_quantization(self):
        capped = audit.decision_math(self.row(error=120., increment=90., budget=90., chosen=45., shadow=30.))
        self.assertEqual(capped['cap_clipping_s'], -75.)
        bounded = audit.decision_math(self.row(error=12., increment=45., budget=45., chosen=10., shadow=5.))
        self.assertEqual((bounded['cap_clipping_s'], bounded['balance_bound_s'], bounded['quantization_s']), (-3., -33., -2.))
        self.assertEqual(sum(bounded[k] for k in audit.COMPONENTS), 5.)

    def test_negative_carry_reduces_current_response(self):
        values = audit.decision_math(self.row(budget=-10., chosen=5.))
        self.assertEqual((values['carry_s'], values['carry_command_delta_s']), (-10., -10.))

    def test_sign_partitions_and_overlapping_cohorts(self):
        totals = defaultdict(lambda: defaultdict(float))
        row = self.row()
        values = audit.decision_math(row)
        audit.accumulate(totals, row, values)
        self.assertEqual(totals['all']['count'], totals['carry_positive']['count'])
        self.assertEqual(totals['all']['count'], totals['innovation_zero']['count'])

    def test_between_prefix_change_is_not_the_direct_carry_effect(self):
        ledger = {(1, 2, True): dict(chosen_s=30., pulse_shadow_s=15., carry_command_delta_s=15.)}
        pulse = {(1, 2, True): dict(chosen_s=20.)}
        result = audit.matched_commands(ledger, pulse)
        self.assertEqual(result['actual_holding_delta_s'], 10.)
        self.assertEqual(result['same_ledger_prefix_carry_response_s'], 15.)
        self.assertEqual(result['between_path_prefix_difference_s'], -5.)
        with self.assertRaises(AssertionError):
            audit.matched_commands(ledger, {})

    def test_changed_trace_command_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit.decision_math(self.row(chosen=45.))


if __name__ == '__main__':
    unittest.main()
