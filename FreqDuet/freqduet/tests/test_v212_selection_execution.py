import copy
import unittest

import numpy as np

import audit_freqduet_v212_selection_execution as v


def row(lower, counts):
    histogram = {str(int(a)): counts.get(int(a), 0) for a in v.ACTIONS}
    total = sum(int(a)*n for a, n in histogram.items())
    return dict(lower=lower, results=dict(action_counts=histogram, dispatch_gap_mean_s=360.,
        dispatch_gap_variance_s2=100., inclusive_launch_delay_s=4., readiness_delay_s=3.,
        journey_min=20., cv_squared=.04), ledger=dict(commanded_holding_s=total, executed_holding_s=total),
        anchor_command=dict(candidate=[0.]), authority=dict(planner=dict(action_ema_alpha=.35)),
        upper=dict(target_s=360., planned_s=7200.))


class SelectionExecutionTests(unittest.TestCase):
    def test_zero_to_zero_has_no_direct_reduction(self):
        r = row([dict(base_bin=0, selected_bin=0, actual_command_s=0.)], {0: 3})
        histogram, gross = v.direct_override(r)
        self.assertEqual(float(histogram @ v.ACTIONS), 0.)
        self.assertEqual(gross, 0.)

    def test_later_feedback_can_reverse_a_five_second_reduction(self):
        control = row([], {5: 2})
        r = row([dict(base_bin=1, selected_bin=0, actual_command_s=0.)], {0: 1, 15: 1})
        values, histograms = v.paired_effect(r, control)
        self.assertEqual(values['direct_s'], -5.)
        self.assertEqual(values['total_command_s'], 5.)
        self.assertEqual(values['rule_response_s'], 10.)
        self.assertEqual(v.reduction_class(values), 'fully_offset_or_reversed')
        np.testing.assert_array_equal(histograms[0]+histograms[2], histograms[1])

    def test_partly_offset_and_amplified_reductions_are_distinct(self):
        control = row([], {5: 2})
        r = row([dict(base_bin=2, selected_bin=0, actual_command_s=0.)], {5: 1})
        values, _ = v.paired_effect(r, control)
        self.assertEqual(v.reduction_class(values), 'partly_offset')
        r['lower'][0]['base_bin'] = 1
        r['results']['action_counts'] = {str(int(a)): 0 for a in v.ACTIONS}
        r['results']['action_counts']['0'] = 1
        r['ledger'].update(commanded_holding_s=0., executed_holding_s=0.)
        values, _ = v.paired_effect(r, control)
        self.assertEqual(v.reduction_class(values), 'amplified_reduction')

    def test_balanced_overrides_are_not_no_intervention(self):
        r = row([dict(base_bin=1, selected_bin=0, actual_command_s=0.),
            dict(base_bin=0, selected_bin=1, actual_command_s=5.)], {0: 1, 5: 1})
        histogram, gross = v.direct_override(r)
        self.assertEqual(float(histogram @ v.ACTIONS), 0.)
        self.assertEqual(gross, 10.)

    def test_anchor_plan_is_not_an_actual_departure_metric(self):
        control = row([], {0: 2})
        r = copy.deepcopy(control)
        r['anchor_command']['candidate'] = [30.]
        r['upper']['target_s'] = 363.
        values, _ = v.paired_effect(r, control)
        self.assertEqual(values['implied_preprojection_ema_delta_s'], 10.5)
        self.assertEqual(values['returned_target_delta_s'], 3.)
        self.assertEqual(values['anchor_plan_delta_s'], 0.)
        self.assertEqual(values['dispatch_mean_delta_s'], 0.)

    def test_missing_or_inconsistent_counts_are_not_zero_filled(self):
        r = row([], {5: 1})
        del r['results']['action_counts']['0']
        with self.assertRaisesRegex(AssertionError, 'missing/extra'):
            v.command_histogram(r)
        r = row([], {5: 1})
        r['ledger']['commanded_holding_s'] = 10.
        with self.assertRaisesRegex(AssertionError, 'reproduce'):
            v.command_histogram(r)

    def test_zero_effect_crossed_and_future_intervals_preserve_pairing(self):
        pairs = [dict(values=dict.fromkeys(v.FIELDS, 0.), histograms=np.zeros((3, 7))) for _ in range(64)]
        result = v.summarized(pairs)
        self.assertEqual(result['n_pairs'], 64)
        self.assertEqual(result['nonzero_direct_override_paths'], 0)
        self.assertEqual(result['any_saved_dispatch_signature_changed'], 0)
        self.assertEqual(result['intervals']['rule_response_s']['crossed_ci95'], [0., 0.])
        self.assertEqual(result['intervals']['rule_response_s']['conditional_future_ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
