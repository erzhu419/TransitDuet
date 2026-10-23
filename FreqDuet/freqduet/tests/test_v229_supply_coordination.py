from pathlib import Path
import unittest

from preregister_freqduet_v229_supply_coordination import (
    ACTIONS, local_gap_cost_change, registration, source_audit)


class SupplyCoordinationTest(unittest.TestCase):
    def test_quadratic_difference_is_exact(self):
        for forward, backward in ((100, 110), (300, 700), (200, 100)):
            for a in ACTIONS:
                for base in ACTIONS:
                    r = lambda h: (forward+h-360)**2+(backward-h-360)**2
                    self.assertEqual(local_gap_cost_change(forward, backward, a, base), r(a)-r(base))

    def test_common_gap_shift_does_not_change_quadratic_difference(self):
        self.assertEqual(local_gap_cost_change(100, 500, 10, 20),
                         local_gap_cost_change(160, 560, 10, 20))

    def test_native_underrelaxation_blocks_smaller_bin_local_nonharm(self):
        comparisons = 0
        for gap_difference in range(1001):
            raw = .25*gap_difference/2
            base = min(ACTIONS, key=lambda h: abs(h-min(45., max(0., raw))))
            for smaller in ACTIONS:
                if smaller < base:
                    comparisons += 1
                    self.assertGreater(local_gap_cost_change(100, 100+gap_difference, smaller, base), 0.)
        self.assertGreater(comparisons, 0)

    def test_no_positive_compensation_for_reverse_gap(self):
        self.assertGreater(local_gap_cost_change(500, 100, 5, 0), 0.)

    def test_native_request_is_on_frontier(self):
        for a in ACTIONS:
            self.assertEqual(local_gap_cost_change(100, 400, a, a), 0.)

    def test_no_execution_or_authority_expansion(self):
        protocol = registration('test')
        self.assertEqual(set(protocol['budget'].values()), {0})
        self.assertEqual(protocol['inherited']['fleet'], 12)
        self.assertEqual(protocol['inherited']['holding_bins_s'], list(ACTIONS))
        self.assertEqual(protocol['inherited']['native_gain'], .25)
        for field in ('runtime_connected', 'selector_defined', 'collection_authorized',
                      'replay_authorized', 'training_authorized', 'main_policy_changed'):
            self.assertIs(protocol[field], False)

    def test_live_query_and_saved_sidecar_are_not_confused(self):
        audit = source_audit(Path(__file__).resolve().parents[1]/'scripts')
        self.assertIn('ready', audit['live_query_fields_available'])
        self.assertTrue({'ready', 'active', 'plans', 'onboard', 'executed_holding_before_s'} <=
                        set(audit['query_sidecar_fields_not_retained']))
        self.assertFalse(audit['archive_field_completeness_verified'])
        self.assertFalse(audit['recording_equivalence_verified'])


if __name__ == '__main__':
    unittest.main()
