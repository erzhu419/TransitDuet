import sys
from pathlib import Path
import unittest
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freqduet_incremental_compensation import ACTIONS, Compensation, bounded_action


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.rule = Compensation()
        self.total = 0.

    def step(self, station, residual, error, predecessor=4, executed=None):
        shadow = ACTIONS[min(range(7), key=lambda i: abs(ACTIONS[i]-min(max(.25*error, 0.), 45.)))]
        key = (9, station, True)
        _, row = self.rule.request(key, predecessor, residual, error, [True]*7, shadow, self.total)
        self.total += row['chosen_s'] if executed is None else executed
        self.rule.settle(key, self.total)
        for value in self.rule.conservation_errors().values():
            self.assertAlmostEqual(value, 0.)
        self.assertLessEqual(row['chosen_s'], max(error, 0.))
        return row

    def test_constant_residual_balanced_multistop(self):
        for station in range(12):
            self.assertEqual(self.step(station, -120., 0.)['chosen_s'], 0.)

    def test_persistent_shock_not_recharged_after_balance(self):
        self.step(0, 0., 0.)
        rows = [self.step(1, -60., 60.), self.step(2, -60., 15.), self.step(3, -60., 0.)]
        self.assertEqual([r['chosen_s'] for r in rows], [45., 15., 0.])
        self.assertEqual(rows[-1]['discarded_s'], 5.)
        for station in range(4, 12):
            self.assertEqual(self.step(station, -60., 0.)['chosen_s'], 0.)

    def test_shock_at_balance_expires_without_resurrection(self):
        self.step(0, 0., 0.)
        row = self.step(1, -60., 0.)
        self.assertEqual(row['discarded_s'], 45.)
        self.assertEqual(self.step(2, -60., 20.)['chosen_s'], 5.)

    def test_multistop_actuator_cap_then_balance(self):
        self.step(0, 0., 0.)
        rows = [self.step(i, -120., d) for i, d in enumerate((120., 75., 30., 0., 0.), 1)]
        self.assertEqual([r['chosen_s'] for r in rows], [45., 45., 30., 0., 0.])
        self.assertEqual(self.total, 120.)

    def test_credit_uses_execution_not_request(self):
        self.step(0, 0., 0.)
        row = self.step(1, -60., 60., executed=20.)
        self.assertEqual(row['chosen_s'], 45.)
        self.assertEqual(row['settled_extra_s'], 5.)
        self.assertEqual(self.step(2, -60., 40.)['unserved_s'], 40.)

    def test_new_predecessor_discards_old_budget(self):
        self.step(0, 0., 0.)
        self.step(1, -60., 60.)
        row = self.step(2, -180., 20., predecessor=5)
        self.assertEqual((row['innovation_s'], row['chosen_s']), (0., 5.))
        self.assertEqual(row['discarded_s'], 15.)

    def test_missing_and_skipped_prefix_reanchor(self):
        self.step(0, 0., 0.)
        self.step(1, -60., 60.)
        self.assertEqual(self.step(3, -180., 20.)['chosen_s'], 5.)
        self.step(4, None, 0., predecessor=None)
        self.assertEqual(self.step(5, -240., 20.)['chosen_s'], 5.)

    def test_quantized_excess_is_debited(self):
        self.step(0, 0., 0.)
        row = self.step(1, -4., 20.)
        self.assertEqual((row['increment_s'], row['chosen_s'], row['settled_extra_s']), (3., 10., 5.))
        self.assertEqual(self.step(2, -4., 20.)['chosen_s'], 5.)
        self.assertEqual(self.step(3, -4., 0.)['chosen_s'], 0.)

    def test_bound_feasibility_and_shorter_ties(self):
        self.assertEqual(ACTIONS[bounded_action(45., 12., [True]*7)], 10.)
        self.assertEqual(ACTIONS[bounded_action(7.5, 20., [True]*7)], 5.)
        self.assertEqual(ACTIONS[bounded_action(45., 20., [True, True, False, False, False, True, True])], 5.)

    def test_zero_budget_recovers_original_feedback(self):
        for error in (-60., 0., 3., 12., 20., 60., 120., 180., 300.):
            shadow = min(range(7), key=lambda i: abs(ACTIONS[i]-min(max(.25*error, 0.), 45.)))
            self.assertEqual(bounded_action(.25*error, error, [True]*7), shadow)

    def test_retained_budget_differs_from_single_step(self):
        self.step(0, 0., 0.)
        self.step(1, -60., 60.)
        retained = self.step(2, -60., 15.)['chosen_s']
        self.rule, self.total = Compensation(retain=False), 0.
        self.step(0, 0., 0.)
        self.step(1, -60., 60.)
        pulse = self.step(2, -60., 15.)['chosen_s']
        self.assertEqual((retained, pulse), (15., 5.))

    def test_pending_cannot_be_reissued_or_settled_twice(self):
        key = (9, 0, True)
        self.rule.request(key, 4, 0., 0., [True]*7, 0., 0.)
        with self.assertRaises(AssertionError):
            self.rule.request((9, 1, True), 4, -60., 60., [True]*7, 15., 0.)
        self.rule.settle(key, 0.)
        with self.assertRaises(KeyError):
            self.rule.settle(key, 0.)

    def test_sampler_metadata_differs_but_outcomes_must_match(self):
        from run_freqduet_v113_incremental_compensation import assert_sampler_invariance
        a = dict(results=dict(sampler_seed=1, journey_min=19., action_counts=[1, 2]),
            contract={}, coverage={}, accounting=dict(executed_s=5.))
        b = deepcopy(a)
        b['results']['sampler_seed'] = 2
        assert_sampler_invariance(a, b)
        b['accounting']['executed_s'] = 10.
        with self.assertRaises(AssertionError):
            assert_sampler_invariance(a, b)


if __name__ == '__main__':
    unittest.main()
