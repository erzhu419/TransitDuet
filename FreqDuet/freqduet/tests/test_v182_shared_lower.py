from pathlib import Path
from itertools import product
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v182_shared_lower as plan


class SharedLowerTests(unittest.TestCase):
    def rows(self):
        return [dict(day=d, scope=s, condition=k, eligible_episodes=0, baseline_fraction=None)
            for d, s, k in product(plan.CALIBRATION, plan.TARGET_SCOPES, ('unavailable', *plan.SIGNS))]

    def test_scope_overlap_and_repeated_queries_do_not_inflate_days(self):
        rows = self.rows()
        for c in rows:
            if c['day'] == plan.CALIBRATION[0] and c['condition'] == 'positive':
                c.update(eligible_episodes=1000, baseline_fraction=1.)
        days, complete = plan.shared_days(rows)
        self.assertEqual(len(days['positive']), 1)
        self.assertTrue(complete)

    def test_original_v172_abstentions_do_not_contribute(self):
        rows = self.rows()
        rows[1].update(eligible_episodes=10, baseline_fraction=0.)
        days, complete = plan.shared_days(rows)
        self.assertEqual(days, {s: set() for s in plan.SIGNS})

    def test_unavailable_prefix_blocks_mask_equivalence(self):
        rows = self.rows()
        rows[0].update(eligible_episodes=1, baseline_fraction=.5)
        days, complete = plan.shared_days(rows)
        result = plan.release([], dict(decision='pass', mode_scopes=[]), complete)
        self.assertEqual(result['decision'], 'insufficient_support')
        self.assertIsNone(result['candidate_mask_equals_v172'])

    def test_nineteen_day_threshold_is_not_relaxed(self):
        for d in (18, 19):
            parameter = dict(actor=1, mode='old', sign='positive', support_eligible=d >= plan.MIN_DAYS)
            result = plan.release([parameter], dict(decision='pass', mode_scopes=[]), True)
            self.assertEqual(result['decision'], 'pass' if d == 19 else 'insufficient_support')

    def test_failed_baseline_scope_cannot_be_overridden(self):
        result = plan.release([], dict(decision='insufficient_support', failed=['old/hour13'], mode_scopes=[]), True)
        self.assertEqual(result['decision'], 'insufficient_support')
        self.assertEqual(result['failed'], ['old/hour13'])

    def test_missing_or_duplicate_metadata_is_not_zero_support(self):
        rows = self.rows()
        with self.assertRaises(AssertionError):
            plan.shared_days(rows[:-1])
        with self.assertRaises(AssertionError):
            plan.shared_days(rows[:-1]+[rows[0]])

    def test_registered_estimator_changes_error_not_v181_branch_keys(self):
        model = dict(actors=list(range(12)), frozen_protocol=dict(frozen=dict(point='V151'),
            conditioning=dict(candidate_key=['actor', 'mode']), support=dict(denominator='all eligible'),
            inference=dict(score='interval score', reduction='equal episodes', bootstrap='crossed'), gates=dict()))
        registered = plan.registration(model)
        self.assertEqual(registered['candidate']['maximum_parameters'], 48)
        self.assertNotIn('extra_field', registered['candidate'])
        self.assertTrue(registered['future_estimator']['error'].startswith('A=L172-T'))
        self.assertEqual(registered['future_estimator']['upper'], 'Unew=U172 exactly')
        self.assertFalse(any(registered['execution'].values()))


if __name__ == '__main__':
    unittest.main()
