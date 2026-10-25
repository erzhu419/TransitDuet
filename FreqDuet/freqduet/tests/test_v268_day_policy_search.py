from types import SimpleNamespace as NS
import unittest

import numpy as np

import freqduet_day_policy_search as method
import run_freqduet_v268_day_policy_search as run


def context(forward=300., follower=460., valid=True, bilateral=True):
    return NS(forward_headway_s=forward, follower_departure_gap_s=follower,
              target_headway_s=360., evidence_valid=valid, follower_evidence_valid=bilateral)


class CompleteDayLearning(unittest.TestCase):
    def test_neutral_matches_frozen_bilateral_including_fallback_and_ties(self):
        for valid in (False, True):
            for bilateral in (False, True):
                for forward in (0., 90., 300., 360., 460., 900.):
                    for follower in (0., 300., 460., 900.):
                        c = context(forward, follower, valid, bilateral)
                        for feasible in (np.ones(7, bool), np.array([1, 1, 1, 1, 0, 0, 0], bool)):
                            expected = run.fixed.prior.rule_action(c, 'bilateral', .25, feasible)
                            actual = method.choose(c, np.ones(6), np.zeros(6), feasible)
                            self.assertEqual(actual, expected)

    def test_state_response_can_increase_and_decrease_holding(self):
        c = context()
        theta = np.array([0., -.8, 0., 0., 0., 0.])
        empty = method.observation(c, 0., .5, 1, 0)
        full = method.observation(c, 1., .5, 1, 0)
        base = method.choose(c, empty, np.zeros(6), np.ones(7, bool))
        self.assertGreater(method.choose(c, empty, theta, np.ones(7, bool)), base)
        self.assertLess(method.choose(c, full, theta, np.ones(7, bool)), base)

    def test_mask_missing_evidence_and_shorter_tie(self):
        self.assertEqual(method.choose(context(valid=False), np.ones(6), np.ones(6), np.ones(7, bool)), 0)
        self.assertEqual(method.choose(context(300., 320.), np.ones(6), np.zeros(6), np.ones(7, bool)), 0)
        feasible = np.array([1, 1, 0, 0, 0, 0, 0], bool)
        self.assertIn(method.choose(context(), np.ones(6), np.ones(6), feasible), (0, 1))

    def test_uniform_has_only_one_degree_and_identical_zero_law(self):
        np.testing.assert_array_equal(method.coefficients([.3], 'uniform'), [.3, 0., 0., 0., 0., 0.])
        np.testing.assert_array_equal(method.coefficients([0.], 'uniform'), np.zeros(6))

    def test_observation_uses_only_current_inputs_and_retains_support(self):
        c = context(bilateral=False)
        x = method.observation(c, .7, .2, 2, 3)
        self.assertEqual(len(x), 6)
        self.assertEqual(x[-1], 0.)
        self.assertAlmostEqual(x[-2], .5)
        self.assertTrue(np.isfinite(x).all())

    def test_nonlinear_actual_constraints_reject_favorable_linear_direction(self):
        seen = []
        def evaluate(theta, index):
            seen.append(tuple(theta))
            return [dict(passenger_person_s=100. - theta[0],
                **{m: 1. + theta[0] ** 2 for m in method.CONSTRAINTS}) for _ in range(4)]
        result = method.search(evaluate, 'uniform')
        self.assertFalse(result['update_identified'])
        self.assertFalse(result['per_day_joint_improvement'])
        self.assertLessEqual(len(seen), 16)
        self.assertEqual(len(seen), len(set(seen)))

    def test_true_full_day_joint_improvement_is_retained_without_q(self):
        def evaluate(theta, index):
            return [dict(passenger_person_s=100. - theta[0],
                **{m: 1. - theta[0] for m in method.CONSTRAINTS}) for _ in range(4)]
        result = method.search(evaluate, 'uniform')
        self.assertTrue(result['update_identified'])
        self.assertTrue(result['per_day_joint_improvement'])
        self.assertTrue(all(v < 0. for v in result['mean_cost_changes']))

    def test_fixed_budget_and_consumed_development_not_confirmation(self):
        self.assertEqual(run.DAYS, (2071001, 2071002, 2071003, 2071004))
        self.assertEqual(run.SEED, 12011)
        self.assertEqual(run.SEARCH_PATH_CAP, 60)
        self.assertEqual(run.PHYSICAL_START_CAP, 123)


if __name__ == '__main__':
    unittest.main()
