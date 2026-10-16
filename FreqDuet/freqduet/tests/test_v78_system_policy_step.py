from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_system_policy_step as step


class SystemPolicyStepTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def days(self):
        rng = np.random.default_rng(78)
        result = []
        for i in range(3):
            result.append(dict(scenario_seed=i, times=np.array([1., 2., 3.]), horizon_s=4.,
                remaining_passenger=np.array([9., 6., 3.]) * (i + 1), cv_squared=.1 * i,
                readiness_delay_s=2. * i, probabilities=np.full((3, 7), 1. / 7),
                features=rng.normal(size=(3, 7, 4)), actions=np.array([0, 1, 2])))
        return result

    def test_features_have_causal_signs_and_cumulative_holding_response(self):
        context = SimpleNamespace(target_headway_s=300., forward_headway_s=150.,
            follower_departure_gap_s=450., evidence_valid=True, follower_evidence_valid=True)
        f = step.action_features(context, .5, 0, 2, 60.)
        np.testing.assert_array_equal(f[0], 0.)
        self.assertGreater(f[-1, 0], 0.)
        self.assertTrue((f[1:, 1:] < 0).all())
        more_ready = step.action_features(context, .5, 2, 2, 60.)
        self.assertGreater(more_ready[-1, 2], f[-1, 2])
        more_holding = step.action_features(context, .5, 0, 2, 120.)
        self.assertLess(more_holding[-1, 3], f[-1, 3])
        context.follower_evidence_valid = False
        np.testing.assert_array_equal(step.action_features(context, .5, 0, 2, 60.)[:, 0], 0.)

    def test_zero_adjustment_exactly_preserves_existing_categorical_probabilities(self):
        from lower.resac_lagrangian import CategoricalPolicy

        policy = CategoricalPolicy(3, step.ACTIONS, hidden_dim=8)
        states = torch.tensor([[.2, .4, .6]])
        expected = policy.dist_info(states)[0].detach()
        actual = step.probabilities(policy, states, np.arange(28.).reshape(7, 4), np.zeros(4))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_complete_day_gradient_uses_other_days_not_same_day_reference(self):
        days = self.days()
        gradients = step.gradient_days(days)
        day = days[0]
        scores = day["features"][np.arange(3), day["actions"]] - day["features"].mean(1)
        baseline = (days[1]["remaining_passenger"] + days[2]["remaining_passenger"]) / 2.
        np.testing.assert_allclose(gradients[0, 0], scores.T @ (day["remaining_passenger"] - baseline) / step.PASSENGER_SCALE)
        np.testing.assert_allclose(gradients[0, 1], scores.sum(0) * (-.15))
        np.testing.assert_allclose(gradients[0, 2], scores.sum(0) * (-3.))

    def test_empirical_kl_gradient_matches_finite_differences(self):
        days = self.days()
        theta = np.array([.01, .02, .03, .04])
        value, gradient = step.empirical_kl(theta, days)
        self.assertGreater(value, 0.)
        numeric = []
        for i in range(4):
            direction = np.eye(4)[i] * 1e-6
            numeric.append((step.empirical_kl(theta + direction, days)[0] -
                            step.empirical_kl(theta - direction, days)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-9)

    def test_feasible_cost_reduction_respects_cv_ready_and_kl_constraints(self):
        gradients = np.tile(np.array([[-1., -.5, .2, .1], [-1., 1., 1., 1.],
                                     [-1., 1., 1., 1.]])[None, :, :], (3, 1, 1))
        with patch.object(step, "gradient_days", return_value=gradients):
            result = step.fit_step(self.days())
        self.assertTrue(result["update_identified"])
        self.assertLess(result["directional_cost_changes"][0], 0.)
        self.assertTrue(np.all(np.asarray(result["directional_cost_changes"])[1:] <= 1e-8))
        self.assertLessEqual(result["training_mean_kl"], step.KL_LIMIT + 1e-9)

    def test_no_feasible_improvement_is_no_update_not_an_identical_control_win(self):
        gradients = np.tile(np.array([[-1.] * 4, [1.] * 4, [1.] * 4])[None, :, :], (3, 1, 1))
        with patch.object(step, "gradient_days", return_value=gradients):
            result = step.fit_step(self.days())
        self.assertFalse(result["update_identified"])
        np.testing.assert_array_equal(result["theta"], 0.)


if __name__ == "__main__":
    unittest.main()
