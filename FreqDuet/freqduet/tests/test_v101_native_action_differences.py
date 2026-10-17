import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_freqduet_v101_native_action_differences import boltzmann, entropy, objective_effect


class ActionDifferenceTests(unittest.TestCase):
    def test_exact_critic_gain_is_alpha_kl(self):
        p = np.full(7, 1 / 7)
        q, cost, pessimism = np.arange(7) / 20, np.arange(7) / 40, np.arange(7) / 100
        alpha, lam = .1, .5
        b = boltzmann(q - pessimism - lam * cost, alpha)
        effect = objective_effect(p, b, q, cost, pessimism, alpha, lam)
        self.assertAlmostEqual(effect['frozen_pessimism_soft_objective'], alpha * np.sum(p * np.log(p / b)))

    def test_common_value_bias_does_not_change_action_difference(self):
        p = np.full(7, 1 / 7)
        b = boltzmann(np.arange(7) / 10, .1)
        a = objective_effect(p, b, np.arange(7), np.arange(7) / 5, np.zeros(7), .1, .5)
        c = objective_effect(p, b, np.arange(7) + 100, np.arange(7) / 5 + 200, np.zeros(7), .1, .5)
        self.assertAlmostEqual(a['native_soft_objective'], c['native_soft_objective'])

    def test_pessimism_is_not_a_realized_passenger_cost(self):
        p, b = np.asarray([1., 0.]), np.asarray([0., 1.])
        effect = objective_effect(p, b, [0., 1.], [0., 0.], np.asarray([0., 2.]), .1, 1.)
        self.assertEqual(effect['native_soft_objective'], 1.)
        self.assertEqual(effect['frozen_pessimism_soft_objective'], -1.)
        self.assertEqual(entropy(b), 0.)


if __name__ == '__main__':
    unittest.main()
