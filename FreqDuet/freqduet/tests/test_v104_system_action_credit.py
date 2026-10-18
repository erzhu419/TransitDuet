import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import freqduet_system_action_credit as model


class SystemActionCreditTest(unittest.TestCase):
    def test_common_level_cancels_from_all_signed_differences(self):
        rng = np.random.default_rng(104)
        y = rng.normal(size=(8, 4, 7, 3))
        p = np.full((8, 7), 1 / 7)
        shift = rng.normal(size=(8, 4, 1, 3)) * 10000
        np.testing.assert_allclose(model.center(y, p[:, None]), model.center(y + shift, p[:, None]), atol=1e-10)
        np.testing.assert_allclose(model.differences(y), model.differences(y + shift), atol=1e-10)
        self.assertEqual(model.differences(y).shape, (8, 4, 21, 3))

    def test_value_and_goals_have_independent_parameters(self):
        action, value = model.GoalHeads(5, 7), model.GoalHeads(5, 1)
        value(torch.ones(2, 5, dtype=torch.float64)).sum().backward()
        self.assertTrue(all(p.grad is None for p in action.parameters()))
        action(torch.ones(2, 5, dtype=torch.float64))[..., 0].sum().backward()
        self.assertTrue(all(p.grad is None for h in action.heads[1:] for p in h.parameters()))

    def test_signed_three_goal_proposal_and_zero_baseline(self):
        p = np.full(7, 1 / 7)
        a = np.zeros((7, 3))
        np.testing.assert_array_equal(model.proposal(a, p, np.ones(3)), p)
        a[0] = [-4, 1, 1]
        a[1] = [-3, -1, -1]
        a = model.center(a, p)
        b = model.proposal(a, p, np.ones(3))
        self.assertLess((b - p) @ a[:, 0], 0)
        self.assertTrue(np.all((b - p) @ a[:, 1:] <= 1e-9))
        np.testing.assert_allclose(b.sum(), 1.)

    def test_level_shift_does_not_change_advantage_fit(self):
        rng = np.random.default_rng(4)
        x, p = rng.normal(size=(4, 5)), np.full((4, 7), 1 / 7)
        y = rng.normal(size=(4, 4, 7, 3))
        a = model.fit(x, p, y, steps=3)
        b = model.fit(x, p, y + 100., steps=3)
        for key, weights in a['networks']['separated_a'].items():
            torch.testing.assert_close(weights, b['networks']['separated_a'][key], atol=1e-12, rtol=0)
        before = copy.deepcopy(a)
        prediction = model.predict(a, x, p)
        np.testing.assert_allclose((prediction['separated_a'] * p[..., None]).sum(1), 0, atol=1e-12)
        for key in a['networks']:
            for name, weights in a['networks'][key].items():
                torch.testing.assert_close(weights, before['networks'][key][name], atol=0, rtol=0)

    def test_finite_future_noise_not_value_level(self):
        rng = np.random.default_rng(20)
        p, y = np.full((2, 7), 1 / 7), rng.normal(size=(2, 4, 7, 3))
        pred = {k: np.zeros((2, 7, 3)) for k in ('direct_q', 'separated_a', 'zero')}
        pred['level_v'] = (y.mean(1) * p[..., None]).sum(1)
        e = model.errors(pred, y, p)
        np.testing.assert_array_equal(e['level_v']['level_squared_error'], 0)
        self.assertTrue(np.all(e['zero']['difference_mse'] > 0))
        np.testing.assert_allclose(e['signal']['difference_energy_minus_noise'],
                                  e['zero']['difference_mse'] - e['signal']['difference_noise_mse'])


if __name__ == '__main__':
    unittest.main()
