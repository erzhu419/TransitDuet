from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_conditional_response as response
import run_freqduet_v77_conditional_response as runner


class ConditionalResponseTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_action_basis_has_exact_native_anchor_and_signed_changes(self):
        native = np.arange(7)
        basis = response.action_basis(native)
        np.testing.assert_array_equal(basis[np.arange(7), native], 0.)
        self.assertLess(basis[3, 0, 0], 0.)
        self.assertGreater(basis[3, 6, 0], 0.)

    def test_features_use_current_state_and_known_policy_not_future_or_day_id(self):
        data = dict(states=np.ones((4, 2)), resources=np.zeros((4, 3)), cases=np.arange(4),
                    outcomes=np.zeros((4, 32, 7, 11)), days=np.arange(4))
        original = response.features(data, True)
        np.testing.assert_array_equal(original[:, -4:], np.eye(4))
        data["outcomes"] += 1e9
        data["days"] += 1000000
        np.testing.assert_array_equal(original, response.features(data, True))
        np.testing.assert_array_equal(original[:, :-4], response.features(data, False))

    def test_entire_demand_day_excluded_across_all_actors_and_windows(self):
        days = np.repeat([1, 2, 3], 12)
        training, testing = runner.partition(dict(days=days), 2)
        self.assertEqual(len(testing), 12)
        self.assertEqual(set(days[training]), {1, 3})
        self.assertEqual(set(days[testing]), {2})

    def test_small_fit_learns_effects_preserving_zero_count_outputs_and_anchor(self):
        rng = np.random.default_rng(77)
        states = rng.normal(size=(24, 3))
        native, days = np.arange(24) % 7, np.repeat(np.arange(4), 6)
        coefficient = np.zeros((24, 2, 5))
        coefficient[:, 0, 0] = 2. + np.tanh(states[:, 0])
        coefficient[:, 0, 1] = .001
        coefficient[:, 1, 2] = .2
        targets = np.einsum("nai,nim->nam", response.action_basis(native), coefficient)
        fitted, info = response.fit_model(states, native, targets, days, steps=200)
        prediction = fitted.predict(states, native)
        self.assertLess(info["final_normalized_mse"], .25 * info["initial_normalized_mse"])
        np.testing.assert_array_equal(prediction[np.arange(24), native], 0.)
        np.testing.assert_array_equal(prediction[:, :, 3:], 0.)
        self.assertTrue(np.isfinite(prediction).all())

    def test_fitting_is_reproducible_for_the_fixed_initialization(self):
        x = np.array([[0.], [1.], [2.], [3.]])
        n = np.zeros(4, dtype=int)
        y = np.ones((4, 7, 5))
        a, _ = response.fit_model(x, n, y, np.arange(4), steps=2)
        b, _ = response.fit_model(x, n, y, np.arange(4), steps=2)
        np.testing.assert_array_equal(a.predict(x, n), b.predict(x, n))


if __name__ == "__main__":
    unittest.main()
