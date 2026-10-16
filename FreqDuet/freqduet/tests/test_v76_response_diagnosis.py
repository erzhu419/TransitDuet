from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v76_response_diagnosis as audit


class ResponseDiagnosisTest(unittest.TestCase):
    def test_exact_quadratic_has_zero_shape_error_and_native_anchor(self):
        x = audit.action_design(3)
        truth = x @ np.arange(10.).reshape(2, 5)
        prediction = x @ np.arange(10., 20.).reshape(2, 5)
        projected, total, shape, mapping = audit.decompose(prediction, truth, 3)
        np.testing.assert_allclose(projected, truth, atol=1e-12)
        np.testing.assert_allclose(shape, 0., atol=1e-24)
        np.testing.assert_allclose(total, mapping, atol=1e-12)
        np.testing.assert_array_equal(projected[3], 0.)

    def test_nonquadratic_shape_is_orthogonal_to_curve_mapping_error(self):
        rng = np.random.default_rng(76)
        truth = rng.normal(size=(7, 5))
        truth[4] = 0.
        prediction = audit.action_design(4) @ rng.normal(size=(2, 5))
        projected, total, shape, mapping = audit.decompose(prediction, truth, 4)
        self.assertTrue((shape > 0).all())
        np.testing.assert_allclose(total, shape + mapping, atol=1e-12)
        valid = np.arange(7) != 4
        np.testing.assert_allclose(audit.action_design(4)[valid].T @ (projected - truth)[valid], 0., atol=1e-12)

    def test_day_sources_recompose_frozen_ridge_without_refitting_candidates(self):
        rng = np.random.default_rng(77)
        states = rng.normal(size=(8, 3))
        native, days = np.arange(8) % 7, np.repeat(np.arange(4), 2)
        outcomes = rng.normal(size=(8, 7, 5))
        fitted = audit.model.fit_model(states, native, outcomes, days, "state_quadratic")
        groups = [days == day for day in range(4)]
        groups.append(np.arange(8) < 2)
        coefficients = audit.source_coefficients(states, native, outcomes[:, :, 0], days, fitted, groups)
        np.testing.assert_allclose(coefficients[:, :4].sum(axis=1), fitted.coefficients[:, 0], atol=1e-12)
        np.testing.assert_allclose(coefficients[:, 0], coefficients[:, 4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
