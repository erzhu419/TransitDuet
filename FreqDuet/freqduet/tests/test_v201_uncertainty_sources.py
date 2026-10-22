import unittest

import numpy as np

import audit_freqduet_v201_uncertainty_sources as v


class UncertaintySourceTests(unittest.TestCase):
    def test_key_readouts_sum_to_v200(self):
        rng = np.random.default_rng(19)
        first, last = rng.normal(size=(2, 5, 3, 9, 4))
        first[..., 0], last[..., 0] = np.abs(first[..., 0])+.2, np.abs(last[..., 0])+.2
        keep = rng.random((3, 9)) > .3
        e, t = v.previous.evaluate(first, last, keep)
        k = v.key_terms(first, last, keep).sum(axis=1)
        np.testing.assert_allclose(k[:, :6], t, atol=1e-12)
        np.testing.assert_allclose(k[:, 6:], e[:, 1, 3:6]-e[:, 0, 3:6], atol=1e-12)

    def test_pure_actor_day_and_interaction_variances(self):
        actor = np.array([[-1., -1.], [1., 1.]])
        day = np.array([[-1., 1.], [-1., 1.]])
        interaction = np.array([[-1., 1.], [1., -1.]])
        for data, expected in ((actor, [.5, 0., 0.]), (day, [0., 1., 0.]), (interaction, [0., 0., .5])):
            np.testing.assert_allclose(v.variance_channels(data[..., None])[:, 0], expected)
        combined = actor+2*day+3*interaction
        np.testing.assert_allclose(v.variance_channels(combined[..., None])[:, 0], [.5, 4., 4.5])

    def test_signed_allocation_keeps_negative_cancellation(self):
        total = np.array([-1., 1., -1., 1.])[:, None]
        parts = np.stack((2*total, -total), axis=1)
        allocation = v.covariance_allocation(parts, total)
        np.testing.assert_allclose(allocation[:, 0], [2., -1.])
        self.assertNotEqual(parts.var(axis=0).sum(), total.var())

    def test_shared_day_deletion_sign_and_fixed_mask(self):
        values = np.zeros((64, 1, 9, 4))
        values[:, 0, 0, 0] = 1.
        values[:, 0, 0, 3] = np.arange(64.)
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        original = keep.copy()
        point = v.key_terms(values[:32].mean(axis=0, keepdims=True), values[32:].mean(axis=0, keepdims=True), keep).sum(axis=1)[0]
        shift = v.leave_day(values, keep)-point
        np.testing.assert_allclose(shift[:32, -1], (np.arange(32.)-15.5)/31.)
        np.testing.assert_allclose(shift[32:, -1], (47.5-np.arange(32., 64.))/31.)
        np.testing.assert_array_equal(keep, original)

    def test_zero_supported_resample_stays_unknown(self):
        first, last = np.zeros((1, 1, 9, 4)), np.zeros((1, 1, 9, 4))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        last[0, 0, 0] = [1., 0., 0., 3.]
        first[0, 0, 8] = [1., 0., 0., 7.]
        result = v.key_terms(first, last, keep)
        self.assertTrue(np.isnan(result[..., :6]).all())
        np.testing.assert_allclose(result[0, 0, 6:], [3., -7., -4.])


if __name__ == '__main__':
    unittest.main()
