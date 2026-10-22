import unittest

import numpy as np

import audit_freqduet_v207_shared_day as v


def fixture():
    d, a, k, b = np.meshgrid(np.arange(64), np.arange(3), np.arange(2), np.arange(9), indexing='ij')
    mass = (1.+d*.002+a*.03+b*.01)/27.
    x = a*.2+b*.1+np.sin(d/5.)
    y = -a*.1+b*.2+np.sin(d/7.)+k*.1
    cov = 1.+.2*np.cos(d/3.)
    raw = np.stack((mass, mass*x, mass*y, mass*(x*y+cov)), axis=-1)
    keep = np.ones((3, 2, 9), bool)
    keep[..., [0, 5, 8]] = False
    return raw, keep


class SharedDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.keep = fixture()

    def test_common_location_shift_cancels_but_parent_scatter_changes(self):
        x = 4.*np.sin(np.arange(64.)/2.)[:, None, None, None]
        y = 3.*np.cos(np.arange(64.)/3.)[:, None, None, None]
        changed = self.raw.copy()
        changed[..., 3] += self.raw[..., 1]*y+self.raw[..., 2]*x+self.raw[..., 0]*x*y
        changed[..., 1] += self.raw[..., 0]*x
        changed[..., 2] += self.raw[..., 0]*y
        a, b = v.profiles(self.raw, self.keep), v.profiles(changed, self.keep)
        np.testing.assert_allclose(a[0].sum(axis=-1), b[0].sum(axis=-1), atol=1e-12)
        np.testing.assert_allclose(a[0][..., 0], b[0][..., 0], atol=1e-12)
        self.assertGreater(np.max(np.abs(a[3][0]['shared']-b[3][0]['shared'])), .1)

    def test_within_day_covariance_changes_do_not_become_mean_shift_effects(self):
        changed = self.raw.copy()
        changed[..., 3] += self.raw[..., 0]*.5
        a, b = v.profiles(self.raw, self.keep)[0], v.profiles(changed, self.keep)[0]
        np.testing.assert_allclose(a[..., 1:], b[..., 1:], atol=1e-12)
        self.assertGreater(np.min(b[..., 0]-a[..., 0]), 0.)

    def test_point_mass_coefficients_reproduce_v203_conditional_not_raw_delta(self):
        daily = v.profiles(self.raw, self.keep)[0]
        point = daily[32:].mean(axis=(0, 1))-daily[:32].mean(axis=(0, 1))
        original = v.scalar_point(self.raw, self.keep)
        np.testing.assert_allclose(point.sum(axis=-1), original.sum(axis=-1), atol=1e-12)
        np.testing.assert_allclose(point[:, 0], original[:, 0], atol=1e-12)

    def test_shared_vs_actor_specific_profile_covariance(self):
        values = np.zeros((64, 3, 2, 13))
        signal = np.sin(np.arange(64.))[:, None, None]
        values[..., 0], values[..., 3] = signal, .5*signal
        for s in v.profile_structure(values):
            np.testing.assert_allclose(s['total'], s['shared'], atol=1e-12)
            np.testing.assert_allclose(s['specific'], 0., atol=1e-12)
        values[:, 1] *= -1.
        values[:, 2] = 0.
        for s in v.profile_structure(values):
            np.testing.assert_allclose(s['shared'], 0., atol=1e-12)
            np.testing.assert_allclose(s['total'], s['specific'], atol=1e-12)

    def test_unsupported_queries_still_enter_full_parent_center(self):
        changed = self.raw.copy()
        changed[..., 0, 3] += 2.*self.raw[..., 0, 2]
        changed[..., 0, 1] += 2.*self.raw[..., 0, 0]
        a, b = v.profiles(self.raw, self.keep)[0], v.profiles(changed, self.keep)[0]
        self.assertGreater(np.max(np.abs(a.sum(axis=-1)-b.sum(axis=-1))), 1e-5)
        np.testing.assert_array_equal(self.keep, fixture()[1])

    def test_fixed_reference_and_estimator_remainder_reproduce_saved_day_response(self):
        rng = np.random.default_rng(207)
        frozen, point = rng.normal(size=(5, 2, 13)), rng.normal(size=(2, 13))
        target = rng.normal(size=(5, 2))
        pieces = v.response_parts(frozen, target, point)
        np.testing.assert_allclose(pieces.sum(axis=-1), target-point.sum(axis=-1), atol=1e-12)
        np.testing.assert_allclose(pieces[..., 2], target-frozen.sum(axis=-1), atol=1e-12)

    def test_nominal_scaling_does_not_renormalize_selected_mass(self):
        a, b = v.profiles(self.raw, self.keep), v.profiles(2.7*self.raw, self.keep)
        np.testing.assert_allclose(2.7*a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)

    def test_zero_exposure_is_marked_not_an_observed_zero_shift(self):
        raw = self.raw.copy()
        raw[0] = 0.
        daily, _, available, _, _ = v.profiles(raw, self.keep)
        self.assertFalse(available[0].any())
        self.assertTrue(available[1:].all())
        np.testing.assert_allclose(daily[0], 0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
