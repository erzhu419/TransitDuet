import unittest

import numpy as np

import audit_freqduet_v202_key_day_covariance as v


def atom(mass, j, a, y, jy=0., ay=0.):
    return np.array([mass, mass*j, mass*(j+a), mass*a, mass*y,
        mass*(j*y+jy), mass*((j+a)*y+jy+ay), mass*(a*y+ay)])


class KeyDayCovarianceTests(unittest.TestCase):
    def test_parent_centering_keeps_daily_within_and_i_minus_a(self):
        raw = np.zeros((64, 1, 9, 8))
        for d in range(64):
            raw[d, 0, 0] = atom(.2, 1.+d, 3., -2.+d, 4., -1.)
            raw[d, 0, 8] = atom(.8, 8.+d, 2., 6.+d, -2., 2.)
        centered = v.center(raw, raw.sum(axis=2))
        np.testing.assert_allclose(centered[..., 2], centered[..., 1]+centered[..., 3], atol=1e-12)
        np.testing.assert_allclose(centered[..., 6], centered[..., 5]+centered[..., 7], atol=1e-11)
        np.testing.assert_allclose(v.within(raw), v.within(centered), atol=1e-11)

    def test_raw_parent_within_and_between_are_distinct(self):
        raw = np.zeros((64, 1, 9, 8))
        for d in range(64):
            raw[d, 0, 0] = atom(1., float(d), 0., 2.*d, 3.)
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        centered = v.center(raw, raw.sum(axis=2))
        metrics, endpoints = v.evaluate(raw, centered, keep, np.full((2, 1, 32), 1./32))
        self.assertAlmostEqual(endpoints[0, 0, 0, 4, 0], 3.)
        self.assertAlmostEqual(endpoints[0, 0, 0, 5, 0], 2*np.var(np.arange(32.)))
        self.assertAlmostEqual(endpoints[0, 0, 0, 2, 0], 0.)
        np.testing.assert_allclose(metrics[..., 0], metrics[..., 1]-metrics[..., 2], atol=1e-10)

    def test_centered_between_day_cell_means_can_cancel_within(self):
        raw = np.zeros((64, 1, 9, 8))
        for d in range(64):
            j = 1. if d % 2 else -1.
            raw[d, 0, 0] = atom(.5, j, 0., -2*j, 1.)
            raw[d, 0, 8] = atom(.5, -j, 0., 2*j, 1.)
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        _, endpoints = v.evaluate(raw, v.center(raw, raw.sum(axis=2)), keep, np.full((2, 1, 32), 1./32))
        np.testing.assert_allclose(endpoints[0, 0, 0, :3, 0], [.5, -1., 0.])
        self.assertAlmostEqual(endpoints[0, 0, 0, 3, 0], -.5)

    def test_fixed_key_totals_reproduce_v201_with_unsupported_mass(self):
        rng = np.random.default_rng(8)
        raw = np.zeros((64, 1, 9, 8))
        for d in range(64):
            for b in (0, 1, 8):
                raw[d, 0, b] = atom(.1+b/20., *rng.normal(size=3), *rng.normal(size=2))
        centered = v.center(raw, raw.sum(axis=2))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, :2] = True
        frozen = keep.copy()
        weights = np.full((2, 1, 32), 1./32)
        metrics, _ = v.evaluate(raw, centered, keep, weights)
        simple = centered[..., [0, 1, 4, 5]]
        first, last = (v.base.average_days(z, weights[h]) for h, z in enumerate((simple[:32], simple[32:])))
        old = v.previous.key_terms(first, last, keep)
        np.testing.assert_allclose(metrics[..., 0, 0], old[..., 0])
        np.testing.assert_allclose(metrics[..., 1, 0], old[..., 1])
        np.testing.assert_allclose(metrics[..., 2, 0]+metrics[..., 3, 0], old[..., 2])
        np.testing.assert_allclose(metrics[..., 4, 0], old[..., 3])
        np.testing.assert_allclose(metrics[..., 6, 0], old[..., 7])
        np.testing.assert_array_equal(keep, frozen)


if __name__ == '__main__':
    unittest.main()
