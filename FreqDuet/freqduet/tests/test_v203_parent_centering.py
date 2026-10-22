import unittest

import numpy as np

import audit_freqduet_v203_parent_centering as v


def fixture(common):
    raw = np.zeros((64, 1, 9, 4))
    for d in range(64):
        j, y = (d*.2, d*.4) if common else (0., 0.)
        offset = 1. if common else (-1. if d % 2 else 1.)
        for bi, sign in ((0, 1.), (8, -1.)):
            x, z = j+sign*offset, y+sign*2*offset
            raw[d, 0, bi] = [.5, .5*x, .5*z, .5*(x*z+1.)]
    parent = raw.sum(axis=2)
    mu = parent[..., 1:3]/parent[..., :1]
    centered = raw.copy()
    centered[..., 1:3] -= raw[..., :1]*mu[:, :, None, :]
    centered[..., 3] -= raw[..., 1]*mu[..., 1, None]+raw[..., 2]*mu[..., 0, None]
    centered[..., 3] += raw[..., 0]*mu[..., 0, None]*mu[..., 1, None]
    keep = np.zeros((1, 9), dtype=bool)
    keep[0, 0] = True
    return raw, centered, keep


class ParentCenteringTests(unittest.TestCase):
    def test_common_parent_day_shift_cancels_in_four_between_terms(self):
        raw, centered, keep = fixture(True)
        pooled, parts = v.half(v.day_terms(raw, centered)[:32], np.full((1, 32), 1./32))
        b = parts[0, 0, 0, 1:5]
        expected = .5*.2*.4*np.var(np.arange(32.))
        np.testing.assert_allclose(b, [expected, -expected, -expected, expected], atol=1e-12)
        self.assertAlmostEqual(b.sum(), 0.)
        self.assertAlmostEqual(parts[0, 0, 0, 0], .5)
        self.assertAlmostEqual(parts[0, 0, 0, 5:9].sum(), 1.)

    def test_cell_daily_mean_covariance_without_parent_shift(self):
        raw, centered, keep = fixture(False)
        _, parts = v.half(v.day_terms(raw, centered)[:32], np.full((1, 32), 1./32))
        np.testing.assert_allclose(parts[0, 0, 0, 1:5], [1., 0., 0., 0.], atol=1e-12)

    def test_fixed_nominal_weight_is_not_applied_twice(self):
        raw, centered, keep = fixture(True)
        frozen = keep.copy()
        weights = np.full((2, 1, 32), 1./32)
        result = v.evaluate(v.day_terms(raw, centered), keep, weights)
        scaled = v.evaluate(v.day_terms(2.7*raw, 2.7*centered), keep, weights)
        for a, b in zip(result, scaled):
            np.testing.assert_allclose(2.7*a, b, atol=1e-11)
        np.testing.assert_array_equal(keep, frozen)

    def test_parts_match_v202_and_retain_unsupported_contribution(self):
        raw, centered, keep = fixture(True)
        weights = np.full((2, 1, 32), 1./32)
        parts, endpoints, extra = v.evaluate(v.day_terms(raw, centered), keep, weights)
        def extend(z):
            out = np.zeros((*z.shape[:-1], 8))
            out[..., 0], out[..., 1], out[..., 2], out[..., 4], out[..., 5], out[..., 6] = (
                z[..., 0], z[..., 1], z[..., 1], z[..., 2], z[..., 3], z[..., 3])
            return out
        old, _ = v.previous.evaluate(extend(raw), extend(centered), keep, weights)
        np.testing.assert_allclose(parts.sum(axis=-1), old[..., 1, 0], atol=1e-11)
        np.testing.assert_allclose(parts[..., 1:5].sum(axis=-1), old[..., 3, 0], atol=1e-11)
        np.testing.assert_allclose(parts[..., 5:9].sum(axis=-1), old[..., 4, 0], atol=1e-11)
        np.testing.assert_allclose(extra[..., 2], old[..., 6, 0], atol=1e-11)

    def test_reject_centering_on_selected_cells(self):
        raw, centered, keep = fixture(True)
        wrong = centered.copy()
        wrong[:, 0, 0, 1:3] = 0.
        with self.assertRaisesRegex(AssertionError, 'full-parent/day centering'):
            v.day_terms(raw, wrong)


if __name__ == '__main__':
    unittest.main()
