import unittest

import numpy as np

import audit_freqduet_v205_linear_states as v


def fixture(shift=False):
    rng = np.random.default_rng(205)
    result = np.zeros((64, 2, 9, 78))
    for d in range(64):
        for key in range(2):
            for cell in range(9):
                z = rng.normal(size=(8, v.Z))
                z[:, 1] = z[:, 0]
                z[:, -1] = 0.
                z[:, 2] *= 1.+d/100.
                j = z[:, 0]+.7*z[:, 2]+rng.normal(size=8)+cell*.1
                y = -.4*z[:, 0]+z[:, 2]+.3*j+rng.normal(size=8)+d/50.
                values = np.column_stack((z, j, y))
                if shift:
                    values += np.arange(1., v.N+1)*d*.01
                mass = (1.+cell/20.+d/300.)/18.
                result[d, key, cell] = mass*np.r_[1., values.mean(axis=0), (values.T@values/8.)[v.TRI]]
    keep = np.ones((2, 9), dtype=bool)
    keep[:, [3, 7, 8]] = False
    return result, keep


def weights(draws=5):
    rng = np.random.default_rng(201)
    return np.stack([rng.multinomial(32, np.full(32, 1./32), size=draws)/32. for _ in range(2)])


def scalar_days(values):
    mass, first, second = v.unpack(values)
    raw = np.stack((mass, first[..., v.J], first[..., v.Y], second[..., v.J, v.Y]), axis=-1)
    parent = raw.sum(axis=2)
    mu = parent[..., 1:3]/parent[..., :1]
    centered = raw.copy()
    centered[..., 1:3] -= raw[..., :1]*mu[:, :, None]
    centered[..., 3] -= raw[..., 1]*mu[..., 1, None]+raw[..., 2]*mu[..., 0, None]
    centered[..., 3] += raw[..., 0]*mu[..., 0, None]*mu[..., 1, None]
    return v.old.day_terms(raw, centered)


class LinearStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values, cls.keep = fixture()
        cls.cache = v.prepare(cls.values, cls.keep)

    def test_matrix_parts_and_each_draw_reproduce_v203_scalar_algebra(self):
        dw = weights()
        result = v.aggregate([self.cache], np.ones((5, 1)), dw)
        p, ep, ex = v.old.evaluate(scalar_days(self.values), self.keep, dw)
        np.testing.assert_allclose(result[0][..., v.J, v.Y], p, atol=1e-12)
        np.testing.assert_allclose(result[2][..., v.J, v.Y], ep, atol=1e-12)
        np.testing.assert_allclose(result[3], ex[..., 2], atol=1e-12)
        ref = dict(boot=p, endpoints_boot=ep, extra_boot=ex)
        self.assertLess(v.reproduce(result, ref, slice(None)), 1e-12)

    def test_all_four_pairs_add_and_operator_is_refitted_per_draw(self):
        result = v.aggregate([self.cache], np.ones((5, 1)), weights())
        pairs, mix, endpoints, beta, active, rank, valid, error = v.decompose(result)
        self.assertTrue(valid.all())
        self.assertLess(error, 1e-11)
        self.assertGreater(float(np.max(np.abs(beta-beta[:1]))), 1e-5)
        self.assertFalse(active[:, -1].any())
        self.assertTrue((rank < v.Z).all())
        np.testing.assert_allclose(pairs+mix, endpoints[:, 1]-endpoints[:, 0], atol=1e-11)

    def test_reference_is_within_cell_day_not_parent_centered_second_moment(self):
        changed, keep = fixture(shift=True)
        dw = weights(1)
        a = v.aggregate([self.cache], np.ones((1, 1)), dw)
        b = v.aggregate([v.prepare(changed, keep)], np.ones((1, 1)), dw)
        np.testing.assert_allclose(a[-1], b[-1], atol=1e-11)

    def test_unsupported_cells_never_get_state_matrices_but_affect_parent_means(self):
        self.assertTrue(np.all(self.cache['matrices'][:, ~self.keep] == 0.))
        self.assertTrue(np.any(self.cache['unsupported'] != 0.))
        changed = self.values.copy()
        # Removing unsupported mass changes full parent/day means,not the mask.
        changed[:, ~self.keep] = 0.
        other = v.prepare(changed, self.keep)
        self.assertGreater(np.max(np.abs(other['first']-self.cache['first'])), .001)
        np.testing.assert_array_equal(other['keep'], self.keep)

    def test_nominal_mass_scaling_is_not_selected_mass_renormalization(self):
        dw = weights(1)
        a = v.aggregate([self.cache], np.ones((1, 1)), dw)
        b = v.aggregate([v.prepare(2.7*self.values, self.keep)], np.ones((1, 1)), dw)
        for original, scaled in zip(a, b):
            np.testing.assert_allclose(2.7*original, scaled, atol=1e-11)
        np.testing.assert_allclose(v.projection(a[-1])[2], v.projection(b[-1])[2], atol=1e-10)

    def test_no_state_variation_is_unknown_not_zero_effect(self):
        reference = np.zeros((1, v.N, v.N))
        reference[0, v.J, v.J] = reference[0, v.Y, v.Y] = 1.
        left, right, _, active, rank, valid = v.projection(reference)
        self.assertFalse(valid[0])
        self.assertEqual(rank[0], 0)
        self.assertFalse(active.any())
        self.assertTrue(np.isnan(left).all() and np.isnan(right).all())


if __name__ == '__main__':
    unittest.main()
