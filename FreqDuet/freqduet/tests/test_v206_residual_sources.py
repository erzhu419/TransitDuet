import unittest

import numpy as np

import audit_freqduet_v206_residual_sources as v


class ResidualSourceTests(unittest.TestCase):
    def test_saved_beta_projection_equals_v205_residual_pair(self):
        rng = np.random.default_rng(206)
        a = rng.normal(size=(4, v.previous.N, v.previous.N))
        reference = a@np.swapaxes(a, -1, -2)
        matrices = rng.normal(size=(4, 2, 9, v.previous.N, v.previous.N))
        left, right, beta, _, _, _ = v.previous.projection(reference)
        expected = v.previous.projected(matrices, left, right)[..., 3]
        np.testing.assert_allclose(v.residual(matrices, beta), expected, atol=1e-12)

    def test_shared_operator_actor_parts_sum_without_actor_specific_fits(self):
        rng = np.random.default_rng(207)
        matrices = rng.normal(size=(3, 2, 9, v.previous.N, v.previous.N))
        beta = rng.normal(size=(1, 2, v.previous.Z))
        weights = np.array([[.25, .5, .25]])
        total = v.fixed_actor_sum((matrices,), weights)[0]
        parts = np.stack([v.residual(m[None], beta)[0] for m in matrices])
        expected = np.tensordot(weights, parts, axes=(1, 0))
        np.testing.assert_allclose(v.residual(total, beta), expected, atol=1e-12)

    def test_point_anchored_nonlinear_responses_are_exact_not_linear_anova(self):
        rng = np.random.default_rng(208)
        a, d = rng.normal(size=(2, 100))
        f = lambda x, y: x*x+2*y+x*y
        joint, actor, day = f(a, d), f(a, 0.), f(0., d)
        pieces = v.anchors(joint, actor, day, 0.)
        np.testing.assert_allclose(pieces[:, 2], a*d, atol=1e-12)
        allocation = v.partition(pieces, joint, list(v.CHANNELS))
        self.assertAlmostEqual(sum(allocation['signed_allocation']), joint.var())

    def test_signed_negative_cancellation_is_retained(self):
        x = np.arange(30.)-14.5
        parts = np.stack((3*x, -2*x, .5*x), axis=1)
        allocation = v.partition(parts, parts.sum(axis=1), ['a', 'b', 'c'])
        self.assertLess(allocation['signed_allocation'][1], 0.)
        self.assertGreater(allocation['shares'][0], 1.)
        self.assertNotAlmostEqual(allocation['sum_own_variances'], allocation['variance'])

    def test_temporal_parts_add_without_absorbing_mean_product_into_between(self):
        parts = np.arange(1., 10.).reshape(1, 1, 9)
        np.testing.assert_allclose(v.metrics(parts), [[[45., 1., 14., 30.]]])

    def test_delete_one_weights_share_days_and_do_not_rebuild_cell_support(self):
        aw, dw = v.deletion_weights(12)
        np.testing.assert_allclose(aw.sum(axis=1), 1.)
        np.testing.assert_allclose(np.diag(aw), 0.)
        np.testing.assert_allclose(dw.sum(axis=2), 1.)
        for h in range(2):
            np.testing.assert_allclose(dw[h, h*32:(h+1)*32].diagonal(), 0.)
            np.testing.assert_allclose(dw[h, (1-h)*32:(2-h)*32], 1./32)


if __name__ == '__main__':
    unittest.main()
