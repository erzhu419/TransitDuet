from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v197_key_states as stage


class KeyStateTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(197)
        a = rng.normal(size=(2, stage.N, stage.N))
        c = a @ np.swapaxes(a, -1, -2)
        return c, stage.base.projection(c.mean(axis=0))

    def test_state_mixed_residual_and_all_cross_blocks_add(self):
        c, r = self.fixture()
        p = stage.components(c, r)
        old = stage.previous.decompose(c, r, blocks=False)
        np.testing.assert_allclose(p['state_covariance'], old['joint_projection'])
        np.testing.assert_allclose(p['residual'], old['adjusted'])
        np.testing.assert_allclose(p['raw'], p['state_covariance']+p['mixed']+p['residual'])
        for name in ('state', 'mixed'):
            np.testing.assert_allclose(sum(v for k, v in p.items() if k.startswith(name+'|')),
                p['state_covariance' if name == 'state' else 'mixed'])

    def test_pooled_mixed_zero_is_projection_identity_not_validation(self):
        c, r = self.fixture()
        p = stage.components(c, r)
        self.assertGreater(abs(p['mixed'][1]-p['mixed'][0]), .01)
        self.assertAlmostEqual(float(p['mixed'].mean()), 0., places=10)

    def test_day_mean_translation_changes_means_not_within_covariance(self):
        rng = np.random.default_rng(73)
        v = rng.normal(size=(4, 2, 8, stage.N))
        def moments(x):
            return np.array([[[stage.saved.moment(row) for row in k] for k in d] for d in x]).mean(axis=2)
        m, shifted = moments(v), moments(v+np.arange(4)[:, None, None, None]*3.)
        c, w, sums = stage.key_moments(m)
        cc, ww, ss = stage.key_moments(shifted)
        np.testing.assert_allclose(w, ww, atol=1e-12)
        self.assertGreater(np.max(abs(ss-sums)), 1.)
        self.assertGreater(np.max(abs(cc-c)), 1.)
        np.testing.assert_allclose(c.sum(axis=0), stage.base.weighted_covariance(
            m[..., :1+stage.N], stage.base.second(m.sum(axis=1)), np.full((1, 4), .25))[0][0], atol=1e-12)

    def test_state_units_variances_and_correlations(self):
        c, _ = self.fixture()
        sums = np.ones((2, 1+stage.N))
        sums[:, 0] = 2.
        values, shifts = stage.state_values(c, c*.75, sums)
        ix = stage.IX[0]
        self.assertAlmostEqual(values['within_variance'][0, 0], .75*c[0, ix, ix]/2.)
        self.assertAlmostEqual(values['between_variance'][0, 0], .25*c[0, ix, ix]/2.)
        self.assertAlmostEqual(values['corr_J'][0, 0], c[0, ix, stage.previous.J]/np.sqrt(c[0, ix, ix]*c[0, stage.previous.J, stage.previous.J]))
        np.testing.assert_allclose(shifts, 0.)

    def test_missing_variance_is_unknown_and_finite_zero_means_are_valid(self):
        c = np.zeros((2, stage.N, stage.N))
        sums = np.zeros((2, 1+stage.N))
        sums[:, 0] = 1.
        values, shifts = stage.state_values(c, c, sums)
        np.testing.assert_allclose(values['mean'], 0.)
        self.assertTrue(np.isnan(values['corr_J']).all() and np.isnan(shifts).all())

    def test_batched_state_delta_and_component_partition(self):
        c, _ = self.fixture()
        sums = np.ones((2, 1+stage.N))
        bc, bs = np.repeat(c[:, None], 5, axis=1), np.repeat(sums[:, None], 5, axis=1)
        result = stage.state_summary(c, c*.6, sums, bc, bc*.6, bs)
        for state in result.values():
            for metric in state['metrics'].values():
                if metric['delta']['mean'] is not None:
                    self.assertAlmostEqual(metric['delta']['mean'], metric['last32']-metric['first32'])
        stats, _ = stage.summarize(c, c*.6, bc, bc*.6, 'clearance')
        self.assertIn('delta_family', stats['within_day']['state_covariance'])
        self.assertNotIn('delta_family', stats['between_day']['state_covariance'])


if __name__ == '__main__':
    unittest.main()
