import unittest

import numpy as np

import audit_freqduet_v178_lower_width as audit


class LowerWidthTest(unittest.TestCase):
    def example(self, truth):
        q = dict(point_s=120., query_time_s=1.)
        p = dict(bounds=[20., 120.], temporal_bounds=[20., 100.])
        r = dict(label=dict(status='resolved', full_eta_s=truth, available_time_s=2., target_ready_time_s=2.),
            residual_s=dict(running=120.-truth, service=0., residence=0.))
        return q, p, r, dict(qhigh_s=100.)

    def test_width_saving_and_score_identity(self):
        v = dict(zip(audit.METRICS, audit.diagnostics(*self.example(105.))))
        self.assertEqual((v['expansion_s'], v['upper_saving_s'], v['envelope_score_delta_s']), (20., 50., -30.))
        self.assertEqual(v['repaired_above'], 1.)
        v = dict(zip(audit.METRICS, audit.diagnostics(*self.example(70.))))
        self.assertEqual(v['envelope_score_delta_s'], 20.)

    def test_lower_error_unrepaired(self):
        v = dict(zip(audit.METRICS, audit.diagnostics(*self.example(15.))))
        self.assertEqual((v['below'], v['lower_excess_s'], v['below_running_s']), (1., 5., 105.))
        self.assertEqual(v['envelope_score_delta_s'], 20.)

    def test_missing_components_are_not_zero(self):
        args = self.example(15.)
        args[2]['residual_s'] = None
        with self.assertRaises(AssertionError):
            audit.diagnostics(*args)

    def test_equal_episode_not_query_weight_and_partition(self):
        n = len(audit.METRICS)
        def row(count):
            return np.r_[count, count, np.ones(n)*count]
        base = {'a': np.array([2., 2.]), 'b': np.array([1., 1.])}
        groups = {'a': {('all', 'all'): row(2), ('stage', 'A'): row(1), ('stage', 'B'): row(1)},
            'b': {('all', 'all'): row(1), ('stage', 'B'): row(1)}}
        result = {(r['axis'], r['value']): r['values'] for r in audit.weighted_profiles(base, groups)}
        self.assertAlmostEqual(result['stage', 'A'][2], .25)
        self.assertAlmostEqual(result['stage', 'B'][2], .75)
        self.assertAlmostEqual(result['all', 'all'][3], 1.)

    def test_shared_day_bootstrap_matches_direct_draws(self):
        weights = audit.bootstrap_weights(2, 3)
        rng = np.random.RandomState(20260803)
        a = rng.randint(0, 2, size=(5000, 2))
        d = rng.randint(0, 3, size=(5000, 3))
        values = np.arange(6.).reshape(2, 3)
        np.testing.assert_allclose(weights@values.ravel(), values[a[:, :, None], d[:, None, :]].mean(axis=(1, 2)))

    def test_no_finite_subgroup_retains_abstention(self):
        matrix = np.zeros((4, 3+len(audit.METRICS)))
        matrix[:, 0] = 1.
        result = audit.profile_readout(matrix, audit.bootstrap_weights(2, 2))
        self.assertEqual(result['finite_fraction']['mean'], 0.)
        self.assertIsNone(result['conditional']['envelope_score_delta_s']['mean'])
        self.assertIsNone(result['conditional']['envelope_score_delta_s']['ci95'])


if __name__ == '__main__':
    unittest.main()
