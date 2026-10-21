from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v195_mechanism as stage


class MechanismReadoutTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(71)
        v = rng.normal(size=(4, 2, 7, stage.N))
        # Distinct day means expose between-day recentering mistakes.
        v += np.arange(4)[:, None, None, None]*2.
        v[..., 30] = v[..., 0]+v[..., 31]
        v[..., 38] = v[..., 32:38].sum(axis=-1)
        return v

    def moments(self, v):
        return np.array([[[stage.saved.moment(x) for x in key] for key in day] for day in v]).mean(axis=2)

    def test_packed_moments_and_within_between_partition(self):
        v = self.sample()
        m = self.moments(v)
        f, s, w = stage.day_parts(m)
        c, mass = stage.weighted_covariance(f, s, np.full((1, 4), .25))
        direct = sum(np.cov(v[:, k].reshape(-1, stage.N), rowvar=False, bias=True) for k in range(2))
        within = sum(np.cov(v[d, k], rowvar=False, bias=True) for d in range(4) for k in range(2))/4
        between = sum(np.cov(v[:, k].mean(axis=1), rowvar=False, bias=True) for k in range(2))
        np.testing.assert_allclose(c[0], direct, atol=1e-12)
        np.testing.assert_allclose(w.mean(axis=0), within, atol=1e-12)
        np.testing.assert_allclose(c[0]-w.mean(axis=0), between, atol=1e-12)
        self.assertEqual(mass[0], 2.)

    def test_resampled_key_centers_match_direct_duplicated_days(self):
        v = self.sample()
        f, s, _ = stage.day_parts(self.moments(v))
        c, _ = stage.weighted_covariance(f, s, np.array([[.5, 0., .25, .25]]))
        duplicate = v[[0, 0, 2, 3]]
        direct = sum(np.cov(duplicate[:, k].reshape(-1, stage.N), rowvar=False, bias=True) for k in range(2))
        np.testing.assert_allclose(c[0], direct, atol=1e-12)

    def test_common_projection_preserves_component_and_innovation_identities(self):
        _, _, w = stage.day_parts(self.moments(self.sample()))
        reference = w.mean(axis=0)
        operator = stage.projection(reference)
        residual = stage.projected(reference, operator)
        stage.identities(residual)
        cov, _ = stage.measures(residual)
        self.assertAlmostEqual(cov[6], cov[7]+cov[8])
        self.assertAlmostEqual(cov[10], cov[1]+cov[8])
        np.testing.assert_allclose((operator @ reference)[:, stage.CI], 0., atol=1e-10)

    def test_control_operator_uses_within_not_between_day(self):
        v = self.sample()
        f, s, w = stage.day_parts(self.moments(v))
        c, _ = stage.weighted_covariance(f, s, np.full((1, 4), .25))
        within = w.mean(axis=0)
        self.assertGreater(np.max(np.abs(stage.projection(c[0])-stage.projection(within))), .01)
        operator = stage.projection(within)
        np.testing.assert_allclose(stage.projected(c[0], operator),
            stage.projected(within, operator)+stage.projected(c[0]-within, operator), atol=1e-10)

    def test_batched_common_projection_all64_and_direct_half_difference(self):
        v = self.sample()
        f, s, w = stage.day_parts(self.moments(v))
        c, _ = stage.weighted_covariance(f, s, np.full((1, 4), .25))
        pc = np.stack((c[0], c[0]*1.7))
        pw = np.stack((w.mean(axis=0), w.mean(axis=0)*1.4))
        bc, bw = np.repeat(pc[:, None], 3, axis=1), np.repeat(pw[:, None], 3, axis=1)
        result, _ = stage.summarize(pc, pw, bc, bw, 'clearance')
        for part in result.values():
            for target in stage.TARGETS:
                estimates = {k: part[k][target]['covariance']['mean'] for k in part}
                self.assertAlmostEqual(estimates['all64'], (estimates['first32']+estimates['last32'])/2)
                self.assertAlmostEqual(estimates['delta'], estimates['last32']-estimates['first32'])

    def test_crossed_bootstrap_shares_actors_not_half_days(self):
        a, d = stage.common.weights(12)
        self.assertEqual(a.shape, (10000, 12))
        self.assertEqual(d.shape, (2, 10000, 32))
        self.assertFalse(np.array_equal(d[0], d[1]))
        np.testing.assert_allclose(a.sum(axis=1), 1.)
        np.testing.assert_allclose(d.sum(axis=2), 1.)

    def test_missing_variance_is_unknown_and_not_zero(self):
        c, rho = stage.measures(np.zeros((len(stage.SELECTED),)*2))
        self.assertTrue(np.isnan(c).all() and np.isnan(rho).all())
        interval = stage.prior.interval(1., np.array([1., np.nan]))
        self.assertIsNone(interval['ci'])
        self.assertIsNone(stage.sign_check(interval, 1))

    def test_family_and_both_halves_required(self):
        results = []
        for mode in ('old', 'handoff'):
            for scope in stage.PRIMARY:
                stats = {split: {target: {name: dict(mean=float(sign), ci=[.5, 1.5] if sign > 0 else [-1.5, -.5])
                    for name in ('covariance', 'family_covariance')} for target, sign in stage.TESTS.values()}
                    for split in ('first32', 'last32', 'all64')}
                results.append(dict(mode=mode, scope=scope, stats=dict(within_day=stats)))
        self.assertTrue(all(g['status'] == 'replicated' for g in stage.gates(results).values()))
        results[0]['stats']['within_day']['first32']['first_running_under_s']['covariance']['ci'] = [-.1, .2]
        self.assertEqual(stage.gates(results)['clearance|H1']['status'], 'not_replicated')
        results[0]['stats']['within_day']['all64']['non_running_under_s']['family_covariance']['ci'] = None
        self.assertEqual(stage.gates(results)['clearance|H3']['status'], 'inconclusive')
        self.assertEqual(stage.FAMILY_ALPHA, .00625)


if __name__ == '__main__':
    unittest.main()
