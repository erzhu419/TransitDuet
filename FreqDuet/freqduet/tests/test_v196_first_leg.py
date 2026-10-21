from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v196_first_leg as stage


class FirstRoadTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(196)
        a = rng.normal(size=(2, stage.N, stage.N))
        c = a @ np.swapaxes(a, -1, -2)
        return c, stage.parent.projection(c.mean(axis=0))

    def test_four_term_and_cross_block_identities(self):
        c, r = self.fixture()
        terms = stage.decompose(c, r)
        np.testing.assert_allclose(terms['raw']+sum(terms[k] for k in stage.TERMS[1:4]), terms['adjusted'])
        np.testing.assert_allclose(sum(v for k, v in terms.items() if k.startswith('block|')), terms['projection'])
        self.assertGreater(abs(float(terms['joint_projection'][0])), .01)

    def test_common_operator_is_not_refitted_per_half_or_group(self):
        c, r = self.fixture()
        whole = stage.decompose(c, r)
        parts = [stage.decompose(c*f, r) for f in (.2, .3, .5)]
        for key in whole:
            np.testing.assert_allclose(sum(p[key] for p in parts), whole[key], atol=1e-12)
        separate = stage.decompose(c, stage.parent.projection(c))['adjusted']
        self.assertGreater(np.max(abs(separate-whole['adjusted'])), .01)

    def test_day_key_centering_preserves_weight_and_adds_to_old_within(self):
        rng = np.random.default_rng(9)
        v = rng.normal(size=(4, 3, 5, stage.N))
        v += np.arange(4)[:, None, None, None]*5.
        m = np.array([[[stage.saved.moment(x) for x in k] for k in d] for d in v]).mean(axis=2)
        m *= np.array([.1, .3, .7])[None, :, None]
        keys = stage.key_within(m)
        np.testing.assert_allclose(keys.sum(axis=1), stage.parent.day_parts(m)[2], atol=1e-12)
        for d in range(4):
            direct = sum(np.cov(v[d, k], rowvar=False, bias=True)*weight for k, weight in enumerate((.1, .3, .7)))
            np.testing.assert_allclose(keys[d].sum(axis=0), direct, atol=1e-12)

    def test_batched_terms_and_direct_half_delta(self):
        c, r = self.fixture()
        b = np.repeat(c[:, None], 7, axis=1)
        br = np.repeat(r[None], 7, axis=0)
        s = stage.summarize(stage.decompose(c, r), stage.decompose(b, br), primary=True)
        for k, values in s.items():
            self.assertAlmostEqual(values['delta']['mean'], values['last32']['mean']-values['first32']['mean'])
            self.assertAlmostEqual(values['all64']['mean'], (values['last32']['mean']+values['first32']['mean'])/2)
            self.assertEqual('delta_family' in values, k in ('raw', 'projection'))
        np.testing.assert_allclose(s['raw']['delta_family']['ci'], [s['raw']['delta']['mean']]*2)

    def test_between_and_within_use_same_operator_and_add(self):
        c, r = self.fixture()
        within, between = stage.decompose(c*.6, r), stage.decompose(c*.4, r)
        total = stage.decompose(c, r)
        for key in total:
            np.testing.assert_allclose(total[key], within[key]+between[key], atol=1e-12)

    def test_contributions_and_leave_out_are_signed_not_renormalized(self):
        c, r = self.fixture()
        base = stage.decompose(c, r, blocks=False)
        stats = stage.summarize(base, {k: np.repeat(v[:, None], 3, axis=1) for k, v in base.items()})
        pieces = [dict(actor=i, terms=stage.point_record({k: v*f for k, v in base.items()}))
            for i, f in enumerate((1.2, -.4, .2))]
        out = stage.contribution_summary(pieces, stats)
        self.assertAlmostEqual(out['positive_delta_sum']+out['negative_delta_sum'], stats['adjusted']['delta']['mean'])
        self.assertEqual(out['groups'], 3)
        direct = [stats['adjusted']['delta']['mean']-p['terms']['adjusted']['delta'] for p in pieces]
        np.testing.assert_allclose(out['leave_one_out']['adjusted']['delta_range'], [min(direct), max(direct)])


if __name__ == '__main__':
    unittest.main()
