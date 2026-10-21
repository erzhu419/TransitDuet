from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v190_leg_risk as stage


class LegRiskTests(unittest.TestCase):
    def test_single_completion_rms_and_strict_clocks(self):
        rows = {(True,2):[(100.,0.,100.),(300.,150.,150.),(500.,350.,150.)]}
        history = stage.clocks.native.CompletedMeans([(('running',True,2),end,value)
            for end,start,value in rows[True,2]])
        prefix = stage.RiskPrefix(rows,history)
        self.assertIsNone(prefix.query((True,2,0),300.,125.))
        value = prefix.query((True,2,0),301.,125.)
        self.assertEqual(value['n'],1)
        self.assertEqual(value['rms'],50.)
        self.assertEqual(value['remaining'],25.)
        reference = stage.support.Innovations(rows,history).query((True,2,0),501.,400./3.)
        later = prefix.query((True,2,0),501.,400./3.)
        self.assertEqual((reference['n'],reference['latest']),(later['n'],later['latest']))
        self.assertEqual(reference['remaining'],later['remaining'])
        self.assertIsNone(prefix.query((True,2,1),501.,400./3.))
        self.assertIsNone(prefix.query((True,2,0),4101.,400./3.))

    def test_exact19_day_support_and_missing_is_not_zero_effect(self):
        mass = np.zeros((64,3))
        mass[:18,0],mass[:19,1] = 1.,.01
        np.testing.assert_array_equal(stage.supported_keys(mass),[False,True,False])
        cov,weight = stage.actor_covariance(np.zeros((64,0,1+stage.N)),np.zeros((64,stage.N,stage.N)),
            np.full((3,64),1./64.))
        self.assertEqual(weight.sum(),0.)
        self.assertTrue(np.isnan(stage.partial(cov,0)).all())
        self.assertIsNone(stage.interval(np.nan,np.array([np.nan]*3))['mean'])
        self.assertIsNone(stage.interval(.1,np.array([.1,np.nan]))['ci'])

    def test_optimized_key_centering_matches_direct_resampling(self):
        rng = np.random.default_rng(8)
        m = np.zeros((4,3,stage.SIZE))
        for day,key in product(range(4),range(3)):
            for _ in range(5):
                m[day,key] += stage.moment(rng.normal(size=stage.N))/5.
        m[0,1] = 0.
        w = np.array([[.25]*4,[.5,.0,.5,.0],[0.,0.,0.,1.]])
        cov,mass = stage.actor_covariance(m[:,:,:1+stage.N],stage.second(m.sum(axis=1)),w)
        for i,weights in enumerate(w):
            pooled = np.einsum('d,dks->ks',weights,m)
            numerator = pooled[:,1:1+stage.N]
            corrected = stage.second(pooled).sum(axis=0)
            for j,p in enumerate(pooled[:,0]):
                if p:
                    corrected -= np.outer(numerator[j],numerator[j])/p
            np.testing.assert_allclose(cov[i],corrected,atol=1e-12)
            self.assertAlmostEqual(mass[i],pooled[:,0].sum())

    def test_controls_remove_common_predictor_not_between_day_by_definition(self):
        moments = []
        for c,z,t,w in product((-1.,1.),repeat=4):
            v = np.zeros(stage.N)
            v[0:2] = [c+z,w]
            v[stage.VARS.index('query_offset_min')] = c
            v[stage.VARS.index('above')] = 2*c+t
            moments.append(stage.moment(v))
        m = np.mean(moments,axis=0)
        cov = stage.second(m)-np.outer(m[1:1+stage.N],m[1:1+stage.N])
        j = stage.TARGETS.index('above')
        self.assertGreater(stage.partial(cov,0,False)[j],.5)
        self.assertAlmostEqual(float(stage.partial(cov,0)[j]),0.,places=10)
        self.assertTrue(np.isnan(stage.partial(cov,0)[stage.TARGETS.index('below')]))

    def test_preregistered_replication_requires_both_modes_and_halves(self):
        results = []
        for mode,scope,split in product(('old','handoff'),stage.PRIMARY,('all64','first32','last32')):
            estimates = {}
            for signal in stage.SIGNALS:
                estimates[signal] = {}
                for target in ('below','above'):
                    negative = signal == 'signed_remaining_s' and target == 'below'
                    interval = dict(mean=-.2 if negative else .2,ci=[-.3,-.1] if negative else [.1,.3])
                    estimates[signal][target] = dict(family=dict(interval),partial=dict(interval))
            results.append(dict(mode=mode,scope=scope,split=split,estimates=estimates))
        self.assertTrue(all(d['decision'] == 'replicated_calibration_association' for d in stage.evidence(results)))
        results[-1]['estimates'][stage.SIGNALS[0]]['below']['partial']['ci'] = None
        d = next(d for d in stage.evidence(results) if
            (d['scope'],d['signal'],d['target']) == (stage.PRIMARY[-1],stage.SIGNALS[0],'below'))
        self.assertEqual(d['decision'],'inconclusive')


if __name__ == '__main__':
    unittest.main()
