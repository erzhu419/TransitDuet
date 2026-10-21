from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v192_state_day as stage


class StateDayTests(unittest.TestCase):
    def test_common_keys_require19_days_in_both_halves(self):
        m = np.zeros((64,3,stage.SIZE))
        m[:19,0,0],m[32:50,0,0] = 1.,1.
        m[:18,1,0],m[32:51,1,0] = 1.,1.
        m[:19,2,0],m[32:51,2,0] = 1.,1.
        np.testing.assert_array_equal(stage.parent.masks(m)[2],[False,False,True])

    def test_day_covariance_with_missing_days_and_unequal_mass(self):
        m = np.zeros((4,2,stage.SIZE))
        for d in range(4):
            for k in range(2):
                if (d,k)==(0,1):
                    continue
                for z in (-1.,1.):
                    v = np.zeros(stage.N)
                    v[1],v[stage.TARGET] = 100.*k+d+z,10.*k+2.*d+3.*z
                    m[d,k] += (d+1)*stage.prior.moment(v)/2.
        within,between = stage.day_matrices(m)
        cov,_ = stage.prior.actor_covariance(m[:,:,:1+stage.N],stage.prior.second(m.sum(axis=1)),np.full((1,4),.25))
        np.testing.assert_allclose((within+between).mean(axis=0),cov[0],atol=1e-9)
        np.testing.assert_allclose(between,np.swapaxes(between,-1,-2),atol=1e-9)

    def test_group_algebra_with_correlated_and_duplicate_controls(self):
        rng = np.random.default_rng(20260803)
        x = rng.normal(size=(200,stage.N))
        x[:,10],x[:,0] = x[:,9],x[:,1]+x[:,9]
        cov = np.cov(x,rowvar=False)
        ref = stage.parent.measure(cov,1,True)
        for c in (cov,2.*cov,np.stack((cov,2.*cov))):
            r = stage.decomposition(c,ref)
            np.testing.assert_allclose(sum(r[g] for g in ('raw','target_adjustment','signal_adjustment','joint_adjustment')),
                stage.parent.contribution(c,ref),atol=1e-10)
            np.testing.assert_allclose(r['raw']+sum(r['groups'].values()),r['adjusted'],atol=1e-10)
        missing = stage.decomposition(np.zeros_like(cov),stage.parent.measure(np.zeros_like(cov),1,True))
        self.assertIsNone(stage.parent.number(missing['adjusted']))

    def test_common_reference_does_not_use_half_specific_projections(self):
        rng = np.random.default_rng(17)
        a,b = rng.normal(size=(300,stage.N)),rng.normal(size=(300,stage.N))
        a[:,stage.TARGET] += 2*a[:,9]
        b[:,stage.TARGET] -= 3*b[:,9]
        c = np.stack((np.cov(a,rowvar=False),np.cov(b,rowvar=False)))
        ref = stage.parent.measure(c.mean(axis=0),1,True)
        values = stage.decomposition(c,ref)['adjusted']
        self.assertAlmostEqual(float(values.mean()),float(ref[0]))
        self.assertEqual(stage.reference_correlation(c,ref).shape,(2,))
        pairs = stage.block_pairs(c[0],ref)
        self.assertAlmostEqual(sum(sum(r.values()) for r in pairs.values()),
            float(stage.decomposition(c[0],ref)['joint_adjustment']))

    def test_resampling_recenters_keys_before_between_day_partition(self):
        m = np.zeros((3,2,stage.SIZE))
        for d in range(3):
            for k in range(2):
                for z in (-1.,1.):
                    v = np.zeros(stage.N)
                    v[1],v[stage.TARGET] = d+2*z+10*k,3*d+z-5*k
                    m[d,k] += (d+k+1)*stage.prior.moment(v)/2.
        within,old_between = stage.day_matrices(m)
        weights = np.array([[.1,.3,.6],[.6,.3,.1]])
        cov,wd,_ = stage.resample(m[:,:,:1+stage.N],stage.prior.second(m.sum(axis=1)),within,weights)
        for j,w in enumerate(weights):
            new = m*(3*w[:,None,None])
            nw,nb = stage.day_matrices(new)
            np.testing.assert_allclose(cov[j],(nw+nb).mean(axis=0),atol=1e-9)
            np.testing.assert_allclose(cov[j]-wd[j],nb.mean(axis=0),atol=1e-9)
        self.assertFalse(np.allclose(cov[0]-wd[0],np.einsum('d,dij->ij',weights[0],old_between)))

    def test_state_shift_uses_key_centered_reference_sd(self):
        c,s = np.zeros((2,stage.N,stage.N)),np.zeros((2,1+stage.N))
        c[:,1,1],s[:,0],s[:,2] = [8.,8.],[2.,2.],[2.,6.]
        mu,delta,scaled = stage.state_shifts(c,s)
        np.testing.assert_allclose(mu[:,1],[1.,3.])
        self.assertEqual(delta[1],2.)
        self.assertEqual(scaled[1],1.)
        self.assertTrue(np.isnan(scaled[0]))

    def test_bootstrap_batched_reference_and_state_shapes(self):
        rng = np.random.default_rng(9)
        base = np.cov(rng.normal(size=(200,stage.N)),rowvar=False)
        cov = np.broadcast_to(base,(2,3,stage.N,stage.N)).copy()
        ref = stage.parent.measure(cov.mean(axis=0),1,True)
        values = stage.decomposition(cov[0],ref)
        self.assertEqual(values['adjusted'].shape,(3,))
        s = np.ones((2,3,1+stage.N))
        s[:,:,0] = 2.
        _,delta,smd = stage.state_shifts(cov,s)
        self.assertEqual(delta.shape,(3,stage.N))
        self.assertEqual(smd.shape,(3,stage.N))


if __name__=='__main__':
    unittest.main()
