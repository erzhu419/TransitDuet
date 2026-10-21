from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v193_within_day as stage


class WithinDayTests(unittest.TestCase):
    def test_only_common_keys_with19_days_in_each_half(self):
        m = np.zeros((64,3,stage.SIZE))
        m[:19,0,0],m[32:50,0,0] = 1.,1.
        m[:18,1,0],m[32:51,1,0] = 1.,1.
        m[:19,2,0],m[32:51,2,0] = 1.,1.
        np.testing.assert_array_equal(stage.common.masks(m)[2],[False,False,True])

    def test_event_cell_partition_and_variance(self):
        m = np.zeros((1,3,stage.SIZE))
        for k,ys in enumerate(((0.,1.),(0.,0.),(1.,1.))):
            for x,y in zip((1.,3.),ys):
                v = np.zeros(stage.N)
                v[1],v[stage.TARGET] = x,y
                m[0,k] += stage.prior.moment(v)/2.
        np.testing.assert_allclose(stage.cell_masses(m),[[3.,1.5,1.,.5,1.,1.,1.]])
        within,_ = stage.parent.day_matrices(m)
        self.assertAlmostEqual(float(within[0,1,1]),3.)

    def test_concentration_equal_or_single_day_and_missing_events(self):
        weights = np.full((1,32),1./32.)
        eff,maximum = stage.concentration(np.ones((1,32)),weights)
        np.testing.assert_allclose(eff,[32.])
        np.testing.assert_allclose(maximum,[1./32.])
        one = np.zeros((1,32))
        one[0,0] = 1.
        eff,maximum = stage.concentration(one,weights)
        np.testing.assert_allclose(eff,[1.])
        np.testing.assert_allclose(maximum,[1.])
        missing,_ = stage.concentration(np.zeros((1,32)),weights)
        self.assertTrue(np.isnan(missing[0]))

    def test_duplicate_bootstrap_days_are_occurrences_not_squared_weights(self):
        weights = np.zeros((1,32))
        weights[0,0],weights[0,1] = .75,.25
        eff,maximum = stage.concentration(np.ones((1,32)),weights)
        np.testing.assert_allclose(eff,[32.])
        np.testing.assert_allclose(maximum,[1./32.])

    def test_daily_correlations_preserve_unknown_when_variance_missing(self):
        c = np.zeros((3,stage.N,stage.N))
        c[:2,1,1],c[:2,stage.TARGET,stage.TARGET] = 4.,1.
        c[:2,1,stage.TARGET],c[:2,stage.TARGET,1] = [1.,-1.],[1.,-1.]
        x,y = np.zeros(stage.N),np.zeros(stage.N)
        x[1],y[stage.TARGET] = 1.,1.
        values = stage.correlations(c,x,y)
        np.testing.assert_allclose(values[:2],[.5,-.5])
        self.assertTrue(np.isnan(values[2]))

    def test_batched_metrics_and_covariance_reproduction(self):
        rng = np.random.default_rng(41)
        z = rng.normal(size=(200,stage.N))
        c = np.cov(z,rowvar=False)
        cov = np.broadcast_to(c,(2,stage.N,stage.N)).copy()
        daily = np.broadcast_to(c,(2,32,stage.N,stage.N)).copy()
        masses = np.ones((2,32,len(stage.FIELDS)))
        masses[:,:,1] = .1
        weights = np.full((2,32),1./32.)
        ref = stage.common.measure(cov,1,True)
        m,d = stage.metrics(cov,daily,masses,weights,ref)
        np.testing.assert_allclose(m['v192_within_contribution'],ref[0])
        np.testing.assert_allclose(m['raw_within_innovation_variance_fraction'],1.)
        np.testing.assert_allclose(m['effective_upper_mass_days'],32.)
        np.testing.assert_allclose(m['adjusted_positive_day_fraction']+m['adjusted_negative_day_fraction']+
            m['adjusted_zero_day_fraction']+m['adjusted_unknown_day_fraction'],1.)
        self.assertEqual(d['adjusted'].shape,(2,32))


if __name__=='__main__':
    unittest.main()
