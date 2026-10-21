from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v191_common_keys as stage


class CommonKeyTests(unittest.TestCase):
    def test_intersection_requires19_distinct_days_in_each_half(self):
        m = np.zeros((64,3,stage.SIZE))
        m[:19,0,0],m[32:50,0,0] = 1.,1.
        m[:18,1,0],m[32:51,1,0] = 1.,1.
        m[:19,2,0],m[32:51,2,0] = 1.,1.
        a,b,c = stage.masks(m)
        np.testing.assert_array_equal(a,[True,False,True])
        np.testing.assert_array_equal(b,[False,True,True])
        np.testing.assert_array_equal(c,[False,False,True])

    def test_balancing_equalizes_support_mass_without_mutating_input(self):
        a,b = np.zeros((32,2,stage.SIZE)),np.zeros((32,2,stage.SIZE))
        a[:,:,0],b[:,:,0] = [1.,3.],[4.,2.]
        original = a.copy(),b.copy()
        x,y = stage.balance(a,b)
        np.testing.assert_array_equal(a,original[0])
        np.testing.assert_array_equal(b,original[1])
        np.testing.assert_allclose(x[:,:,0].mean(axis=0),[2.5,2.5])
        np.testing.assert_allclose(y[:,:,0].mean(axis=0),[2.5,2.5])
        self.assertEqual(stage.balance(a[:,:0],b[:,:0])[0].shape[1],0)

    def test_day_partition_retains_between_day_covariance(self):
        m = np.zeros((3,2,stage.SIZE))
        for day in range(3):
            for key in range(2):
                for z in (-1.,1.):
                    v = np.zeros(stage.N)
                    v[0],v[stage.TARGET] = day+z+100*key,day+2*z
                    m[day,key] += stage.prior.moment(v)/2.
        cov,mass,within,days = stage.day_partition(m)
        self.assertEqual(mass,2.)
        self.assertAlmostEqual(within.mean(axis=0)[0,stage.TARGET],4.)
        self.assertAlmostEqual((cov-within.mean(axis=0))[0,stage.TARGET],4./3.)
        measurement = stage.measure(cov,0,True)
        w = stage.contribution(within.mean(axis=0),measurement)
        b = stage.contribution(cov-within.mean(axis=0),measurement)
        self.assertAlmostEqual(float(w+b),float(measurement[0]))
        self.assertAlmostEqual(float(stage.contribution(days,measurement).mean()),float(measurement[0]))

    def test_residual_vectors_match_original_partial_with_duplicate_controls(self):
        rng = np.random.default_rng(21)
        values = rng.normal(size=(100,stage.N))
        values[:,10] = values[:,9]
        cov = np.cov(values,rowvar=False)
        for signal in (0,1):
            for unique in (False,True):
                expected = stage.prior.partial(cov,signal,other_signal=unique)[stage.prior.TARGETS.index('above')]
                self.assertAlmostEqual(float(stage.measure(cov,signal,unique)[0]),float(expected),places=10)
        missing = stage.measure(np.zeros((stage.N,stage.N)),0,True)[0]
        self.assertIsNone(stage.number(missing))

    def test_correlation_normalization_identity(self):
        a = (None,np.array([1.,-.3]),np.array([2.,3.]))
        b = (None,np.array([2.,-.1]),np.array([5.,1.]))
        covariance,normalization = stage.normalization(a,b)
        np.testing.assert_allclose(covariance+normalization,b[1]/b[2]-a[1]/a[2],atol=1e-12)

    def test_halves_share_actors_but_not_arbitrary_day_positions(self):
        a,days = stage.weights(12)
        np.testing.assert_allclose(a.sum(axis=1),1.)
        np.testing.assert_allclose(days.sum(axis=2),1.)
        self.assertFalse(np.array_equal(days[0],days[1]))


if __name__ == '__main__':
    unittest.main()
