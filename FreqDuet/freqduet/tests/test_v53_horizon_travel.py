import unittest

import numpy as np
import pandas as pd

from freqduet_conditioned_travel import ConditionedTravelPath
from freqduet_horizon_travel import HorizonTravelPath, fit_reversion, horizon_basis, reverted_mean


class HorizonTravelTests(unittest.TestCase):
    def test_convex_and_smooth_by_lead(self):
        leads = np.arange(0, 1801)
        weights = horizon_basis(leads)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)
        mean = reverted_mean(10., 20., leads, [.1, .9])
        self.assertTrue(np.all((mean >= 10.) & (mean <= 20.)))
        self.assertLess(np.diff(mean).max(), .01)

    def test_completed_strict_prefix(self):
        rows = pd.DataFrame([dict(origin=0., start_s=lead, end_s=1000.+lead, actual=actual,
            current=10., historical=20.) for lead, actual in [(100., 10.), (1400., 20.), (1500., 999.)]])
        fit = fit_reversion(rows, 2500.)
        self.assertEqual(fit['samples'], 2)
        self.assertLess(fit['latest_completion_s'], 2500.)
        np.testing.assert_allclose(fit['beta'], [0., 1.], atol=2e-5)
        rows.loc[2, 'actual'] = -999.
        self.assertEqual(fit, fit_reversion(rows, 2500.))

    def test_unissued_training_prediction_rejected(self):
        rows = pd.DataFrame([dict(origin=100., start_s=100., end_s=200., actual=10., current=10., historical=20.)])
        with self.assertRaises(ValueError):
            fit_reversion(rows, 300.)

    def test_zero_reversion_preserves_sampler_draws(self):
        model = dict(now_s=100., dt_s=1., horizon_s=1800, bin_s=300, keys=['True:0'],
            means={'True:0': 10.}, historical_means={'True:0': 30.}, reversion={'beta': [0., 0.]},
            shared_factor=[[1.]], shared_fraction=.5, ratios={'True:0': [.5, 1., 1.5]})
        vehicle = dict(direction=True, travel_station_id=0)
        for mode in ('conditioned_independent', 'conditioned_shared'):
            old, new = ConditionedTravelPath(model, mode, 0), HorizonTravelPath(model, mode, 0)
            for time, fraction in [(100., .001), (101., .5), (1000., 1.), (1800., 1.)]:
                self.assertEqual(old(vehicle, time, fraction), new(vehicle, time, fraction))


if __name__ == '__main__':
    unittest.main()
