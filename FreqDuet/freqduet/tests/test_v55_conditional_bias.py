import unittest

import pandas as pd

from freqduet_conditional_bias import (BiasCalibratedTravelPath, attach_issued_means,
    bias_coefficients, calibrated_mean, fit_completed_bias, predicted_bias)
from freqduet_conditional_persistence import ConditionalPersistencePath


class ConditionalBiasTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(road=road, origin=0., start_s=t, end_s=t+10.,
            actual=20.+bias, issued_mean=20.)
            for road, bias in [('True:0', 10.), ('True:1', -10.)] for t in range(100, 1500, 50)])

    def test_strict_completed_prefix(self):
        rows = self.rows()
        rows.loc[len(rows)] = dict(road='True:0', origin=2000., start_s=2100., end_s=2200., actual=999., issued_mean=1.)
        fit = fit_completed_bias(rows, 2200.)
        rows.loc[len(rows)-1, ['actual', 'issued_mean']] = [-999., 999.]
        self.assertEqual(fit, fit_completed_bias(rows, 2200.))
        self.assertLess(fit['latest_completion_s'], 2200.)
        rows.loc[0, 'origin'] = rows.loc[0, 'start_s']
        with self.assertRaises(ValueError):
            fit_completed_bias(rows, 2200.)

    def test_residual_signs_follow_the_road(self):
        fit = fit_completed_bias(self.rows(), 2000.)
        self.assertGreater(predicted_bias(fit, 'True:0', 0., 1400.), 0.)
        self.assertLess(predicted_bias(fit, 'True:1', 0., 1400.), 0.)

    def test_unobserved_horizon_retains_parent_bias(self):
        self.assertEqual(bias_coefficients(self.rows().iloc[:1], [0., 7.])[1], 7.)

    def test_each_residual_uses_its_original_issued_model(self):
        rows = pd.DataFrame([dict(road='True:0', origin=o, start_s=o+100., end_s=o+150.,
            actual=30., current=10., historical=20.) for o in [0., 1000.]])
        models = {str(o): dict(now_s=o, persistence=dict(global_beta=[beta, beta], roads={}, contexts={},
            latest_completion_s=None)) for o, beta in [(0., 0.), (1000., 1.)]}
        self.assertEqual(attach_issued_means(rows, models).issued_mean.tolist(), [10., 20.])
        models['0.0']['persistence']['latest_completion_s'] = 0.
        with self.assertRaises(ValueError):
            attach_issued_means(rows, models)

    def test_zero_bias_exactly_preserves_existing_draws(self):
        zero = dict(global_bias=[0., 0.], roads={}, contexts={})
        model = dict(now_s=100., dt_s=1., horizon_s=1800, bin_s=300, keys=['True:0'],
            means={'True:0': 10.}, historical_means={'True:0': 30.},
            persistence=dict(global_beta=[.2, .8], roads={}, contexts={}), bias_calibration=zero,
            shared_factor=[[1.]], shared_fraction=.5, ratios={'True:0': [.5, 1., 1.5]})
        vehicle = dict(direction=True, travel_station_id=0)
        for mode in ('conditioned_independent', 'conditioned_shared'):
            old, new = ConditionalPersistencePath(model, mode, 0), BiasCalibratedTravelPath(model, mode, 0)
            for time, fraction in [(100., .001), (101., .5), (1000., 1.), (1800., 1.)]:
                self.assertEqual(old(vehicle, time, fraction), new(vehicle, time, fraction))
        negative = dict(zero, global_bias=[-100., -100.])
        self.assertEqual(calibrated_mean(10., negative, 'True:0', 100., 1000., 1.), 1.)


if __name__ == '__main__':
    unittest.main()
