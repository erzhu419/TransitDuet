import unittest

import numpy as np
import pandas as pd

from freqduet_conditional_persistence import (ConditionalPersistencePath, conditional_beta,
    conditional_mean, fit_conditional_persistence, pooled_fit, time_context)
from freqduet_conditioned_travel import ConditionedTravelPath


class ConditionalPersistenceTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(road=road, origin=0., start_s=t, end_s=t+10.,
            current=10., historical=20., actual=y)
            for road, y in [('True:0', 10.), ('True:1', 20.)] for t in range(100, 1500, 50)])

    def test_strict_completion_prefix_and_future_issuance(self):
        rows = self.rows()
        rows.loc[len(rows)] = dict(road='True:0', origin=2000., start_s=2100., end_s=2200., current=10., historical=20., actual=999.)
        fit = fit_conditional_persistence(rows, 2200.)
        rows.iloc[-1, rows.columns.get_loc('actual')] = -999.
        self.assertEqual(fit, fit_conditional_persistence(rows, 2200.))
        self.assertLess(fit['latest_completion_s'], 2200.)
        rows.loc[0, 'origin'] = rows.loc[0, 'start_s']
        with self.assertRaises(ValueError):
            fit_conditional_persistence(rows, 2200.)

    def test_road_effects_are_distinct_and_bounded(self):
        fit = fit_conditional_persistence(self.rows(), 2000.)
        first = conditional_mean(fit, 'True:0', 10., 20., 0., 1400.)
        second = conditional_mean(fit, 'True:1', 10., 20., 0., 1400.)
        self.assertTrue(10. <= first < second <= 20.)

    def test_sparse_group_is_shrunk_to_parent(self):
        rows = self.rows().loc[lambda f: f.road == 'True:1'].iloc[:1]
        beta = pooled_fit(rows, [0., 0.])
        self.assertGreater(beta[0], 0.)
        self.assertLess(beta[0], .2)

    def test_unobserved_horizon_retains_its_parent(self):
        rows = self.rows().iloc[:1]
        self.assertEqual(pooled_fit(rows, [.2, .75])[1], .75)

    def test_only_calendar_clock_determines_time_context(self):
        self.assertEqual(time_context(6300., 7199.), '3:0')
        self.assertEqual(time_context(6300., 7200.), '3:1')
        self.assertEqual(time_context(7200., 8100.), '0:0')
        model = dict(global_beta=[.1, .2], roads={'True:0': dict(beta=[.3, .4])},
            contexts={'True:0/3:1': dict(beta=[.8, .9])})
        self.assertEqual(conditional_beta(model, 'True:0', 6300., 7200.), [.8, .9])
        self.assertEqual(conditional_beta(model, 'True:0', 6300., 7199.), [.3, .4])

    def test_zero_correction_preserves_base_sampler(self):
        model = dict(now_s=100., dt_s=1., horizon_s=1800, bin_s=300, keys=['True:0'],
            means={'True:0': 10.}, historical_means={'True:0': 30.},
            persistence=dict(global_beta=[0., 0.], roads={}, contexts={}),
            shared_factor=[[1.]], shared_fraction=.5, ratios={'True:0': [.5, 1., 1.5]})
        vehicle = dict(direction=True, travel_station_id=0)
        for mode in ('conditioned_independent', 'conditioned_shared'):
            old, new = ConditionedTravelPath(model, mode, 0), ConditionalPersistencePath(model, mode, 0)
            for time, fraction in [(100., .001), (101., .5), (1000., 1.), (1800., 1.)]:
                self.assertEqual(old(vehicle, time, fraction), new(vehicle, time, fraction))


if __name__ == '__main__':
    unittest.main()
