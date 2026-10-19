import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v120_component_eta as run


class ComponentEtaTests(unittest.TestCase):
    def route(self):
        return dict(ids=[0, 1, 2], positions=np.array([0., 100., 200.]), index={0: 0, 1: 1, 2: 2})

    def test_completed_mean_strictly_excludes_simultaneous_and_future(self):
        key = ('running', True, 1)
        h = run.CompletedMeans([(key, 10., 20.), (key, 30., 40.), (key, 40., 1000.)])
        self.assertEqual(h.estimate(key, 30.)['mean'], 20.)
        self.assertEqual(h.estimate(key, 31.)['mean'], 30.)
        self.assertIsNone(h.estimate(key, 10.))
        self.assertTrue(h.estimate(key, 4000.)['historical_fallback'])

    def test_recent_window_is_fixed_and_old_labels_are_explicit_fallback(self):
        key = ('service', False, 2)
        h = run.CompletedMeans([(key, 1., 10.), (key, 3602., 30.)])
        self.assertEqual(h.estimate(key, 3603.)['mean'], 30.)
        self.assertFalse(h.estimate(key, 3603.)['historical_fallback'])
        self.assertEqual(h.estimate(key, 8000.)['mean'], 20.)
        self.assertTrue(h.estimate(key, 8000.)['historical_fallback'])

    def test_causal_motion_requires_past_departure_and_position_consistency(self):
        p = run.MovingPrefix([dict(time_s=20., station_id=1)],
            [dict(time_s=0., station_id=0), dict(time_s=30., station_id=1)])
        self.assertEqual(p.origin(10., 50., self.route()), 0)
        self.assertIsNone(p.origin(25., 100., self.route()))
        self.assertIsNone(p.origin(30., 100., self.route()))
        self.assertEqual(p.origin(31., 110., self.route()), 1)
        self.assertIsNone(p.origin(20., 100., self.route()))

    def sample_history(self, future_value=999.):
        values = [('running', 1, 20.), ('running', 2, 40.), ('service', 1, 10.),
            ('holding', 1, 5.), ('ticks', 1, 2.), ('service', 2, 12.)]
        return run.CompletedMeans([((p, True, s), end, value) for p, s, old in values
            for end, value in ((5., old), (100., future_value))])

    def test_remaining_running_service_holding_and_target_ready_are_separate(self):
        pred = run.component_prediction(self.sample_history(), self.route(), True, 0, 2, 50., 50.)
        self.assertEqual(pred['running_s'], 50.)
        self.assertEqual(pred['service_s'], 10.)
        self.assertEqual(pred['holding_s'], 5.)
        self.assertEqual(pred['ticks_s'], 2.)
        self.assertEqual(pred['target_service_s'], 12.)
        changed = run.component_prediction(self.sample_history(1.), self.route(), True, 0, 2, 50., 50.)
        self.assertEqual(pred, changed)

    def test_missing_component_is_not_imputed_zero(self):
        h = run.CompletedMeans([(('running', True, 1), 5., 20.)])
        self.assertIsNone(run.component_prediction(h, self.route(), True, 0, 2, 50., 50.))


if __name__ == '__main__':
    unittest.main()
