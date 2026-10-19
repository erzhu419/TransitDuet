import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v141_ready_eta as run


class ReadyEtaTests(unittest.TestCase):
    def route(self):
        return dict(ids=[0, 1, 2], positions=np.array([0., 100., 200.]), index={0: 0, 1: 1, 2: 2})

    def history(self, future):
        return run.model.CompletedMeans([((channel, True, 2), t, v)
            for channel, past in (('running', 40.), ('service', 12.))
            for t, v in ((5., past), (100., future))])

    def test_unknown_command_uses_only_past_ready_residence(self):
        d = run.ServiceResiduals([(True, 1, 5., 2.), (True, 1, 6., 11.)])
        r = run.predict_ready(self.history(999.), d, self.route(), True, 1, 2, 100., 99.)
        self.assertEqual(r['gap_s'], 5.5+40.+12.)
        self.assertLess(r['latest_label_s'], 100.)

    def test_simultaneous_and_future_labels_cannot_change_prediction(self):
        predictions = []
        for value in (0., 999.):
            d = run.ServiceResiduals([(True, 1, 5., 11.), (True, 1, 100., value), (True, 1, 101., value)])
            predictions.append(run.predict_ready(self.history(value), d, self.route(), True, 1, 2, 100., 99.))
        self.assertEqual(predictions[0], predictions[1])

    def test_ready_at_query_time_is_not_observed(self):
        r = run.predict_ready(self.history(999.), run.ServiceResiduals([]), self.route(), True, 1, 2, 100., 100.)
        self.assertFalse(r['applied'])

    def test_no_survivor_is_not_zero_wait(self):
        r = run.predict_ready(self.history(999.), run.ServiceResiduals([(True, 1, 5., 2.)]),
            self.route(), True, 1, 2, 100., 90.)
        self.assertEqual(r['status'], 'no_completed_ready_residence_survivor')
        self.assertNotIn('gap_s', r)

    def test_completed_target_ready_is_not_forecast_as_future_departure(self):
        r = run.predict_ready(self.history(999.), run.ServiceResiduals([]), self.route(), True, 1, 1, 100., 99.)
        self.assertFalse(r['applied'])

    def test_interpolated_small_sample_interval_can_have_zero_self_containment(self):
        r = run.interval_row(np.array([0., 10.]), 1., 9., 5., [1], 10.)
        self.assertEqual(r['empirical_containment'], 0.)
        self.assertTrue(r['covered'])
        self.assertFalse(r['outside_empirical'])

    def test_tail_and_duplicate_episode_diagnostics_are_distinct(self):
        a = run.interval_row(np.array([0., 1., 3.]), .2, 2.6, 1., [1], 10.)
        b = run.interval_row(np.array([0., 1., 3.]), .2, 2.6, 4., [2], 10.)
        self.assertTrue(b['above'])
        self.assertTrue(b['outside_empirical'])
        r = run.interval_metrics([a, a, a, b])
        self.assertEqual(r['coverage'], .75)
        self.assertEqual(r['episode_weighted_coverage'], .5)
        self.assertEqual(r['earliest_query_coverage'], .5)

    def test_missing_interval_is_not_zero_coverage(self):
        self.assertIsNone(run.interval_metrics([])['coverage'])


if __name__ == '__main__':
    unittest.main()
