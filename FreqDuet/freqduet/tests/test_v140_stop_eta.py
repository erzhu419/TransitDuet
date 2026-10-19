import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import freqduet_stop_phase_eta as run


class StopEtaTests(unittest.TestCase):
    def route(self):
        return dict(ids=[0, 1, 2], positions=np.array([0., 100., 200.]), index={0: 0, 1: 1, 2: 2})

    def history(self, future=999.):
        values = [('running', 2, 40.), ('service', 2, 12.), ('holding', 1, 5.), ('ticks', 1, 2.)]
        return run.model.CompletedMeans([((p, True, s), t, v) for p, s, old in values
            for t, v in ((5., old), (100., future))])

    def test_conditional_residual_excludes_future_and_simultaneous(self):
        samples = [(True, 1, 5., 10.), (True, 1, 6., 30.), (True, 1, 100., 999.)]
        r = run.ServiceResiduals(samples).estimate(True, 1, 100., 20.)
        self.assertEqual(r['mean'], 10.)
        self.assertEqual(r['count'], 1)
        self.assertIsNone(run.ServiceResiduals(samples).estimate(True, 1, 100., 40.))

    def test_equal_age_is_supported_before_simultaneous_ready(self):
        s = run.ServiceResiduals([(True, 1, 5., 20.)])
        self.assertEqual(s.estimate(True, 1, 100., 20.)['mean'], 0.)

    def test_window_and_explicit_all_past_fallback_match_original(self):
        s = run.ServiceResiduals([(True, 1, 1., 10.), (True, 1, 3602., 30.)])
        self.assertEqual(s.estimate(True, 1, 3603., 0.)['mean'], 30.)
        self.assertTrue(s.estimate(True, 1, 8000., 0.)['historical_fallback'])

    def test_service_tail_and_future_command_do_not_leak(self):
        s = run.ServiceResiduals([(True, 1, 5., 30.)])
        a = run.from_stop(self.history(), s, self.route(), True, 1, 2, 100., 80.,
            ready_s=1000., command=dict(policy_time_s=1001., chosen_s=45.))
        b = run.from_stop(self.history(1.), s, self.route(), True, 1, 2, 100., 80.,
            ready_s=2000., command=dict(policy_time_s=2001., chosen_s=0.))
        self.assertEqual(a, b)
        self.assertEqual(a['gap_s'], 10.+5.+2.+40.+12.)

    def test_known_positive_and_zero_commands_use_execution_clock(self):
        s = run.ServiceResiduals([])
        a = run.from_stop(self.history(), s, self.route(), True, 1, 2, 100., 80.,
            90., dict(policy_time_s=91., chosen_s=20.))
        self.assertEqual(a['current_remaining_s'], 11.)
        b = run.from_stop(self.history(), s, self.route(), True, 1, 2, 92., 80.,
            90., dict(policy_time_s=91., chosen_s=0.))
        self.assertEqual(b['current_remaining_s'], 0.)

    def test_simultaneous_command_is_not_observed(self):
        r = run.from_stop(self.history(), run.ServiceResiduals([]), self.route(), True,
            1, 2, 100., 80., 90., dict(policy_time_s=100., chosen_s=45.))
        self.assertFalse(r['applied'])

    def test_no_survivor_is_not_zero_imputation(self):
        r = run.from_stop(self.history(), run.ServiceResiduals([(True, 1, 5., 10.)]),
            self.route(), True, 1, 2, 100., 80.)
        self.assertEqual(r['status'], 'no_completed_service_survivor')
        self.assertNotIn('gap_s', r)

    def test_target_stop_forecasts_ready_not_post_ready_holding(self):
        r = run.from_stop(self.history(), run.ServiceResiduals([(True, 1, 5., 30.)]),
            self.route(), True, 1, 1, 100., 80.)
        self.assertEqual(r['gap_s'], 10.)


if __name__ == '__main__':
    unittest.main()
