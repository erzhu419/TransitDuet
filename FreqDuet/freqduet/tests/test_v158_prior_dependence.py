from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v158_prior_dependence as audit


class PriorDependenceTests(unittest.TestCase):
    def test_between_and_within_day_variance_are_distinct(self):
        value = audit.day_moments([dict(mean=10., second=104.), dict(mean=30., second=916.)])
        self.assertEqual(value['mean'], 20.)
        self.assertEqual(value['between_day_variance_s2'], 100.)
        self.assertEqual(value['within_day_variance_s2'], 10.)

    def test_raw_cross_terms_and_error_sum(self):
        errors = dict(running=3., service=-2., holding=4., execution_wait=-1.)
        record = audit.error_record([1], errors, 4.)
        self.assertEqual(record['m_running__service'], -6.)
        mu, second = audit.matrices(record, audit.IDENTIFIED)
        np.testing.assert_array_equal(mu, [3., -2., 4., -1.])
        self.assertEqual(second.sum(), record['full_mse_s2'])
        with self.assertRaises(AssertionError):
            audit.error_record([1], errors, 5.)

    def test_joint_residence_is_not_pure_holding(self):
        predicted, actual = audit.audit.parts.zeros(), audit.audit.parts.zeros()
        predicted.update(current_ready_joint=10., holding=4.)
        actual.update(holding=8., execution_wait=2.)
        views = audit.error_views(predicted, actual, (1, 2, True))
        self.assertEqual(views, dict(coarse=dict(running=0., service=0., residence=4.)))
        predicted['current_ready_joint'] = 0.
        self.assertIn('holding_identified', audit.error_views(predicted, actual, None))

    def test_cell_day_actor_and_interaction_covariance_close(self):
        actors, days = np.array([-1., 1.]), np.array([-2., 0., 2.])
        means = np.empty((2, 3, 3))
        within = np.array([[4., -2., 1.], [-2., 9., 0.], [1., 0., 1.]])
        for i, a in enumerate(actors):
            for j, d in enumerate(days):
                means[i, j] = [10.+a+d+a*d, -3.+2*d, 2.-a]
        seconds = np.einsum('sdi,sdj->sdij', means, means)+within
        result = audit.covariance_report(means, seconds)
        np.testing.assert_allclose(result['within_cell_covariance_s2'], within)
        expected_day = np.array([[8./3, 16./3, 0.], [16./3, 32./3, 0.], [0., 0., 0.]])
        np.testing.assert_allclose(result['day_mean_covariance_s2'], expected_day)
        self.assertAlmostEqual(result['full_mse_s2'], result['full_variance_s2']+9.**2)

    def test_constant_component_has_undefined_correlation(self):
        means = np.zeros((2, 2, 2))
        raw = np.zeros((2, 2, 2, 2))
        raw[:, :, 0, 0] = 4.
        result = audit.covariance_report(means, raw)
        self.assertIsNone(result['correlation'][0][1])

    def test_missing_crossed_support_is_not_a_zero_effect(self):
        cells = [dict(seed=1, day=1, modes=dict(old=dict(groups={'coarse/all': {}}))),
            dict(seed=1, day=2, modes=dict(old=dict(groups={}))) ]
        result = audit.dependence(cells, 'old', 'coarse/all', audit.COARSE)
        self.assertEqual(result['decision'], 'inconclusive_crossed_support')
        self.assertEqual(result['contributing_cells'], 1)
        self.assertNotIn('correlation_ci95', result)

    def test_complete_crossed_moments_use_shared_bootstrap(self):
        cells = []
        for actor in (1, 2):
            for day in (1, 2, 3):
                errors = dict(running=float(day), service=float(2*day), residence=float(actor))
                record = audit.error_record([1], errors, sum(errors.values()))
                value = audit.audit.stats.describe([record])
                cells.append(dict(seed=actor, day=day,
                    modes=dict(old=dict(groups={'coarse/all': value}))))
        result = audit.dependence(cells, 'old', 'coarse/all', audit.COARSE)
        self.assertAlmostEqual(result['correlation'][0][1], 1.)
        self.assertIn('running__service', result['covariance_ci95'])
        self.assertGreater(result['correlation_ci95']['running__service']['defined_draws'], 0)

    def test_future_clock_is_a_label_not_the_prior_lookup(self):
        prefix = audit.audit.frozen.prior.RunningPrefix.__new__(audit.audit.frozen.prior.RunningPrefix)
        prefix.rows = {(True, 2): [(10., 0., 10.)]}
        prefix.times = {(True, 2): [10.]}
        law = audit.launch.RecordedRunning([dict(direction=True, station=2, hour=0,
            days=[dict(day=3, mean=40., second=1600.)])], 5, prefix)
        law.estimate(True, 2, 20., 3500., dict(mean=10.))
        path = dict(dispatch={42: dict(actual_s=3600.)}, departures={},
            arrivals={(42, 2, True): dict(time_s=3700.)})
        row = dict(prediction_time_s=20., evidence=dict(predicted_follower_trip_id=42, predicted_follower_bus_id=5))
        result = audit.prior_records(path, row, law, {True: dict(ids=[1, 2], index={1: 0, 2: 1})}, {})
        self.assertEqual(result['prior_leg_bias_s'], -60.)
        self.assertEqual(result['frozen_leg_bias_s'], -75.)
        self.assertEqual(result['entry_hour_mismatch_fraction'], 1.)
        self.assertEqual(result['latest_completion_age_s'], 10.)


if __name__ == '__main__':
    unittest.main()
