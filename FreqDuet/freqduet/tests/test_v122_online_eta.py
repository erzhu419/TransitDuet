import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from freqduet_online_component_eta import OnlineMeans, OnlineComponentETA, model
from run_freqduet_v122_online_eta import compare_prediction


class OnlineEtaTests(unittest.TestCase):
    def geometry(self):
        return {True: dict(ids=[0, 1, 2], positions=np.array([0., 100., 200.]), index={0: 0, 1: 1, 2: 2})}

    def test_incremental_means_equal_full_index_including_same_time_ties(self):
        key = ('service', True, 1)
        rows = [(key, 10., 4.), (key, 20., 8.), (key, 20., 2.), (key, 4000., 20.)]
        offline, online = model.CompletedMeans(rows), OnlineMeans()
        index = 0
        for now in (0., 10., 11., 20., 21., 3999., 4000., 4001., 8000.):
            while index < len(rows) and rows[index][1] <= now:
                online.add(*rows[index])
                index += 1
            self.assertEqual(online.estimate(key, now), offline.estimate(key, now))

    def prefix(self):
        p = OnlineComponentETA(self.geometry())
        p.launch(1, True, 0.)
        p.arrival((1, 1, True), 20., 0.)
        p.ready((1, 1, True), 25.)
        p.departure((1, 1, True), 30., 3.)
        p.arrival((1, 2, True), 60., 3.)
        p.ready((1, 2, True), 66.)
        p.departure((1, 2, True), 68., 3.)
        p.launch(2, True, 80.)
        return p

    def evidence(self):
        return dict(ready_time_s=100., follower_gap_prediction_s=66.,
            follower_source=model.clock.JOURNEY, predicted_follower_trip_id=2,
            speed_mps=2.5, spatial_gap_m=150.)

    def test_ready_and_departure_labels_mature_at_different_events(self):
        p = OnlineComponentETA(self.geometry())
        p.launch(1, True, 0.)
        p.arrival((1, 1, True), 20., 0.)
        p.ready((1, 1, True), 25.)
        self.assertIsNone(p.history.estimate(('service', True, 1), 25.))
        self.assertEqual(p.history.estimate(('service', True, 1), 26.)['mean'], 5.)
        self.assertIsNone(p.history.estimate(('holding', True, 1), 26.))
        p.departure((1, 1, True), 30., 3.)
        self.assertIsNone(p.history.estimate(('holding', True, 1), 30.))
        self.assertEqual(p.history.estimate(('holding', True, 1), 31.)['mean'], 3.)
        self.assertEqual(p.history.estimate(('ticks', True, 1), 31.)['mean'], 2.)

    def test_live_prefix_predicts_remaining_components_without_future_paths(self):
        p, e = self.prefix(), self.evidence()
        value = p.predict(e, True, 2)
        self.assertTrue(value['component_used'])
        self.assertEqual(value['gap_s'], 56.)
        self.assertEqual(value['running_s'], 40.)
        self.assertEqual(value['service_s'], 5.)
        self.assertEqual(value['holding_s'], 3.)
        self.assertEqual(value['ticks_s'], 2.)
        self.assertEqual(value['target_service_s'], 6.)
        self.assertEqual(value['latest_completion_s'], 66.)
        p.arrival((2, 1, True), 120., 0.)
        p.ready((2, 1, True), 125.)
        self.assertEqual(p.predict(e, True, 2), value)

    def test_pending_and_exact_arrival_boundary_keep_raw_gap(self):
        p, e = self.prefix(), self.evidence()
        p.arrival((2, 1, True), 120., 0.)
        for now in (120., 121.):
            snapshot = dict(e, ready_time_s=now, speed_mps=100./(now-80.), spatial_gap_m=100.)
            value = p.predict(snapshot, True, 2)
            self.assertFalse(value['component_used'])
            self.assertEqual(value['gap_s'], 66.)

    def test_missing_support_and_nonjourney_sources_are_not_zero_predictions(self):
        p = OnlineComponentETA(self.geometry())
        p.launch(2, True, 80.)
        self.assertEqual(p.predict(self.evidence(), True, 2)['status'], 'missing_completed_component')
        e = dict(self.evidence(), follower_source='same_time_avl_route_speed_eta')
        self.assertEqual(p.predict(e, True, 2)['gap_s'], 66.)
        self.assertIsNone(p.predict(dict(e, follower_gap_prediction_s=None), True, 2)['gap_s'])

    def test_comparison_requires_ready_clock_eligibility_and_completed_support(self):
        e = self.evidence()
        live = self.prefix().predict(e, True, 2)
        compare_prediction(live, live, e)
        with self.assertRaises(AssertionError):
            compare_prediction(dict(live, prediction_time_s=101.), live, e)
        with self.assertRaises(AssertionError):
            compare_prediction(dict(live, component_used=False), live, e)
        with self.assertRaises(AssertionError):
            compare_prediction(dict(live, latest_completion_s=100.), live, e)


if __name__ == '__main__':
    unittest.main()
