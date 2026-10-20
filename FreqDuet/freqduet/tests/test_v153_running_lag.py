import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v153_running_lag as audit


class RunningLagTests(unittest.TestCase):
    def test_prefix_excludes_same_clock_and_counts_old_entry(self):
        p = audit.RunningPrefix.__new__(audit.RunningPrefix)
        p.rows = {(True, 2): [(7201., 7190., 11.), (7211., 7205., 6.), (7220., 7215., 5.)]}
        p.times = {(True, 2): [7201., 7211., 7220.]}
        r = p.selected(True, 2, 7220.)
        self.assertEqual(r['count'], 2)
        self.assertEqual(r['before_hour_fraction'], .5)
        self.assertEqual(r['current_hour_mean_s'], 6.)

    def test_running_average_removes_only_observed_stop_time(self):
        path = dict(dispatch={1: dict(actual_s=0.)},
            arrivals={(1, 2, True): dict(time_s=10.), (1, 3, True): dict(time_s=100.)},
            departures={(1, 2, True): dict(time_s=30.)})
        row = dict(prediction_time_s=40., station_id=3, direction=True,
            evidence=dict(predicted_follower_trip_id=1, predicted_follower_bus_id=8,
                spatial_gap_m=50., follower_source='journey', speed_mps=3.75))
        route = dict(ids=[1, 2, 3], positions=[0., 100., 200.], index={1: 0, 2: 1, 3: 2})
        r = audit.observed_motion(path, row, route)
        self.assertEqual(r['observed_running_s'], 20.)
        self.assertEqual(r['running_average_mps'], 7.5)
        self.assertEqual(r['current_segment_average_mps'], 5.)
        path['arrivals'][1, 3, True]['time_s'] = 200.
        self.assertEqual(audit.observed_motion(path, row, route), r)

    def test_launch_does_not_fabricate_observed_running_speed(self):
        route = dict(ids=[1, 2], positions=[0., 100.], index={1: 0, 2: 1})
        path = dict(dispatch={1: dict(actual_s=50.)}, arrivals={}, departures={})
        row = dict(prediction_time_s=50., station_id=2, direction=True,
            evidence=dict(predicted_follower_trip_id=1, predicted_follower_bus_id=8,
                spatial_gap_m=100., follower_source='route_speed_eta', speed_mps=2.))
        r = audit.observed_motion(path, row, route)
        self.assertIsNone(r['running_average_mps'])
        self.assertIsNone(r['current_segment_average_mps'])

    def test_segment_budget_keeps_query_episode_weights(self):
        rows = [dict(episode=[1], segments={'a': -4.}), dict(episode=[1], segments={'b': -2.}),
            dict(episode=[2], segments={'a': 3.})]
        result = audit.budget(rows, ['a', 'b'])
        self.assertEqual(result['a']['metrics']['bias_contribution_s'], .5)
        self.assertEqual(result['b']['metrics']['bias_contribution_s'], -.5)


if __name__ == '__main__':
    unittest.main()
