import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v139_fallback_phase import FollowerTimeline


def timeline():
    arrival = dict(trip_id=1, station_id=2, direction=True, time_s=10.)
    departure = dict(arrival, time_s=20.)
    ready = {(1, 2, True): dict(ready_time_s=15.)}
    route = dict(ids=[1, 2, 3], positions=[0., 100., 200.], index={1: 0, 2: 1, 3: 2})
    return FollowerTimeline([arrival], [departure], ready, 0., route)


class FallbackPhaseTests(unittest.TestCase):
    def test_moving_prefix_and_stop_phases_are_distinct(self):
        t = timeline()
        self.assertEqual(t.inspect(5., 50.)['reason'], 'moving_segment')
        self.assertEqual(t.inspect(12., 100.)['strict_phase'], 'mandatory_service')
        self.assertEqual(t.inspect(17., 100.)['strict_phase'], 'ready_stop')
        self.assertEqual(t.inspect(17., 100.)['reason'], 'pending_stop')

    def test_same_second_arrival_is_not_in_strict_prefix(self):
        r = timeline().inspect(10., 100.)
        self.assertEqual(r['completed_arrivals'], 0)
        self.assertEqual(r['reason'], 'progress_at_or_beyond_next_stop')
        self.assertIn('arrival', r['same_clock_events'])

    def test_same_second_departure_stays_blocked(self):
        t = timeline()
        self.assertEqual(t.inspect(20., 100.)['reason'], 'pending_stop')
        self.assertIn('departure', t.inspect(20., 100.)['same_clock_events'])
        self.assertEqual(t.inspect(21., 110.)['reason'], 'moving_segment')

    def test_ready_boundary_does_not_claim_elapsed_holding(self):
        r = timeline().inspect(15., 100.)
        self.assertEqual(r['strict_phase'], 'ready_boundary')
        self.assertEqual(r['same_clock_events'], ['ready'])

    def test_position_mismatch_is_not_called_pending_service(self):
        r = timeline().inspect(21., 99.)
        self.assertEqual(r['reason'], 'progress_before_completed_segment')
        self.assertEqual(r['strict_phase'], 'in_transit')

    def test_launch_boundary_has_no_completed_departure(self):
        r = timeline().inspect(0., 0.)
        self.assertEqual(r['reason'], 'no_completed_departure')
        self.assertEqual(r['completed_departures'], 0)


if __name__ == '__main__':
    unittest.main()
