from types import SimpleNamespace as NS
import unittest

import freqduet_v216_observation as v


def environment():
    buses = [NS(bus_id=i, trip_id=i+20, direction=False, on_route=True, state=NS(name='HOLDING'),
        last_station=NS(station_id=3), next_station=NS(station_id=2), effective_route=[NS(distance=1000.)],
        travel_distance=100.+i) for i in range(12)]
    plan = NS(launch_turn=40, direction=True, launched=False, launch_time=7400., target_headway=360.)
    plan._freqduet_scheduled_launch = 7450.
    return NS(bus_all=buses, timetables=[plan], current_time=7500.)


def observation():
    env, clock = environment(), v.EventClock()
    clock.stamp('prepare', 7499., bus_id=0, trip_id=20, station_id=3, direction=False)
    stamp = clock.stamp('lower_query', 7500., bus_id=0, trip_id=20, station_id=3, direction=False)
    q = dict(v.snapshot(env, stamp), bus_id=0, trip_id=20, station_id=3, direction=False,
        state_captured_s=7499., prepare_event_seq=0, terminal_masked=False, forward_valid=True,
        follower_valid=True, base_bin=3, feasible_bins=list(range(7)), executed_holding_before_s=12.)
    return q, clock


class ObservationTests(unittest.TestCase):
    def test_snapshot_reads_current_executable_queue_and_no_active_return_label(self):
        env = environment()
        frame = v.snapshot(env, dict(time_s=7500., event_seq=3))
        self.assertEqual(frame['plans'][0]['planned_s'], 7450.)
        self.assertEqual(frame['plans'][0]['public_s'], 7400.)
        self.assertEqual(frame['active'][0]['remaining_distance_m'], 900.)
        self.assertNotIn('return_s', frame['active'][0])
        env.bus_all[0].on_route = False
        env.bus_all[0].back_to_terminal_time = 7499.
        frame = v.snapshot(env, dict(time_s=7500., event_seq=4))
        self.assertEqual(frame['ready'][0]['last_observed_return_s'], 7499.)
        env.timetables[0].launched = True
        self.assertEqual(v.plans(env), {})

    def test_one_global_order_distinguishes_same_clock_native_events(self):
        c = v.EventClock()
        for kind in ('prepare', 'lower_query', 'return', 'launch', 'fresh_upper', 'plan_update'):
            c.stamp(kind, 7500.)
        self.assertEqual([e['event_seq'] for e in c.events], list(range(6)))
        with self.assertRaises(AssertionError):
            c.stamp('lower_query', 7499.)

    def test_native_prepare_identity_clock_and_live_snapshot_required(self):
        q, clock = observation()
        self.assertTrue(v.validate_capture([q], [], clock)['prepare_queries_linked'])
        for field, bad in (('state_captured_s', 7501.), ('snapshot_event_seq', 0), ('trip_id', 99)):
            changed = dict(q, **{field: bad})
            with self.assertRaises(AssertionError):
                v.validate_capture([changed], [], clock)
        changed = dict(q)
        del changed['plans']
        with self.assertRaises(AssertionError):
            v.validate_capture([changed], [], clock)

    def label_data(self, first_trip=40):
        anchor = dict(bus_id=0, trip_id=20, station_id=3, inbound_direction=False,
                      target_direction=True, due_trip_ids=[40], base_bin=3)
        returns = [dict(bus_id=0, trip_id=20, return_s=8000.)]
        launches = [dict(trip=first_trip, bus=0, direction=True, actual_s=8000.),
                    dict(trip=40, bus=0, direction=True, actual_s=9000.)]
        events = [dict(event_seq=0, kind='launch', bus_id=0, trip_id=20, time_s=7000.),
                  dict(event_seq=1, kind='return', bus_id=0, trip_id=20, time_s=8000.),
                  dict(event_seq=2, kind='launch', bus_id=0, trip_id=first_trip, time_s=8000.),
                  dict(event_seq=3, kind='launch', bus_id=0, trip_id=40, time_s=9000.)]
        departure = [dict(trip_id=20, station_id=3, direction=False, realized_holding=27.)]
        return anchor, returns, launches, events, departure, dict(executed_holding_before_s=12.)

    def test_supply_label_preserves_same_vehicle_trip_and_same_clock_order(self):
        result = v.supply_label(*self.label_data())
        self.assertTrue(result['same_vehicle_due_supply'])
        self.assertTrue(result['anchor_holding_execution_matches'])
        self.assertEqual(result['executed_at_anchor_s'], 15.)
        self.assertLess(result['return_event_seq'], result['next_launch_event_seq'])

    def test_supply_label_does_not_skip_unmatched_first_launch_to_find_a_winner(self):
        result = v.supply_label(*self.label_data(first_trip=99))
        self.assertFalse(result['same_vehicle_due_supply'])
        self.assertEqual(result['next_same_vehicle_launch']['trip'], 99)

    def test_missing_return_or_launch_is_retained_not_replaced_or_zero_filled(self):
        args = list(self.label_data())
        args[1] = []
        self.assertFalse(v.supply_label(*args)['returned'])
        args = list(self.label_data())
        args[3] = args[3][:2]
        result = v.supply_label(*args)
        self.assertIsNone(result['next_same_vehicle_launch'])
        self.assertFalse(result['same_vehicle_due_supply'])
        self.assertEqual(v.supply_label(None, [], [], [], [], None), dict(anchor_observed=False))


if __name__ == '__main__':
    unittest.main()
