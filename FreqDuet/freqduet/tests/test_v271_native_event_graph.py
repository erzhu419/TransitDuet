from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import freqduet_causal_event_graph as graph
from freqduet_native_event_graph import CompletedPrefix, native_routes, snapshot


def fake_env():
    stations = [NS(station_id=i, station_type=0 if i in (0, 3) else 1) for i in (0, 1, 2, 3, 2, 1, 0)]
    buses = []
    for i in range(12):
        active = i < 2
        buses.append(NS(bus_id=i, trip_id=i if active else -1, on_route=active, direction=True,
            state=NS(name='WAITING_ACTION' if active else 'TRAVEL'), passengers=[None]*10 if active else [],
            capacity=50., travel_distance=500., current_speed=0., launch_time=100., back_to_terminal_time=-1.,
            dwelling_time=0., holding_time=0., last_station=stations[1], next_station=stations[2],
            last_board_count=0, last_board_time=150., next_station_dis=500., current_route=NS(distance=500.),
            episode_hold_vehicle_seconds=20.))
    return NS(current_time=200., max_agent_num=12, bus_all=buses, timetables=[], stations=stations,
              routes=[NS(distance=500.)]*6, protocol=NS(evaluation_end_time_s=5000.))


class NativeBridgeTests(unittest.TestCase):
    def test_current_stop_and_pending_command_are_not_future_variables(self):
        env = fake_env()
        prefix = CompletedPrefix({i: dict(time_s=100., holding_counter_at_launch=0.) for i in (0, 1)})
        for bus in env.bus_all[:2]:
            prefix.arrival(bus, 150., 1, True, 10)
        pending = NS(time_s=200., bus_id=1, trip_id=1, station_id=1, action_s=10.)
        mask = {(1, 1, 1): (0, 10)}
        value, locks = snapshot(env, prefix, native_routes(env), 0, (0, 5, 10), 5, [pending], mask)
        own, prior = value.vehicles[:2]
        self.assertEqual((own.index, own.phase, own.legal), (0, 'prepare', (0, 5, 10)))
        self.assertEqual((prior.phase, prior.remaining_s, locks[1, 1, 1]), ('hold', 11., 210.))
        self.assertEqual(env.bus_all[1].state.name, 'WAITING_ACTION')
        self.assertEqual(env.bus_all[1].dwelling_time, 0.)
        self.assertEqual(own.executed_h_s, 20.)
        self.assertEqual(len(value.vehicles), 12)

    def test_zero_pending_uses_native_transition_tick(self):
        env = fake_env()
        prefix = CompletedPrefix({i: dict(time_s=100., holding_counter_at_launch=0.) for i in (0, 1)})
        for bus in env.bus_all[:2]:
            prefix.arrival(bus, 150., 1, True, 10)
        pending = NS(time_s=200., bus_id=1, trip_id=1, station_id=1, action_s=0.)
        value, locks = snapshot(env, prefix, native_routes(env), 0, (0, 5), 0, [pending], {})
        self.assertEqual((value.vehicles[1].remaining_s, locks[1, 1, 1]), (2., 201.))

    def test_travel_uses_next_stop_and_observed_remaining_distance(self):
        env = fake_env()
        prefix = CompletedPrefix({i: dict(time_s=100., holding_counter_at_launch=0.) for i in (0, 1)})
        prefix.arrival(env.bus_all[0], 150., 1, True, 10)
        env.bus_all[1].state.name = 'TRAVEL'
        env.bus_all[1].next_station_dis = 125.
        value, _ = snapshot(env, prefix, native_routes(env), 0, (0, 5), 0, [], {})
        self.assertEqual((value.vehicles[1].index, value.vehicles[1].remaining_fraction), (1, .25))
        self.assertIsNone(value.vehicles[1].arrival_s)

    def test_prefix_requires_both_clocks_strictly_completed(self):
        prefix = CompletedPrefix({i: dict(time_s=100.) for i in (0, 1)})
        bus = NS(bus_id=0, trip_id=0, last_board_count=3, passengers=[None]*8, capacity=50.)
        prefix.arrival(bus, 150., 1, True, 10)
        prefix.complete(0, 0, 1, True, 160.)
        self.assertEqual(prefix.first_visits, 1)
        bus.bus_id, bus.trip_id = 1, 1
        prefix.arrival(bus, 180., 1, True, 10)
        prefix.complete(1, 1, 1, True, 190.)
        self.assertEqual(len(prefix.strict(190.)), 0)
        self.assertEqual(len(prefix.strict(191.)), 1)
        self.assertEqual((prefix.visits[0].incoming_load, prefix.visits[0].boarded, prefix.visits[0].alighted), (10., 3., 5.))
        prefix.visits[0] = replace(prefix.visits[0], available_s=195.)
        self.assertEqual(len(prefix.strict(195.)), 0)
        self.assertEqual(len(prefix.strict(196.)), 1)

    def test_completion_and_terminal_recorder_do_not_duplicate_visit(self):
        prefix = CompletedPrefix({i: dict(time_s=100.) for i in (0, 1)})
        for i, arrival in enumerate((150., 180.)):
            bus = NS(bus_id=i, trip_id=i, last_board_count=0, passengers=[], capacity=50.)
            prefix.arrival(bus, arrival, 3, True, 10)
            prefix.complete(i, i, 3, True, arrival)
            prefix.complete(i, i, 3, True, arrival)
        self.assertEqual(len(prefix.finished), 2)
        self.assertEqual(len(prefix.visits), 1)

    def test_same_tick_arrival_gap_is_preserved_not_imputed(self):
        prefix = CompletedPrefix({i: dict(time_s=100.) for i in (0, 1)})
        for bid in (0, 1):
            bus = NS(bus_id=bid, trip_id=bid, last_board_count=0, passengers=[], capacity=50.)
            prefix.arrival(bus, 150., 1, True, 0)
            prefix.complete(bid, bid, 1, True, 160.)
        self.assertEqual(prefix.visits[0].gap_s, 0.)
        self.assertEqual(prefix.visits[0].available_s, 150.)


if __name__ == '__main__':
    unittest.main()
