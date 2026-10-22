from types import SimpleNamespace as NS
import unittest

import run_freqduet_v214_inbound_smoke as v


def bus(index, state='TRAVEL', *, direction=1, on_route=True):
    stops = [NS(station_id=i, station_type=0 if i in (0, 5) else 1) for i in range(6)]
    return NS(bus_id=2, trip_id=4, direction=direction, on_route=on_route, effective_station=stops,
        last_station=stops[index], next_station=stops[min(5, index+1)], state=NS(name=state),
        effective_route=[NS(distance=500.) for _ in range(5)], travel_distance=500.*index)


class InboundSmokeTests(unittest.TestCase):
    def test_public_position_and_modes_count_only_remaining_controlled_stops(self):
        self.assertTrue(v.controlled_stop_ahead(bus(0)))
        self.assertTrue(v.controlled_stop_ahead(bus(3, 'HOLDING')))
        self.assertTrue(v.controlled_stop_ahead(bus(3, 'WAITING_ACTION')))
        self.assertTrue(v.controlled_stop_ahead(bus(3, 'DWELLING')))
        self.assertTrue(v.controlled_stop_ahead(bus(4, 'HOLDING')))
        self.assertFalse(v.controlled_stop_ahead(bus(4, 'TRAVEL')))
        self.assertFalse(v.controlled_stop_ahead(bus(5, 'HOLDING')))

    def test_frame_keeps_query_position_clock_and_not_future_return(self):
        b = bus(2)
        b.back_to_terminal_time = 999999.
        trip = NS(launch_turn=8, direction=0, launch_time=7200., launched=False)
        env = NS(current_time=7300., bus_all=[b, bus(0, on_route=False, direction=1)], timetables=[trip])
        f = v.supply_frame(env, trip)
        self.assertEqual(f['active'][0]['remaining_distance_m'], 1500.)
        self.assertEqual(f['active'][0]['position_sample_s'], 7300.)
        self.assertNotIn('back_to_terminal_time', f['active'][0])
        self.assertEqual(len(v.protocol.choose_supply(f)), 1)
        env.bus_all[1].direction = 0
        self.assertEqual(v.protocol.choose_supply(v.supply_frame(env, trip)), [])

    def test_candidate_map_reuses_correct_upper_only_recorder(self):
        for cid, (shift, _) in enumerate(v.protocol.CANDIDATES):
            old = 5*(cid//2)
            self.assertEqual(v.prior.UPPER[v.prior.CANDIDATES[old][0]], shift)
        self.assertEqual(v.protocol.CANDIDATES[v.SMOKE_CANDIDATE], (-30, 1))
        self.assertNotIn(v.SMOKE_FUTURE, [v.protocol.future_seed(0, f) for f in range(32)])

    def test_upper_target_change_alone_is_not_a_plan_clock_change(self):
        base = dict(changes=[dict(trip_id=4, before=dict(planned_s=100.), after=dict(planned_s=100., target_s=360.))])
        joint = dict(changes=[])
        self.assertEqual(v.changed_plan_clocks(base, joint), [])
        joint['changes'] = [dict(trip_id=4, before=dict(planned_s=100.), after=dict(planned_s=95.))]
        self.assertEqual(v.changed_plan_clocks(base, joint)[0]['delta_s'], -5.)


if __name__ == '__main__':
    unittest.main()
