from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import run_freqduet_v214_inbound_smoke as v


def bus(index, state='TRAVEL', *, direction=1, on_route=True):
    stops = [NS(station_id=i, station_type=0 if i in (0, 5) else 1) for i in range(6)]
    return NS(bus_id=2, trip_id=4, direction=direction, on_route=on_route, effective_station=stops,
        last_station=stops[index], next_station=stops[min(5, index+1)], state=NS(name=state),
        effective_route=[NS(distance=500.) for _ in range(5)], travel_distance=500.*index)


class InboundSmokeTests(unittest.TestCase):
    def test_v214_finish_keeps_zero_or_one_opportunity_and_closes_ledger(self):
        for count in (0, 1, 2):
            a = object.__new__(v.InboundAudit)
            a.anchor, a.applied_upper = dict(inbound=[{}]), {}
            a.launches = {0: dict(time_s=0., holding_counter_at_launch=0.)}
            a.return_ledger = {0: dict(launch_s=0., return_s=100., holding_s=10.*count,
                                     other_elapsed_s=100.-10.*count)}
            a.dispatch_ledger = [dict(actual_s=0., public_baseline_s=0.)]
            a.frozen_scalars, a.anchor_frames, a.upper_calls, a.arrivals, a.departures = (1., 2., 3.), [], [], [], []
            a.lower_changes = [dict(base_bin=2, selected_bin=2) for _ in range(count)]
            a.queries = [dict(trip_id=0, station_id=i+1, direction=True, time_s=float(i), command_s=10.)
                         for i in range(count)]
            records = [dict(q, action=2) for q in a.queries]
            runner = NS(env=NS(bus_all=[NS(on_route=False) for _ in range(12)], _done_reason='clear'), cfg={},
                lower_trainer=NS(alpha=1., lambda_param=2.), upper_trainer=NS(alpha=3.))
            with patch.object(v.prior.prior.factorial.prior, 'physical_contract', return_value={}), \
                 patch.object(v.prior.prior.factorial.prior, 'dispatch_metrics', return_value={}), \
                 patch.object(v.prior.prior.frozen.source.experiment.LogitCross, 'finish'):
                a.finish(runner, records, [dict(action_s=10.) for _ in range(count)])
            self.assertEqual(a.execution['opportunities'], count)
            self.assertEqual(a.ledger['executed_holding_s'], 10.*count)
            self.assertEqual(a.ledger['unexecuted_command_s'], 0.)

    def test_ready_clock_comes_from_native_prepare_and_hook_is_reversible(self):
        class FrozenBus:
            def _prepare_for_action(self, current_time, buses):
                return 'native_result'
        original = FrozenBus._prepare_for_action
        audit = object.__new__(v.InboundAudit)
        audit.observation_times = {}
        runner = NS(upper_trainer=NS(policy_net=NS(get_action=None)))
        def attach(a, r):
            a.bus_class = FrozenBus
        with patch.object(v.passive.PassiveAudit, 'attach', attach), patch.object(v.passive.PassiveAudit, 'detach'):
            audit.attach(runner)
            b = FrozenBus()
            b.bus_id, b.trip_id, b.direction, b.last_station = 2, 4, True, NS(station_id=3)
            self.assertEqual(b._prepare_for_action(123., []), 'native_result')
            self.assertEqual(audit.observation_times[2, 4, 3, True], 123.)
            self.assertFalse(hasattr(b, 'pre_action_time_s'))
            audit.detach()
        self.assertIs(FrozenBus._prepare_for_action, original)

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
