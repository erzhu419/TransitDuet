import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v127_turnover_replay as replay


class TurnoverTests(unittest.TestCase):
    def test_eligibility_and_selected_vehicle_visibility_are_distinct(self):
        self.assertEqual(replay.dispatch_parts(100., 130., 121.), [100., 21., 9.])
        self.assertEqual(replay.dispatch_parts(100., 105., 90.), [100., 0., 5.])
        self.assertEqual(replay.dispatch_parts(100., 100., None), [100., 0., 0.])
        with self.assertRaises(AssertionError):
            replay.dispatch_parts(100., 110., 111.)

    def test_return_is_visible_in_next_dispatch_phase_and_reset_is_preserved(self):
        class Bus:
            def drive(self):
                self.on_route = False
                self.back_to_terminal_time = 100.
                self.last_completed_trip_id = 4
                self.direction = False
                return 'original_drive'
        bus = Bus()
        bus.bus_id, bus.trip_id, bus.on_route = 7, 4, True
        bus.episode_hold_vehicle_seconds = 25.
        env = SimpleNamespace(current_time=100., bus_in_terminal=[])
        original_drive = Bus.drive
        def launch(trip, actual_launch=None):
            bus.trip_id, bus.on_route, bus.back_to_terminal_time = trip.launch_turn, True, None
            return bus
        env.step, env.launch_bus = lambda: 'original_step', launch
        original_step, original_launch = env.step, env.launch_bus
        audit = replay.TurnoverAudit(12011, {}, replay.frozen.PRIMARY)
        audit.env, audit.bus_type = env, Bus
        audit.launches[4] = dict(time_s=20., holding_counter_at_launch=5.)
        with patch.object(replay.frozen.BilateralAudit, 'attach'), patch.object(replay.frozen.BilateralAudit, 'detach'):
            audit.attach(None)
            self.assertEqual(bus.drive(), 'original_drive')
            self.assertIsNone(audit.returns[4]['first_dispatch_visible_s'])
            env.current_time, env.bus_in_terminal = 101., [bus]
            self.assertEqual(env.step(), 'original_step')
            self.assertEqual(audit.returns[4]['first_dispatch_visible_s'], 101.)
            trip = SimpleNamespace(launch_turn=8, direction=False, _freqduet_first_eligible_time=90.)
            self.assertIs(env.launch_bus(trip), bus)
            self.assertEqual(audit.dispatches[0]['prior_trip_id'], 4)
            self.assertEqual(audit.dispatches[0]['coordinates'], [90., 11., 0.])
            self.assertEqual(audit.returns[4]['holding_s'], 20.)
            self.assertEqual(bus.trip_id, 8)
            audit.detach()
        self.assertIs(Bus.drive, original_drive)
        self.assertIs(env.step, original_step)
        self.assertIs(env.launch_bus, original_launch)

    def test_paired_dispatch_decomposition_and_interval_contributions(self):
        def path(times):
            rows = [dict(trip_id=i, bus_id=i, direction=True, prior_trip_id=-1,
                eligible_s=i*100., actual_s=t, coordinates=replay.dispatch_parts(i*100., t, None))
                for i, t in enumerate(times)]
            return dict(dispatches=rows, returns=[])
        base = path([0., 110., 220., 330., 440., 550., 660., 770.])
        candidate = path([0., 100., 240., 320., 450., 560., 660., 770.])
        result = replay.paired_details(base, candidate)
        self.assertEqual(result['unmatched_vehicle_or_prior_trip'], [])
        self.assertEqual(result['availability_groups']['initial_pool']['n'], 8)
        self.assertEqual(len(result['localized_intervals']), 6)
        self.assertGreater(result['dispatch_effects']['cv_squared'], 0.)


if __name__ == '__main__':
    unittest.main()
