import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v130_plan_terminal as experiment


class PlanTerminalTests(unittest.TestCase):
    def test_half_gap_and_independent_plan_deadline(self):
        self.assertEqual(experiment.planned_command(360., 360., 20., 1), 370.)
        self.assertEqual(experiment.planned_command(360., 360., 100., 1), 390.)
        self.assertEqual(experiment.planned_command(360., 360., 0., 1), 360.)

    def test_no_recursive_growth_and_recovery_after_one_supply_shock(self):
        last = 100.
        waits = []
        for i in range(1, 101):
            published = i*360.
            last = experiment.planned_command(published, published, last, 1)
            waits.append(last-published)
        self.assertEqual(waits[:6], [30., 15., 7., 3., 1., 0.])
        self.assertEqual(waits[6:], [0.]*94)

    def test_no_ready_or_expired_budget_does_not_restart_waiting(self):
        self.assertEqual(experiment.planned_command(360., 360., 100., 0), 360.)
        self.assertEqual(experiment.planned_command(360., 500., 490., 1), 360.)
        self.assertEqual(experiment.planned_command(360., 360., -10000., 1), 360.)
        with self.assertRaises(AssertionError):
            experiment.planned_command(360., 360., 361., 1)

    def test_command_is_once_only_and_has_no_hard_spacing_gate(self):
        audit = experiment.PlanAudit(12011, {}, experiment.frozen.PRIMARY, True)
        env = SimpleNamespace(current_time=100., _last_dispatch_time={True: 90.},
            _ready_vehicle_count=lambda direction: 1, step=lambda: 'original_step')
        def upper(state, trip):
            trip._original_launch, trip._delta_t = trip.launch_time, 0.
            return 360.
        runner = SimpleNamespace(_upper_callback_v2=upper)
        audit.env, audit.runner = env, runner
        base = experiment.prior.prior.TurnoverAudit
        with patch.object(base, 'attach'), patch.object(base, 'detach'):
            audit.attach(runner)
            trip = SimpleNamespace(launch_turn=2, launch_time=100., direction=True)
            self.assertEqual(runner._upper_callback_v2([], trip), 360.)
            self.assertEqual(trip._freqduet_scheduled_launch, 130.)
            self.assertFalse(hasattr(trip, '_freqduet_min_dispatch_headway'))
            env.current_time, env._last_dispatch_time[True] = 200., 190.
            env.step()
            self.assertEqual(trip._freqduet_scheduled_launch, 130.)
            audit.detach()
        self.assertIs(runner._upper_callback_v2, upper)

    def test_total_delay_includes_control_but_does_not_hide_fleet_delay(self):
        rows = [dict(trip_id=0, direction=False, actual_s=0., eligible_s=0.),
            dict(trip_id=1, direction=True, actual_s=400., eligible_s=130.),
            dict(trip_id=2, direction=False, actual_s=360., eligible_s=360.),
            dict(trip_id=3, direction=True, actual_s=760., eligible_s=760.),
            dict(trip_id=4, direction=False, actual_s=720., eligible_s=720.),
            dict(trip_id=5, direction=True, actual_s=1120., eligible_s=1120.)]
        clocks = {0: 0., 1: 100., 2: 360., 3: 760., 4: 720., 5: 1120.}
        plans = [dict(command_s=r['eligible_s'], published_release_s=clocks[r['trip_id']]) for r in rows]
        result = experiment.terminal_stats(rows, clocks, plans, True)
        self.assertEqual(result['inclusive_terminal_delay_s'], 50.)
        self.assertEqual(result['native_readiness_delay_s'], 45.)
        self.assertEqual(result['qualification_shift_s'], 5.)
        with self.assertRaises(AssertionError):
            experiment.terminal_stats(rows, {**clocks, 1: 99.}, plans, True)


if __name__ == '__main__':
    unittest.main()
