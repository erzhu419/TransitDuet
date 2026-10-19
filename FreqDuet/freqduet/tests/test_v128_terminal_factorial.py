import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v128_terminal_factorial as experiment


class TerminalFactorialTests(unittest.TestCase):
    def test_native_spacing_preserves_original_release_and_lower_goal(self):
        trip = SimpleNamespace(launch_time=100., _original_launch=100., _delta_t=20., target_headway=362.)
        self.assertEqual(experiment.install_spacing(trip), 120.)
        self.assertTrue(trip._freqduet_terminal_dispatch)
        self.assertEqual(trip._freqduet_scheduled_launch, 120.)
        self.assertEqual(trip._freqduet_min_dispatch_headway, 360.)
        self.assertEqual(trip.target_headway, 362.)
        self.assertEqual(experiment.published_release(trip), 120.)

    def test_terminal_wait_cannot_disappear_into_readiness_definition(self):
        rows = [dict(trip_id=0, direction=True, actual_s=100., eligible_s=100.),
            dict(trip_id=2, direction=True, actual_s=600., eligible_s=460.)]
        result = experiment.terminal_metrics(rows, {0: 100., 2: 200.}, True)
        self.assertEqual(result['inclusive_terminal_delay_s'], 200.)
        self.assertEqual(result['native_readiness_delay_s'], 70.)
        self.assertEqual(result['qualification_shift_s'], 130.)
        with self.assertRaises(AssertionError):
            experiment.terminal_metrics(rows, {0: 100., 2: 200.}, False)

    def test_original_callback_and_reference_clock_are_preserved(self):
        audit = experiment.TerminalAudit(12011, {}, experiment.frozen.PRIMARY, True)
        env = SimpleNamespace(current_time=100., _last_dispatch_time={True: 0.},
            _ready_vehicle_count=lambda direction: 1)
        def step():
            return 'original_step'
        def upper(state, trip):
            trip._original_launch, trip._delta_t = trip.launch_time, 2.
            trip.target_headway = 362.
            return 362.
        env.step = step
        runner = SimpleNamespace(_upper_callback_v2=upper)
        audit.env, audit.runner = env, runner
        with patch.object(experiment.prior.TurnoverAudit, 'attach'), patch.object(experiment.prior.TurnoverAudit, 'detach'):
            audit.attach(runner)
            trip = SimpleNamespace(launch_turn=2, launch_time=100., direction=True)
            self.assertEqual(runner._upper_callback_v2([], trip), 362.)
            self.assertNotIn(2, audit.reference_clocks)
            env.current_time = 102.
            self.assertEqual(env.step(), 'original_step')
            self.assertEqual(audit.reference_clocks[2], 102.)
            audit.detach()
        self.assertIs(runner._upper_callback_v2, upper)
        self.assertIs(env.step, step)

    def test_factorial_sign_and_joint_gate_use_inclusive_delay(self):
        levels = {method: {k: float(i) for k in experiment.METRICS}
            for i, method in enumerate(experiment.METHODS)}
        effects = experiment.factorial_effects(levels)
        self.assertEqual(effects['split_spacing_vs_raw']['cv_squared'], 3.)
        self.assertEqual(effects['terminal_x_lower']['cv_squared'], 0.)
        stats = {k: dict(ci95=[-2., -1.]) for k in experiment.METRICS}
        self.assertTrue(experiment.joint_gate(stats))
        stats['inclusive_terminal_delay_s']['ci95'][1] = 1.
        self.assertFalse(experiment.joint_gate(stats))


if __name__ == '__main__':
    unittest.main()
