from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v134_handoff_pilot as run
import test_v133_launch_handoff as fixtures


class HandoffPilotTests(unittest.TestCase):
    def stream(self):
        prefix = fixtures.LaunchHandoffTests().prefix()
        active = run.ActiveLaunchETA(prefix.geometry)
        for name in ('history', 'launches', 'motion', 'departures', 'pending'):
            setattr(active, name, getattr(prefix, name))
        return active

    def test_active_path_returns_proposal_and_retains_same_state_original(self):
        e = fixtures.LaunchHandoffTests().evidence()
        p = self.stream()
        result = p.predict(e, True, 2)
        self.assertEqual(result['gap_s'], 66.)
        key = run.prior.forecast_key(e, True, 2)
        self.assertEqual(p.original_predictions[key]['gap_s'], 140.)

    def test_ordinary_journey_forecast_is_not_rewritten(self):
        e = fixtures.LaunchHandoffTests().evidence(ready_time_s=100., spatial_gap_m=150., speed_mps=2.5,
            follower_source=run.prior.model.clock.JOURNEY)
        p = self.stream()
        result = p.predict(e, True, 2)
        self.assertIs(result, p.original_predictions[run.prior.forecast_key(e, True, 2)])

    def test_joint_gate_includes_control_wait_not_only_native_readiness(self):
        delta = dict(passenger_person_s=-100., cv_squared=-.001, inclusive_terminal_delay_s=1.,
            trips_unlaunched=0, trips_incomplete=0)
        self.assertFalse(run.joint_improvement(delta))
        self.assertTrue(run.joint_improvement(dict(delta, inclusive_terminal_delay_s=-1.)))


if __name__ == '__main__':
    unittest.main()
