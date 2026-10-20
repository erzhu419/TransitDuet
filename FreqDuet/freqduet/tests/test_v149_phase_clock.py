from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import freqduet_phase_clock_eta as eta
import run_freqduet_v149_phase_clock as run


ROUTE = dict(ids=[0, 1, 2], index={0: 0, 1: 1, 2: 2}, positions=[0., 100., 200.])


def history():
    return eta.model.CompletedMeans([
        ((channel, True, stop), 20., value) for stop in (1, 2)
        for channel, value in (('running', 10.), ('service', 20.), ('holding', 3.), ('ticks', 1.))])


class PhaseClockTests(unittest.TestCase):
    def test_arrival_phase_not_completion_phase(self):
        arrivals = {(1, 1, True): dict(time_s=99.), (2, 1, True): dict(time_s=101.)}
        ready = {(1, 1, True): dict(ready_time_s=110.), (2, 1, True): dict(ready_time_s=107.)}
        phases = eta.PhaseServices(arrivals, ready, 100.)
        self.assertEqual(phases.estimate(True, 1, 120., 99.)['mean'], 11.)
        self.assertEqual(phases.estimate(True, 1, 120., 100.)['mean'], 6.)

    def test_strict_completed_prefix_excludes_equal_and_future(self):
        a = {(1, 1, True): dict(time_s=101.), (2, 1, True): dict(time_s=102.)}
        r = {(1, 1, True): dict(ready_time_s=107.), (2, 1, True): dict(ready_time_s=99999.)}
        phases = eta.PhaseServices(a, r, 100.)
        self.assertIsNone(phases.estimate(True, 1, 107., 120.))
        self.assertEqual(phases.estimate(True, 1, 108., 120.)['mean'], 6.)

    def test_window_and_all_past_fallback_are_not_tuned(self):
        a = {(1, 1, True): dict(time_s=1.), (2, 1, True): dict(time_s=3990.)}
        r = {(1, 1, True): dict(ready_time_s=10.), (2, 1, True): dict(ready_time_s=4000.)}
        phases = eta.PhaseServices(a, r, 0.)
        self.assertEqual(phases.estimate(True, 1, 4001., 5000.)['mean'], 10.)
        self.assertTrue(phases.estimate(True, 1, 8000., 9000.)['historical_fallback'])

    def test_missing_clearance_support_retains_native_service(self):
        phases = eta.PhaseServices({}, {}, 100.)
        result = eta.tail(history(), phases, ROUTE, True, 0, 2, 0., 99., phase_service=True)
        self.assertEqual(result['gap_s'], 64.)
        self.assertEqual(result['phase_unsupported_stops'], 2)
        self.assertEqual(result['service_delta_s'], 0.)

    def test_phase_uses_predicted_arrival_and_retains_nonzero_service(self):
        phases = eta.PhaseServices({(1, 1, True): dict(time_s=80.),
            (2, 2, True): dict(time_s=101.)}, {(1, 1, True): dict(ready_time_s=86.),
                (2, 2, True): dict(ready_time_s=107.)}, 100.)
        # First arrival is 105, so the active-only stop1 sample cannot be used.
        result = eta.tail(history(), phases, ROUTE, True, 0, 2, 0., 95., phase_service=True)
        self.assertEqual((result['gap_s'], result['phase_supported_stops']), (64., 0))
        result = eta.tail(history(), phases, ROUTE, True, 0, 2, 0., 108., phase_service=True)
        self.assertEqual((result['gap_s'], result['phase_supported_stops']), (50., 1))
        self.assertEqual(result['phase_unsupported_stops'], 1)

    def test_current_residence_preserved_target_holding_excluded(self):
        result = eta.tail(history(), None, ROUTE, True, 1, 2, 100., 95., initial_s=7.)
        self.assertEqual(result['gap_s'], 37.)
        current_target = eta.tail(history(), None, ROUTE, True, 2, 2, 200., 95., initial_s=7.)
        self.assertEqual(current_target['gap_s'], 7.)

    def test_all_speed_sources_use_same_physical_position(self):
        m = eta.model.MovingPrefix([], [dict(station_id=0, time_s=0.)])
        indexed = {2: SimpleNamespace(motion=m)}
        row = dict(station_id=2, direction=True, prediction_time_s=50., evidence=dict(
            predicted_follower_trip_id=2, follower_gap_prediction_s=1000., spatial_gap_m=150.,
            follower_source='same_time_avl_route_speed_eta', speed_mps=999.))
        position, _ = eta.state({}, row, indexed, {True: ROUTE}, None, None, None)
        self.assertEqual((position['origin'], position['progress']), (0, 50.))
        row['evidence']['follower_source'] = 'same_time_avl_journey_speed_eta'
        self.assertEqual(position, eta.state({}, row, indexed, {True: ROUTE}, None, None, None)[0])

    def test_simultaneous_launch_is_not_strictly_available(self):
        m = eta.model.MovingPrefix([], [dict(station_id=0, time_s=50.)])
        row = dict(station_id=2, direction=True, prediction_time_s=50., evidence=dict(
            predicted_follower_trip_id=2, follower_gap_prediction_s=1000., spatial_gap_m=200.))
        self.assertIsNone(eta.state({}, row, {2: SimpleNamespace(motion=m)},
            {True: ROUTE}, None, None, None)[0])

    def test_unsupported_clock_does_not_invent_zero_prediction(self):
        with patch.object(eta, 'state', return_value=(None, 'unsupported')):
            values, _ = eta.forecasts({}, {}, dict(v141_s=123.), {}, (None,)*4, None)
        self.assertEqual(values, dict.fromkeys(run.ARMS, 123.))

    def test_structured_baseline_must_reproduce_exactly(self):
        row = dict(online=dict(component_used=True))
        saved = dict(v141_s=99.)
        with patch.object(eta, 'state', return_value=({}, 'structured')), patch.object(
                eta, 'tail', return_value=dict(gap_s=98., phase_supported_stops=0, phase_unsupported_stops=0)):
            with self.assertRaises(AssertionError):
                eta.forecasts({}, row, saved, {}, (None,)*4, None)

    def test_same_frozen_control_serves_all_shadow_arms(self):
        values = dict(base=10., service=9., clock=8., joint=7.)
        r = run.record([1], values, 8.)
        self.assertEqual((r['service_delta_mae_s'], r['clock_delta_mae_s'], r['joint_delta_mae_s']),
            (-1., -2., -1.))
        plan = run.registration()
        self.assertEqual((plan['fitted_parameters'], plan['actor_updates'], plan['critic_updates'],
            plan['dual_updates'], plan['main_policy_changed']), (0, 0, 0, 0, False))


if __name__ == '__main__':
    unittest.main()
