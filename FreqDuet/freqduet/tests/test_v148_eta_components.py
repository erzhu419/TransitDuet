from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v148_eta_components as run


ROUTE = dict(ids=[0, 1, 2], index={0: 0, 1: 1, 2: 2}, positions=[0., 100., 200.])
TAIL = dict(running_s=10., service_s=0., holding_s=0., ticks_s=0., target_service_s=5.)


def row(now=12.):
    return dict(trip_id=1, station_id=2, direction=True, prediction_time_s=now,
        evidence=dict(predicted_follower_trip_id=2, predicted_follower_bus_id=22),
        online=dict(status='not_causally_moving', component_used=False))


def path():
    return dict(dispatch={2: dict(actual_s=0.)},
        arrivals={(2, 1, True): dict(time_s=10., realized_holding=0.),
            (2, 2, True): dict(time_s=30., realized_holding=4.)},
        ready={(2, 1, True): dict(ready_time_s=14.), (2, 2, True): dict(ready_time_s=35.)},
        departures={(2, 1, True): dict(time_s=20., realized_holding=4.),
            (2, 2, True): dict(time_s=40., realized_holding=7.)},
        trace={(2, 1, True): dict(chosen_s=5.)})


class History:
    def estimate(self, key, now):
        return dict(mean={'holding': 4., 'ticks': 2.}[key[0]])


class EtaComponentTests(unittest.TestCase):
    def test_actual_service_running_holding_and_execution_partition(self):
        p = run.actual_parts(path(), row(), ROUTE)
        self.assertEqual(p, dict(current_service=2., running=10., downstream_service=5.,
            holding=4., execution_wait=2., current_ready_joint=0., unallocated=0.))

    def test_joint_ready_residence_is_not_arbitrarily_split(self):
        p = run.actual_parts(path(), row(17.), ROUTE, (2, 1, True))
        self.assertEqual((p['current_ready_joint'], p['holding'], p['execution_wait']), (3., 0., 0.))
        self.assertEqual(sum(p.values()), 18.)

    def test_moving_remainder_and_target_post_ready_holding_exclusion(self):
        p = run.actual_parts(path(), row(22.), ROUTE)
        self.assertEqual((p['running'], p['downstream_service'], p['holding']), (8., 5., 0.))
        self.assertEqual(sum(p.values()), 13.)

    def test_simultaneous_arrival_is_not_a_strictly_observed_current_service(self):
        p = run.actual_parts(path(), row(10.), ROUTE)
        self.assertEqual(p['current_service'], 0.)
        self.assertEqual(p['downstream_service'], 9.)

    def test_service_packet_preserves_frozen_total(self):
        stop = dict(applied=True, status='passive_stop_component', phase='service_in_progress',
            current_stop_key=[2, 1, True], gap_s=23., current_remaining_s=8., service_residual=dict(mean=2.))
        with patch.object(run.frozen.frozen, 'predict', return_value=stop), patch.object(
                run.frozen.model, 'component_prediction', return_value=TAIL):
            packet, joint, _ = run.component_packet(path(), row(), dict(v140_s=23., v141_s=23.),
                {True: ROUTE}, History(), None, None, None)
        self.assertEqual(sum(packet.values()), 23.)
        self.assertIsNone(joint)

    def test_known_command_remaining_has_no_new_holding_estimate(self):
        stop = dict(applied=True, status='passive_stop_component', phase='known_command',
            current_stop_key=[2, 1, True], gap_s=18., current_remaining_s=3.)
        with patch.object(run.frozen.frozen, 'predict', return_value=stop), patch.object(
                run.frozen.model, 'component_prediction', return_value=TAIL):
            packet, _, _ = run.component_packet(path(), row(17.), dict(v140_s=18., v141_s=18.),
                {True: ROUTE}, None, None, None, None)
        self.assertEqual(packet['holding'], 3.)
        self.assertEqual(sum(packet.values()), 18.)

    def test_ready_joint_prediction_and_truth_use_same_joint_channel(self):
        stop = dict(applied=False, status='ready_without_prior_command', current_stop_key=[2, 1, True])
        ready = dict(applied=True, gap_s=18., ready_residual=dict(mean=3.))
        with patch.object(run.frozen.frozen, 'predict', return_value=stop), patch.object(
                run.frozen, 'predict_ready', return_value=ready), patch.object(
                run.frozen.model, 'component_prediction', return_value=TAIL):
            packet, joint, _ = run.component_packet(path(), row(17.), dict(v141_s=18.),
                {True: ROUTE}, None, None, None, None)
        self.assertEqual(joint, (2, 1, True))
        self.assertEqual(packet['current_ready_joint'], 3.)

    def test_opaque_fallback_retains_whole_error(self):
        with patch.object(run.frozen.frozen, 'predict', return_value=dict(applied=False, status='unchanged')):
            parts, structured, _ = run.decompose(path(), row(), dict(v141_s=30.), 23.,
                {True: ROUTE}, None, None, None, None)
        self.assertFalse(structured)
        self.assertEqual(parts['unallocated'], 7.)
        self.assertEqual(sum(parts.values()), 7.)

    def test_saved_moving_components_are_used_without_new_estimates(self):
        r = row(22.)
        r['online'] = dict(TAIL, status='predicted', component_used=True)
        with patch.object(run.frozen.frozen, 'predict', return_value=dict(applied=False, status='unchanged')):
            packet, joint, _ = run.component_packet(path(), r, dict(v141_s=15.),
                {True: ROUTE}, None, None, None, None)
        self.assertEqual((packet['running'], packet['downstream_service']), (10., 5.))
        self.assertIsNone(joint)

    def test_component_total_must_match_the_saved_forecast(self):
        r = row(22.)
        r['online'] = dict(TAIL, status='predicted', component_used=True)
        with patch.object(run.frozen.frozen, 'predict', return_value=dict(applied=False, status='unchanged')):
            with self.assertRaises(AssertionError):
                run.component_packet(path(), r, dict(v141_s=99.), {True: ROUTE}, None, None, None, None)

    def test_signed_mae_allocation_is_additive_not_absolute_component_sum(self):
        parts = run.zeros()
        parts.update(running=10., holding=-6.)
        record = run.error_record([1], 4., parts, True)
        self.assertEqual(record['holding_mae_allocation_s'], -6.)
        self.assertEqual(sum(record[p+'_mae_allocation_s'] for p in run.PARTS), record['full_mae_s'])
        self.assertEqual(record['post_ready_total_bias_s'], -6.)

    def test_observed_no_switch_has_zero_burden_not_a_fake_conditional_row(self):
        result = run.burden([], 10)
        self.assertEqual(result['queries'], 10)
        self.assertTrue(all(v == 0 for v in result['metrics'].values()))
        self.assertEqual(run.prior.describe([])['metrics'], {})


if __name__ == '__main__':
    unittest.main()
