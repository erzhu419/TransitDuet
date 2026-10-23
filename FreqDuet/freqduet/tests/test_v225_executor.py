import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_freqduet_v225_action_smoke as run
from test_v224_prepare_capture import fixture


def candidate(changed=True):
    capture, _, q = fixture()
    q.update(command_s=15., executed_holding_before_s=17.)
    if changed:
        q['context']['follower_departure_gap_s'] = 401.
        packet = capture.prepares[0]
        packet['native']['follower_gap_s'] = 401.
        packet['phases']['current'].update(native_plan_s=151., native_tick_s=151.,
            reserved_bus_id=3, status='supported', phase_s=-1.)
    capture.query(q)
    audit = run.JointAudit.__new__(run.JointAudit)
    audit.handoff, audit.queries = capture, [q]
    audit.execution, audit.first = [], None
    audit.reference, audit.archived, audit.actor = {}, {}, 12037
    audit.last_progress = time.monotonic()
    native = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
    return audit, native


def execute(audit, native):
    with patch.object(run.RecordingAudit, '__call__', return_value=native), patch.object(
            run, 'live_prefix', return_value=dict(prefix_exact=True)) as prefix:
        result = audit(None, None, None, None)
        return result, prefix.call_count


def completed(audit):
    command = audit.execution[0]
    action = dict(time_s=500., trip_id=7, station_id=2, direction=True,
        action=command['selected_bin'], probabilities=[float(i == command['selected_bin']) for i in range(7)])
    data = dict(trace=dict(lower_actions=[action],
        issued_holding=[dict(time_s=500., bus_id=3, trip_id=7, station_id=2,
                             action_s=command['command_s'])],
        actual_launches=[dict(trip=7, bus=3)],
        departures=[dict(trip_id=7, station_id=2, direction=True, time_s=520.,
                         realized_holding=17.+command['command_s'])]))
    side = dict(audit.handoff.data(), complete=True, execution=audit.execution)
    return data, side


class ExecutorTest(unittest.TestCase):
    def test_changed_request_returns_one_hot_and_updates_legacy_command(self):
        audit, native = candidate()
        result, calls = execute(audit, native)
        self.assertEqual(calls, 1)
        self.assertEqual(result.tolist(), [[0., 0., 1., 0., 0., 0., 0.]])
        self.assertEqual(native.tolist(), [[0., 0., 0., 1., 0., 0., 0.]])
        self.assertEqual(audit.queries[-1]['command_s'], 10.)
        self.assertTrue(audit.handoff.queries[-1]['executed_shadow'])
        self.assertEqual(audit.handoff.checked['v225_apply'], 1)

    def test_native_request_preserves_tensor_and_does_not_open_first_gate(self):
        audit, native = candidate(False)
        result, calls = execute(audit, native)
        self.assertIs(result, native)
        self.assertEqual(calls, 0)
        self.assertIsNone(audit.first)
        self.assertEqual(audit.execution[0]['selected_bin'], 3)

    def test_modified_shadow_request_cannot_pass_execution_validation(self):
        audit, native = candidate()
        audit.handoff.queries[-1]['shadow_bin'] = 6
        with self.assertRaises(AssertionError):
            execute(audit, native)
            run.holding_execution(*completed(audit))

    def test_issued_and_executed_holding_match_on_same_vehicle(self):
        audit, native = candidate()
        execute(audit, native)
        result = run.holding_execution(*completed(audit))
        self.assertEqual(result['executed_changes'], 1)
        self.assertEqual(result['executed_changed_s'], 10.)
        self.assertEqual(result['direct_vs_same_state_native_s'], -5.)
        self.assertEqual(result['decreases'], 1)

    def test_wrong_cumulative_holding_or_vehicle_cannot_pass(self):
        audit, native = candidate()
        execute(audit, native)
        for field in ('cumulative', 'vehicle', 'issued', 'prepare'):
            data, side = completed(audit)
            side = copy.deepcopy(side)
            if field == 'cumulative':
                data['trace']['departures'][0]['realized_holding'] += 1.
            elif field == 'vehicle':
                data['trace']['actual_launches'][0]['bus'] = 4
            elif field == 'issued':
                data['trace']['issued_holding'][0]['action_s'] = 15.
            else:
                side['execution'][0]['prepare_order'] += 1
            with self.assertRaises(AssertionError, msg=field):
                run.holding_execution(data, side)

    def test_selected_bin_cannot_be_supported_by_a_later_phase_change(self):
        audit, native = candidate()
        execute(audit, native)
        data, side = completed(audit)
        side = copy.deepcopy(side)
        side['prepares'][0]['phases']['current']['phase_s'] = 0.
        with self.assertRaises(AssertionError):
            run.holding_execution(data, side)

    def test_live_prefix_uses_exact_first_packet_and_native_lists(self):
        audit, _ = candidate()
        audit.clock = SimpleNamespace(events=[dict(time_s=500., event_seq=88)])
        for name in ('policy_inputs', 'commands', 'callbacks', 'dispatches', 'returns',
                     'arrivals', 'departures', 'cycles', 'fresh_queries'):
            setattr(audit, name, [])
        audit.credit = SimpleNamespace(holds=[])
        reference = dict(events=audit.clock.events, cycles=[], fresh_upper=[],
            trace={k: [] for k in ('lower_policy_inputs', 'upper_commands', 'upper_callbacks',
                'issued_holding', 'actual_launches', 'vehicle_returns', 'arrivals', 'departures')})
        archived = json.loads(json.dumps(audit.handoff.data()))
        result = run.live_prefix(audit, reference, archived)
        self.assertTrue(result['first_shadow_exact'])
        archived['prepares'][0]['native']['forward_gap_s'] += 1.
        with self.assertRaisesRegex(AssertionError, 'first prepared packet'):
            run.live_prefix(audit, reference, archived)

    def test_first_uniform_stays_same_except_executed_action(self):
        action = dict(time_s=500., features=[1., 2.], context=dict(x=3), action=3,
                      probabilities=[0., 0., 0., 1., 0., 0., 0.])
        uniform = [[7, 2, 1, 0], dict(uniform=.4, order=0, time_s=500., action_s=15.)]
        old = dict(trace=dict(lower_actions=[action], action_uniforms=[uniform]))
        data = copy.deepcopy(old)
        data['trace']['lower_actions'][0].update(action=2, probabilities=[0., 0., 1., 0., 0., 0., 0.])
        data['trace']['action_uniforms'][0][1]['action_s'] = 10.
        side = dict(first_change=dict(prefix_exact=True, first_shadow_exact=True,
                                     query_index=0, selected_bin=2))
        run.prefix_after_run(data, side, old)
        data['trace']['action_uniforms'][0][1]['uniform'] = .5
        with self.assertRaises(AssertionError):
            run.prefix_after_run(data, side, old)

    def test_started_path_cannot_silently_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'progress').mkdir()
            (root/'progress/c0_joint.json').write_text('{"started": true}')
            with patch.object(run.legacy.prior, 'read', return_value={}), patch.object(
                    run.legacy, 'run_item') as rollout:
                result = run.run_path(root, root/'archive', 0)
            rollout.assert_not_called()
            self.assertFalse(result['engineering_passed'])
            self.assertIn('started slot cannot replay', result['failure'])


if __name__ == '__main__':
    unittest.main()
