import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import freqduet_v255_passive_pulse as pulse
import freqduet_v248_prepare_timing_capture as timing
from freqduet_v235_passive_cycle_capture import recording_audit as cycle_audit
from freqduet_v241_passive_requests import recording_audit as request_audit
from test_v248_prepare_timing_capture import Fixture as TimingFixture, NativeAudit


class Fixture(TimingFixture):
    """Synthetic native producer: same real six-observer chain, no transit runtime."""
    def __init__(self, mode=None, supported=True):
        base = timing.recording_audit(request_audit(cycle_audit(NativeAudit)))
        self.audit = (base if mode is None else pulse.recording_audit(base, mode))(0, 'v107')
        recorder = self.audit.recorder_type()
        recorder.result = object()
        plans = [(1, False, 0.), (2, False, 201.), (3, True, 100.), (4, False, 200.), (5, True, 300.)]
        self.env = SimpleNamespace(time_step=1., current_time=0., headway_events=recorder,
            bus_all=[SimpleNamespace(bus_id=b) for b in range(12)],
            timetables=[SimpleNamespace(launch_turn=t, direction=d, launched=False,
                launch_time=s, target_headway=100.) for t,d,s in plans])
        self.audit.attach(SimpleNamespace(env=self.env))
        self.physical, self.running, self.return_times, self.draws = {}, {}, {}, []
        self.ready_directions = dict.fromkeys(range(12), False)
        self.supported = supported
        for _ in range(20):
            self.audit.handoff.mark('tick')
        self.audit.clock.stamp('fresh_upper', 0., trip_id=1, direction=False)
        self.replan(0., 1, False)

    def prepare(self, bus, station, now, held, front=None):
        original = self.audit.bus_type._prepare_for_action
        a = self.audit
        def native_inputs(vehicle, *args, **kwargs):
            if front is not None and self.supported:
                vehicle.pre_action_follower_departure_gap = 220.
                vehicle.pre_action_follower_source = 'same_time_avl_journey_speed_eta'
                vehicle.pre_action_follower_bus_id = 1
                identities = dict(time_s=now, order=a.handoff.mark('avl_identities'),
                    vehicles=[dict(bus_id=1, trip_id=2, direction=False, on_route=True)])
                snapshot = dict(time_s=now, order=a.handoff.mark('avl_sample'),
                    origin='native_inline_snapshot', identity_sample=identities,
                    vehicles=[dict(bus_id=1, direction=False, on_route=True)])
                a.handoff.estimates[bus] = dict(time_s=now, order=a.handoff.mark('estimate'),
                    snapshot=snapshot, result=dict(follower_bus_id=1))
            return original(vehicle, *args, **kwargs)
        a.bus_type._prepare_for_action = native_inputs
        try:
            super().prepare(bus, station, now, held, front)
        finally:
            a.bus_type._prepare_for_action = original

    def complete(self):
        self.launch(0, 1, False, 0.)
        self.launch(2, 4, False, 200.)
        self.launch(1, 2, False, 201.)
        self.departure(0, 7, 7090., 40.)
        self.prepare(2, 7, 7190., 20., front=1)
        self.query(2, 7195., 25.)
        before = copy.deepcopy(self.audit.resource.plans[3])
        self.replan(7196., 3, True, [dict(trip_id=3, before=before, after=dict(before, planned_s=90.))])
        self.departure(0, 8, 7197., 50.)
        self.prepare(2, 8, 7200., 30., front=1)
        self.query(2, 7200., 35.)
        self.query(2, 7201., 35.)  # A second occurrence is not another pulse.
        self.departure(0, 9, 7210., 55.)
        self.prepare(2, 9, 7310., 40., front=1)
        self.query(2, 7311., 45.)
        self.prepare(2, 10, 7320., 45.)  # Unused PREPARE stays in its old payload.
        self.returned(0, 7330., 55.)
        self.returned(2, 7331., 45.)
        self.returned(1, 7332., 0.)  # Zero-query physical cycle.
        self.launch(0, 3, True, 7340.)
        self.prepare(0, 1, 7345., 0.)
        self.query(0, 7345., 0., masked=True)
        self.returned(0, 7350., 0.)
        self.launch(2, 5, True, 7360.)
        self.prepare(2, 1, 7365., 0.)  # Open,zero-query cycle.

    def payloads(self):
        data = self.originals()
        # Real legacy lower_actions are actions,NOT full query snapshots.
        data['legacy']['trace']['lower_actions'] = [dict(bin=q['base_bin'], time_s=q['time_s'])
                                                    for q in self.audit.queries]
        data['v248'] = self.audit.prepare_timing_capture.data(True)
        return data


class PassivePulseTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.fixtures = []

    def tearDown(self):
        for f in reversed(self.fixtures):
            f.audit.detach()
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])

    def make(self, mode=None, supported=True):
        random.seed(255)
        np.random.seed(255)
        torch.manual_seed(255)
        f = Fixture(mode, supported)
        self.fixtures.append(f)
        f.complete()
        return f

    def pair(self, mode='observer_only'):
        reference, current = self.make(), self.make(mode)
        return reference.payloads(), current.payloads(), current.audit.pulse_capture.data(True)

    def test_both_modes_preserve_six_complete_payloads_and_endpoints(self):
        for mode in pulse.MODES:
            with self.subTest(mode=mode):
                result = pulse.validate_capture(*self.pair(mode))
                self.assertTrue(result['six_payloads_exact'])
                self.assertEqual(result['queries'], 5)
                self.assertEqual(result['anchor_query_index'], 1)
                self.assertEqual(result['command_records'], int(mode == 'forced_native_once'))

    def test_first_anchor_keeps_prepare_query_h_and_clock_domains(self):
        f = self.make('forced_native_once')
        data = f.audit.pulse_capture.data(True)
        row = data['anchor']['query']
        self.assertEqual(row['clock']['query_s'], 7200.)
        self.assertEqual((row['clock']['prepare_holding_s'], row['clock']['query_holding_s']), (30.,35.))
        self.assertNotEqual(row['clock']['prepare_order'], row['clock']['prepare_seq'])
        self.assertEqual(data['queries'][1]['clock']['prepare_seq'], data['queries'][2]['clock']['prepare_seq'])
        self.assertNotEqual(data['queries'][1]['clock']['query_occurrence'], data['queries'][2]['clock']['query_occurrence'])
        self.assertEqual(len(data['commands']), 1)
        self.assertEqual(data['queries'][0]['reasons'], ['before_registered_time'])
        self.assertIn('terminal_masked', data['queries'][-1]['reasons'])

    def test_noop_returns_same_action_object_once_without_new_native_calls(self):
        f = self.make('forced_native_once')  # Fixture.query asserts object identity on EVERY call.
        self.assertEqual(f.audit.handoff.calls['lower_rule'], 5)
        self.assertEqual(len(f.audit.queries), 5)
        self.assertEqual(len(f.audit.clock.events), len(self.make().audit.clock.events))
        self.assertEqual(f.draws, self.make().draws)

    def test_no_anchor_is_explicit_and_fails_admission(self):
        reference, f = self.make(supported=False), self.make('forced_native_once', False)
        data = f.audit.pulse_capture.data(True)
        self.assertEqual(data['status'], 'missing_anchor')
        self.assertEqual(data['commands'], [])
        with self.assertRaisesRegex(AssertionError, 'anchor absent'):
            pulse.validate_capture(reference.payloads(), f.payloads(), data)

    def test_anchor_is_copied_not_backfilled_from_later_state(self):
        f = self.make('observer_only')
        before = f.audit.pulse_capture.data(True)['anchor']
        self.assertEqual(before['prepared']['own']['holding_s'], 30.)
        f.audit.prepare_timing_capture.prepares[1]['mapped']['own']['holding_s'] = 999.
        f.audit.cycle_capture.queries[1]['view']['own_association'] = None
        self.assertEqual(f.audit.pulse_capture.data(True)['anchor'], before)

    def test_old_types_values_keysets_and_order_are_exact(self):
        ref, old, data = self.pair()
        for change in ('type','key','order','endpoint'):
            bad = copy.deepcopy(old)
            if change == 'type':
                bad['legacy']['results']['journey'] = 18  # Numerically equal is insufficient.
            elif change == 'key':
                bad['v241']['queries'][0]['extra'] = 0
            elif change == 'order':
                bad['v248']['queries'].reverse()
            else:
                bad['legacy']['results']['journey'] += 1.
            with self.subTest(change=change), self.assertRaises(AssertionError):
                pulse.validate_capture(ref, bad, data)

    def test_missing_repeated_query_wrong_anchor_or_command_stops(self):
        ref, old, data = self.pair('forced_native_once')
        for change in ('query','anchor','clock','H','command','missing_command','extra_command'):
            bad = copy.deepcopy(data)
            if change == 'query':
                bad['queries'].pop(2)
            elif change == 'anchor':
                bad['anchor']['query'] = bad['queries'][2]
            elif change == 'clock':
                bad['queries'][1]['clock']['prepare_seq'] = bad['queries'][1]['clock']['prepare_order']
            elif change == 'H':
                bad['anchor']['prepared']['own']['holding_s'] = 35.
            elif change == 'command':
                bad['commands'][0]['command_bin'] = 0
            elif change == 'missing_command':
                bad['commands'] = []
            else:
                bad['commands'] *= 2
            with self.subTest(change=change), self.assertRaises(AssertionError):
                pulse.validate_capture(ref, old, bad)

    def test_observer_cannot_record_a_command(self):
        ref, old, data = self.pair()
        data['commands'] = [dict(command_bin=0)]
        with self.assertRaises(AssertionError):
            pulse.validate_capture(ref, old, data)

    def test_side_effect_guards_reject_rng_draws_and_native_calls(self):
        f = self.make('observer_only')
        for kind in ('python','numpy','torch','native_call'):
            with self.subTest(kind=kind), self.assertRaisesRegex(AssertionError, 'pulse'):
                with f.audit.pulse_capture.read():
                    if kind == 'python':
                        random.random()
                    elif kind == 'numpy':
                        np.random.random()
                    elif kind == 'torch':
                        torch.rand(())
                    else:
                        f.audit.handoff.calls['extra'] += 1

    def test_other_arms_and_action_selection_are_not_interfaces(self):
        with self.assertRaises(AssertionError):
            pulse.recording_audit(NativeAudit, 'reduce')
        with self.assertRaises(AssertionError):
            pulse.recording_audit(NativeAudit, 'observer_only')(0, 'upper')


if __name__ == '__main__':
    unittest.main()
