import copy
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import freqduet_v235_passive_cycle_capture as frozen
import test_v235_passive_cycle_capture as fixture
from freqduet_v230_resource_capture import ResourceCapture
from test_v238_persistent_requests import snapshot
from freqduet_v241_passive_requests import (
    JOIN_FIELDS, recording_audit, request_packet, validate_capture)


def packet_inputs(unassociated=False):
    state, row, meta = snapshot()
    if unassociated:
        row['plans'], row['pending_plan_version_refs'] = [], {}
    row.update(dict(zip(JOIN_FIELDS, (3, 5, 2, 0))))
    cycle = dict({k: row[k] for k in JOIN_FIELDS}, view=state.query(row, meta),
                 identity=dict(bus_id=2, trip_id=1, direction=False, station_id=3))
    prepare = dict(clock=dict(handoff_order=2, prepare_s=9), current=cycle['identity'])
    return row, cycle, prepare


def originals(f):
    return dict(f.originals(), v235=f.audit.cycle_capture.data(True))


class PassiveRequestsTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def tearDown(self):
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])

    def make(self, requests):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        if requests:
            with patch.object(fixture, 'recording_audit',
                              lambda base: recording_audit(frozen.recording_audit(base))):
                f = fixture.Fixture()
        else:
            f = fixture.Fixture()
        f.complete()
        return f

    def test_frozen_hook_chain_keeps_all_four_original_payloads_exact(self):
        ref, current = self.make(False), self.make(True)
        result = validate_capture(originals(ref), originals(current), current.audit.request_capture.data(True))
        self.assertTrue(result['original_payloads_exact'] and result['v235_sidecar_exact'])
        self.assertEqual(result['queries'], 3)

    def test_native_action_method_and_tensor_return_are_not_replaced(self):
        base = frozen.recording_audit(fixture.BaseAudit)
        self.assertIs(recording_audit(base).__call__, base.__call__)

        class Native:
            def __init__(self, seed, arm):
                self.cycle_capture = SimpleNamespace(resource=SimpleNamespace())

            def __call__(self, value, *args):
                return value

        audit = recording_audit(Native)(0, 'v107')
        value = torch.tensor([[.25, .75]])
        before = value.clone()
        self.assertIs(audit(value), value)
        self.assertTrue(torch.equal(before, value))

    def test_changed_requests_remain_unexecuted_and_h0_is_only_an_alias(self):
        row, cycle, prepare = packet_inputs()
        before = copy.deepcopy((row, cycle, prepare))
        p = request_packet(0, row, cycle, prepare)
        self.assertEqual((p['native_bin'], p['supply_request']['shadow_s'],
                          p['blind_request']['shadow_s']), (4, 10, 10))
        self.assertEqual(p['H0_bins'], dict(supply=4, blind=4))
        self.assertEqual(p['returned_rule'], 'native')
        self.assertFalse(p['executed_shadow'])
        self.assertEqual((row, cycle, prepare), before)

    def test_unassociated_query_keeps_supply_and_blind_requests_distinct(self):
        p = request_packet(0, *packet_inputs(unassociated=True))
        self.assertEqual((p['supply_request']['shadow_s'], p['blind_request']['shadow_s']), (20, 10))
        self.assertFalse(p['executed_shadow'])

    def test_replan_holding_and_actual_new_cycle_remain_frozen(self):
        f = self.make(True)
        qs = f.audit.request_capture.data(True)['queries']
        self.assertEqual([p['executed_holding_before_s'] for p in qs], [0, 5, 0])
        self.assertEqual(qs[0]['cycle_key'], qs[1]['cycle_key'])
        self.assertNotEqual(qs[1]['cycle_key'], qs[2]['cycle_key'])
        self.assertNotEqual(qs[0]['pending_plan_version_refs'], qs[1]['pending_plan_version_refs'])
        self.assertEqual(qs[2]['supply_request']['reason'], 'terminal_masked')

    def test_old_prepare_clock_is_kept_and_wrong_clocks_or_occurrences_stop(self):
        row, cycle, prepare = packet_inputs()
        p = request_packet(0, row, cycle, prepare)
        self.assertEqual((p['prepare_s'], p['query_s']), (9, 10))
        for bad in ('late_prepare', 'wrong_join'):
            r, c, old = copy.deepcopy((row, cycle, prepare))
            if bad == 'late_prepare':
                old['clock']['prepare_s'] = 11
            else:
                c['query_occurrence'] = 1
            with self.assertRaises(AssertionError):
                request_packet(0, r, c, old)

    def test_missing_changed_or_future_filled_packets_fail_acceptance(self):
        ref, current = self.make(False), self.make(True)
        a, b, data = originals(ref), originals(current), current.audit.request_capture.data(True)
        for kind in ('missing', 'bin', 'clock', 'executed', 'future', 'old_counter'):
            bad, old = copy.deepcopy(data), copy.deepcopy(b)
            if kind == 'missing':
                bad['queries'].pop()
            elif kind == 'bin':
                bad['queries'][0]['blind_request']['shadow_bin'] = 6
            elif kind == 'clock':
                bad['queries'][0]['query_s'] += 1
            elif kind == 'executed':
                bad['queries'][0]['executed_shadow'] = True
            elif kind == 'future':
                bad['queries'][0]['actual_next_trip'] = 999
            else:
                old['v235']['observer_checks']['query'] += 1
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                validate_capture(a, old, bad)

    def test_observer_rejects_rng_draws_and_native_calls(self):
        f = self.make(True)
        for mutate in (random.random, np.random.random, lambda: torch.rand(()),
                       lambda: f.audit.handoff.calls.update(['estimator'])):
            with self.assertRaises(AssertionError):
                with f.audit.request_capture.read():
                    mutate()

    def test_serialization_and_detach_preserve_old_hooks_and_counters(self):
        f = self.make(True)
        before = originals(f)
        data = f.audit.request_capture.data(True)
        self.assertEqual(json.loads(json.dumps(data)), data)
        self.assertEqual(before, originals(f))
        f.audit.detach()
        self.assertEqual(f.audit.clock.stamp, f.audit.native_stamp)
        self.assertIs(f.audit.resource.query.__func__, ResourceCapture.query)
        self.assertTrue(f.audit.detached)

    def test_incomplete_capture_and_active_legacy_arms_are_not_accepted(self):
        ref, current = self.make(False), self.make(True)
        with self.assertRaises(AssertionError):
            validate_capture(originals(ref), originals(current), current.audit.request_capture.data(False))
        with self.assertRaises(AssertionError):
            recording_audit(frozen.recording_audit(fixture.BaseAudit))(0, 'joint')


if __name__ == '__main__':
    unittest.main()
