"""V254 observer/native-no-op callback; no rollout, future fork or action selector."""

from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

import freqduet_v248_prepare_timing_capture as timing
from preregister_freqduet_v224_prepare_handoff import require_legacy_equal
from preregister_freqduet_v246_compensation_supply import ACTIONS
from preregister_freqduet_v230_resource_capture import PAYLOAD_FIELDS
from seal_freqduet_v254_action_manifest import PAYLOADS


MODES = ('observer_only', 'forced_native_once')


def exact_types(reference, recorded):
    assert type(reference) is type(recorded), 'old payload type changed'
    if isinstance(reference, dict):
        assert reference.keys() == recorded.keys(), 'old payload keys changed'
        for key in reference:
            exact_types(reference[key], recorded[key])
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(recorded), 'old payload length changed'
        for left, right in zip(reference, recorded):
            exact_types(left, right)
    else:
        assert reference == recorded, 'old payload value changed'


def query_packet(q, old, resource, cycle, prepared, joined):
    assert q['event_seq'] == old['legacy_query_event_seq'] == joined['query_seq']
    assert all(q[k] == prepared['identity'][k] == old['identity'][k]
               for k in ('bus_id', 'trip_id', 'station_id', 'direction'))
    assert cycle['identity'] == old['identity']
    assert cycle['view']['cycle_key'] == prepared['cycle_key']
    assert joined == timing.join_query(prepared, old, q, cycle['view'])
    assert q['base_bin'] == old['native_bin'] == resource['shadow_frontier']['native_bin']
    assert q['feasible_bins'] == old['feasible_bins']
    bins = q['feasible_bins']
    assert bins and len(bins) == len(set(bins)) and all(b in range(7) for b in bins)
    assert q['base_bin'] in bins
    assert resource['snapshot_read_s'] == q['snapshot_read_s'] == q['time_s']
    assert resource['snapshot_event_seq'] == q['event_seq']
    assert prepared['prepare_s'] <= q['time_s']
    supported = prepared['forward_s'] is not None and prepared['backward_s'] is not None
    assert supported == (q['context']['evidence_valid'] and q['context']['follower_evidence_valid'])
    if prepared['forward_s'] is not None:
        assert prepared['front_status'] == 'completed_at_prepare', 'missing completed front timing'
    reasons = []
    if q['time_s'] < 7200.:
        reasons.append('before_registered_time')
    if q['terminal_masked']:
        reasons.append('terminal_masked')
    if ACTIONS[q['base_bin']] == 0:
        reasons.append('zero_native_holding')
    if not supported:
        reasons.append('unsupported_prepare_gaps')
    if len(bins) < 2:
        reasons.append('fewer_than_two_feasible_bins')
    return copy.deepcopy(dict(identity=old['identity'], cycle_key=prepared['cycle_key'],
        clock=joined, native_bin=q['base_bin'], native_holding_s=ACTIONS[q['base_bin']],
        feasible_bins=bins, terminal_masked=q['terminal_masked'],
        forward_s=prepared['forward_s'], backward_s=prepared['backward_s'],
        eligible=not reasons, reasons=reasons))


def anchor_snapshot(q, packet, prepare, prepared, cycle):
    assert prepared['identity'] == prepare['current']
    return copy.deepcopy(dict(query=packet, prepared=prepared,
        prepare_neighbors=dict(front=prepare['front'], follower=prepare['follower'],
                               phases=prepare['phases']),
        resource_snapshot=dict(read_s=q['snapshot_read_s'], event_seq=q['snapshot_event_seq'],
            ready=q['ready'], incoming=q['active'], plans=q['plans']),
        physical_cycle_view=cycle['view']))


class PulseCapture:
    def __init__(self, audit, mode):
        assert mode in MODES
        self.audit, self.mode = audit, mode
        self.queries, self.anchor, self.commands = [], None, []
        self.checks = 0

    @contextmanager
    def read(self):
        calls = self.audit.handoff.calls
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), calls.copy()
        try:
            yield
        finally:
            after = np.random.get_state()
            assert random.getstate() == before[0], 'pulse changed Python RNG'
            assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:], \
                'pulse changed NumPy RNG'
            assert torch.equal(before[2], torch.get_rng_state()), 'pulse changed Torch RNG'
            assert calls == before[3], 'pulse invoked native computation'
            self.checks += 1

    def observe(self, native):
        a = self.audit
        with self.read():
            n = len(self.queries)+1
            assert n == len(a.queries) == len(a.handoff.queries) == len(a.resource.rows)
            assert n == len(a.cycle_capture.queries) == len(a.prepare_timing_capture.queries)
            assert n == len(a.request_capture.queries)
            q, old = a.queries[-1], a.handoff.queries[-1]
            assert native.shape == (1, 7) and int(native.argmax(dim=1).item()) == q['base_bin']
            joined = a.prepare_timing_capture.queries[-1]
            prepared = a.prepare_timing_capture.prepares[joined['prepare_index']]['mapped']
            cycle = a.cycle_capture.queries[-1]
            packet = query_packet(q, old, a.resource.rows[-1], cycle, prepared, joined['join'])
            packet['query_index'] = n-1
            self.queries.append(packet)
            if self.anchor is None and packet['eligible']:
                prepare = a.handoff.by_prepare_seq[q['prepare_event_seq']]
                self.anchor = anchor_snapshot(q, packet, prepare, prepared, cycle)
                if self.mode == 'forced_native_once':
                    self.commands.append(dict(query_index=n-1, query_seq=q['event_seq'],
                        native_bin=q['base_bin'], command_bin=q['base_bin'],
                        kind='engineering_native_passthrough', changed=False))
        return native

    def data(self, complete):
        return copy.deepcopy(dict(complete=complete, mode=self.mode, queries=self.queries,
            anchor=self.anchor, commands=self.commands, query_checks=self.checks,
            status='anchor_captured' if self.anchor is not None else 'missing_anchor',
            observer_native_call_delta=0, observer_rng_unchanged=True,
            future_switches=0, action_changes=0, selector_defined=False))


def recording_audit(base, mode):
    """Wrap the completed six-payload callback once; no attach hooks or runner imports."""
    assert mode in MODES
    class RecordingAudit(base):
        def __init__(self, seed, arm):
            assert arm == 'v107', 'V254 passive slots are c0/V107 only'
            super().__init__(seed, arm)
            self.pulse_capture = PulseCapture(self, mode)

        def __call__(self, policy, states, features, theta):
            native = super().__call__(policy, states, features, theta)
            return self.pulse_capture.observe(native)
    return RecordingAudit


def validate_capture(reference, recorded, data):
    assert set(reference) == set(recorded) == set(PAYLOADS)
    for name in PAYLOADS:
        require_legacy_equal(reference[name], recorded[name])
        exact_types(reference[name], recorded[name])
    timing.validate_capture({k: reference[k] for k in timing.OLD_PAYLOADS},
                            {k: recorded[k] for k in timing.OLD_PAYLOADS}, recorded['v248'])
    assert data['complete'] and data['mode'] in MODES
    assert data['observer_rng_unchanged'] is True
    assert data['observer_native_call_delta'] == data['future_switches'] == data['action_changes'] == 0
    assert data['selector_defined'] is False
    native_queries = recorded['legacy']['trace']['lower_actions']
    old, resources, cycles, times = (recorded[k] for k in ('v224', 'v230', 'v235', 'v248'))
    lengths = [len(x) for x in (native_queries, old['queries'], resources['query_extensions'],
                               cycles['queries'], times['queries'], data['queries'])]
    assert len(set(lengths)) == 1 and data['query_checks'] == lengths[0]
    expected_anchor = None
    prepares = {p['clock']['legacy_prepare_event_seq']: p for p in old['prepares']}
    for i, (handoff, resource, cycle, joined, saved) in enumerate(zip(
            old['queries'], resources['query_extensions'], cycles['queries'], times['queries'], data['queries'])):
        prepared = times['prepares'][joined['prepare_index']]['mapped']
        prepare = prepares[prepared['prepare_seq']]
        # Native action traces are not query snapshots; reconstruct from the old sidecars.
        q = dict(handoff['identity'], **{k: resource[k] for k in PAYLOAD_FIELDS},
            time_s=handoff['query_s'], event_seq=handoff['legacy_query_event_seq'],
            prepare_event_seq=prepared['prepare_seq'], state_captured_s=prepared['prepare_s'],
            context=prepare['context'], base_bin=handoff['native_bin'], feasible_bins=handoff['feasible_bins'])
        expected = dict(query_packet(q, handoff, resource, cycle, prepared, joined['join']), query_index=i)
        exact_types(expected, saved)
        if expected_anchor is None and expected['eligible']:
            expected_anchor = anchor_snapshot(q, expected, prepare, prepared, cycle)
    assert expected_anchor is not None, 'registered first qualified anchor absent; no replacement'
    exact_types(expected_anchor, data['anchor'])
    assert data['status'] == 'anchor_captured'
    anchor = expected_anchor['query']
    commands = [] if data['mode'] == 'observer_only' else [dict(query_index=anchor['query_index'],
        query_seq=anchor['clock']['query_seq'], native_bin=anchor['native_bin'],
        command_bin=anchor['native_bin'], kind='engineering_native_passthrough', changed=False)]
    exact_types(commands, data['commands'])
    return dict(six_payloads_exact=True, recording_complete=True, clock_domains_preserved=True,
        native_calls_and_rng_unchanged=True, queries=len(native_queries),
        anchor_query_index=anchor['query_index'], anchor_query_seq=anchor['clock']['query_seq'],
        command_records=len(commands), action_changes=0, future_switches=0)
