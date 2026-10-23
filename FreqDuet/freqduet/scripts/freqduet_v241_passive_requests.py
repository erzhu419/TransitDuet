"""Independent fifth sidecar for V240; the native action callback is inherited."""

from collections import Counter
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from preregister_freqduet_v224_prepare_handoff import require_legacy_equal
from preregister_freqduet_v234_passive_cycle_preflight import JOIN_FIELDS
from preregister_freqduet_v238_persistent_requests import shadow_request
from preregister_freqduet_v240_paired_coordination import blind_request


SOURCES = dict(protocol='e6788e6109', supply='b4b1d00203', blind='e6788e6109',
               cycle_capture='6c3397ff7f')
ORIGINALS = ('legacy', 'v224', 'v230', 'v235')


def request_packet(index, row, cycle_query, prepare):
    view, identity = cycle_query['view'], cycle_query['identity']
    assert all(row[k] == cycle_query[k] for k in JOIN_FIELDS)
    assert prepare['clock']['handoff_order'] == row['prepare_order'] < row['query_order']
    assert prepare['clock']['prepare_s'] <= row['snapshot_read_s'] == view['query_s']
    assert all(prepare['current'][k] == identity[k] for k in
               ('bus_id', 'trip_id', 'direction', 'station_id'))
    assert view['cycle_key'][:2] == [identity['bus_id'], identity['trip_id']]
    supply = shadow_request(view, row)
    blind = blind_request(view['executed_holding_s'], row['shadow_frontier'])
    supply_h0 = shadow_request(dict(view, executed_holding_s=0),
                               dict(row, executed_holding_before_s=0))
    blind_h0 = blind_request(0, row['shadow_frontier'])
    native = row['shadow_frontier']['native_bin']
    assert supply_h0['shadow_bin'] == blind_h0['shadow_bin'] == native
    return copy.deepcopy(dict(query_index=index, **{k: row[k] for k in JOIN_FIELDS},
        identity=identity, prepare_s=prepare['clock']['prepare_s'], query_s=view['query_s'],
        native_event_seq=view['native_event_seq'], cycle_key=view['cycle_key'],
        executed_holding_before_s=view['executed_holding_s'],
        message_refs=view['message_refs'], pending_plan_version_refs=view['pending_plan_version_refs'],
        own_association=view['own_association'], native_bin=native,
        supply_request=supply, blind_request=blind,
        H0_bins=dict(supply=supply_h0['shadow_bin'], blind=blind_h0['shadow_bin']),
        returned_rule='native', executed_shadow=False))


class RequestCapture:
    def __init__(self, cycle_capture, arm):
        self.cycle = cycle_capture
        self.resource = cycle_capture.resource
        self.arm, self.queries, self.checks = arm, [], Counter()

    @contextmanager
    def read(self):
        calls = self.resource.audit.handoff.calls
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), calls.copy()
        try:
            yield
        finally:
            after = np.random.get_state()
            assert random.getstate() == before[0], 'request observer changed Python RNG'
            assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:], \
                'request observer changed NumPy RNG'
            assert torch.equal(before[2], torch.get_rng_state()), 'request observer changed Torch RNG'
            assert calls == before[3], 'request observer invoked native computation'
            self.checks['query'] += 1

    def attach(self):
        audit = self.resource.audit
        assert not audit.clock.events, 'request observer must start before native events'
        self.original_query = self.resource.query

        def query(q):
            value = self.original_query(q)
            with self.read():
                assert len(self.resource.rows) == len(self.cycle.queries) == len(self.queries)+1
                row, cycle_query, old = self.resource.rows[-1], self.cycle.queries[-1], audit.handoff.queries[-1]
                assert all(row[k] == old[k] for k in JOIN_FIELDS)
                assert cycle_query['identity'] == old['identity']
                prepare = audit.handoff.by_prepare_seq[q['prepare_event_seq']]
                self.queries.append(request_packet(len(self.queries), row, cycle_query, prepare))
            return value

        self.resource.query = query

    def detach(self):
        self.resource.query = self.original_query

    def data(self, complete):
        return copy.deepcopy(dict(complete=complete, sources=SOURCES, engine_arm=self.arm,
            queries=self.queries, observer_checks=dict(self.checks),
            observer_native_call_delta=0, observer_rng_unchanged=True,
            returned_rule='native', shadow_actions_executed=0))


def recording_audit(base):
    """Append to the frozen V235 recording class; never override its __call__."""
    class RecordingAudit(base):
        def __init__(self, seed, arm):
            assert arm in ('v107', 'upper'), 'passive V240 uses only frozen V107/upper arms'
            super().__init__(seed, arm)
            self.request_capture = RequestCapture(self.cycle_capture, arm)

        def attach(self, runner):
            super().attach(runner)
            self.request_capture.attach()

        def detach(self):
            self.request_capture.detach()
            super().detach()
    return RecordingAudit


def validate_capture(reference, recorded, requests):
    """Recompute requests from unchanged old sidecars,not the new observer rows."""
    assert set(reference) == set(recorded) == set(ORIGINALS)
    for name in ORIGINALS:
        require_legacy_equal(reference[name], recorded[name])
    assert requests['complete'] and requests['sources'] == SOURCES
    assert requests['engine_arm'] in ('v107', 'upper')
    rows, cycles, handoff = (recorded[name] for name in ('v230', 'v235', 'v224'))
    resource_rows, cycle_rows, old_rows = rows['query_extensions'], cycles['queries'], handoff['queries']
    assert len(requests['queries']) == len(resource_rows) == len(cycle_rows) == len(old_rows)
    prepares = {p['clock']['handoff_order']: p for p in handoff['prepares']}
    assert len(prepares) == len(handoff['prepares'])
    for i, (row, cycle, old) in enumerate(zip(resource_rows, cycle_rows, old_rows)):
        assert all(row[k] == old[k] for k in JOIN_FIELDS) and cycle['identity'] == old['identity']
        expected = request_packet(i, row, cycle, prepares[row['prepare_order']])
        require_legacy_equal(expected, requests['queries'][i])
    expected_checks = dict(query=len(old_rows)) if old_rows else {}
    assert requests['observer_checks'] == expected_checks
    assert requests['observer_native_call_delta'] == requests['shadow_actions_executed'] == 0
    assert requests['observer_rng_unchanged'] and requests['returned_rule'] == 'native'
    return dict(original_payloads_exact=True, v235_sidecar_exact=True,
        queries=len(old_rows), request_occurrences_complete=True, prepared_clocks_preserved=True,
        frozen_requests_exact=True, H0_aliases_native=True,
        observer_rng_and_calls_unchanged=True, shadow_never_executed=True)
