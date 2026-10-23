"""Fourth-sidecar observer for frozen V230; no rollout or action entrypoint."""

from collections import Counter
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

import preregister_freqduet_v232_persistent_coordination as frozen
from freqduet_v230_resource_capture import ResourceCapture, validate as validate_resource
from preregister_freqduet_v234_passive_cycle_preflight import (
    JOIN_FIELDS, PAIR_CHECKS, check_originals, check_queries)


class CycleCapture:
    def __init__(self, resource):
        self.resource = resource
        self.state = frozen.CycleState()
        self.metadata = None
        self.queries, self.transitions, self.links = [], [], []
        self.cycles, self.checks = {}, Counter()

    @contextmanager
    def read(self, label):
        calls = self.resource.audit.handoff.calls
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), calls.copy()
        try:
            yield
        finally:
            after = np.random.get_state()
            assert random.getstate() == before[0], 'cycle observer changed Python RNG'
            assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:], \
                'cycle observer changed NumPy RNG'
            assert torch.equal(before[2], torch.get_rng_state()), 'cycle observer changed Torch RNG'
            assert before[3] == calls, 'cycle observer invoked native computation'
            self.checks[label] += 1

    def attach(self):
        audit = self.resource.audit
        assert not audit.clock.events, 'cycle capture must start before native events'
        self.original_stamp, self.original_query = audit.clock.stamp, self.resource.query

        def stamp(kind, now, **identity):
            value = self.original_stamp(kind, now, **identity)
            if kind in ('launch', 'return'):
                with self.read(kind):
                    self.event(self.resource.cycle_events[-1])
            return value

        def query(q):
            value = self.original_query(q)
            with self.read('query'):
                self.query(self.resource.rows[-1], audit.handoff.queries[-1])
            return value

        audit.clock.stamp, self.resource.query = stamp, query

    def detach(self):
        self.resource.audit.clock.stamp = self.original_stamp
        self.resource.query = self.original_query

    def event(self, event):
        if self.metadata is None:
            self.metadata = dict(copy.deepcopy(self.resource.metadata), frozen_cycle_source='7fdf968a5f')
        e = event
        bus, trip, seq, now = e['bus_id'], e['trip_id'], e['native_event_seq'], e['time_s']
        if e['kind'] == 'launch':
            view = self.state.launch(bus, trip, e['direction'], seq, now)
            key = tuple(view['cycle_key'])
            self.cycles[key] = dict(cycle_key=list(key), direction=e['direction'], launch_s=now,
                queries=0, query_increment_s=0., after_last_query_s=None,
                executed_holding_s=None, return_s=None, return_event_seq=None, status='open')
            if view['observed_previous_cycle'] is not None:
                self.links.append(dict(observed_at_seq=seq, current_cycle=list(key),
                                       previous=copy.deepcopy(view['observed_previous_cycle'])))
        else:
            assert e['kind'] == 'return'
            key = tuple(self.state.active[bus]['cycle_key'])
            prefix = self.state.active[bus]['executed_holding_s']
            view = self.state.returned_to_terminal(bus, trip, seq, now, e['executed_holding_s'])
            cycle = self.cycles[key]
            remainder = view['executed_holding_s']-prefix
            assert remainder >= 0 and cycle['query_increment_s']+remainder == view['executed_holding_s']
            cycle.update(after_last_query_s=remainder, executed_holding_s=view['executed_holding_s'],
                         return_s=now, return_event_seq=seq, status='returned')
        self.transitions.append(dict(event=copy.deepcopy(e), view=copy.deepcopy(view)))

    def query(self, row, old):
        assert row['snapshot_read_s'] == old['query_s'] == self.resource.audit.env.current_time
        assert all(row[k] == old[k] for k in JOIN_FIELDS)
        view = self.state.query(row, self.metadata)
        self.queries.append(dict({k: old[k] for k in JOIN_FIELDS},
                                 identity=copy.deepcopy(old['identity']), view=copy.deepcopy(view)))
        cycle = self.cycles[tuple(view['cycle_key'])]
        cycle['queries'] += 1
        cycle['query_increment_s'] += view['delta_executed_since_last_sample_s']

    def data(self, complete):
        return copy.deepcopy(dict(complete=complete, metadata=self.metadata,
            queries=self.queries, transitions=self.transitions, observed_links=self.links,
            cycles=list(self.cycles.values()), terminal=dict(open_cycles=list(self.state.active.values()),
                returned_without_next_launch=list(self.state.returned.values())),
            observer_checks=dict(self.checks), observer_native_call_delta=0,
            observer_rng_unchanged=True, shadow_actions_executed=0))


def recording_audit(base):
    """V230's unchanged construction order, with the independent observer last."""
    class RecordingAudit(base):
        def __init__(self, seed, arm):
            super().__init__(seed, arm)
            self.resource = ResourceCapture(self)
            self.cycle_capture = CycleCapture(self.resource)

        def attach(self, runner):
            super().attach(runner)
            self.resource.attach()
            self.cycle_capture.attach()

        def __call__(self, policy, states, features, theta):
            value = super().__call__(policy, states, features, theta)
            self.resource.query(self.queries[-1])
            return value

        def detach(self):
            self.cycle_capture.detach()
            self.resource.detach()
            super().detach()
    return RecordingAudit


def validate_capture(reference, recorded, capture):
    """Independently reconstruct from original events, never observer transitions."""
    check_originals(reference, recorded)
    legacy, old, resource = (recorded[k] for k in ('legacy', 'v224', 'v230'))
    validate_resource(resource, old, legacy)
    assert capture['complete'] and capture['metadata'] == dict(resource['metadata'], frozen_cycle_source='7fdf968a5f')
    state = frozen.CycleState()
    events = {e['native_event_seq']: e for e in resource['cycle_events']}
    queries = {q['snapshot_event_seq']: q for q in resource['query_extensions']}
    assert len(events) == len(resource['cycle_events']) and len(queries) == len(resource['query_extensions'])
    expected_transitions, expected_views, expected_links, cycles = [], [], [], {}
    seen = set()
    for native in legacy['events']:
        kind, seq, now = native['kind'], native['event_seq'], native['time_s']
        if kind == 'lower_query':
            row = queries[seq]
            assert row['snapshot_read_s'] == now
            view = state.query(row, resource['metadata'])
            expected_views.append(view)
            cycles[tuple(view['cycle_key'])]['views'].append(view)
        elif kind in ('launch', 'return'):
            e = events[seq]
            assert (e['kind'], e['time_s'], e['bus_id'], e['trip_id']) == (kind, now, native['bus_id'], native['trip_id'])
            seen.add(seq)
            if kind == 'launch':
                view = state.launch(e['bus_id'], e['trip_id'], e['direction'], seq, now)
                key = tuple(view['cycle_key'])
                cycles[key] = dict(direction=e['direction'], launch_s=now, views=[], returned=None)
                if view['observed_previous_cycle'] is not None:
                    expected_links.append(dict(observed_at_seq=seq, current_cycle=list(key),
                                               previous=copy.deepcopy(view['observed_previous_cycle'])))
            else:
                key = tuple(state.active[e['bus_id']]['cycle_key'])
                view = state.returned_to_terminal(e['bus_id'], e['trip_id'], seq, now, e['executed_holding_s'])
                cycles[key]['returned'] = view
            expected_transitions.append(dict(event=e, view=view))
    assert seen == events.keys()
    check_queries(capture['queries'], resource['query_extensions'], old['queries'], expected_views)
    assert capture['transitions'] == expected_transitions and capture['observed_links'] == expected_links
    assert capture['terminal'] == dict(open_cycles=list(state.active.values()),
                                      returned_without_next_launch=list(state.returned.values()))
    expected_cycles = []
    for key, cycle in cycles.items():
        views, ret = cycle['views'], cycle['returned']
        total = sum(v['delta_executed_since_last_sample_s'] for v in views)
        assert total == (views[-1]['executed_holding_s'] if views else 0.)
        tail = None if ret is None else ret['executed_holding_s']-total
        assert tail is None or tail >= 0
        expected_cycles.append(dict(cycle_key=list(key), direction=cycle['direction'], launch_s=cycle['launch_s'],
            queries=len(views), query_increment_s=total, after_last_query_s=tail,
            executed_holding_s=None if ret is None else ret['executed_holding_s'],
            return_s=None if ret is None else ret['return_s'],
            return_event_seq=None if ret is None else ret['return_event_seq'],
            status='open' if ret is None else 'returned'))
    assert capture['cycles'] == expected_cycles
    closed = [c for c in expected_cycles if c['status'] == 'returned']
    assert sum(c['executed_holding_s'] for c in closed) == legacy['ledger']['executed_holding_s']
    expected_checks = dict(query=len(expected_views), launch=sum(e['event']['kind'] == 'launch' for e in expected_transitions),
                           **({'return': len(closed)} if closed else {}))
    assert {k: v for k, v in expected_checks.items() if v} == capture['observer_checks']
    assert capture['observer_native_call_delta'] == capture['shadow_actions_executed'] == 0
    assert capture['observer_rng_unchanged'] is True
    return dict({k: True for k in PAIR_CHECKS}, queries=len(expected_views), cycles=len(cycles),
        closed_cycles=len(closed), zero_query_cycles=sum(not c['queries'] for c in expected_cycles),
        open_cycles=len(state.active), returned_without_next_launch=len(state.returned),
        observed_links=len(expected_links), executed_holding_s=sum(c['executed_holding_s'] for c in closed))
