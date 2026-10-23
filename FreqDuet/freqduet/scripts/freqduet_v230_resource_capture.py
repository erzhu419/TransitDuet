"""Independent, read-only extension of the frozen V224 capture."""

from collections import Counter
from contextlib import contextmanager
import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import torch

from preregister_freqduet_v229_supply_coordination import ACTIONS, local_gap_cost_change
from preregister_freqduet_v230_resource_capture import (
    LINK_FIELDS, PAYLOAD_FIELDS, check_cycle_progress, check_query_link)


def frontier(q):
    ctx = q['context']
    status = ('terminal_masked' if q['terminal_masked'] else
              'missing_forward' if not ctx['evidence_valid'] else
              'missing_follower' if not ctx['follower_evidence_valid'] else 'supported')
    native = ACTIONS[q['base_bin']]
    return dict(status=status, native_bin=q['base_bin'], candidates=[dict(bin=b,
        holding_s=ACTIONS[b], delta_holding_s=ACTIONS[b]-native,
        immediate_onboard_delta_person_s=q['onboard']*(ACTIONS[b]-native),
        local_gap_cost_delta_s2=(local_gap_cost_change(ctx['forward_headway_s'],
            ctx['follower_departure_gap_s'], ACTIONS[b], native) if status == 'supported' else None))
        for b in q['feasible_bins']])


def ready_coverage(row, metadata):
    """Conditional next-pass coverage under the currently posted plans only."""
    dt, now = metadata['native_dt_s'], row['snapshot_read_s']
    tick = (math.floor(now/dt)+1)*dt
    rank = {bus: i for i,bus in enumerate(metadata['native_inventory_order'])}
    stock = {}
    for direction in (False, True):
        ready = [b for b in row['ready'] if b['direction'] == direction and
                 (not b['return_observed'] or b['last_observed_return_s']+dt <= tick)]
        ready.sort(key=lambda b: (b['last_observed_return_s'] if b['return_observed'] else -math.inf,
                                  rank[b['bus_id']]))
        stock[direction] = ready
    covered, due, uncovered = {}, {'false': [], 'true': []}, {'false': [], 'true': []}
    for plan in row['plans']:
        if math.ceil(plan['planned_s']/dt)*dt > tick:
            continue
        direction = bool(plan['direction'])
        name = str(direction).lower()
        due[name].append(plan['trip_id'])
        if stock[direction]:
            covered[str(plan['trip_id'])] = stock[direction].pop(0)['bus_id']
        else:
            uncovered[name].append(plan['trip_id'])
    return dict(next_dispatch_s=tick, due=due, uncovered=uncovered, covered_by_ready_bus=covered,
        interpretation='Current-plan/current-inventory conditional coverage,not a reservation or forecast.')


class ResourceCapture:
    def __init__(self, audit):
        self.audit = audit
        self.metadata = None
        self.versions = {'initial': dict(version_id='initial', initial=True,
            native_event_seq=None, time_s=0., direction=None, fresh_upper_event_seq=None,
            callback_trip_id=None, changes=[])}
        self.latest = {'false': 'initial', 'true': 'initial'}
        self.plan_versions, self.plans = {}, {}
        self.launches, self.open_cycles, self.last_returns, self.previous = {}, {}, {}, {}
        self.rows, self.cycle_events, self.outcomes = [], [], []
        self.pending_fresh, self.original_stamp = None, None
        self.checks, self.stats, self.support, self.bin_counts = Counter(), Counter(), Counter(), Counter()

    @contextmanager
    def read(self, label):
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), self.audit.handoff.calls.copy()
        yield
        after = np.random.get_state()
        assert random.getstate() == before[0], ('extension changed Python RNG', label)
        assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:]
        assert torch.equal(before[2], torch.get_rng_state()), ('extension changed Torch RNG', label)
        assert before[3] == self.audit.handoff.calls, ('extension invoked native computation', label)
        self.checks[label] += 1

    def initialize(self):
        env = self.audit.env
        assert len(env.bus_all) == 12 and not self.audit.clock.events[:-1]
        assert not any(t.launched for t in env.timetables), 'initial plans were not observed before launch'
        plans = [dict(trip_id=int(t.launch_turn), direction=bool(t.direction), launched=False,
            planned_s=float(getattr(t, '_freqduet_scheduled_launch', t.launch_time)),
            public_s=float(getattr(t, '_freqduet_base_launch', getattr(t, '_original_launch', t.launch_time))),
            target_s=float(t.target_headway)) for t in env.timetables]
        self.metadata = dict(native_dt_s=float(env.time_step),
            native_inventory_order=[int(b.bus_id) for b in env.bus_all],
            native_timetable_order=[p['trip_id'] for p in plans],
            initial_published_plans=copy.deepcopy(plans), initial_version_id='initial')
        self.plans = {p['trip_id']: p for p in plans}
        self.plan_versions = dict.fromkeys(self.plans, 'initial')

    def attach(self):
        self.original_stamp = self.audit.clock.stamp
        def stamp(kind, now, **identity):
            value = self.original_stamp(kind, now, **identity)
            if kind in ('fresh_upper', 'plan_update', 'launch', 'return'):
                with self.read(kind):
                    if self.metadata is None:
                        self.initialize()
                    self.event(self.audit.clock.events[-1])
            return value
        self.audit.clock.stamp = stamp

    def detach(self):
        self.audit.clock.stamp = self.original_stamp

    def event(self, event):
        kind, seq, trip = event['kind'], event['event_seq'], event.get('trip_id')
        if kind == 'fresh_upper':
            assert self.pending_fresh is None
            self.pending_fresh = dict(event)
        elif kind == 'plan_update':
            callback = self.audit.callbacks[-1]
            assert (callback['trip_id'], callback['time_s']) == (trip, event['time_s'])
            fresh = self.pending_fresh
            if fresh is not None:
                assert (fresh['trip_id'], fresh['direction'], fresh['time_s']) == (
                    trip, callback['direction'], callback['time_s'])
            version = dict(version_id=seq, initial=False, native_event_seq=seq,
                time_s=event['time_s'], direction=callback['direction'], callback_trip_id=trip,
                fresh_upper_event_seq=None if fresh is None else fresh['event_seq'],
                changes=copy.deepcopy(callback['changes']))
            for changed in version['changes']:
                tid = changed['trip_id']
                assert self.plans[tid] == changed['before'], 'plan before-image differs from observed prefix'
                self.plans[tid] = copy.deepcopy(changed['after'])
                self.plan_versions[tid] = seq
            self.versions[seq] = version
            self.latest[str(bool(callback['direction'])).lower()] = seq
            self.pending_fresh = None
        elif kind == 'launch':
            actual = self.audit.dispatches[-1]
            bus = event['bus_id']
            assert (actual['trip'], actual['bus'], actual['actual_s']) == (trip, bus, event['time_s'])
            assert bus not in self.open_cycles and trip not in self.launches
            self.launches[trip] = dict(event)
            self.open_cycles[bus] = [bus, trip, seq]
            self.cycle_events.append(dict(kind='launch', native_event_seq=seq, time_s=event['time_s'],
                bus_id=bus, trip_id=trip, direction=actual['direction'],
                launch_native_event_seq=seq, executed_holding_s=0.))
            parent = self.last_returns.pop(bus, None)
            self.outcomes.append(dict(bus_id=bus, inbound_trip_id=None if parent is None else parent['trip_id'],
                inbound_return_event_seq=None if parent is None else parent['native_event_seq'],
                next_trip_id=trip, next_launch_event_seq=seq,
                status='initial_stock' if parent is None else 'first_observed_same_bus_launch'))
            self.plans.pop(trip)
            self.plan_versions.pop(trip)
        elif kind == 'return':
            ret = self.audit.returns[-1]
            bus = event['bus_id']
            assert (ret['trip_id'], ret['bus_id'], ret['return_s']) == (trip, bus, event['time_s'])
            cycle = self.open_cycles.pop(bus)
            assert cycle[:2] == [bus, trip]
            row = dict(kind='return', native_event_seq=seq, time_s=event['time_s'], bus_id=bus,
                trip_id=trip, direction=ret['inbound_direction'], launch_native_event_seq=cycle[2],
                executed_holding_s=ret['holding_s'], account=copy.deepcopy(ret))
            self.cycle_events.append(row)
            self.last_returns[bus] = row

    def query(self, q):
        with self.read('query'):
            old = self.audit.handoff.queries[-1]
            prepare = self.audit.handoff.by_prepare_seq[q['prepare_event_seq']]
            assert q['time_s'] == old['query_s'] == self.audit.env.current_time
            assert q['context'] == prepare['context'] and q['state_captured_s'] == prepare['clock']['prepare_s']
            assert q['plans'] == list(self.plans.values()), 'query plans do not reconstruct from actual writes'
            row = {k: copy.deepcopy(q[k]) for k in PAYLOAD_FIELDS}
            row.update({k: old[k] for k in LINK_FIELDS})
            row.update(cycle_ref=list(self.open_cycles[q['bus_id']]), message_refs=dict(self.latest),
                pending_plan_version_refs={str(p['trip_id']): self.plan_versions[p['trip_id']] for p in q['plans']},
                shadow_frontier=frontier(q))
            assert all(row[k] == q[k] for k in PAYLOAD_FIELDS)
            check_query_link(row, old, prepare, self.launches[q['trip_id']], self.versions)
            if q['bus_id'] in self.previous:
                check_cycle_progress(self.previous[q['bus_id']], row)
            self.previous[q['bus_id']] = row
            ranks = {t: i for i,t in enumerate(self.metadata['native_timetable_order'])}
            assert [ranks[p['trip_id']] for p in row['plans']] == sorted(ranks[p['trip_id']] for p in row['plans'])
            inv = {b: i for i,b in enumerate(self.metadata['native_inventory_order'])}
            for entries in (row['ready'], row['active']):
                assert [inv[b['bus_id']] for b in entries] == sorted(inv[b['bus_id']] for b in entries)
            for b in row['ready']:
                assert not b['return_observed'] or b['last_observed_return_s'] <= q['time_s']
            coverage = ready_coverage(row, self.metadata)
            row['ready_coverage'] = coverage
            assert len(coverage['covered_by_ready_bus'].values()) == len(set(coverage['covered_by_ready_bus'].values()))
            target = str(not bool(q['direction'])).lower()
            self.stats['pressure_queries'] += bool(coverage['uncovered'][target])
            self.stats['pressure_positive_native_overlap'] += bool(coverage['uncovered'][target]) and ACTIONS[q['base_bin']] > 0
            self.stats['positive_native_queries'] += ACTIONS[q['base_bin']] > 0
            self.stats['masked_queries'] += bool(q['terminal_masked'])
            self.support[row['shadow_frontier']['status']] += 1
            self.bin_counts[str(len(q['feasible_bins']))] += 1
            self.rows.append(row)

    def data(self, complete):
        return dict(complete=complete, metadata=self.metadata, query_extensions=self.rows,
            upper_versions=list(self.versions.values()), cycle_events=self.cycle_events,
            supply_outcomes=self.outcomes,
            terminal_cycles=[dict(bus_id=b, trip_id=r['trip_id'], status='returned_no_later_launch',
                native_return_event_seq=r['native_event_seq']) for b,r in self.last_returns.items()],
            open_cycles=list(self.open_cycles.values()), observer_checks=dict(self.checks),
            counts=dict(self.stats), shadow_support=dict(self.support), feasible_bin_counts=dict(self.bin_counts),
            observer_native_call_delta=0, observer_rng_unchanged=True, shadow_actions_executed=0)


def validate(data, sidecar, legacy):
    assert data['complete'] and data['metadata'] is not None
    rows, old = data['query_extensions'], sidecar['queries']
    assert len(rows) == len(old) == len(legacy['trace']['lower_actions'])
    versions = {v['version_id']: v for v in data['upper_versions']}
    prepares = {p['clock']['handoff_order']: p for p in sidecar['prepares']}
    launches = {e['trip_id']: e for e in legacy['events'] if e['kind'] == 'launch'}
    callbacks = iter(legacy['trace']['upper_callbacks'])
    for event in legacy['events']:
        if event['kind'] == 'plan_update':
            callback = next(callbacks)
            assert versions[event['event_seq']]['changes'] == callback['changes']
    previous = {}
    for row, query in zip(rows, old):
        trip = query['identity']['trip_id']
        check_query_link(row, query, prepares[query['prepare_order']], launches[trip], versions)
        assert row['ready_coverage'] == ready_coverage(row, data['metadata'])
        bus = query['identity']['bus_id']
        if bus in previous:
            check_cycle_progress(previous[bus], row)
        previous[bus] = row
    returns = {r['trip_id']: r for r in legacy['trace']['vehicle_returns']}
    cycle_returns = {r['trip_id']: r for r in data['cycle_events'] if r['kind'] == 'return'}
    assert returns.keys() == cycle_returns.keys()
    for trip, ret in returns.items():
        assert cycle_returns[trip]['account'] == ret
        assert ret['return_s']-ret['launch_s'] == ret['holding_s']+ret['other_elapsed_s']
    assert sum(r['holding_s'] for r in returns.values()) == legacy['ledger']['executed_holding_s']
    active = {}
    expected = []
    for e in legacy['events']:
        if e['kind'] == 'return':
            active[e['bus_id']] = e
        elif e['kind'] == 'launch':
            parent = active.pop(e['bus_id'], None)
            expected.append((e['bus_id'], None if parent is None else parent['trip_id'], e['trip_id']))
    assert expected == [(r['bus_id'], r['inbound_trip_id'], r['next_trip_id']) for r in data['supply_outcomes']]
    assert len(data['cycle_events']) == len(launches)+len(returns)
    assert data['observer_native_call_delta'] == data['shadow_actions_executed'] == 0
    assert data['observer_rng_unchanged'] and data['observer_checks']['query'] == len(rows)
    return dict(query_payload_complete=True, query_prepare_clocks_valid=True,
        upper_versions_strict_prefix=True, physical_cycles_valid=True,
        ready_and_plan_orders_preserved=True, ready_dispatch_boundary_valid=True,
        observer_rng_and_native_calls_unchanged=True, shadow_never_executed=True,
        queries=len(rows), upper_versions=len(versions)-1, completed_cycles=len(returns),
        remaining_returned_cycles=len(data['terminal_cycles']), open_cycles=len(data['open_cycles']),
        stats=data['counts'], shadow_support=data['shadow_support'], feasible_bin_counts=data['feasible_bin_counts'])


def dry_archive_prefix(legacy):
    """Check observable event/plan interfaces before spending a passive path."""
    first = legacy['fresh_upper'][0]
    assert first['event_seq'] == 0 and not first['active'] and len(first['ready']) == 12
    audit = SimpleNamespace(callbacks=[], dispatches=[], returns=[])
    capture = ResourceCapture(audit)
    capture.metadata = dict(native_dt_s=legacy['physical_contract']['dt_s'],
        native_inventory_order=[b['bus_id'] for b in first['ready']],
        native_timetable_order=[p['trip_id'] for p in first['plans']],
        initial_published_plans=copy.deepcopy(first['plans']), initial_version_id='initial')
    capture.plans = {p['trip_id']: copy.deepcopy(p) for p in first['plans']}
    capture.plan_versions = dict.fromkeys(capture.plans, 'initial')
    snapshots = {q['event_seq']: q for q in legacy['fresh_upper']}
    callbacks = iter(legacy['trace']['upper_callbacks'])
    launches = {r['trip']: r for r in legacy['trace']['actual_launches']}
    returns = {r['trip_id']: r for r in legacy['trace']['vehicle_returns']}
    for event in legacy['events']:
        kind = event['kind']
        if kind == 'fresh_upper':
            assert list(capture.plans.values()) == snapshots[event['event_seq']]['plans'], \
                'posted plans have writes missing from callback history'
        elif kind == 'plan_update':
            audit.callbacks.append(next(callbacks))
        elif kind == 'launch':
            audit.dispatches.append(launches[event['trip_id']])
        elif kind == 'return':
            audit.returns.append(returns[event['trip_id']])
        if kind in ('fresh_upper', 'plan_update', 'launch', 'return'):
            capture.event(event)
    assert not capture.open_cycles and not capture.plans
    return dict(fresh_snapshots=len(snapshots), callbacks=len(audit.callbacks),
        physical_launches=len(audit.dispatches), physical_returns=len(audit.returns),
        replay_paths=0, all_archived_plan_prefixes_reconstructed=True)
