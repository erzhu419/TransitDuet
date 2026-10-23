"""Passive prepare-time sidecar for the frozen V221 controller."""

from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import asdict
import importlib
import math
import random
from types import SimpleNamespace

import numpy as np
import torch

import freqduet_v223_common_reference as shadow
from preregister_freqduet_v224_prepare_handoff import registration


FIELDS = registration('schema')['prepare_fields']


def empty_phase(reason):
    return dict(dict.fromkeys(FIELDS['role_phase']), status=reason, phase_s=None)


def seal_launch_phase(basis, trip, bus, direction, plan, actual, order, dt):
    basis = basis or dict(native_plan_s=plan, proposed_plan_s=plan, applied_plan_s=plan,
                          admission_order=None, admission_s=0., reserved_bus_id=None)
    row = dict(basis, trip_id=trip, bus_id=bus, direction=direction, actual_bus_id=bus,
        actual_launch_s=actual, plan_at_launch_s=plan, launch_order=order, launch_event_seq=order)
    row['native_tick_s'] = max(basis['admission_s'], math.ceil(basis['native_plan_s']/dt)*dt)
    row['coordinated_tick_s'] = max(basis['admission_s'], math.ceil(basis['applied_plan_s']/dt)*dt)
    if basis.get('invalidated') or plan != basis['applied_plan_s']:
        row.update(status='undelivered_plan', phase_s=None)
    else:
        value = shadow.launch_phase(row, (trip, bus, direction), order+1)
        row.update(status=('undelivered_plan' if value is None else
                           'known_zero' if value == 0. else 'supported'), phase_s=value)
    return row


def cached_context(bus):
    def optional(value, positive=False):
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) and (value > 0. if positive else value >= 0.) else None
    f = optional(bus.pre_action_forward_headway)
    b = optional(bus.pre_action_follower_departure_gap)
    h = optional(bus._target_headway, True)
    fs = str(bus.pre_action_forward_headway_source or 'unavailable').strip().lower()
    bs = str(bus.pre_action_follower_source or 'unavailable').strip().lower()
    return dict(forward_headway_s=f, target_headway_s=h, evidence_source=fs,
        evidence_valid=fs == 'matched_departure_event' and f is not None and h is not None,
        follower_departure_gap_s=b, follower_evidence_source=bs,
        follower_evidence_valid=bs.startswith('same_time_avl_') and b is not None)


def check_packet(packet):
    for section in ('clock', 'current', 'native', 'front', 'follower'):
        assert set(FIELDS[section]) <= packet[section].keys(), ('missing field', section)
    clock, own = packet['clock'], packet['current']
    now, order = clock['prepare_s'], clock['handoff_order']
    assert now == clock['record_read_s'] == clock['pre_action_time_s']
    if packet['native']['forward_valid']:
        front = packet['front']
        assert front['status'] == 'supported' and front['direction'] == own['direction']
        assert front['station_id'] == own['station_id'] and front['departure_order'] < order
        assert now-front['departure_s'] == packet['native']['forward_gap_s']
    if packet['native']['follower_valid']:
        follower = packet['follower']
        assert follower['status'] == 'supported' and follower['direction'] == own['direction']
        assert follower['avl_sample_s'] == now and follower['avl_sample_order'] < order
    for role in ('front', 'current', 'follower'):
        phase = packet['phases'][role]
        assert set(FIELDS['role_phase']) <= phase.keys(), ('missing phase field', role)
        if phase['status'] in ('supported', 'known_zero'):
            assert phase['launch_order'] < order
            assert all(phase[k] == packet[role][k] for k in ('trip_id', 'bus_id', 'direction'))
            value = shadow.launch_phase(phase, tuple(packet[role][k] for k in
                ('trip_id', 'bus_id', 'direction')), order)
            assert value == phase['phase_s'] and value is not None
        else:
            assert phase['phase_s'] is None


class PrepareCapture:
    def __init__(self, audit):
        self.audit = audit
        self.events, self.prepares, self.queries, self.samples = [], [], [], []
        self.launches, self.departures, self.plans = {}, {}, {}
        self.by_prepare_seq, self.prepare_counts, self.query_counts = {}, Counter(), Counter()
        self.calls, self.checked, self.patches = Counter(), Counter(), []
        self.avl_objects, self.estimates, self.pending_projection = [], {}, None
        self.sample_time, self.policies = None, set()

    def mark(self, kind, **values):
        row = dict(order=len(self.events), kind=kind, time_s=float(self.env.current_time), **values)
        assert not self.events or row['time_s'] >= self.events[-1]['time_s'], 'sidecar clock reversed'
        self.events.append(row)
        return row['order']

    @contextmanager
    def read(self, label):
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), self.calls.copy()
        yield
        after = np.random.get_state()
        assert random.getstate() == before[0], ('observer changed Python RNG', label)
        assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:]
        assert torch.equal(before[2], torch.get_rng_state()), ('observer changed Torch RNG', label)
        assert before[3] == self.calls, ('observer invoked native computation', label)
        self.checked[label] += 1

    def watch(self, owner, name, kind, after=None, before=None):
        original = getattr(owner, name)
        def watched(*args, **kwargs):
            self.calls[kind] += 1
            if before is not None:
                with self.read(kind+'_before'):
                    before(args, kwargs)
            value = original(*args, **kwargs)
            if after is not None:
                with self.read(kind):
                    after(value, args, kwargs)
            return value
        self.patches.append((owner, name, original))
        setattr(owner, name, watched)

    def attach(self, runner, legacy):
        from env.bus import Bus
        from env.scenario import ScenarioTape
        self.runner, self.env = runner, runner.env
        bus_module = importlib.import_module(Bus.__module__)
        sim_module = importlib.import_module(type(self.env).__module__)
        self.watch(sim_module, 'freeze_avl_vehicle_snapshots', 'avl_tick',
                   lambda v, a, k: self.sample(v, 'native_tick_start'))
        self.watch(bus_module, 'freeze_avl_vehicle_snapshots', 'avl_inline',
                   lambda v, a, k: self.sample(v, 'native_inline_fallback'))
        self.watch(bus_module, 'estimate_follower_departure_gap', 'estimator', self.estimate)
        self.watch(Bus, '_prepare_for_action', 'prepare', self.prepare)
        self.watch(type(self.env.headway_events), 'record_departure', 'departure', self.departure)
        self.watch(legacy.supply, 'project', 'supply_projection', self.projection)
        self.watch(runner.timetable_planner, 'apply', 'planner', self.admission)
        self.watch(self.env, 'launch_bus', 'launch', self.launch)
        self.watch(self.env, 'step', 'step', before=lambda a, k: self.mark('tick'))
        self.watch(runner.upper_trainer.policy_net, 'get_action', 'upper_policy')
        self.watch(legacy.pilot.event.EventUniforms, 'draw', 'sampler')
        # Every ScenarioTape draw obtains its RNG through these two native methods.
        self.watch(ScenarioTape, '_rng', 'scenario_scalar_rng')
        self.watch(ScenarioTape, '_stream_rng', 'scenario_stream_rng')

    def detach(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def bind_policy(self, policy):
        if id(policy) not in self.policies:
            self.policies.add(id(policy))
            self.watch(policy, 'forward', 'lower_policy')

    def sample(self, vehicles, origin):
        now = float(self.env.current_time)
        if self.sample_time != now:
            self.avl_objects.clear()
            self.sample_time = now
        order = self.mark('avl_sample', origin=origin)
        row = dict(order=order, time_s=now, origin=origin, vehicles=[asdict(v) for v in vehicles])
        self.samples.append(row)
        self.avl_objects.append((vehicles, row))

    def estimate(self, value, args, kwargs):
        snapshot = next((r for v, r in self.avl_objects if v is kwargs['vehicles']), None)
        assert snapshot is not None, 'native estimator snapshot was not captured'
        key = int(kwargs['current_bus_id'])
        self.estimates[key] = dict(time_s=float(self.env.current_time),
            order=self.mark('follower_estimate', bus_id=key), snapshot=snapshot, result=asdict(value))

    def projection(self, value, args, kwargs):
        _, query, _, native, _ = args
        solution, _ = value
        self.pending_projection = dict(query_seq=query['event_seq'], native=copy.deepcopy(native),
                                       proposed=copy.deepcopy(solution))

    def admission(self, value, args, kwargs):
        query = self.audit.fresh_queries[-1]
        order = self.mark('plan_admission', trip_id=query['trip_id'])
        projection = self.pending_projection
        projected = projection is not None and projection['query_seq'] == query['event_seq']
        applied = projected and self.audit.cycles[-1]['applied']
        resources = self.audit.cycles[-1]['tasks'] if projected else []
        reserved = {t['trip_id']: t['reserved_bus_id'] for t in resources}
        timetables = {int(t.launch_turn): t for t in self.env.timetables}
        for pending in query['plans']:
            trip = pending['trip_id']
            actual = float(getattr(timetables[trip], '_freqduet_scheduled_launch', timetables[trip].launch_time))
            native = projection['native'][trip]['planned_s'] if projected else actual
            proposed = projection['proposed'][trip]['planned_s'] if projected else actual
            assert actual == (proposed if applied else native)
            previous = self.plans.get(trip)
            invalidated = bool(previous and previous['applied_plan_s'] != previous['native_plan_s']
                               and actual == native)
            self.plans[trip] = dict(admission_order=order, admission_s=float(self.env.current_time),
                legacy_admission_event_seq=query['event_seq'], native_plan_s=native,
                proposed_plan_s=proposed, applied_plan_s=actual, reserved_bus_id=reserved.get(trip),
                invalidated=invalidated)

    def launch(self, bus, args, kwargs):
        if bus is None:
            return
        trip = args[0]
        key = int(trip.launch_turn)
        order = self.mark('launch', trip_id=key, bus_id=int(bus.bus_id), direction=bool(trip.direction))
        assert key not in self.launches, 'duplicate trip launch'
        plan = float(getattr(trip, '_freqduet_scheduled_launch', trip.launch_time))
        self.launches[key] = seal_launch_phase(self.plans.get(key), key, int(bus.bus_id),
            bool(trip.direction), plan, float(self.env.current_time), order, self.env.time_step)

    def departure(self, value, args, kwargs):
        _, station, direction, now, trip = args
        phase = self.launches[int(trip)]
        self.departures[int(trip), int(station), bool(direction)] = dict(status='supported',
            trip_id=int(trip), bus_id=phase['bus_id'], direction=bool(direction), station_id=int(station),
            departure_s=float(now), departure_order=self.mark('departure', trip_id=int(trip),
                station_id=int(station), direction=bool(direction)), launch_order=phase['launch_order'])

    def phase(self, identity, reason):
        if identity['trip_id'] is None or identity['bus_id'] is None:
            return empty_phase(reason)
        row = self.launches.get(identity['trip_id'])
        if row is None:
            return empty_phase('missing_launch')
        if any(row[k] != identity[k] for k in ('trip_id', 'bus_id', 'direction')):
            return empty_phase('identity_unresolved')
        return copy.deepcopy(row)

    def prepare(self, value, args, kwargs):
        bus, now = args[:2]
        identity = dict(trip_id=int(bus.trip_id), bus_id=int(bus.bus_id),
                        direction=bool(bus.direction), station_id=int(bus.last_station.station_id))
        key = tuple(identity[k] for k in ('bus_id', 'trip_id', 'station_id', 'direction'))
        legacy = self.audit.observed_ready[key]
        ctx = cached_context(bus)
        order = self.mark('prepare', **identity)
        clock = dict(prepare_s=float(now), record_read_s=float(self.env.current_time),
            pre_action_time_s=float(bus.pre_action_time_s), handoff_order=order,
            legacy_prepare_event_seq=legacy['event_seq'], prepare_occurrence=self.prepare_counts[key])
        self.prepare_counts[key] += 1
        front = dict(dict.fromkeys(FIELDS['front']), status='no_native_front')
        if ctx['evidence_valid']:
            trip = int(bus.forward_predecessor_trip_id)
            front = copy.deepcopy(self.departures[trip, identity['station_id'], identity['direction']])
        estimate = self.estimates.get(identity['bus_id'])
        follower = dict(dict.fromkeys(FIELDS['follower']), status='no_native_follower')
        if estimate is not None and estimate['time_s'] == now:
            snap = estimate['snapshot']
            follower.update(avl_sample_s=snap['time_s'], avl_sample_order=snap['order'],
                            avl_sample_origin=snap['origin'], estimate_order=estimate['order'])
        if ctx['follower_evidence_valid']:
            assert estimate is not None and estimate['time_s'] == now
            fid, bid = int(bus.pre_action_follower_trip_id), int(bus.pre_action_follower_bus_id)
            assert (fid, bid) == (estimate['result']['follower_trip_id'], estimate['result']['follower_bus_id'])
            frozen = next(v for v in estimate['snapshot']['vehicles'] if v['bus_id'] == bid)
            assert frozen['on_route'] and frozen['trip_id'] == fid and frozen['direction'] == identity['direction']
            follower.update(status='supported', trip_id=fid, bus_id=bid, direction=identity['direction'],
                            launch_order=self.launches[fid]['launch_order'])
        native = dict(forward_gap_s=ctx['forward_headway_s'], forward_valid=ctx['evidence_valid'],
            forward_source=ctx['evidence_source'], follower_gap_s=ctx['follower_departure_gap_s'],
            follower_valid=ctx['follower_evidence_valid'], follower_source=ctx['follower_evidence_source'],
            target_headway_s=ctx['target_headway_s'])
        packet = dict(clock=clock, current=identity, native=native, context=ctx, front=front,
            follower=follower, phases={role: self.phase(who, who.get('status', 'missing_launch'))
                for role, who in (('current', identity), ('front', front), ('follower', follower))})
        check_packet(packet)
        assert legacy['event_seq'] not in self.by_prepare_seq
        self.by_prepare_seq[legacy['event_seq']] = packet
        self.prepares.append(packet)

    def query(self, q):
        with self.read('query'):
            assert q['time_s'] == self.env.current_time
            packet = self.by_prepare_seq[q['prepare_event_seq']]
            assert all(packet['current'][k] == q[k] for k in ('trip_id', 'bus_id', 'direction', 'station_id'))
            assert packet['context'] == q['context'], 'query changed cached prepare context'
            assert packet['clock']['prepare_s'] == q['state_captured_s']
            phases = tuple(packet['phases'][role]['phase_s'] if packet['phases'][role]['status']
                in ('supported', 'known_zero') else None for role in ('front', 'current', 'follower'))
            ctx = SimpleNamespace(**packet['context'])
            native = shadow.command(ctx, q['feasible_bins'])
            result = shadow.command(ctx, q['feasible_bins'], phases)
            assert native['bin'] == q['base_bin'], 'shadow native rule differs from frozen controller'
            key = tuple(q[k] for k in ('trip_id', 'station_id', 'direction'))
            row = dict(query_s=q['time_s'], query_order=self.mark('query', **packet['current']),
                legacy_query_event_seq=q['event_seq'], prepare_order=packet['clock']['handoff_order'],
                query_occurrence=self.query_counts[key], feasible_bins=list(q['feasible_bins']),
                native_bin=q['base_bin'], shadow_bin=result['bin'], shadow_support_reason=result['mode'],
                identity=packet['current'], phases=phases, shadow=result, executed_shadow=False)
            self.query_counts[key] += 1
            assert set(FIELDS['query_join']) <= row.keys()
            assert row['prepare_order'] < row['query_order'] and packet['clock']['prepare_s'] <= row['query_s']
            self.queries.append(row)

    def summary(self, legacy):
        assert len(self.queries) == len(legacy['trace']['lower_actions'])
        assert self.checked['prepare'] == len(self.prepares)
        assert self.checked['query'] == len(self.queries)
        used = {q['prepare_order'] for q in self.queries}
        for p in self.prepares:
            check_packet(p)
        corrections = [q['shadow']['phase_correction_s'] for q in self.queries]
        return dict(recording_complete=True, prepares=len(self.prepares), queries=len(self.queries),
            unused_prepares=len(self.prepares)-len(used), observer_checks=dict(self.checked),
            native_calls=dict(self.calls), global_rng_unchanged=True, observer_native_call_delta=0,
            role_phase_support={role: dict(Counter(p['phases'][role]['status'] for p in self.prepares))
                                for role in ('front', 'current', 'follower')},
            shadow_modes=dict(Counter(q['shadow_support_reason'] for q in self.queries)),
            shadow_bin_differences=sum(q['shadow_bin'] != q['native_bin'] for q in self.queries),
            shadow_continuous_s=dict(positive=sum(x > 0 for x in corrections),
                negative=sum(x < 0 for x in corrections), zero=sum(x == 0 for x in corrections),
                total=float(sum(corrections))), shadow_actions_executed=0)

    def data(self):
        return dict(events=self.events, prepares=self.prepares, queries=self.queries,
                    avl_samples=self.samples, launch_phases=list(self.launches.values()),
                    observer_checks=dict(self.checked), native_calls=dict(self.calls))


def validate_saved_capture(data, legacy):
    assert data['complete'], 'only complete saved paths admit analysis recovery'
    capture = PrepareCapture(None)
    capture.prepares, capture.queries = data['prepares'], data['queries']
    capture.checked, capture.calls = Counter(data['observer_checks']), Counter(data['native_calls'])
    return capture.summary(legacy)
