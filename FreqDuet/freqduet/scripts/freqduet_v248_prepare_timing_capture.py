"""Independent PREPARE timing observer; no runner, rollout or action entrypoint."""

from bisect import bisect_left
from collections import Counter
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from preregister_freqduet_v224_prepare_handoff import require_legacy_equal
from preregister_freqduet_v246_compensation_supply import persistent_change
from preregister_freqduet_v247_prepare_timing import (
    IDENTITY, OLD_PAYLOADS, join_query, map_prepare, prepare_sample)


class PrepareTimingCapture:
    def __init__(self, audit):
        self.audit = audit
        self.launches, self.departures, self.samples, self.prepares, self.queries = [], [], [], [], []
        self.completed, self.sample_by_seq, self.prepare_by_seq, self.previous = {}, {}, {}, {}
        self.checks = Counter()

    @contextmanager
    def read(self, label):
        calls = self.audit.handoff.calls
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), calls.copy()
        try:
            yield
        finally:
            after = np.random.get_state()
            assert random.getstate() == before[0], 'prepare timing observer changed Python RNG'
            assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:]
            assert torch.equal(before[2], torch.get_rng_state()), 'prepare timing observer changed Torch RNG'
            assert calls == before[3], 'prepare timing observer invoked native computation'
            self.checks[label] += 1

    def attach(self):
        a = self.audit
        assert not a.clock.events, 'attach before native events'
        self.original_stamp, self.original_prepare = a.clock.stamp, a.bus_type._prepare_for_action
        self.recorder_type = type(a.env.headway_events)
        self.original_departure, self.original_query = self.recorder_type.record_departure, a.resource.query

        def stamp(kind, now, **identity):
            value = self.original_stamp(kind, now, **identity)
            if kind in ('launch', 'prepare'):
                with self.read(kind):
                    self.native_event(a.clock.events[-1])
            return value

        def departure(recorder, station_id, direction, departure_time_s, trip_id):
            value = self.original_departure(recorder, station_id, direction, departure_time_s, trip_id)
            with self.read('departure'):
                self.completed_departure(trip_id, station_id, direction, departure_time_s)
            return value

        def prepare(bus, *args, **kwargs):
            value = self.original_prepare(bus, *args, **kwargs)
            with self.read('prepare_join'):
                self.prepared(a.handoff.prepares[-1])
            return value

        def query(q):
            value = self.original_query(q)
            with self.read('query'):
                self.queried(q, a.handoff.queries[-1], a.cycle_capture.queries[-1]['view'])
            return value

        a.clock.stamp, a.bus_type._prepare_for_action = stamp, prepare
        self.recorder_type.record_departure, a.resource.query = departure, query

    def detach(self):
        a = self.audit
        a.clock.stamp, a.bus_type._prepare_for_action = self.original_stamp, self.original_prepare
        self.recorder_type.record_departure, a.resource.query = self.original_departure, self.original_query

    def native_event(self, event):
        a, trip = self.audit, event['trip_id']
        assert event['time_s'] == a.env.current_time
        launch = a.launches[trip]
        if event['kind'] == 'launch':
            self.launches.append(dict(trip_id=trip, native_event_seq=event['event_seq'],
                bus_id=launch['bus_id'], time_s=launch['time_s'],
                holding_counter_at_launch=launch['holding_counter_at_launch']))
            return
        bus = a.buses[trip]
        assert (bus.bus_id, bus.trip_id, bool(bus.direction), bus.last_station.station_id) == tuple(
            event[k] for k in ('bus_id', 'trip_id', 'direction', 'station_id'))
        front_trip = bus.forward_predecessor_trip_id
        key = None if front_trip is None else (int(front_trip), event['station_id'], event['direction'])
        index = self.completed.get(key)
        completed = None if index is None else self.departures[index]['completed']
        cycle = a.cycle_capture.state.active[event['bus_id']]['cycle_key']
        sample = prepare_sample(event, launch, cycle, float(bus.episode_hold_vehicle_seconds),
                                float(a.env.current_time), completed)
        seq = event['event_seq']
        assert seq not in self.sample_by_seq
        self.sample_by_seq[seq] = len(self.samples)
        self.samples.append(dict(sample=sample, completed_departure_count=len(self.departures),
                                 front_departure_index=index))

    def completed_departure(self, trip, station, direction, now):
        a, key = self.audit, (int(trip), int(station), bool(direction))
        timing = a.departures[-1]
        old = a.handoff.departures[key]
        assert tuple(timing[k] for k in ('trip_id', 'station_id', 'direction', 'time_s')) == (*key, now)
        assert old['departure_s'] == now == a.env.current_time
        launch = a.launches[key[0]]
        assert old['bus_id'] == launch['bus_id'] and key not in self.completed
        self.completed[key] = len(self.departures)
        self.departures.append(dict(identity={k: old[k] for k in IDENTITY},
            departure_order=old['departure_order'], completed=dict(timing=copy.deepcopy(timing),
                launch=dict(bus_id=launch['bus_id'], time_s=launch['time_s']))))

    def prepared(self, old):
        seq = old['clock']['legacy_prepare_event_seq']
        assert seq not in self.prepare_by_seq
        index = self.sample_by_seq[seq]
        mapped = map_prepare(old, self.samples[index]['sample'])
        cycle = tuple(mapped['cycle_key'])
        previous = self.previous.get(cycle)
        if previous is not None:
            assert mapped['own']['holding_s'] >= previous['own']['holding_s']
        self.prepare_by_seq[seq] = len(self.prepares)
        self.prepares.append(dict(sample_index=index, mapped=mapped,
                                 change=persistent_change(previous, mapped)))
        self.previous[cycle] = mapped

    def queried(self, q, old, view):
        index = self.prepare_by_seq[q['prepare_event_seq']]
        self.queries.append(dict(prepare_index=index,
            join=join_query(self.prepares[index]['mapped'], old, q, view)))

    def data(self, complete):
        return copy.deepcopy(dict(complete=complete, protocol_source='e74d381e62',
            launches=self.launches, departures=self.departures, samples=self.samples,
            prepares=self.prepares, queries=self.queries, observer_checks=dict(self.checks),
            observer_native_call_delta=0, observer_rng_unchanged=True, shadow_actions_executed=0))


def recording_audit(base):
    """Append to the V241 class; inherit its action callback unchanged."""
    class RecordingAudit(base):
        def __init__(self, seed, arm):
            assert arm in ('v107', 'upper')
            super().__init__(seed, arm)
            self.prepare_timing_capture = PrepareTimingCapture(self)

        def attach(self, runner):
            super().attach(runner)
            self.prepare_timing_capture.attach()

        def detach(self):
            self.prepare_timing_capture.detach()
            super().detach()
    return RecordingAudit


def validate_capture(reference, recorded, data):
    """Server-side acceptance from full old payloads and the new separate sidecar."""
    assert set(reference) == set(recorded) == set(OLD_PAYLOADS)
    for name in OLD_PAYLOADS:
        require_legacy_equal(reference[name], recorded[name])
    assert data['complete'] and data['protocol_source'] == 'e74d381e62'
    assert data['observer_native_call_delta'] == data['shadow_actions_executed'] == 0
    assert data['observer_rng_unchanged'] is True
    legacy, old, resource, cycles = (recorded[k] for k in ('legacy', 'v224', 'v230', 'v235'))
    native_launches = [e for e in legacy['events'] if e['kind'] == 'launch']
    assert len(data['launches']) == len(native_launches)
    launches = {}
    for row, event in zip(data['launches'], native_launches):
        assert (row['trip_id'], row['bus_id'], row['time_s'], row['native_event_seq']) == (
            event['trip_id'], event['bus_id'], event['time_s'], event['event_seq'])
        assert row['trip_id'] not in launches
        launches[row['trip_id']] = row
    phases = {r['trip_id']: r for r in old['launch_phases']}
    departure_events = [e for e in old['events'] if e['kind'] == 'departure']
    assert len(data['departures']) == len(departure_events) == len(legacy['trace']['departures'])
    for row, event, timing in zip(data['departures'], departure_events, legacy['trace']['departures']):
        assert row['departure_order'] == event['order']
        assert all(row['identity'][k] == event[k] == timing[k] for k in ('trip_id','station_id','direction'))
        assert row['completed']['timing'] == timing and timing['time_s'] == event['time_s']
        phase = phases[timing['trip_id']]
        assert row['identity']['bus_id'] == phase['bus_id']
        assert row['completed']['launch'] == dict(bus_id=phase['bus_id'], time_s=phase['actual_launch_s'])
    native_prepares = [e for e in legacy['events'] if e['kind'] == 'prepare']
    assert len(data['samples']) == len(data['prepares']) == len(old['prepares']) == len(native_prepares)
    dep_orders = [e['order'] for e in departure_events]
    previous = {}
    for i, (saved, row, packet, native) in enumerate(zip(
            data['samples'], data['prepares'], old['prepares'], native_prepares)):
        sample = saved['sample']
        assert sample['native'] == native
        launch = launches[native['trip_id']]
        assert sample['cycle_key'] == [native['bus_id'], native['trip_id'], launch['native_event_seq']]
        assert saved['completed_departure_count'] == bisect_left(dep_orders, packet['clock']['handoff_order'])
        front_index = saved['front_departure_index']
        if front_index is None:
            assert sample['completed_front'] is None
        else:
            assert 0 <= front_index < saved['completed_departure_count']
            assert sample['completed_front'] == data['departures'][front_index]['completed']
        require_legacy_equal(prepare_sample(native, launch, sample['cycle_key'],
            sample['counter_s'], sample['read_s'], sample['completed_front']), sample)
        mapped = map_prepare(packet, sample)
        assert row['sample_index'] == i
        require_legacy_equal(mapped, row['mapped'])
        assert not packet['native']['forward_valid'] or mapped['front_status'] == 'completed_at_prepare'
        cycle = tuple(mapped['cycle_key'])
        prior = previous.get(cycle)
        if prior is not None:
            assert mapped['own']['holding_s'] >= prior['own']['holding_s']
        require_legacy_equal(persistent_change(prior, mapped), row['change'])
        previous[cycle] = mapped
    indices = {r['mapped']['prepare_seq']: i for i, r in enumerate(data['prepares'])}
    assert len(data['queries']) == len(old['queries']) == len(resource['query_extensions']) == len(cycles['queries'])
    for row, query, res, cycle in zip(data['queries'], old['queries'], resource['query_extensions'], cycles['queries']):
        mapped = data['prepares'][row['prepare_index']]['mapped']
        assert indices[mapped['prepare_seq']] == row['prepare_index']
        assert cycle['identity'] == query['identity']
        q = dict(query['identity'], time_s=query['query_s'], event_seq=query['legacy_query_event_seq'],
            prepare_event_seq=old['prepares'][row['prepare_index']]['clock']['legacy_prepare_event_seq'],
            state_captured_s=mapped['prepare_s'], executed_holding_before_s=res['executed_holding_before_s'])
        require_legacy_equal(join_query(mapped, query, q, cycle['view']), row['join'])
    expected = dict(launch=len(native_launches), departure=len(departure_events),
                    prepare=len(native_prepares), prepare_join=len(native_prepares), query=len(old['queries']))
    assert data['observer_checks'] == {k: v for k, v in expected.items() if v}
    used = {r['prepare_index'] for r in data['queries']}
    return dict(all_five_old_payloads_exact=True, prepare_occurrences=len(data['prepares']),
        query_occurrences=len(data['queries']), unused_prepares=len(data['prepares'])-len(used),
        zero_holding_prepares=sum(r['mapped']['own']['holding_s'] == 0 for r in data['prepares']),
        supported_fronts=sum(r['mapped']['front_status'] == 'completed_at_prepare' for r in data['prepares']),
        unknown_fronts=sum(r['mapped']['front_status'] == 'no_native_front' for r in data['prepares']),
        zero_query_cycles=sum(c['queries'] == 0 for c in cycles['cycles']),
        open_cycles=len(cycles['terminal']['open_cycles']),
        clock_domains_separated=True, observer_rng_and_calls_unchanged=True)
