#!/usr/bin/env python3
"""Read-only acceptance of the six completed V230 paths; no replay imports."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import unittest


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def audit_prefix(data, sidecar, legacy):
    """Reconstruct each query from original ordered writes and physical events."""
    meta, first = data['metadata'], legacy['fresh_upper'][0]
    assert meta['initial_published_plans'] == first['plans']
    assert meta['native_inventory_order'] == [b['bus_id'] for b in first['ready']]
    assert meta['native_timetable_order'] == [p['trip_id'] for p in first['plans']]
    assert meta['native_dt_s'] == legacy['physical_contract']['dt_s']
    plans = {p['trip_id']: p for p in first['plans']}
    writes = dict.fromkeys(plans, 'initial')
    latest = {'false': 'initial', 'true': 'initial'}
    queries = {q['legacy_query_event_seq']: q for q in sidecar['queries']}
    rows = {q['legacy_query_event_seq']: q for q in data['query_extensions']}
    assert len(rows) == len(data['query_extensions']) == len(queries) == len(sidecar['queries'])
    versions = {v['version_id']: v for v in data['upper_versions']}
    assert len(versions) == len(data['upper_versions'])
    arrivals = {(r['trip_id'], r['station_id'], r['direction']): r
                for r in legacy['trace']['arrivals']}
    actual = {r['trip']: r for r in legacy['trace']['actual_launches']}
    returns = {r['trip_id']: r for r in legacy['trace']['vehicle_returns']}
    callbacks = iter(legacy['trace']['upper_callbacks'])
    ready = {b['bus_id']: dict(b) for b in first['ready']}
    active, last_return, cycles, outcomes = {}, {}, [], []
    pending_fresh, seen_versions, seen_queries = None, ['initial'], []
    stats = Counter()
    for ordinal, e in enumerate(legacy['events']):
        assert e['event_seq'] == ordinal
        kind, seq, trip = e['kind'], e['event_seq'], e.get('trip_id')
        if kind == 'fresh_upper':
            assert pending_fresh is None
            pending_fresh = e
        elif kind == 'plan_update':
            callback = next(callbacks)
            assert (callback['trip_id'], callback['time_s'], callback['direction']) == (
                trip, e['time_s'], e['direction'])
            if pending_fresh is not None:
                assert all(pending_fresh[k] == e[k] for k in ('trip_id', 'time_s', 'direction'))
            expected = dict(version_id=seq, initial=False, native_event_seq=seq,
                time_s=e['time_s'], direction=e['direction'], callback_trip_id=trip,
                fresh_upper_event_seq=None if pending_fresh is None else pending_fresh['event_seq'],
                changes=callback['changes'])
            assert versions[seq] == expected, 'callback version/fresh provenance differs'
            stats['cached_callbacks' if pending_fresh is None else 'fresh_callbacks'] += 1
            for change in callback['changes']:
                tid = change['trip_id']
                assert plans[tid] == change['before'], 'plan before-image differs'
                plans[tid], writes[tid] = change['after'], seq
                stats['changed_plan_rows'] += 1
            latest[str(e['direction']).lower()] = seq
            seen_versions.append(seq)
            pending_fresh = None
        elif kind == 'launch':
            bus, launch = e['bus_id'], actual[trip]
            assert (launch['bus'], launch['actual_s']) == (bus, e['time_s'])
            assert bus in ready and bus not in active
            assert ready.pop(bus)['direction'] == launch['direction'] == plans.pop(trip)['direction']
            writes.pop(trip)
            active[bus] = dict(e, direction=launch['direction'])
            parent = last_return.pop(bus, None)
            outcomes.append(dict(bus_id=bus, inbound_trip_id=None if parent is None else parent['trip_id'],
                inbound_return_event_seq=None if parent is None else parent['native_event_seq'],
                next_trip_id=trip, next_launch_event_seq=seq,
                status='initial_stock' if parent is None else 'first_observed_same_bus_launch'))
            cycles.append(dict(kind='launch', native_event_seq=seq, time_s=e['time_s'],
                bus_id=bus, trip_id=trip, direction=launch['direction'],
                launch_native_event_seq=seq, executed_holding_s=0.))
        elif kind == 'return':
            bus, ret = e['bus_id'], returns[trip]
            launched = active.pop(bus)
            assert (launched['trip_id'], ret['bus_id'], ret['return_s']) == (trip, bus, e['time_s'])
            assert ret['inbound_direction'] == launched['direction'] != ret['target_direction']
            row = dict(kind='return', native_event_seq=seq, time_s=e['time_s'], bus_id=bus,
                trip_id=trip, direction=ret['inbound_direction'],
                launch_native_event_seq=launched['event_seq'], executed_holding_s=ret['holding_s'], account=ret)
            cycles.append(row)
            last_return[bus] = row
            ready[bus] = dict(bus_id=bus, direction=ret['target_direction'],
                return_observed=True, last_observed_return_s=e['time_s'])
        elif kind == 'lower_query':
            row, query = rows[seq], queries[seq]
            identity = query['identity']
            assert all(e[k] == identity[k] for k in ('bus_id', 'trip_id', 'direction', 'station_id'))
            assert row['snapshot_read_s'] == query['query_s'] == e['time_s']
            assert row['snapshot_event_seq'] == seq
            assert row['plans'] == list(plans.values()), 'query posted plans differ from prior writes'
            assert row['pending_plan_version_refs'] == {str(t): v for t, v in writes.items()}, \
                'pending trip does not reference its actual last write'
            assert row['message_refs'] == latest, 'direction message is not the completed prefix'
            launched = active[identity['bus_id']]
            assert row['cycle_ref'] == [identity['bus_id'], trip, launched['event_seq']]
            arrival = arrivals[trip, identity['station_id'], identity['direction']]
            assert arrival['time_s'] <= query['query_s']
            assert row['executed_holding_before_s'] == arrival['realized_holding'], \
                'query executed holding differs from actual arrival prefix'
            for entries, expected in ((row['ready'], ready), (row['active'], active)):
                assert [b['bus_id'] for b in entries] == [b for b in meta['native_inventory_order'] if b in expected]
                for b in entries:
                    observed = expected[b['bus_id']]
                    assert b['direction'] == observed['direction'] and b['position_sample_s'] == query['query_s']
                    fields = ('return_observed', 'last_observed_return_s') if expected is ready else ('trip_id',)
                    assert all(b[k] == observed[k] for k in fields), 'inventory is not the observed physical prefix'
            seen_queries.append(seq)
    assert next(callbacks, None) is None and pending_fresh is None and not plans and not active
    assert seen_queries == list(rows) == list(queries)
    assert seen_versions == [v['version_id'] for v in data['upper_versions']]
    assert cycles == data['cycle_events'] and outcomes == data['supply_outcomes']
    assert data['terminal_cycles'] == [dict(bus_id=b, trip_id=r['trip_id'],
        status='returned_no_later_launch', native_return_event_seq=r['native_event_seq'])
        for b, r in last_return.items()]
    assert data['open_cycles'] == []
    return dict(query_plan_prefixes=len(seen_queries), actual_holding_prefixes=len(seen_queries),
        initial_stock_launches=sum(r['status'] == 'initial_stock' for r in outcomes),
        next_same_bus_launches=sum(r['status'] == 'first_observed_same_bus_launch' for r in outcomes),
        terminal_returns=len(last_return), **stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'reference-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v230_resource_audit.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    from freqduet_v230_resource_capture import validate, frontier
    from preregister_freqduet_v224_prepare_handoff import require_legacy_equal
    from preregister_freqduet_v230_resource_capture import FIRST_PAIR_CHECKS, registration
    summary = read(a.collection_root/'summary/status.json')
    assert summary['status'] == 'v230_remaining_six_passed' and summary['all_checked_passed']
    assert summary['started_paths'] == summary['checked_paths'] == summary['max_started_paths'] == 8
    assert summary['source'] == 'eec7a20c71'
    slots = registration('cf86ee990a')['reserved_passive']['slots']
    assert [r['slot'] for r in summary['paths']] == [s['slot'] for s in slots]
    assert all(r[k] is True for r in summary['paths'] for k in FIRST_PAIR_CHECKS)
    a.out_root.mkdir(parents=True, exist_ok=False)
    results = []
    for item in slots[2:]:
        slot = item['slot']
        legacy = read(a.collection_root/'records'/f'{slot}.json.gz')
        require_legacy_equal(read(a.reference_root/'records'/f'{slot}.json.gz'), legacy)
        sidecar = read(a.collection_root/'sidecars'/f'{slot}.json.gz')
        require_legacy_equal(read(a.reference_root/'sidecars'/f'{slot}.json.gz'), sidecar)
        data = read(a.collection_root/'resource_sidecars'/f'{slot}.json.gz')
        checked = validate(data, sidecar, legacy)
        assert all(checked[k] == summary['paths'][len(results)+2][k] for k in checked)
        prepares = {r['clock']['handoff_order']: r for r in sidecar['prepares']}
        for row, q in zip(data['query_extensions'], sidecar['queries']):
            prepared = prepares[q['prepare_order']]
            assert row['shadow_frontier'] == frontier(dict(context=prepared['context'],
                base_bin=q['native_bin'], onboard=row['onboard'], terminal_masked=row['terminal_masked'],
                feasible_bins=q['feasible_bins']))
        result = dict(slot=slot, legacy_exact=True, old_sidecar_exact=True,
            shadow_frontier_exact=True, **audit_prefix(data, sidecar, legacy), **checked)
        results.append(result)
        print('V230_READ_ONLY_PATH '+json.dumps(result), flush=True)
        del data, legacy, sidecar
    result = dict(status='v230_remaining_six_read_only_acceptance_passed', source=a.source,
        collection_source=summary['source'], tests=tests.testsRun, newly_audited_paths=6,
        prior_c0_pair_reused=True, all_eight_passed=True, paths=results,
        new_replays=0, training_updates=0, shadow_actions_executed=0, main_policy_changed=False,
        next_step='Read-only resource-pressure/shadow-support analysis on saved paths;no action selector admitted.')
    (a.out_root/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    print('V230_READ_ONLY_ACCEPTANCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
