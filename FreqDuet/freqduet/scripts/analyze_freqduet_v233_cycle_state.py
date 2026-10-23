#!/usr/bin/env python3
"""Read saved V230 events through frozen V232; evaluate future labels separately."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import time
import unittest

import preregister_freqduet_v232_persistent_coordination as frozen


CONTEXTS = ((12037, 2151001), (12037, 2151002), (933083, 2151001), (933083, 2151002))


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write_detail(path, value):
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, separators=(',', ':'))


def close(a, b, label):
    assert abs(a-b) <= 1e-6, (label, a, b)


def obligation(view):
    return view['own_association']['obligation']


def transitions(previous, view, row, callback_delta, write_delta):
    old, new = obligation(previous), obligation(view)
    old_id, new_id = (None if p is None else p['trip_id'] for p in (old, new))
    pending_version = None if old is None else row['pending_plan_version_refs'].get(str(old_id))
    return dict(callback_interval=callback_delta > 0, actual_write_interval=write_delta > 0,
        cache_only_interval=callback_delta > 0 and write_delta == 0,
        previous_H_positive=previous['executed_holding_s'] > 0,
        association_changed=old_id != new_id, association_lost=old is not None and new is None,
        association_gained=old is None and new is not None,
        position_rank_changed=previous['own_association']['distance_rank'] != view['own_association']['distance_rank'],
        prior_obligation_revised=pending_version is not None and old is not None
            and pending_version != old['last_write_version'],
        prior_obligation_no_longer_pending=old is not None and pending_version is None,
        same_obligation_future_to_due=old is not None and new is not None and old_id == new_id
            and old['kind'] == 'future' and new['kind'] == 'due',
        due_pressure_cleared=bool(previous['target_profile']['uncovered_due'])
            and not view['target_profile']['uncovered_due'])


def walk_saved(events, cycle_events, queries, versions, metadata):
    """No supply_outcomes, future launch table or action labels enter this pass."""
    state = frozen.CycleState()
    cycle_by_seq = {e['native_event_seq']: e for e in cycle_events}
    query_by_seq = {q['snapshot_event_seq']: q for q in queries}
    version_by_seq = {v['native_event_seq']: v for v in versions if not v['initial']}
    assert len(query_by_seq) == len(queries) and len(cycle_by_seq) == len(cycle_events)
    callbacks, writes = 0, 0
    cycles, samples, observed_links = {}, [], []
    interval_counts, interval_cycles = Counter(), defaultdict(set)
    for event in events:
        kind, seq, now = event['kind'], event['event_seq'], event['time_s']
        if kind == 'plan_update':
            callbacks += 1
            writes += bool(version_by_seq[seq]['changes'])
        elif kind == 'launch':
            e = cycle_by_seq[seq]
            opened = state.launch(e['bus_id'], e['trip_id'], e['direction'], seq, now)
            if opened['observed_previous_cycle'] is not None:
                observed_links.append(dict(observed_at_seq=seq, current_cycle=opened['cycle_key'],
                                           previous=opened['observed_previous_cycle']))
            key = tuple(opened['cycle_key'])
            assert key not in cycles
            cycles[key] = dict(cycle_key=list(key), queries=0, query_increment_s=0.,
                last_callbacks=callbacks, last_writes=writes)
        elif kind == 'lower_query':
            row = query_by_seq[seq]
            assert row['snapshot_read_s'] == now, 'query snapshot clock differs from its native event'
            bus = row['cycle_ref'][0]
            previous = state.active[bus]['last_profile']
            view = state.query(row, metadata)
            assert view['executed_holding_s'] == row['executed_holding_before_s']
            key = tuple(view['cycle_key'])
            cycle = cycles[key]
            flags = {} if previous is None else transitions(previous, view, row,
                callbacks-cycle['last_callbacks'], writes-cycle['last_writes'])
            if previous is not None:
                interval_counts['intervals'] += 1
                for name, value in flags.items():
                    interval_counts[name] += bool(value)
                    if value:
                        interval_cycles[name].add(key)
                if flags['actual_write_interval'] and flags['previous_H_positive']:
                    interval_counts['write_intervals_carrying_positive_H'] += 1
            p = obligation(view)
            samples.append(dict(native_event_seq=seq, query_s=now, cycle_key=list(key),
                H_s=view['executed_holding_s'], delta_H_s=view['delta_executed_since_last_sample_s'],
                association=None if p is None else dict(p),
                position_rank=view['own_association']['distance_rank'],
                incoming_current=[dict(bus_id=b['bus_id'], trip_id=b['inbound_trip_id'])
                                  for b in view['target_profile']['position_associations']],
                message_refs=view['message_refs'], interval_flags=flags))
            cycle['queries'] += 1
            cycle['query_increment_s'] += view['delta_executed_since_last_sample_s']
            cycle.update(last_callbacks=callbacks, last_writes=writes)
        elif kind == 'return':
            e = cycle_by_seq[seq]
            key = tuple(state.active[e['bus_id']]['cycle_key'])
            last_h = state.active[e['bus_id']]['executed_holding_s']
            closed = state.returned_to_terminal(e['bus_id'], e['trip_id'], seq, now, e['executed_holding_s'])
            tail = closed['executed_holding_s']-last_h
            cycle = cycles[key]
            close(cycle['query_increment_s']+tail, closed['executed_holding_s'], 'cycle prefix plus return remainder')
            cycle.update(return_event_seq=seq, return_s=now, holding_s=closed['executed_holding_s'],
                         after_last_query_s=tail)
    assert len(samples) == len(queries) and not state.active
    assert all('holding_s' in c for c in cycles.values()), 'open cycle cannot be filled with zero'
    return dict(samples=samples, cycles=list(cycles.values()), observed_parent_links=observed_links,
        ledger=dict(queries=len(samples), closed_cycles=len(cycles), open_cycles=len(state.active),
            returned_without_next_launch=len(state.returned),
            query_increment_s=math.fsum(c['query_increment_s'] for c in cycles.values()),
            after_last_query_s=math.fsum(c['after_last_query_s'] for c in cycles.values()),
            executed_holding_s=math.fsum(c['holding_s'] for c in cycles.values())),
        intervals=dict(interval_counts), cycles_with_interval={k: len(v) for k, v in sorted(interval_cycles.items())})


def label_sample(sample, next_trip, launches, returns):
    """Future-only evaluation, never called by the state transition pass."""
    p = sample['association']
    if next_trip is None:
        status = 'missing_next_launch'
    elif p is None:
        status = 'labelled_without_association'
    else:
        status = 'match' if p['trip_id'] == next_trip else 'mismatch'
    ranks = [returns.get((b['bus_id'], b['trip_id'])) for b in sample['incoming_current']]
    known = all(r is not None for r in ranks)
    own_key = tuple(sample['cycle_key'][:2])
    own_return = returns[own_key]
    assert own_return['native_event_seq'] > sample['native_event_seq']
    return_rank = None if not known else 1+sum(r['native_event_seq'] < own_return['native_event_seq'] for r in ranks)
    predicted = None if p is None else launches.get(p['trip_id'])
    reason = None
    if status == 'mismatch':
        reason = ('predicted_slot_not_observed' if predicted is None else
                  'predicted_slot_served_by_other_bus' if predicted['bus_id'] != own_key[0] else
                  'same_bus_but_not_first_next_launch')
    return dict(native_event_seq=sample['native_event_seq'], cycle_key=sample['cycle_key'],
        kind='unassociated' if p is None else p['kind'], status=status, mismatch_reason=reason,
        actual_next_trip=next_trip, predicted_trip=None if p is None else p['trip_id'],
        future_return_rank=return_rank, distance_rank_correct=None if return_rank is None else
            return_rank == sample['position_rank'],
        predicted_slot_left_before_own_return=None if predicted is None else
            predicted['native_event_seq'] < own_return['native_event_seq'])


def tally_labels(rows):
    counts = Counter(r['status'] for r in rows)
    counts['queries'] = len(rows)
    counts['association_labelled'] = counts['match']+counts['mismatch']
    counts['distance_rank_labelled'] = sum(r['distance_rank_correct'] is not None for r in rows)
    counts['distance_rank_correct'] = sum(r['distance_rank_correct'] is True for r in rows)
    return dict(counts, match_fraction=counts['match']/counts['association_labelled']
                if counts['association_labelled'] else None)


def evaluate(model, outcomes, cycle_events):
    returns = {(e['bus_id'], e['trip_id']): e for e in cycle_events if e['kind'] == 'return'}
    launches = {e['trip_id']: e for e in cycle_events if e['kind'] == 'launch'}
    next_by_parent = {(r['bus_id'], r['inbound_trip_id']): r['next_trip_id'] for r in outcomes
                      if r['inbound_trip_id'] is not None}
    observed_next = {tuple(r['previous']['cycle_key'][:2]): r['current_cycle'][1]
                     for r in model['observed_parent_links']}
    assert next_by_parent == observed_next, 'future labels differ from first observed same-bus launch'
    labels = [label_sample(r, next_by_parent.get(tuple(r['cycle_key'][:2])), launches, returns)
              for r in model['samples']]
    groups, per_cycle = defaultdict(list), defaultdict(list)
    mismatch = Counter()
    for label in labels:
        groups[label['kind']].append(label)
        per_cycle[tuple(label['cycle_key'])].append(label)
        if label['status'] == 'mismatch':
            mismatch[label['mismatch_reason']] += 1
            mismatch['distance_rank_correct' if label['distance_rank_correct'] is True else
                     'distance_rank_changed' if label['distance_rank_correct'] is False else 'distance_rank_unknown'] += 1
            mismatch['predicted_slot_left_before_own_return'] += label['predicted_slot_left_before_own_return'] is True
    first, last = [v[0] for v in per_cycle.values()], [v[-1] for v in per_cycle.values()]
    return dict(all_queries=tally_labels(labels), kinds={k: tally_labels(v) for k, v in sorted(groups.items())},
        mismatch_diagnostics=dict(mismatch), fixed_first_query=tally_labels(first),
        fixed_last_query=tally_labels(last), cycles_without_query=len(model['cycles'])-len(per_cycle)), labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'v231-root', 'v232-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert 'source_7fdf968a5f' in frozen.__file__, 'use sealed V232,not mutable reference code'
    existing = a.out_root/'summary/status.json'
    if existing.exists():
        saved = read(existing)
        assert saved['source'] == a.source and saved['status'] == 'v233_frozen_cycle_state_complete'
        print('V233_EXISTING_COMPLETE_ANALYSIS_REUSED', flush=True)
        return
    assert read(a.v232_root/'protocol.json') == frozen.registration('7fdf968a5f')
    assert read(a.v232_root/'summary/status.json')['status'] == 'v232_persistent_coordination_protocol_sealed'
    parent = read(a.collection_root/'summary/status.json')
    prior = read(a.v231_root/'summary/status.json')
    assert parent['status'] == 'v230_remaining_six_passed' and parent['source'] == 'eec7a20c71'
    assert parent['started_paths'] == parent['checked_paths'] == 8 and parent['all_checked_passed']
    assert prior['status'] == 'v231_supply_overlap_complete' and prior['source'] == '2dc70e4bf7'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v233_cycle_state.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'details').mkdir()
    spec = dict(stage='v233_frozen_cycle_state_read_only_audit', source=a.source, frozen_source='7fdf968a5f',
        authorization='User approved all eight existing V230 paths under unchanged V232.',
        paths=8, archive_passes_per_path=1, new_replays=0, training_updates=0, model_fits=0,
        shadow_actions_executed=0, new_futures=0, main_policy_changed=False,
        scope='Query state and future labels are separate passes/files. Inter-query holding delta '
            'is a ledger increment,not an attribution to current risk. Match fractions are '
            'descriptive on two consumed actors/two days,not efficacy or query-level CI.')
    (a.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    contexts, start = [], time.monotonic()
    for ci, (actor, day) in enumerate(CONTEXTS):
        paths = {}
        for arm in ('v107', 'upper'):
            slot = f'c{ci}_{arm}'
            data = read(a.collection_root/'records'/f'{slot}.json.gz')
            resource = read(a.collection_root/'resource_sidecars'/f'{slot}.json.gz')
            assert (data['context'], data['arm']) == (ci, arm)
            assert (data['results']['train_seed'], data['results']['scenario_seed']) == (actor, day)
            model = walk_saved(data['events'], resource['cycle_events'], resource['query_extensions'],
                               resource['upper_versions'], resource['metadata'])
            close(model['ledger']['executed_holding_s'], data['ledger']['executed_holding_s'], 'full executed ledger')
            assert model['ledger']['queries'] == 5200 and model['ledger']['closed_cycles'] == 262
            assert len(model['observed_parent_links']) == 250
            assert model['ledger']['returned_without_next_launch'] == 12
            write_detail(a.out_root/'details'/f'{slot}_causal_state.json.gz', model)
            evaluated, labels = evaluate(model, resource['supply_outcomes'], resource['cycle_events'])
            write_detail(a.out_root/'details'/f'{slot}_future_labels.json.gz', labels)
            old = prior['contexts'][ci]['paths'][arm]
            close(model['ledger']['executed_holding_s'], old['all_queries']['holding_s'], 'V231 holding endpoint')
            assert old['physical_cycles'] == model['ledger']['closed_cycles']
            paths[arm] = dict(ledger=model['ledger'], intervals=model['intervals'],
                cycles_with_interval=model['cycles_with_interval'], evaluation=evaluated,
                frozen_endpoints_match_v231=True)
            del data, resource, model, labels
        contexts.append(dict(context=ci, actor=actor, demand_day=day, paths=paths))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=ci+1, total=4, unit='context',
            elapsed_seconds=time.monotonic()-start)), flush=True)
    summary = dict(spec, status='v233_frozen_cycle_state_complete', tests=tests.testsRun, contexts=contexts)
    (a.out_root/'summary').mkdir()
    existing.write_text(json.dumps(summary, separators=(',', ':'))+'\n')
    print('V233_FROZEN_CYCLE_STATE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
