#!/usr/bin/env python3
"""Preregister V219 and count state-only opportunities in four saved V216 paths."""

import argparse
from collections import Counter
import gzip
import json
from math import ceil
import os
from pathlib import Path
import subprocess
import sys

import freqduet_v219_service_task as law


def positive_query(q):
    return (q['base_bin'] > 0 and q['base_bin'] in q['feasible_bins']
            and any(i < q['base_bin'] for i in q['feasible_bins'])
            and q['forward_valid'] and q['follower_valid'] and not q['terminal_masked'])


def audit(data, context):
    capture, trace = data['capture'], data['trace']
    callbacks = {(c['time_s'], c['trip_id']): c for c in trace['upper_callbacks'] if c['commands']}
    launches = {r['trip']: r for r in trace['actual_launches']}
    launch_events = {e['trip_id']: e for e in capture['events'] if e['kind'] == 'launch'}
    updates = {(e['time_s'], e['trip_id']): e['event_seq'] for e in capture['events']
               if e['kind'] == 'plan_update'}
    plan_history = [(updates[c['time_s'], c['trip_id']], x['trip_id'], x['after']['planned_s'])
                    for c in trace['upper_callbacks'] for x in c['changes']]
    query_by_trip = {}
    for q in capture['queries']:
        if positive_query(q):
            query_by_trip.setdefault(q['trip_id'], []).append(q)
    cycles, labels = [], Counter()
    for q in capture['fresh_upper']:
        if q['time_s'] < law.START_S:
            continue
        assert q['snapshot_event_seq'] == q['event_seq'] and q['snapshot_read_s'] == q['time_s']
        assert len(q['ready'])+len(q['active']) == 12
        resource = law.ready_prefix(q)
        assert len(set(resource['ready_bus_by_trip'].values())) == len(resource['ready_bus_by_trip'])
        callback = callbacks[q['time_s'], q['trip_id']]
        assert callback['commands'][0]['coefficient'] == q['native_coefficient']
        planned = {p['trip_id']: p['planned_s'] for p in q['plans']}
        planned.update({c['trip_id']: c['after']['planned_s'] for c in callback['changes']})
        potentials = [t for t in resource['mutable_ready_trip_ids']
                      if q['time_s'] < planned[t] < resource['expiry_public_s']]
        positive = []
        for trip in potentials:
            launch, event = launches[trip], launch_events[trip]
            later = [r for r in query_by_trip.get(trip, []) if r['event_seq'] > event['event_seq']
                     and r['time_s'] >= launch['actual_s']]
            # These are labels AFTER the immutable state-only opportunity count.
            matched = launch['bus'] == resource['ready_bus_by_trip'][trip]
            final_plan = planned[trip]
            for seq, changed_trip, value in plan_history:
                if changed_trip == trip and q['event_seq'] < seq < event['event_seq']:
                    final_plan = value
            tick = data['result']['physical_contract']['dt_s']
            planned_tick = ceil(final_plan/tick)*tick
            plan_reached = launch['actual_s'] == planned_tick
            first_launch = next(e for e in capture['events'] if e['kind'] == 'launch'
                and e['bus_id'] == resource['ready_bus_by_trip'][trip] and e['event_seq'] > q['event_seq'])
            stayed_ready = matched and first_launch['trip_id'] == trip
            labels['potential_tasks'] += 1
            labels['actual_same_reserved_bus'] += int(matched)
            labels['actual_at_final_plan_tick'] += int(plan_reached)
            labels['continuously_ready_from_query'] += int(stayed_ready)
            labels['strict_plan_wait_label'] += int(stayed_ready and plan_reached
                                                   and planned_tick > q['time_s']+tick)
            labels['positive_query_tasks'] += int(bool(later))
            labels['positive_queries'] += len(later)
            if later:
                positive.append(trip)
        cycles.append(dict(time_s=q['time_s'], trip_id=q['trip_id'], event_seq=q['event_seq'],
            elapsed_hour=int(q['time_s']//3600), ready_counts=[sum(b['direction'] == d for b in q['ready'])
                                                           for d in (False, True)],
            mutable_ready_trip_ids=resource['mutable_ready_trip_ids'], potential_task_ids=potentials,
            future_positive_query_task_ids=positive))
    seed, day = law.CONTEXTS[context]
    task_ids = sorted({t for c in cycles for t in c['potential_task_ids']})
    positive_ids = sorted({t for c in cycles for t in c['future_positive_query_task_ids']})
    return dict(context=context, actor=seed, day=day, native_cycles=len(cycles),
        cycles_with_mutable_ready_prefix=sum(bool(c['mutable_ready_trip_ids']) for c in cycles),
        cycles_with_potential_task=sum(bool(c['potential_task_ids']) for c in cycles),
        unique_potential_tasks=len(task_ids), unique_positive_query_tasks=len(positive_ids),
        opportunity_by_elapsed_hour=dict(Counter(c['elapsed_hour'] for c in cycles if c['potential_task_ids'])),
        future_labels=dict(labels), implementation_admission=bool(positive_ids),
        cycles=cycles)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v219_service_task.py', '-v'], check=True)
    a.out_root.mkdir(parents=True, exist_ok=True)
    spec = law.registration(os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    protocol = a.out_root/'protocol.json'
    assert not protocol.exists(), 'inspect original attempt; no automatic protocol overwrite'
    protocol.write_text(json.dumps(spec, indent=2)+'\n')
    rows = []
    for ci in range(len(law.CONTEXTS)):
        with gzip.open(a.archive_root/'records'/f'v216_c{ci}_capture1.json.gz', 'rt') as f:
            row = audit(json.load(f), ci)
        detail = a.out_root/'details'
        detail.mkdir(exist_ok=True)
        (detail/f'c{ci}.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append({k: v for k, v in row.items() if k != 'cycles'})
    result = dict(status='v219_state_only_opportunity_audit_complete', contexts=rows,
        implementation_admission=all(r['implementation_admission'] for r in rows),
        native_preview_executed=False, proposed_controller_run=False,
        new_replays=0, new_training=0, independent_futures_opened=False, main_policy_changed=False,
        limitations='Existing consumed contexts; positive queries and actual assignment are future labels, '
            'not state admission inputs. Repeated tasks across cycles are not independent samples. '
            'No proposed-plan efficacy or completed-predecessor credit eligibility is established.')
    summary = a.out_root/'summary'
    summary.mkdir(exist_ok=True)
    (summary/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
