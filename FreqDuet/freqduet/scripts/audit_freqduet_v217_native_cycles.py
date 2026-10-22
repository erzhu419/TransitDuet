#!/usr/bin/env python3
"""Upper-first opportunity coverage from existing V216 passive records only."""

import argparse
from bisect import bisect_right
import gzip
import json
from pathlib import Path

import preregister_freqduet_v215_positive_event as v


def eligible(q):
    return (v.current_snapshot(q) and not q['terminal_masked'] and q['forward_valid']
        and q['follower_valid'] and q['base_bin'] > 0
        and q['base_bin'] in q['feasible_bins'] and any(i < q['base_bin'] for i in q['feasible_bins']))


def summarize(data):
    capture, trace = data['capture'], data['trace']
    fresh = capture['fresh_upper']
    assert fresh == sorted(fresh, key=lambda u: u['event_seq'])
    sequences = [u['event_seq'] for u in fresh]
    cycles = [dict(upper_s=u['time_s'], event_seq=u['event_seq'], direction=u['direction'],
        positive_queries=0, queue_linked_queries=0, opposite_callback_direction=0,
        realized_same_vehicle_due_supply=0, supply_label_missing=0, linked_holding_s=0.,
        changed_plan_trip_ids=[], changed_plan_trips_actually_launched=0) for u in fresh]
    callbacks = [c for c in trace['upper_callbacks'] if c['commands']]
    assert len(callbacks) == len(fresh)
    launches = {r['trip']: r for r in trace['actual_launches']}
    for row, u, c in zip(cycles, fresh, callbacks):
        assert (u['trip_id'], u['time_s']) == (c['trip_id'], c['time_s'])
        row['changed_plan_trip_ids'] = [x['trip_id'] for x in c['changes']
            if x['before']['planned_s'] != x['after']['planned_s']]
        row['changed_plan_trips_actually_launched'] = sum(t in launches for t in row['changed_plan_trip_ids'])
    before_first = 0
    for q in capture['queries']:
        if not eligible(q):
            continue
        ci = bisect_right(sequences, q['event_seq'])-1
        if ci < 0:
            before_first += 1
            continue
        row = cycles[ci]
        row['positive_queries'] += 1
        link = v.resource_link(q, q['bus_id'], q['trip_id'], q['direction'])
        if link is None:
            continue
        row['queue_linked_queries'] += 1
        row['opposite_callback_direction'] += row['direction'] != link['target_direction']
        row['linked_holding_s'] += q['command_s']
        returned = next((r for r in trace['vehicle_returns'] if (r['bus_id'], r['trip_id']) ==
                         (q['bus_id'], q['trip_id'])), None)
        if returned is None:
            row['supply_label_missing'] += 1
            continue
        later = [e for e in capture['events'] if e['kind'] == 'return' and e['trip_id'] == q['trip_id']]
        assert len(later) == 1
        event = next((e for e in capture['events'] if e['kind'] == 'launch' and e['bus_id'] == q['bus_id']
                      and e['event_seq'] > later[0]['event_seq']), None)
        launch = launches.get(event['trip_id']) if event else None
        row['realized_same_vehicle_due_supply'] += bool(launch is not None
            and launch['direction'] == link['target_direction'] and launch['trip'] in link['due_trip_ids'])
    return dict(native_cycles=len(cycles), positive_before_first_native_cycle=before_first,
        cycles_with_positive=sum(r['positive_queries'] > 0 for r in cycles),
        cycles_with_queue_link=sum(r['queue_linked_queries'] > 0 for r in cycles),
        positive_queries=sum(r['positive_queries'] for r in cycles),
        queue_linked_queries=sum(r['queue_linked_queries'] for r in cycles),
        same_vehicle_due_supply_labels=sum(r['realized_same_vehicle_due_supply'] for r in cycles), cycles=cycles)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for ci in range(4):
        with gzip.open(a.capture_root/'records'/f'v216_c{ci}_capture1.json.gz', 'rt') as f:
            rows.append(dict(context=ci, **summarize(json.load(f))))
    a.out_root.mkdir(parents=True, exist_ok=True)
    (a.out_root/'status.json').write_text(json.dumps(dict(contexts=[{k: val for k, val in r.items()
        if k != 'cycles'} for r in rows], new_rollouts=0, main_policy_changed=False,
        interpretation='coverage only; eventual launch is a future label, not proof of a changed departure or benefit',
        shared_upper='latest fresh shared coefficient across both directions; callback direction does not limit authority'),
        indent=2)+'\n')
    (a.out_root.parent/'cycles.json').write_text(json.dumps(rows)+'\n')
    print('V217_NATIVE_CYCLE_COVERAGE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
