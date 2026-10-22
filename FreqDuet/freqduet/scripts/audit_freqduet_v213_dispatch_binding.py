#!/usr/bin/env python3
"""Read only the existing upper-only paths to locate dispatch binding."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v213_coordination as review


def frame(data):
    t = data['trace']
    launches = {r['trip']: r for r in t['actual_launches']}
    latest, previous, availability = {}, {}, {}
    for launch in t['actual_launches']:
        bus, trip = launch['bus'], launch['trip']
        before = previous.get(bus)
        returned = t['vehicle_returns'][str(before)]['return_s'] if before is not None else None
        availability[trip] = dict(previous_trip=before, return_s=returned)
        if returned is not None:
            review.require(returned <= launch['actual_s'], 'vehicle launched before its previous return')
        previous[bus] = trip
    for call in t['upper_callbacks']:
        for change in call['changes']:
            trip = change['trip_id']
            if trip in launches and call['time_s'] <= launches[trip]['actual_s']:
                latest[trip] = change['after']['planned_s']
    return dict(data=data, launches=launches, latest_plan=latest, availability=availability)


def binding_records(base, candidate):
    a, b = review.anchor_call(base), review.anchor_call(candidate)
    am, bm = ({v['trip_id']: v for v in call['changes']} for call in (a, b))
    result = []
    for trip in am.keys() | bm.keys():
        before = (am.get(trip) or bm[trip])['before']['planned_s']
        original, changed = (m[trip]['after']['planned_s'] if trip in m else before for m in (am, bm))
        if abs(changed-original) <= review.EPS:
            continue
        rows = []
        for p in (base, candidate):
            launch, available = p['launches'][trip], p['availability'][trip]
            rows.append(dict(actual_s=launch['actual_s'], planned_s=p['latest_plan'][trip],
                eligible_s=launch['eligible_s'], previous_return_s=available['return_s'],
                previous_trip=available['previous_trip'], bus=launch['bus'],
                registered_lower_hits_inbound_trip=any(v['identity'][0] == available['previous_trip']
                    for v in p['data']['result']['lower']),
                registered_lower_hits_assigned_vehicle=any(v['identity'][3] == launch['bus']
                    for v in p['data']['result']['lower']),
                after_eligible_s=launch['actual_s']-launch['eligible_s'],
                after_plan_s=launch['actual_s']-p['latest_plan'][trip],
                after_return_s=launch['actual_s']-available['return_s'] if available['return_s'] is not None else None))
        result.append(dict(trip=trip, anchor_plan_delta_s=changed-original, base=rows[0], candidate=rows[1]))
    return result


def describe(values):
    return dict(n=len(values), min=min(values) if values else None,
        mean=sum(values)/len(values) if values else None, max=max(values) if values else None,
        histogram=dict(Counter(str(v) for v in values)))


def run(root, out):
    values = {u: [] for u in (1, 2)}
    for context in range(4):
        for future in range(16):
            base = frame(review.read(root/'records'/f'c{context}_f{future}_a0.json.gz'))
            for u in (1, 2):
                candidate = frame(review.read(root/'records'/f'c{context}_f{future}_a{5*u}.json.gz'))
                rows = binding_records(base, candidate)
                values[u].extend(dict(context=context, future=future, **r) for r in rows)
    result = {}
    for u, rows in values.items():
        counts = Counter()
        for r in rows:
            b, c = r['base'], r['candidate']
            counts['same_actual_launch'] += abs(c['actual_s']-b['actual_s']) <= review.EPS
            counts['same_vehicle'] += c['bus'] == b['bus']
            counts['both_wait_after_eligibility'] += min(b['after_eligible_s'], c['after_eligible_s']) > review.EPS
            counts['both_wait_after_plan'] += min(b['after_plan_s'], c['after_plan_s']) > review.EPS
            counts['registered_lower_hits_inbound_trip'] += c['registered_lower_hits_inbound_trip']
            counts['registered_lower_hits_assigned_vehicle'] += c['registered_lower_hits_assigned_vehicle']
            supported = b['previous_return_s'] is not None and c['previous_return_s'] is not None
            counts['previous_return_observed'] += supported
            if supported:
                counts['same_previous_trip_and_return'] += (b['previous_trip'], b['previous_return_s']) == (c['previous_trip'], c['previous_return_s'])
        result[str(u)] = dict(affected_trips=len(rows), counts=dict(counts),
            by_role={role: {field: describe([r[role][field] for r in rows if r[role][field] is not None])
                for field in ('after_eligible_s', 'after_plan_s', 'after_return_s')} for role in ('base', 'candidate')},
            examples=rows[:4])
    path = out/'summary/status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status='v213_dispatch_binding_review_complete', upper_only=result,
        paths_read=192, new_rollouts=0, training_updates=0, main_policy_changed=False), indent=2)+'\n')
    print('V213_DISPATCH_BINDING_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v213-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v213_coordination_review.py', '-v'], check=True)
    run(a.v213_root, a.out_root)


if __name__ == '__main__':
    main()
