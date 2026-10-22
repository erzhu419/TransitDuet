#!/usr/bin/env python3
"""Explain frozen V215 follow-up support from completed passive records only."""

import argparse
import gzip
import json
from pathlib import Path

import preregister_freqduet_v215_positive_event as v


def describe(anchor, query):
    if query is None:
        return None
    active = any((b['bus_id'], b['trip_id']) == (anchor['bus_id'], anchor['trip_id']) for b in query['active'])
    due = [p['trip_id'] for p in query['plans'] if p['direction'] == anchor['target_direction']
           and not p['launched'] and p['planned_s'] <= query['time_s']]
    return dict(time_s=query['time_s'], event_seq=query['event_seq'], trip_id=query['trip_id'],
        direction=query['direction'], delta_from_anchor_s=query['time_s']-anchor['time_s'],
        within900=query['time_s'] < anchor['time_s']+900, selected_inbound_trip_still_active=active,
        compatible_ready=sum(b['direction'] == anchor['target_direction'] for b in query['ready']),
        original_due_ids_remaining=sorted(set(due) & set(anchor['due_trip_ids'])),
        queue_coverage_link=v.resource_link(query, anchor['bus_id'], anchor['trip_id'], anchor['inbound_direction']),
        frozen_followup_eligible=v.upper_followup(anchor, query))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for ci in range(4):
        with gzip.open(a.capture_root/'records'/f'v216_c{ci}_capture1.json.gz', 'rt') as f:
            data = json.load(f)['capture']
        anchor = data['anchor']
        assert anchor is not None
        later = [q for q in data['fresh_upper'] if (q['time_s'], q['event_seq']) >
                 (anchor['time_s'], anchor['event_seq'])]
        target = [q for q in later if q['direction'] == anchor['target_direction']]
        eligible = [q for q in later if v.upper_followup(anchor, q)]
        assert (eligible[0] if eligible else None) == data['followup'], 'saved follow-up differs from frozen predicate'
        window = [q for q in target if q['time_s'] < anchor['time_s']+900]
        rows.append(dict(context=ci, anchor_s=anchor['time_s'], return_s=data['supply']['actual_return']['return_s'],
            actual_same_vehicle_launch_s=data['supply']['next_same_vehicle_launch']['actual_s'],
            first_later_target_fresh=describe(anchor, target[0] if target else None),
            target_fresh_within900=[describe(anchor, q) for q in window],
            frozen_followup_count=len(eligible)))
    a.out_root.mkdir(parents=True, exist_ok=True)
    (a.out_root/'status.json').write_text(json.dumps(dict(contexts=rows, new_rollouts=0,
        rules_changed=False, main_policy_changed=False, observation_only=True), indent=2)+'\n')
    print('V216_FOLLOWUP_SUPPORT_READONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
