#!/usr/bin/env python3
"""Read-only V274 failed-gate review; never changes the frozen verdict or traffic."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from freqduet_apc_flow_evidence import APCArrival, completed_flow
from freqduet_v274_passive_flow import boundary, WALL_FIELDS
from run_freqduet_protocol_v6_checkpoint_trajectory import assert_same_state as same
from run_freqduet_v274_passive_flow import load_trace
from seal_freqduet_v274_flow_equivalence import REFERENCE, read, write


def review(a):
    start = time.perf_counter()
    root = a.results_root/'protocol_v6_v274_flow_equivalence'
    old = a.results_root/REFERENCE
    result = read(root/'paths/flow_passive_c0.json')
    previous = read(old/'paths/noop_fix.json')
    failure = read(root/'failures/flow_passive_c0.json')
    assert result['source'] == failure['source'] == 'bed2255dca'
    assert previous['source'] == '40eddb8fb6'
    native = load_trace(root/'traces/flow_passive_c0.pkl.gz')
    graph = load_trace(root/'traces/graph_queries_flow_passive_c0.pkl.gz')
    flow = load_trace(root/'traces/flow_flow_passive_c0.pkl.gz')
    old_native = load_trace(old/'traces/noop_fix.pkl.gz')
    old_graph = load_trace(old/'traces/graph_queries_noop_fix.pkl.gz')
    checks, violations, first_errors = {}, Counter(), {}

    def equal(label, left, right):
        checks.setdefault(label,True)
        try:
            same(left,right,label)
        except AssertionError as error:
            checks[label] = False
            violations[label] += 1
            first_errors.setdefault(label,str(error))

    equal('legacy_native_schema',set(native),set(old_native))
    for key in old_native:
        equal('legacy_native_'+key,native[key],old_native[key])
    for key in ('results','contract','coverage','execution'):
        equal('legacy_'+key,result[key],previous[key])
    equal('legacy_graph_schema',set(graph),set(old_graph))
    equal('legacy_graph_visits',graph['completed_visits'],old_graph['completed_visits'])
    clean = lambda rows:[{k:v for k,v in row.items() if k not in WALL_FIELDS} for row in rows]
    equal('legacy_graph_queries_except_wall',clean(graph['queries']),clean(old_graph['queries']))
    events = [APCArrival(**row) for row in flow['events']]
    terms = set(map(tuple,flow['terminals']))
    roster = [(e.trip,e.station,e.direction,e.arrival_s) for e in events]
    observed = [(r['trip_id'],r['station_id'],r['direction'],r['time_s']) for r in native['arrivals']]
    equal('literal_frozen_arrival_roster',roster,observed)
    extra = Counter(roster)-Counter(observed)
    missing = Counter(observed)-Counter(roster)
    extra_groups = Counter()
    for row,count in extra.items():
        extra_groups[row[2],row[1]] += count
    equal('nonterminal_arrival_projection', [r for r in roster if (r[2],r[1]) not in terms],
          [r for r in observed if (r[2],r[1]) not in terms])
    equal('contiguous_flow_order',[e.order for e in events],list(range(len(events))))
    equal('unique_flow_keys',len({(e.bus,e.trip,e.station,e.direction) for e in events}),len(events))
    equal('flow_arrival_count',len(events),5502)
    equal('flow_query_count',len(flow['queries']),5200)
    equal('native_query_count',len(native['queries']),5200)
    launches = {r['trip']:r for r in native['dispatches']}
    departures = {}
    for r in native['departures']:
        key = r['trip_id'],r['station_id'],r['direction']
        if key in departures:
            equal('duplicate_departure_clock',departures[key],r['time_s'])
        departures[key] = r['time_s']
    previous_stop, projected = {}, []
    first, completion_match, terminal_clock_only = Counter(),0,0
    for e in events:
        try:
            e.validate()
        except AssertionError as error:
            checks['seat_and_event_clocks'] = False
            violations['seat_and_event_clocks'] += 1
            first_errors.setdefault('seat_and_event_clocks',str(error))
        checks.setdefault('seat_and_event_clocks',True)
        equal('all_visits_complete',e.completed_s is not None,True)
        equal('actual_trip_vehicle',(e.bus,e.direction),(launches[e.trip]['bus'],launches[e.trip]['direction']))
        equal('APC_availability_clock',e.available_s,e.arrival_s)
        key = e.trip,e.station,e.direction
        if key in departures:
            equal('observed_departure_completion',e.completed_s,departures[key])
            completion_match += 1
        elif e.key in terms:
            equal('terminal_return_completion',e.completed_s,e.arrival_s)
            terminal_clock_only += 1
        else:
            equal('nonterminal_departure_present',False,True)
        if e.key in previous_stop:
            projected.append((e.direction,e.station,e.arrival_s,e.available_s,e.completed_s,e.incoming,
                              e.boarded,e.alighted,e.capacity,e.arrival_s-previous_stop[e.key]))
        else:
            first['terminal' if e.key in terms else 'nonterminal'] += 1
        previous_stop[e.key] = e.arrival_s
    equal('legacy_APC_gap_completion_projection',Counter(projected),Counter(
        (v.direction,v.station,v.arrival_s,v.available_s,v.completed_s,v.incoming_load,v.boarded,
         v.alighted,v.capacity,v.gap_s) for v in old_graph['completed_visits']))
    equal('first_and_legacy_visit_counts',(sum(first.values()),len(projected)),(42,5460))
    equal('actual_command_receipts',
          [(q['time_s'],q['bus'],q['trip'],q['station'],q['action_s']) for q in native['queries']],
          [(h['time_s'],h['bus_id'],h['trip_id'],h['station_id'],h['action_s']) for h in native['holds']])
    by_key = {(e.bus,e.trip,e.station,e.direction):e for e in events}
    last_n,last_time,blocks,unavailable_queries = 0,float('-inf'),0,0
    completion_margin,availability_margin = [],[]
    equal('query_boundary_count',len(flow['queries']),len(native['queries']))
    for i,(captured,q) in enumerate(zip(flow['queries'],native['queries'])):
        n,now = captured['arrival_seen_count'],q['time_s']
        equal('query_order',last_n <= n <= len(events) and now >= last_time,True)
        visible = events[:n]
        equal('no_future_arrival_in_prefix',all(e.arrival_s <= now for e in visible),True)
        equal('no_missing_past_arrival',all(e.arrival_s >= now for e in events[n:]),True)
        equal('exact_query_boundary',captured,boundary(visible,q,i))
        own = by_key[q['bus'],q['trip'],q['station'],q['direction']]
        equal('current_arrival_visible',own.order < n,True)
        equal('query_APC',(q['arrival_s'],q['boarded'],q['load']),
              (own.arrival_s,own.boarded,own.incoming-own.alighted+own.boarded))
        if captured['usable_strict_count']:
            completion_margin.append(now-captured['max_completed_s'])
            availability_margin.append(now-captured['max_available_s'])
        try:
            evidence = completed_flow(visible,now)
            checks.setdefault('frozen_flow_derivation',True)
            unavailable_queries += evidence['unavailable_visits'] > 0
            blocks += len(evidence['exact_blocks'])
            equal('derived_label_clocks_strict',all(b['max_completed_s'] < now and b['max_available_s'] < now
                  for b in evidence['exact_blocks']),True)
        except AssertionError as error:
            checks['frozen_flow_derivation'] = False
            violations['frozen_flow_derivation'] += 1
            first_errors.setdefault('frozen_flow_derivation',str(error))
        last_n,last_time = n,now
    equal('strict_completion_margin',bool(completion_margin) and min(completion_margin) > 0,True)
    equal('strict_availability_margin',bool(availability_margin) and min(availability_margin) > 0,True)
    metrics = ('passenger_person_s','journey_min','cv_squared','readiness_delay_s',
               'public_launch_delay_total_s','holding_s_per_trip','trips_unlaunched','trips_incomplete')
    latency = lambda rows:dict(mean=float(np.mean(rows)),p95=float(np.quantile(rows,.95)),max=float(max(rows)))
    summary = dict(source=a.source,execution_source='bed2255dca',status='read_only_failed_gate_review_complete',
        original_task='t211740',original_error=failure['error'],original_verdict_preserved='failed_slot_consumed',
        checks=checks,violation_counts=dict(violations),first_errors=first_errors,
        endpoint_deltas={m:result['results'][m]-previous['results'][m] for m in metrics},
        counts=dict(flow_arrivals=len(events),native_arrivals=len(observed),queries=len(flow['queries']),
            extra_flow_events=sum(extra.values()),missing_flow_events=sum(missing.values()),
            extra_nonterminal=sum(n for r,n in extra.items() if (r[2],r[1]) not in terms),
            extra_terminal=sum(n for r,n in extra.items() if (r[2],r[1]) in terms),
            first_visits=dict(first),legacy_visits=len(projected),full_visits=sum(not e.cleared for e in events),
            zero_seat_visits=sum(e.free_before_boarding == 0 for e in events),
            same_clock_intervals=sum(r[-1] == 0 for r in projected),
            completions_matched_to_departure=completion_match,terminal_completions_without_departure_record=terminal_clock_only),
        arrival_difference=dict(by_direction_station={str(k):n for k,n in extra_groups.items()},examples=list(extra)[:4]),
        strict_prefix=dict(queries_with_unavailable_visits=unavailable_queries,exact_blocks_across_queries=blocks,
            minimum_completion_margin_s=min(completion_margin) if completion_margin else None,
            minimum_availability_margin_s=min(availability_margin) if availability_margin else None),
        graph_latency_s=latency([q['total_s'] for q in graph['queries']]),
        native_flow_and_whole_callback_latency='not persisted before the original failure; cannot recover without new evidence',
        physical_slots_consumed=1,physical_start_cap=1,new_traffic_paths=0,training_updates=0,
        automatic_retry=False,main_policy_changed=False,controller_admitted=False,
        wall_s=time.perf_counter()-start,
        scope='Read-only archived-path diagnosis; no frozen verdict rewrite, replay, fitting or policy benefit claim.')
    write(a.out_root/'results.json',summary)
    print('V274_READ_ONLY_REVIEW_COMPLETE',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    review(p.parse_args())


if __name__ == '__main__':
    main()
