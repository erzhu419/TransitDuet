"""Passive query boundaries and server-side parity checks for V274."""

from collections import Counter
from dataclasses import asdict

from freqduet_apc_flow_evidence import APCArrival, completed_flow


QUERY_ID = ('time_s', 'bus', 'trip', 'station', 'direction')
WALL_FIELDS = ('build_s', 'total_s')


def boundary(events, query, order):
    now = query['time_s']
    usable = [e for e in events if e.arrival_s < now and e.available_s < now
              and e.completed_s is not None and e.completed_s < now]
    return dict(query_order=order, **{k:query[k] for k in QUERY_ID},
                arrival_seen_count=len(events), usable_strict_count=len(usable),
                max_completed_s=max((e.completed_s for e in usable), default=None),
                max_available_s=max((e.available_s for e in usable), default=None))


class FlowRecorder:
    def __init__(self):
        self.queries = []

    def capture(self, prefix, query):
        self.queries.append(boundary(prefix.flow_events, query, len(self.queries)))

    def bundle(self, prefix, routes):
        return dict(events=[asdict(e) for e in prefix.flow_events], queries=self.queries,
                    terminals=[(d, route[-1].station) for d,route in routes.items()])


def compare_legacy(trace, result, graph, old_trace, old_result, old_graph, same):
    same(trace, old_trace, 'V274_native_trace')
    for field in ('results', 'contract', 'coverage', 'execution'):
        same(result[field], old_result[field], 'V274_'+field)
    same(set(graph), set(old_graph), 'V274_graph_keys')
    same(graph['completed_visits'], old_graph['completed_visits'], 'V274_completed_visits')
    def without_wall(rows):
        return [{k:v for k,v in row.items() if k not in WALL_FIELDS} for row in rows]
    same(without_wall(graph['queries']), without_wall(old_graph['queries']), 'V274_graph_queries')


def validate_flow(data, trace, graph, same):
    events = [APCArrival(**row) for row in data['events']]
    same([e.order for e in events], list(range(len(events))), 'V274_arrival_order')
    keys = [(e.trip,e.station,e.direction) for e in events]
    assert len(set(keys)) == len(keys), 'V274_duplicate_arrival'
    same([(e.trip,e.station,e.direction,e.arrival_s) for e in events],
         [(r['trip_id'],r['station_id'],r['direction'],r['time_s']) for r in trace['arrivals']],
         'V274_native_arrival_roster')
    launches = {r['trip']:r for r in trace['dispatches']}
    departures = {}
    for r in trace['departures']:
        key = r['trip_id'],r['station_id'],r['direction']
        if key in departures:
            same(departures[key], r['time_s'], 'V274_duplicate_departure_clock')
        departures[key] = r['time_s']
    terminals = set(map(tuple, data['terminals']))
    previous, projected, first = {}, [], 0
    for e in events:
        e.validate()
        assert e.completed_s is not None, 'V274_uncompleted_visit_at_cleared_endpoint'
        launch = launches[e.trip]
        same((e.bus,e.direction), (launch['bus'],launch['direction']), 'V274_actual_trip_vehicle')
        same(e.available_s, e.arrival_s, 'V274_native_APC_availability')
        completion = e.arrival_s if e.key in terminals else departures[e.trip,e.station,e.direction]
        same(e.completed_s, completion, 'V274_completion_clock')
        if e.key not in previous:
            first += 1
        else:
            projected.append((e.direction,e.station,e.arrival_s,e.available_s,e.completed_s,
                              e.incoming,e.boarded,e.alighted,e.capacity,e.arrival_s-previous[e.key]))
        previous[e.key] = e.arrival_s
    # Legacy visits are in completion order, not arrival order. Compare their
    # APC/gap multiplicities here; compare_legacy separately preserves list order.
    same(Counter(projected), Counter((v.direction,v.station,v.arrival_s,v.available_s,v.completed_s,
         v.incoming_load,v.boarded,v.alighted,v.capacity,v.gap_s) for v in graph['completed_visits']),
         'V274_legacy_APC_and_gap_records')
    same([(q['time_s'],q['bus'],q['trip'],q['station'],q['action_s']) for q in trace['queries']],
         [(h['time_s'],h['bus_id'],h['trip_id'],h['station_id'],h['action_s']) for h in trace['holds']],
         'V274_command_receipts')
    same(len(data['queries']), len(trace['queries']), 'V274_flow_query_count')
    event_by_key = {(e.bus,e.trip,e.station,e.direction):e for e in events}
    last_seen, last_time, labels = 0, float('-inf'), 0
    for i,(captured,q) in enumerate(zip(data['queries'],trace['queries'])):
        n,now = captured['arrival_seen_count'], q['time_s']
        assert last_seen <= n <= len(events) and now >= last_time, 'V274_query_order'
        assert all(e.arrival_s <= now for e in events[:n]), 'V274_future_arrival_in_visible_prefix'
        assert all(e.arrival_s >= now for e in events[n:]), 'V274_missing_past_arrival'
        same(captured, boundary(events[:n],q,i), 'V274_query_visible_support')
        own = event_by_key[q['bus'],q['trip'],q['station'],q['direction']]
        assert own.order < n, 'V274_current_arrival_not_visible'
        same((q['arrival_s'],q['boarded'],q['load']),
             (own.arrival_s,own.boarded,own.incoming-own.alighted+own.boarded), 'V274_query_APC')
        evidence = completed_flow(events[:n],now)
        for block in evidence['exact_blocks']:
            assert block['max_completed_s'] < now and block['max_available_s'] < now
        labels += len(evidence['exact_blocks'])
        last_seen, last_time = n,now
    same_clock = sum(row[-1] == 0 for row in projected)
    counts = dict(first_arrivals=first, full_visits=sum(not e.cleared for e in events),
                  zero_seat_visits=sum(e.free_before_boarding == 0 for e in events),
                  same_clock_intervals=same_clock)
    return dict(arrivals=len(events), queries=len(data['queries']), completed_visits=len(projected),
                exact_blocks_across_queries=labels, cases=counts,
                case_status={k:'exercised' if n else 'not_exercised' for k,n in counts.items()},
                all_query_clocks_strict=True, legacy_roster_and_APC_match=True)
