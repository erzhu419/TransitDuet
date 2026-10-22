"""Query-time observations and separate realized supply labels for V215."""

import math

import preregister_freqduet_v215_positive_event as protocol


def plans(env):
    return {int(t.launch_turn): dict(trip_id=int(t.launch_turn), direction=bool(t.direction), launched=False,
        planned_s=float(getattr(t, '_freqduet_scheduled_launch', t.launch_time)),
        public_s=float(getattr(t, '_freqduet_base_launch', getattr(t, '_original_launch', t.launch_time))),
        target_s=float(t.target_headway)) for t in env.timetables if not t.launched}


def snapshot(env, stamp):
    ready, active = [], []
    now = float(env.current_time)
    assert stamp['time_s'] == now
    for b in env.bus_all:
        row = dict(bus_id=int(b.bus_id), direction=bool(b.direction), position_sample_s=now)
        if not b.on_route:
            returned = float(b.back_to_terminal_time)
            assert math.isfinite(returned) or returned == -math.inf
            ready.append(dict(row, return_observed=math.isfinite(returned),
                last_observed_return_s=returned if math.isfinite(returned) else None))
        else:
            active.append(dict(row, trip_id=int(b.trip_id), state=b.state.name,
                last_station_id=int(b.last_station.station_id), next_station_id=int(b.next_station.station_id),
                progress_m=float(b.travel_distance),
                remaining_distance_m=float(sum(r.distance for r in b.effective_route))-float(b.travel_distance)))
    assert len(ready)+len(active) == 12
    return dict(stamp, snapshot_read_s=now, snapshot_event_seq=stamp['event_seq'],
                ready=ready, active=active, plans=list(plans(env).values()))


class EventClock:
    def __init__(self):
        self.events = []

    def stamp(self, kind, now, **identity):
        assert not self.events or now >= self.events[-1]['time_s'], 'native event clock reversed'
        row = dict(event_seq=len(self.events), time_s=float(now), kind=kind, **identity)
        self.events.append(row)
        return dict(event_seq=row['event_seq'], time_s=row['time_s'])


def supply_label(anchor, returns, launches, events, departures, query):
    """Future-only label; first same-vehicle launch, never search for a later match."""
    if anchor is None:
        return dict(anchor_observed=False)
    returned = next((r for r in returns if (r['bus_id'], r['trip_id']) ==
                     (anchor['bus_id'], anchor['trip_id'])), None)
    if returned is None:
        return dict(anchor_observed=True, returned=False, same_vehicle_due_supply=False)
    r_event = next(e for e in events if e['kind'] == 'return' and e['trip_id'] == anchor['trip_id'])
    later = [e for e in events if e['kind'] == 'launch' and e['bus_id'] == anchor['bus_id']
             and e['event_seq'] > r_event['event_seq']]
    next_event = later[0] if later else None
    next_launch = next((r for r in launches if next_event is not None and r['trip'] == next_event['trip_id']), None)
    departure = next((r for r in departures if (r['trip_id'], r['station_id'], r['direction']) ==
                      (anchor['trip_id'], anchor['station_id'], anchor['inbound_direction'])), None)
    executed = None if departure is None else departure['realized_holding']-query['executed_holding_before_s']
    matched = (next_launch is not None and next_launch['direction'] == anchor['target_direction']
               and next_launch['trip'] in anchor['due_trip_ids'])
    return dict(anchor_observed=True, returned=True, actual_return=returned,
        next_same_vehicle_launch=next_launch, same_vehicle_due_supply=matched,
        anchor_command_s=protocol.ACTIONS[anchor['base_bin']], executed_at_anchor_s=executed,
        anchor_holding_execution_matches=executed == protocol.ACTIONS[anchor['base_bin']],
        return_event_seq=r_event['event_seq'], next_launch_event_seq=None if next_event is None else next_event['event_seq'],
        interpretation='Observed supply chain only;no counterfactual timing or passenger benefit inferred.')


def validate_capture(queries, fresh, clock):
    events = clock.events
    assert [e['event_seq'] for e in events] == list(range(len(events)))
    assert all(a['time_s'] <= b['time_s'] for a, b in zip(events, events[1:]))
    for q in queries:
        assert all(k in q for k in protocol.LOWER_FIELDS) and protocol.current_snapshot(q)
        p = events[q['prepare_event_seq']]
        assert p['kind'] == 'prepare' and p['event_seq'] < q['event_seq']
        assert p['time_s'] == q['state_captured_s'] <= q['time_s']
        for k in ('bus_id', 'trip_id', 'station_id', 'direction'):
            assert p[k] == q[k], 'query and native observation belong to different vehicle/stop'
        assert len(q['ready'])+len(q['active']) == 12
    assert all(protocol.current_snapshot(q) and q['fresh_upper'] for q in fresh)
    return dict(queries=len(queries), fresh_upper_queries=len(fresh), native_events=len(events),
        prepare_queries_linked=True, all_query_snapshots_complete=True, global_event_order_valid=True)
