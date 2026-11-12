"""Forecast terminal supply at public slots without a live holding decision."""

from collections import defaultdict
import math

import freqduet_causal_event_graph as graph
from freqduet_fleet_observation import capture_fleet
from freqduet_native_continuation import inputs
from frozen_v406 import ReferenceGraph, historical_rates, partial_forecasts
from plan_model import observe_plans


def public_clock(trip):
    return float(getattr(trip, '_freqduet_base_launch', getattr(trip, '_original_launch', trip.launch_time)))


def slot_schedule(trips, leads):
    schedule = defaultdict(list)
    for trip in trips:
        due = public_clock(trip)
        for lead in leads:
            schedule[due-lead].append((int(trip.launch_turn), bool(trip.direction), due, lead))
    return dict(schedule)


def slot_snapshot(env, bridge, captured):
    """Same physical cursor and issued-command clock as the native query snapshot."""
    now = captured['time_s']
    routes, prefix = bridge.routes, bridge.prefix
    index = {d: {s.station: i for i, s in enumerate(route)} for d, route in routes.items()}
    rows = {r['bus_id']: r for r in captured['fleet']}
    actions = tuple(int(a) for a in graph.ACTIONS)
    cursors = {}
    for bus in env.bus_all:
        bid, trip, direction = int(bus.bus_id), int(bus.trip_id), bool(bus.direction)
        if not bus.on_route:
            cursors[bid] = (0, 1., 0., actions, None, None)
            continue
        travel = bus.state.name == 'TRAVEL'
        station = int((bus.next_station if travel else bus.last_station).station_id)
        observed = None if travel else prefix.current[bid, trip, station, direction]
        fraction = min(1., max(0., float(bus.next_station_dis)/float(bus.current_route.distance))) if travel else 1.
        spent = float(bus.episode_hold_vehicle_seconds) - prefix.launches[trip]['holding_counter_at_launch']
        assert spent >= 0
        cursors[bid] = (index[direction][station], fraction, spent,
            bridge.masks.get((bid, trip, station), actions),
            observed['arrival_s'] if observed else None, observed['boarded'] if observed else None)
    for command in bridge.credit.stream.pending_commands:
        row = rows[command.bus_id]
        cursor = cursors[command.bus_id]
        assert command.time_s == now and row['state'] == 'WAITING_ACTION'
        assert row['active_trip_id'] == command.trip_id and command.action_s in cursor[3]
        assert routes[row['direction']][cursor[0]].station == command.station_id
        row.update(state='DWELLING', remaining_command_hold_s=max(1., command.action_s)+1.,
                   remaining_service_s=0.)
    # A sentinel outside the physical fleet prevents any root-action override.
    return graph.from_capture(captured, routes, cursors, -1, 0, now)


def forecast(env, bridge, captured):
    value = slot_snapshot(env, bridge, captured)
    now = value.now_s
    prefix = bridge.prefix
    visits = prefix.strict(now)
    observed = inputs(env, prefix, bridge.evidence)
    assert all(t <= now for _, _, _, t, _ in observed['arrivals'])
    assert all(t < now for _, _, _, t, _ in observed['departures'])
    for v in value.vehicles:
        if v.phase == 'prepare':
            key = v.bus, v.trip, value.routes[v.direction][v.index].station
            packet = observed['packets'].get(key)
            if packet is None or packet['prepare_s'] >= now:
                raise graph.Unsupported('missing strictly prior PREPARE evidence')
    bases, censored = partial_forecasts(value, visits, prefix.flow_events, historical_rates(env, now))
    if not censored:
        bases = bases[:1]
    plans, last = observe_plans(env)
    result = []
    for i, base in enumerate(bases):
        row = ReferenceGraph(base, observed, plans, last, {}).evaluate()
        result.append(dict(scenario=i, failure=row['failure'],
            returns={int(bus): t for bus, t in row['returns'].items()},
            launches={int(trip): [t, int(bus)] for trip, (t, bus) in row['launches'].items()}))
    return result, len(censored)


class PublicSlots:
    def __init__(self, leads):
        self.leads, self.records, self.schedule = tuple(leads), [], None

    def before_step(self, env, bridge):
        now = float(env.current_time)
        if self.schedule is None:
            self.schedule = slot_schedule(env.timetables, self.leads)
        due = self.schedule.get(now, ())
        if not due:
            return
        assert len(env.bus_all) in (0, 12)
        initialized = bool(env.bus_all)
        models, censored, reason = [], None, None
        fleet = capture_fleet(env)
        if initialized:
            try:
                models, censored = forecast(env, bridge, fleet)
            except graph.Unsupported as error:
                reason = str(error)
        else:
            reason = 'initial pool not observed'
        for trip, direction, public_s, lead in due:
            earlier = [int(t.launch_turn) for t in env.timetables if not t.launched
                and bool(t.direction) == direction and (public_clock(t), t.launch_turn) < (public_s, trip)]
            ready = [int(bus.bus_id) for bus in env.bus_in_terminal if bool(bus.direction) == direction]
            cohort = {r['bus_id']: r['active_trip_id'] for r in fleet['fleet']
                      if r['on_route'] and r['direction'] != direction}
            self.records.append(dict(query_s=now, trip=trip, public_s=public_s, lead_s=lead,
                direction=direction, earlier=earlier, ready=ready, cohort=cohort,
                censored_keys=censored, scenarios=models, unavailable=reason))

    def finish(self, start, end):
        expected = {(trip, lead) for tick, slots in self.schedule.items() if start <= tick <= end
                    for trip, _, _, lead in slots}
        actual = {(r['trip'], r['lead_s']) for r in self.records}
        assert expected == actual and len(actual) == len(self.records)
        assert all(math.isfinite(r['query_s']) and r['query_s']+r['lead_s'] == r['public_s']
                   for r in self.records)
