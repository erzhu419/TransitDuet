"""Native APC/AVL bridge for the frozen V270 graph; never executes shadow actions."""

from dataclasses import replace
import time

import numpy as np

import freqduet_causal_event_graph as graph
from freqduet_fleet_observation import capture_fleet


def native_query(env, bus, legal, action, pending):
    return dict(time_s=float(env.current_time), bus=int(bus.bus_id), trip=int(bus.trip_id),
        station=int(bus.last_station.station_id), direction=bool(bus.direction), phase=bus.state.name,
        load=len(bus.passengers), boarded=int(bus.last_board_count), arrival_s=float(bus.last_board_time),
        executed_h_s=float(bus.episode_hold_vehicle_seconds), legal=tuple(int(h) for h in legal),
        action_s=int(action), pending=[(c.time_s, c.bus_id, c.trip_id, c.station_id, c.action_s) for c in pending])


class CompletedPrefix:
    def __init__(self, launches):
        self.launches = launches
        self.arrivals, self.departures, self.current = {}, {}, {}
        self.visits = []
        self.finished = set()
        self.first_visits = 0

    def arrival(self, bus, now, station, direction, incoming):
        key = (int(bus.bus_id), int(bus.trip_id), int(station), bool(direction))
        stop = (bool(direction), int(station))
        previous = self.arrivals.get(stop)
        start = self.departures.get(key[:2], self.launches[key[1]]['time_s'])
        board, after = int(bus.last_board_count), len(bus.passengers)
        row = dict(direction=bool(direction), station=int(station), arrival_s=float(now),
            available_s=float(now), run_s=float(now)-start, incoming_load=float(incoming),
            boarded=float(board), alighted=float(incoming+board-after),
            gap_s=None if previous is None else float(now)-previous, capacity=float(bus.capacity))
        assert row['run_s'] > 0 and 0 <= row['alighted'] <= incoming
        assert 0 <= after <= bus.capacity
        self.current[key] = row
        self.arrivals[stop] = float(now)

    def complete(self, bus, trip, station, direction, now):
        key = (int(bus), int(trip), int(station), bool(direction))
        if key in self.finished:  # Terminal return and terminal recorder share one physical event.
            return
        row = self.current[key]
        assert row['arrival_s'] <= now
        self.finished.add(key)
        self.departures[key[:2]] = float(now)
        if row['gap_s'] is None:
            self.first_visits += 1
        else:
            # Left-behind is not observed from APC; full-capacity visits are
            # censored by the frozen assembler, rather than reading latent queues.
            self.visits.append(graph.Visit(**row, completed_s=float(now)))

    def strict(self, now):
        return tuple(v for v in self.visits if v.completed_s < now and v.available_s < now)


def native_routes(env):
    split = (len(env.stations)+1)//2
    sequences = {True: env.stations[:split], False: env.stations[split-1:]}
    routes = {d: tuple(graph.Stop(int(s.station_id), control=int(s.station_type) != 0,
                                 terminal=int(s.station_type) == 0) for s in seq[1:])
              for d, seq in sequences.items()}
    assert all(route[-1].terminal and not any(s.terminal for s in route[:-1]) for route in routes.values())
    return routes


def snapshot(env, prefix, routes, current_bus, legal, native_action, pending, masks):
    captured = capture_fleet(env)
    assert captured['buses'] == len(captured['fleet']) == 12
    rows = {r['bus_id']: r for r in captured['fleet']}
    indices = {d: {s.station: i for i, s in enumerate(route)} for d, route in routes.items()}
    cursors = {}
    for bus in env.bus_all:
        bid, trip, d = int(bus.bus_id), int(bus.trip_id), bool(bus.direction)
        row = rows[bid]
        if not bus.on_route:
            cursors[bid] = (0, 1., 0., tuple(graph.ACTIONS), None, None)
            continue
        travel = bus.state.name == 'TRAVEL'
        station = int((bus.next_station if travel else bus.last_station).station_id)
        index = indices[d][station]
        fraction = min(1., max(0., float(bus.next_station_dis)/float(bus.current_route.distance))) if travel else 1.
        observed = None if travel else prefix.current[bid, trip, station, d]
        spent = float(bus.episode_hold_vehicle_seconds) - prefix.launches[trip]['holding_counter_at_launch']
        assert spent >= 0
        current_mask = tuple(legal) if bid == current_bus else masks.get((bid, trip, station), tuple(graph.ACTIONS))
        cursors[bid] = (index, fraction, spent, current_mask,
                        observed['arrival_s'] if observed else None, observed['boarded'] if observed else None)
    locks = {}
    for command in pending:
        assert command.time_s == env.current_time and command.bus_id != current_bus
        row = rows[command.bus_id]
        assert row['state'] == 'WAITING_ACTION' and row['active_trip_id'] == command.trip_id
        cursor = cursors[command.bus_id]
        assert routes[row['direction']][cursor[0]].station == command.station_id
        assert command.action_s in cursor[3]
        # Issued this tick, not driven yet. Convert to the equivalent fixed
        # countdown expected by V270's previous-tick DWELLING representation.
        row['state'] = 'DWELLING'
        row['remaining_command_hold_s'] = max(1., command.action_s)+1.
        row['remaining_service_s'] = 0.
        locks[command.bus_id, command.trip_id, command.station_id] = env.current_time+max(1., command.action_s)
    value = graph.from_capture(captured, routes, cursors, current_bus, int(native_action), float(env.current_time))
    own = next(v for v in value.vehicles if v.bus == current_bus)
    assert own.phase == 'prepare' and value.routes[own.direction][own.index].station == int(
        next(b for b in env.bus_all if b.bus_id == current_bus).last_station.station_id)
    assert all(next(v for v in value.vehicles if v.bus == k[0]).phase == 'hold' for k in locks)
    return value, locks


class NativeNoop:
    def __init__(self, env, launches, credit):
        self.env, self.prefix, self.credit = env, CompletedPrefix(launches), credit
        self.routes = native_routes(env)
        self.planner = graph.OnlinePlanner()
        self.masks, self.queries = {}, []

    def attach(self, Bus):
        self.Bus = Bus
        self.recorder = type(self.env.headway_events)
        self.originals = (Bus.exchange_passengers, Bus.drive, self.recorder.record_departure)
        exchange, drive, departure = self.originals

        def exchange_passengers(bus, current_time, debug):
            station, direction, incoming = bus.next_station.station_id, bool(bus.direction), len(bus.passengers)
            result = exchange(bus, current_time, debug)
            self.prefix.arrival(bus, current_time, station, direction, incoming)
            return result

        def completed(recorder, station_id, direction, departure_time_s, trip_id):
            result = departure(recorder, station_id, direction, departure_time_s, trip_id)
            bus = self.prefix.launches[int(trip_id)]['bus_id']
            self.prefix.complete(bus, trip_id, station_id, direction, departure_time_s)
            return result

        def drive_bus(bus, current_time, *args, **kwargs):
            active, trip, direction, terminal = bus.on_route, bus.trip_id, bool(bus.direction), bus.effective_station[-1].station_id
            result = drive(bus, current_time, *args, **kwargs)
            if active and not bus.on_route:
                self.prefix.complete(bus.bus_id, trip, terminal, direction, bus.back_to_terminal_time)
            return result

        Bus.exchange_passengers, Bus.drive, self.recorder.record_departure = exchange_passengers, drive_bus, completed

    def detach(self):
        self.Bus.exchange_passengers, self.Bus.drive, self.recorder.record_departure = self.originals

    def observe(self, bus, legal, native_action):
        start = time.perf_counter()
        now = float(self.env.current_time)
        self.masks[int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id)] = tuple(legal)
        value, locks = snapshot(self.env, self.prefix, self.routes, int(bus.bus_id), legal, native_action,
                                self.credit.stream.pending_commands, self.masks)
        visits = self.prefix.strict(now)
        assert all(v.completed_s < now and v.available_s < now for v in visits)
        build_s = time.perf_counter()-start
        if any(v.gap_s <= 0 for v in visits):
            # Native simultaneous arrivals are valid, but not supported by the
            # frozen positive-gap demand estimator. Keep them, do not impute a gap.
            self.planner.proposals = {}
            result = dict(status='unsupported_nonpositive_gap', action_s=int(native_action), evaluations=0)
        else:
            result = self.planner.choose(value, visits)
        ref = result.get('reference')
        verified = 0
        if ref is not None:
            for key, departure_s in locks.items():
                assert key not in ref['legal'] and key not in result['commands']
                times = [t for t, kind, bid in ref['nodes'] if kind == 'departure' and bid == key[0]]
                if times:
                    assert min(times) == departure_s
                    verified += 1
        profile = dict(time_s=now, bus=int(bus.bus_id), trip=int(bus.trip_id), station=int(bus.last_station.station_id),
            strict_visits=len(visits), max_completed_s=max((v.completed_s for v in visits), default=None),
            max_available_s=max((v.available_s for v in visits), default=None), pending_locks=len(locks),
            model_locks_verified=verified, status=result['status'], evaluations=result['evaluations'],
            shadow_action_s=int(result['action_s']), native_action_s=int(native_action),
            build_s=build_s, total_s=time.perf_counter()-start)
        self.queries.append(profile)
        if len(self.queries) % 250 == 0:
            print('V271_NOOP_QUERIES', len(self.queries), 't', now, 'status', result['status'], flush=True)
        return profile

    def summary(self):
        timings = [r['total_s'] for r in self.queries]
        active = [r['total_s'] for r in self.queries if r['evaluations'] > 1]
        return dict(queries=len(timings), completed_visits=len(self.prefix.visits),
            first_arrivals_without_gap=self.prefix.first_visits,
            nonpositive_gap_visits=sum(v.gap_s <= 0 for v in self.prefix.visits),
            statuses={s: sum(r['status'] == s for r in self.queries) for s in sorted({r['status'] for r in self.queries})},
            pending_locks=sum(r['pending_locks'] for r in self.queries),
            model_locks_verified=sum(r['model_locks_verified'] for r in self.queries),
            shadow_action_differences=sum(r['shadow_action_s'] != r['native_action_s'] for r in self.queries),
            query_latency_s=dict(mean=float(np.mean(timings)), p95=float(np.quantile(timings, .95)),
                                 max=float(max(timings)), total=float(sum(timings))),
            optimizing_query_latency_s=None if not active else dict(count=len(active), mean=float(np.mean(active)),
                p95=float(np.quantile(active, .95)), max=float(max(active))),
            raw_query_records_remain_server_side=True, executed_shadow_actions=0)
