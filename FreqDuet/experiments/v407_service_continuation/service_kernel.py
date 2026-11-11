"""Finite-window service continuation copied from the frozen physical kernel."""

import heapq
import math

import numpy as np

from freqduet_causal_event_graph import ACTIONS, MAX_EVENTS


def simulate_service(graph, commands, homogeneous, continuation):
    s, forecast = graph.snapshot, graph.forecasts
    end = min(graph.horizon_s, graph.evaluation_end_s)
    demand_end = graph.demand_end_s
    now, serial, events = s.now_s, 0, []
    assert end >= now

    def inflow_time(left, right):
        return max(0., min(right, demand_end) - min(left, demand_end))
    vehicles = {v.bus: dict(bus=v.bus, trip=v.trip, d=v.direction, i=v.index,
        phase=v.phase, load=v.load, cap=v.capacity, legal=v.legal, frontier=now) for v in s.vehicles}
    originals = {v.bus: v for v in s.vehicles}
    arrivals = {k: [r.arrival_s for r in rows[-2:]] for k, rows in graph.completed.items()}
    departures = ({k: sorted(r.completed_s for r in rows)[-2:] for k, rows in graph.completed.items()}
                  if continuation is None else None)
    queues = {}
    for k, rows in graph.completed.items():
        if graph.initial_queues is not None:
            queues[k] = graph.initial_queues[k]
            continue
        last, amount = rows[-1].arrival_s, 0.
        observed = sorted((v.arrival_s, v.bus, v.boarded) for v in s.vehicles
                          if v.phase in ('prepare', 'hold', 'service') and
                          (v.direction, s.routes[v.direction][v.index].station) == k and v.arrival_s > last)
        for t, _, boarded in observed:
            amount = max(0., amount + forecast[k]['rate'] * (t - last) - boarded)
            last = t
        queues[k] = amount + forecast[k]['rate'] * (now - last)
    ready = {False: [], True: []}
    last_dispatch = dict(graph.initial_last_dispatch_s)
    eligibility = {p.trip:p.first_eligible_s for p in s.plans if p.first_eligible_s is not None}
    gap_wakes = set()
    due_plans = {p.trip: p for p in s.plans if math.ceil(p.release_s) <= end}
    pending = {d: [p for p in due_plans.values() if p.direction == d] for d in (False, True)}
    initial = {v.bus for v in s.vehicles if v.phase != 'terminal'}
    returned, next_launch, launches, actions = {}, {}, {}, {}
    completed_trips = set()
    nodes, used = [], set()
    boardings, return_events = [], []
    initial_queue = dict(queues)
    waiting = onboard = 0.
    homogeneous_load = float(np.mean([v.load for v in s.vehicles if v.phase != 'terminal'])) if initial else 0.
    total_rate = sum(f['rate'] for f in forecast.values())
    total_queue = sum(queues.values())
    queue_clock = {k: now for k in queues}
    total_load = sum(v.load for v in s.vehicles)
    initial_stock = total_queue + total_load
    delivered = 0.
    active_count = len(initial)
    legal = {}
    fleet_order = {v.bus: i for i, v in enumerate(s.vehicles)}

    def advance(t):
        nonlocal now, waiting, onboard, total_queue
        dt, active = t-now, inflow_time(now, t)
        waiting += total_queue * dt + total_rate * active * (dt - active / 2)
        total_queue += total_rate * active
        onboard += (homogeneous_load * active_count if homogeneous else total_load) * dt
        now = t

    def push(t, priority, kind, bus=-1):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (max(now, float(t)), priority, fleet_order.get(bus, -1), serial, kind, bus))

    def stage(v, t, kind):
        v['frontier'] = t
        push(t, 1, kind, v['bus'])

    for d in pending:
        # Preserve native roster order, including overdue trips; no free assignment.
        for p in pending[d]:
            push(math.ceil(p.release_s), 0, 'dispatch')
    for v in s.vehicles:
        row = vehicles[v.bus]
        if v.phase == 'terminal':
            ready[v.direction].append((v.ready_s, v.bus))
            push(v.ready_s + 1, 0, 'dispatch')
        elif v.phase == 'travel':
            remaining = math.ceil(forecast[v.direction, s.routes[v.direction][v.index].station]['run_s'] * v.remaining_fraction)
            stage(row, now + max(0, remaining - 1), 'arrival')
        elif v.phase == 'service':
            control = s.routes[v.direction][v.index].control
            stage(row, now + max(0, math.ceil(v.remaining_s) - (not control)), 'prepare')
        elif v.phase == 'hold':
            # The remaining part of an irreversible command is never re-optimized.
            stage(row, now + max(0, math.ceil(v.remaining_s) - 1), 'departure')
        else:
            stage(row, now, 'prepare')
        if v.arrival_s is not None:
            arrivals[v.direction, s.routes[v.direction][v.index].station].append(v.arrival_s)
    push(now, 0, 'dispatch')
    if continuation is not None:
        continuation.initialize(vehicles)
    invalid = None
    while events and events[0][0] <= end:
        t, _, _, _, kind, bus = heapq.heappop(events)
        advance(t)
        if len(nodes) >= MAX_EVENTS:
            invalid = 'event_limit'
            break
        nodes.append((t, kind, bus))
        if kind == 'dispatch':
            gap_wakes.discard(now)
            for d in (False, True):
                # Native checks every due trip in roster order, even when no
                # vehicle is ready. A blocked earlier trip does not hide a later
                # trip with a different published minimum gap.
                for p in tuple(pending[d]):
                    if p.release_s > now:
                        continue
                    due = math.ceil(max(p.release_s, last_dispatch[d] + p.min_gap_s
                        if p.min_gap_s is not None else p.release_s))
                    if due > now:
                        if due not in gap_wakes:
                            push(due, 0, 'dispatch')
                            gap_wakes.add(due)
                        continue
                    eligibility.setdefault(p.trip, now)
                    ready[d].sort()
                    if not ready[d] or ready[d][0][0] + 1 > now:
                        continue
                    pending[d].remove(p)
                    _, chosen = ready[d].pop(0)
                    last_dispatch[d] = now
                    v = vehicles[chosen]
                    launches[p.trip] = (now, chosen)
                    if chosen in returned:
                        next_launch.setdefault(chosen, now)
                    v.update(trip=p.trip, d=d, i=0, phase='travel', load=0., legal=tuple(ACTIONS.tolist()))
                    active_count += 1
                    stage(v, now + math.ceil(forecast[d, s.routes[d][0].station]['run_s']), 'arrival')
                    if continuation is not None:
                        continuation.event('launch', v, now)
            continue
        v = vehicles[bus]
        stop = s.routes[v['d']][v['i']]
        key = (v['d'], stop.station)
        action_key = (bus, v['trip'], stop.station)
        if kind == 'arrival':
            arrivals[key].append(now)
            if stop.terminal:
                completed_trips.add(v['trip'])
                delivered += v['load']
                total_load -= v['load']
                active_count -= 1
                v.update(load=0., phase='terminal', d=not v['d'])
                returned.setdefault(bus, now)
                return_events.append((now, bus))
                ready[v['d']].append((now, bus))
                push(now + 1, 0, 'dispatch')
                if continuation is not None:
                    continuation.event('return', v, now)
                continue
            alight = v['load'] * forecast[key]['alight']
            queues[key] += forecast[key]['rate'] * inflow_time(queue_clock[key], now)
            queue_clock[key] = now
            board = min(queues[key], v['cap'] - (v['load'] - alight))
            queues[key] -= board
            total_queue -= board
            v['load'] += board - alight
            total_load += board - alight
            delivered += alight
            boardings.append((now, bus, v['trip'], stop.station, board, v['load'], queues[key]))
            v['phase'] = 'service'
            # At controlled stops service completion prepares the query; the
            # command is supplied on the following step, not at the arrival tick.
            stage(v, now + math.ceil(max(alight, 2 * board) + 4) + stop.control, 'prepare')
            if continuation is not None:
                continuation.event('arrival', v, now, service_s=max(alight, 2 * board) + 4)
        elif kind == 'prepare':
            original = originals[bus]
            initial_node = v['trip'] == original.trip and v['i'] == original.index
            choices = v['legal'] if initial_node else tuple(ACTIONS.tolist())
            if not stop.control:
                h = 0
            else:
                if continuation is not None:
                    native = continuation.action(v, now, choices)
                else:
                    previous = departures[key][-1] if departures[key] else None
                    forward = now - previous if previous is not None else 360.
                    follower = []
                    for other in vehicles.values():
                        if other['bus'] == bus or other['phase'] == 'terminal' or other['d'] != v['d'] or other['i'] > v['i']:
                            continue
                        eta = other['frontier']
                        for j in range(other['i'] + 1, v['i'] + 1):
                            f = forecast[v['d'], s.routes[v['d']][j].station]
                            eta += f['run_s'] + 4
                        if eta > now:
                            follower.append(eta - now)
                    desired = .25 * ((min(follower) - forward) / 2 if follower else (360 - forward))
                    desired = min(45., max(0., desired))
                    native = min(choices, key=lambda a: (abs(a - desired), a))
                if bus == s.current_bus and now == s.now_s:
                    if continuation is not None:
                        assert native == s.native_action_s, 'same-state native continuation mismatch'
                    native = s.native_action_s
                h = commands.get(action_key, native)
                if h not in choices:
                    raise ValueError('requested holding is outside the native legal mask')
                if action_key in commands:
                    used.add(action_key)
                legal[action_key] = tuple(choices)
                actions[action_key] = (now, int(h))
            v['phase'] = 'hold'
            # Native _start_dwelling sets DWELLING even for zero. It cannot
            # depart until _process_dwelling on the next physical tick.
            stage(v, now + max(1, h), 'departure')
        elif kind == 'departure':
            if departures is not None:
                departures[key].append(now)
            v['i'] += 1
            v['phase'] = 'travel'
            target = s.routes[v['d']][v['i']]
            stage(v, now + math.ceil(forecast[v['d'], target.station]['run_s']), 'arrival')
            if continuation is not None:
                continuation.event('departure', v, now)
    if invalid is None:
        advance(end)
    # Historical left boundaries stay fixed; future service order is candidate-specific.
    gaps = np.array([g for times in arrivals.values() for g in np.diff(sorted(times))], dtype=float)
    cv2 = float(np.var(gaps) / np.mean(gaps) ** 2) if len(gaps) and np.mean(gaps) > 0 else math.inf
    releases = {trip: p.release_s for trip, p in due_plans.items()}
    delays = {trip: time - releases[trip] for trip, (time, _) in launches.items()}
    pending_ids = {p.trip for rows in pending.values() for p in rows}
    endpoint_queues = {k: amount + forecast[k]['rate'] * inflow_time(queue_clock[k], now)
                       for k, amount in queues.items()}
    generated = total_rate * inflow_time(s.now_s, now)
    balance_error = initial_stock + generated - delivered - total_queue - total_load
    coverage = (invalid is None and not (set(launches) & pending_ids)
                and set(launches) | pending_ids == set(due_plans)
                and abs(balance_error) < 1e-6
                and abs(sum(endpoint_queues.values()) - total_queue) < 1e-6)
    return dict(cost=waiting + onboard, waiting=waiting, onboard=onboard, cv2=cv2,
        returns={b: returned[b] for b in initial if b in returned},
        next_launch={b: next_launch[b] for b in initial if b in next_launch},
        launches=launches, delays=delays, actions=actions, legal=legal, nodes=nodes,
        first_eligible_s={trip: eligibility.get(trip) for trip in due_plans},
        public_delays={trip:max(0., launches.get(trip, (now, None))[0]-p.public_s)
                       for trip, p in due_plans.items()},
        readiness_delays={trip:max(0., launches.get(trip, (now, None))[0]-eligibility[trip])
                          if trip in eligibility else 0. for trip in due_plans},
        initial_queue=initial_queue, boardings=boardings, return_events=return_events, arrivals=arrivals,
        missing_commands=set(commands) - used, coverage=bool(coverage), failure=invalid,
        endpoint_queue=total_queue, endpoint_load=total_load,
        endpoint_queues=endpoint_queues, endpoint_vehicles=vehicles,
        due_trips=set(due_plans), pending_trips=pending_ids,
        unfinished_trips={v['trip'] for v in vehicles.values() if v['phase'] != 'terminal'},
        initial_stock=initial_stock, generated=generated, delivered=delivered,
        balance_error=balance_error, window_end_s=now, demand_end_s=demand_end)

