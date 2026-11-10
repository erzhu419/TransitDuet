"""Query-local passenger/event model. Inputs are observations, not simulator futures."""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from freqduet_event_graph_design import ACTIONS


MIN_VISITS = 3
BEAM_WIDTH, MAX_VARIABLES, MAX_EVALUATIONS, MAX_EVENTS = 4, 6, 128, 12000


@dataclass(frozen=True)
class Stop:
    station: int
    control: bool = True
    terminal: bool = False


@dataclass(frozen=True)
class Visit:
    direction: bool
    station: int
    arrival_s: float
    completed_s: float
    available_s: float
    run_s: float
    incoming_load: float
    boarded: float
    alighted: float
    gap_s: float
    capacity: float = 50.
    left_behind: float = 0.


@dataclass(frozen=True)
class Vehicle:
    bus: int
    trip: int
    direction: bool
    index: int
    phase: str
    load: float
    capacity: float = 50.
    remaining_s: float = 0.
    remaining_fraction: float = 1.
    ready_s: float = 0.
    executed_h_s: float = 0.
    legal: tuple = (0, 5, 10, 15, 20, 30, 45)
    arrival_s: float | None = None
    boarded: float | None = None


@dataclass(frozen=True)
class Plan:
    trip: int
    direction: bool
    release_s: float


@dataclass(frozen=True)
class Snapshot:
    now_s: float
    sampled_s: float
    routes: dict
    vehicles: tuple
    plans: tuple
    current_bus: int
    native_action_s: int


class Unsupported(ValueError):
    """The causal inputs cannot support a complete comparison; retain native action."""


def from_capture(capture, routes, cursors, current_bus, native_action_s, sampled_s):
    """Use capture_fleet plus same-query route cursors; never read future bus labels.

    Cursor entries are (next route index, remaining link fraction, executed H,
    current legal seconds, observed arrival, observed APC board count).
    Countdown telemetry is already public.
    """
    phases = dict(TRAVEL='travel', HOLDING='service', WAITING_ACTION='prepare', DWELLING='hold', TERMINAL='terminal')
    vehicles = []
    for row in capture['fleet']:
        index, fraction, executed, legal, arrival, boarded = cursors[row['bus_id']]
        vehicles.append(Vehicle(row['bus_id'], row['active_trip_id'] if row['on_route'] else -1,
            row['direction'], index, phases[row['state']], row['load'], row['capacity'],
            row['remaining_command_hold_s'] + row['remaining_service_s'], fraction,
            row['terminal_return_s'] if row['terminal_return_s'] is not None else -1., executed, tuple(legal), arrival, boarded))
    plans = tuple(Plan(p['trip_id'], p['direction'], p['published_release_s'])
                  for p in capture['plans'] if not p['launched'])
    return Snapshot(capture['time_s'], sampled_s, routes, tuple(vehicles), plans, current_bus, native_action_s)


@dataclass
class EventGraph:
    snapshot: Snapshot
    forecasts: dict
    completed: dict
    horizon_s: float
    initial_queues: dict | None = None

    def evaluate(self, commands=None, homogeneous=False):
        return _simulate(self, {} if commands is None else commands, homogeneous)


def assemble(snapshot, visits):
    q = snapshot.now_s
    if q != snapshot.sampled_s:
        raise Unsupported('current state was not sampled at this query')
    vehicles = {v.bus: v for v in snapshot.vehicles}
    if len(vehicles) != len(snapshot.vehicles) or not vehicles:
        raise ValueError('physical vehicle IDs must be unique')
    current = vehicles[snapshot.current_bus]
    if current.phase != 'prepare' or snapshot.native_action_s not in current.legal:
        raise ValueError('current query must be uncommitted and its native action legal')
    if set(snapshot.routes) != {False, True}:
        raise Unsupported('both physical directions are required')
    keys = {(d, s.station) for d, route in snapshot.routes.items() for s in route}
    groups = {k: [] for k in keys}
    for v in visits:
        if v.arrival_s > v.completed_s or v.arrival_s > v.available_s:
            raise ValueError('completed APC/AVL clocks are out of order')
        # A label completing or arriving at this tick is not in the strict prefix.
        if v.completed_s < q and v.available_s < q and (v.direction, v.station) in groups:
            groups[v.direction, v.station].append(v)
    forecasts, completed = {}, {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: r.arrival_s)
        if len(rows) < MIN_VISITS:
            raise Unsupported('fewer than three completed visits at ' + str(key))
        if any(r.run_s <= 0 or r.gap_s <= 0 or not 0 <= r.alighted <= r.incoming_load
               or not 0 <= r.incoming_load - r.alighted + r.boarded <= r.capacity for r in rows):
            raise ValueError('invalid completed run or APC counts')
        # Board counts identify arrivals only when the station was not capacity censored.
        demand_rows = rows[-MIN_VISITS:]
        if any(r.left_behind or r.incoming_load - r.alighted + r.boarded >= r.capacity for r in demand_rows):
            raise Unsupported('capacity-censored APC demand needs an observable arrival forecast')
        incoming = sum(r.incoming_load for r in rows)
        forecasts[key] = dict(run_s=float(np.mean([r.run_s for r in rows])),
            rate=sum(r.boarded for r in demand_rows) / sum(r.gap_s for r in demand_rows),
            alight=sum(r.alighted for r in rows) / incoming if incoming else 0.,
            visits=len(rows), demand_visits=len(demand_rows))
        completed[key] = rows
    return finish_assembly(snapshot, forecasts, completed)


def finish_assembly(snapshot, forecasts, completed):
    """Shared physical validation; demand identification belongs to the assembler."""
    q = snapshot.now_s
    vehicles = {v.bus: v for v in snapshot.vehicles}
    longest = 0.
    for d, route in snapshot.routes.items():
        if not route or not route[-1].terminal or route[-1].control:
            raise ValueError('the terminal is a return event, never a holding control')
        if len({s.station for s in route}) != len(route):
            raise ValueError('route station IDs must be unique within direction')
        max_service = 2 * max(v.capacity for v in snapshot.vehicles) + 4
        longest = max(longest, sum(math.ceil(forecasts[d, s.station]['run_s']) +
            (0 if s.terminal else max_service + 1 + (45 if s.control else 1)) for s in route))
    for v in snapshot.vehicles:
        if v.phase not in ('terminal', 'travel', 'service', 'prepare', 'hold') or not 0 <= v.load <= v.capacity:
            raise ValueError('invalid current phase or APC load')
        if v.remaining_s < 0 or not 0 <= v.remaining_fraction <= 1 or v.ready_s > q:
            raise ValueError('current countdown, position or ready clock is invalid')
        if not v.legal or any(h not in ACTIONS for h in v.legal):
            raise ValueError('native masks contain only seven-bin actions')
        if v.phase != 'terminal' and not 0 <= v.index < len(snapshot.routes[v.direction]):
            raise ValueError('current route cursor is invalid')
        if v.phase in ('prepare', 'hold') and not snapshot.routes[v.direction][v.index].control:
            raise ValueError('no control or command may be invented at an illegal stop')
        if v.phase in ('prepare', 'hold', 'service') and (v.arrival_s is None or v.arrival_s > q):
            raise Unsupported('a current stop needs its observed arrival clock')
        if v.phase in ('prepare', 'hold', 'service') and v.boarded is None:
            raise Unsupported('a current stop needs its observed APC boarding count')
    if len({p.trip for p in snapshot.plans}) != len(snapshot.plans):
        raise ValueError('published trip IDs must be unique')
    # A lookahead bound, not operational slack or additional vehicle authority.
    horizon = math.ceil(q + longest + 360 * len(vehicles) + 1)
    return EventGraph(snapshot, forecasts, completed, horizon)


def _simulate(graph, commands, homogeneous, continuation=None):
    s, forecast, end = graph.snapshot, graph.forecasts, graph.horizon_s
    now, serial, events = s.now_s, 0, []
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
    last_release = s.now_s + 360 * len(vehicles)
    pending = {d: [p for p in s.plans if p.direction == d and p.release_s <= last_release] for d in (False, True)}
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
    active_count = len(initial)
    legal = {}
    fleet_order = {v.bus: i for i, v in enumerate(s.vehicles)}

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
            push(p.release_s, 0, 'dispatch')
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
        dt = t - now
        waiting += total_queue * dt + total_rate * dt * dt / 2
        total_queue += total_rate * dt
        onboard += (homogeneous_load * active_count if homogeneous else total_load) * dt
        now = t
        if len(nodes) >= MAX_EVENTS:
            invalid = 'event_limit'
            break
        nodes.append((t, kind, bus))
        if kind == 'dispatch':
            for d in (False, True):
                while pending[d] and pending[d][0].release_s <= now and ready[d]:
                    ready[d].sort()  # Current FIFO return time, then physical fleet order.
                    if ready[d][0][0] + 1 > now:
                        break
                    p = pending[d].pop(0)
                    _, chosen = ready[d].pop(0)
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
            queues[key] += forecast[key]['rate'] * (now - queue_clock[key])
            queue_clock[key] = now
            board = min(queues[key], v['cap'] - (v['load'] - alight))
            queues[key] -= board
            total_queue -= board
            v['load'] += board - alight
            total_load += board - alight
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
        dt = end - now
        waiting += total_queue * dt + total_rate * dt * dt / 2
        total_queue += total_rate * dt
        onboard += (homogeneous_load * active_count if homogeneous else total_load) * dt
    # Historical left boundaries stay fixed; future service order is candidate-specific.
    gaps = np.array([g for times in arrivals.values() for g in np.diff(sorted(times))], dtype=float)
    cv2 = float(np.var(gaps) / np.mean(gaps) ** 2) if len(gaps) and np.mean(gaps) > 0 else math.inf
    releases = {p.trip: p.release_s for p in s.plans if p.release_s <= last_release}
    delays = {trip: time - releases[trip] for trip, (time, _) in launches.items()}
    coverage = (initial <= returned.keys() and initial <= next_launch.keys()
                and releases.keys() <= launches.keys() and releases.keys() <= completed_trips)
    return dict(cost=waiting + onboard, waiting=waiting, onboard=onboard, cv2=cv2,
        returns={b: returned[b] for b in initial if b in returned},
        next_launch={b: next_launch[b] for b in initial if b in next_launch},
        launches=launches, delays=delays, actions=actions, legal=legal, nodes=nodes,
        initial_queue=initial_queue, boardings=boardings, return_events=return_events, arrivals=arrivals,
        missing_commands=set(commands) - used, coverage=bool(coverage), failure=invalid,
        endpoint_queue=total_queue, endpoint_load=total_load)


def feasible(candidate, reference):
    if candidate['failure'] or not candidate['coverage'] or candidate['missing_commands']:
        return False
    if candidate['cv2'] > reference['cv2'] + 1e-12:
        return False
    # No extra ready/dispatch delay may hide a cheaper passenger prediction.
    return (all(candidate['returns'][b] <= t + 1e-9 for b, t in reference['returns'].items())
            and all(candidate['next_launch'][b] <= t + 1e-9 for b, t in reference['next_launch'].items())
            and all(candidate['delays'].get(trip, math.inf) <= delay + 1e-9 for trip, delay in reference['delays'].items()))


def solve(graph, homogeneous=False, incumbent=None):
    """Bounded joint beam over complete plans; execute only the current first action."""
    ref = graph.evaluate(homogeneous=homogeneous)
    s = graph.snapshot
    current = next(v for v in s.vehicles if v.bus == s.current_bus)
    key = (current.bus, current.trip, s.routes[current.direction][current.index].station)
    fallback = dict(action_s=s.native_action_s, commands={}, evaluations=1,
        status='retain_native', reference=ref, selected=ref, optimality_claim=False)
    if ref['failure'] or not ref['coverage'] or not math.isfinite(ref['cv2']):
        return fallback
    def position(v):
        return v.index - (v.remaining_fraction if v.phase == 'travel' else 0.)

    peers = [v for v in s.vehicles if v.phase != 'terminal' and v.direction == current.direction and v.bus != current.bus]
    peers.sort(key=lambda v: (abs(position(v)-position(current)), v.bus))
    before = next((v for v in peers if position(v) < position(current)), None)
    after = next((v for v in peers if position(v) >= position(current)), None)
    neighbors = [v for v in (before, after) if v is not None]
    neighbors += [v for v in peers if v not in neighbors][:2-len(neighbors)]
    ordered = sorted(ref['actions'], key=lambda k: (ref['actions'][k][0], k))
    keys = [k for v in (current, *neighbors)
            for k in [k for k in ordered if k[:2] == (v.bus, v.trip)][:2]]
    keys = keys[:MAX_VARIABLES]
    beam, best, evaluations = [({}, ref)], ({}, ref), 1

    def consider(commands):
        nonlocal best, evaluations
        result = graph.evaluate(commands, homogeneous)
        evaluations += 1
        if feasible(result, ref) and result['cost'] < best[1]['cost'] - 1e-9:
            best = commands, result
        return commands, result

    if incumbent:
        warm = {k: h for k, h in incumbent.items() if k in ref['legal'] and h in ref['legal'][k]}
        if warm:
            beam.append(consider(warm))
    # Joint transfer proposals keep a useful coupled move from being pruned merely
    # because its one-coordinate intermediate violates a phase constraint.
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if left[:2] != right[:2]:
                continue
            total = ref['actions'][left][1] + ref['actions'][right][1]
            for h in ref['legal'][left]:
                other = total - h
                if other in ref['legal'][right] and evaluations < MAX_EVALUATIONS:
                    proposal = consider({left: h, right: other})
                    beam.append(proposal)

    def ranking(item):
        _, result = item
        if result['failure'] or not result['coverage'] or result['missing_commands']:
            return (2, math.inf, math.inf)
        violation = max(0., result['cv2'] - ref['cv2'])
        violation += sum(max(0., result['returns'][b] - t) for b, t in ref['returns'].items()) / 360.
        violation += sum(max(0., result['next_launch'][b] - t) for b, t in ref['next_launch'].items()) / 360.
        violation += sum(max(0., result['delays'].get(p, math.inf) - t) for p, t in ref['delays'].items()) / 360.
        return (int(violation > 1e-12), violation, result['cost'])

    beam = sorted(beam, key=ranking)[:BEAM_WIDTH]
    for k in keys:
        expanded = []
        for commands, _ in beam:
            for h in ref['legal'][k]:
                if evaluations >= MAX_EVALUATIONS:
                    break
                trial = dict(commands)
                trial[k] = h
                expanded.append(consider(trial))
            if evaluations >= MAX_EVALUATIONS:
                break
        if not expanded:
            break
        # Infeasible intermediate plans may survive: a later joint move can repair them.
        beam = sorted(expanded, key=ranking)[:BEAM_WIDTH]
        if evaluations >= MAX_EVALUATIONS:
            break
    commands, selected = best
    return dict(action_s=commands.get(key, s.native_action_s), commands=commands,
        evaluations=evaluations, status='model_improvement' if commands else 'retain_native',
        reference=ref, selected=selected, optimality_claim=False)


def request(snapshot, visits, homogeneous=False):
    """Pure callback boundary. Unsupported inputs cannot change the native request."""
    try:
        graph = assemble(snapshot, visits)
    except Unsupported as error:
        return dict(action_s=snapshot.native_action_s, commands={}, evaluations=0,
                    status='unsupported_prefix', reason=str(error), optimality_claim=False)
    return solve(graph, homogeneous)


class OnlinePlanner:
    """Future commands are revisable proposals, not permissions to spend past H."""

    def __init__(self, homogeneous=False):
        self.homogeneous = homogeneous
        self.proposals = {}

    def choose(self, snapshot, visits):
        try:
            graph = assemble(snapshot, visits)
        except Unsupported as error:
            self.proposals = {}
            return dict(action_s=snapshot.native_action_s, commands={}, evaluations=0,
                        status='unsupported_prefix', reason=str(error), optimality_claim=False)
        result = solve(graph, self.homogeneous, self.proposals)
        current = next(v for v in snapshot.vehicles if v.bus == snapshot.current_bus)
        current_key = (current.bus, current.trip, snapshot.routes[current.direction][current.index].station)
        self.proposals = {k: h for k, h in result['commands'].items() if k != current_key}
        return result
