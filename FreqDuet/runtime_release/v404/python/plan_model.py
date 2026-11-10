"""Isolated published-terminal-plan model; no slow policy or traffic entrypoint.

The event kernel below is copied from V277 and changes only plan-conditioned
dispatch clocks and native continuation targets. The lookahead stays unchanged.
"""

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np

from fast_continuation import FastArrivalMetricGraph, SparseContinuation
from freqduet_causal_event_graph import ACTIONS, MAX_EVENTS


@dataclass(frozen=True)
class PublishedPlan:
    trip: int
    direction: bool
    release_s: float
    target_s: float
    public_s: float
    min_gap_s: float | None = None
    first_eligible_s: float | None = None


def observe_plans(env):
    """Read current terminal commitments and original public clocks, never futures."""
    plans = []
    for trip in env.timetables:
        if trip.launched:
            continue
        if not getattr(trip, '_freqduet_terminal_dispatch', False):
            raise ValueError('plan prototype supports published terminal schedules only')
        plans.append(PublishedPlan(int(trip.launch_turn), bool(trip.direction),
            float(trip._freqduet_scheduled_launch), float(trip.target_headway),
            float(getattr(trip, '_freqduet_base_launch',
                          getattr(trip, '_original_launch', trip.launch_time))),
            getattr(trip, '_freqduet_min_dispatch_headway', None),
            getattr(trip, '_freqduet_first_eligible_time', None)))
    return tuple(plans), {d:float(env._last_dispatch_time[d]) for d in (False, True)}


class PlanContinuation(SparseContinuation):
    def initialize(self, vehicles):
        super().initialize(vehicles)
        # The actual runtime latches target at dispatch for the entire trip.
        self.targets = {bus:row['target'] for bus,row in self.data['fleet'].items()}

    def event(self, kind, vehicle, now, service_s=None):
        if kind == 'launch':
            self.targets[vehicle['bus']] = self.model.plan_by_trip[vehicle['trip']].target_s
        return super().event(kind, vehicle, now, service_s)

    def action(self, vehicle, command_s, legal):
        s = self.model.snapshot
        station = s.routes[vehicle['d']][vehicle['i']].station
        original = self.model.originals[vehicle['bus']]
        if (original.phase == 'prepare' and original.trip == vehicle['trip']
                and original.index == vehicle['i']):
            packet = self.data['packets'][vehicle['bus'], vehicle['trip'], station]
        else:
            prepare = command_s - 1
            progress = self.station_progress(vehicle['d'], vehicle['i'])
            packet = dict(bus=vehicle['bus'], trip=vehicle['trip'], station=station,
                direction=vehicle['d'], prepare_s=prepare, progress_m=progress,
                service_s=self.service[vehicle['bus']], target_s=self.targets[vehicle['bus']],
                predecessor=self.predecessors[vehicle['bus']],
                avl=self.follower_avl(prepare, vehicle['bus'], vehicle['d'], progress),
                departure=self.departure_at((vehicle['d'], station), prepare, vehicle['bus']))
        return self.model.native_rule(packet, legal)[0]


class PlanConsistentGraph(FastArrivalMetricGraph):
    """Evaluate published plans with native eligibility and fixed-trip targets.

    `delays` retains the planned-release delay used by V277. Public and native
    readiness delays are separate outputs; a later policy must explicitly use
    their intended constraints rather than equating these three clocks.
    """
    def __init__(self, base, observed, plans, last_dispatch_s):
        plans = tuple(plans)
        if [(p.trip, p.direction) for p in plans] != [(p.trip, p.direction) for p in base.snapshot.plans]:
            raise ValueError('published plans must retain the current unlaunched roster')
        for plan in plans:
            if not math.isfinite(plan.target_s) or plan.target_s <= 0:
                raise ValueError('published target must be positive and finite')
            if plan.min_gap_s is not None and (not math.isfinite(plan.min_gap_s) or plan.min_gap_s < 0):
                raise ValueError('published minimum dispatch gap must be nonnegative')
            if plan.first_eligible_s is not None and plan.first_eligible_s > base.snapshot.now_s:
                raise ValueError('first eligibility must already have been observed')
        super().__init__(base, observed)
        self.snapshot = replace(self.snapshot, plans=plans)
        self.plan_by_trip = {p.trip:p for p in plans}
        self.initial_last_dispatch_s = {d:float(last_dispatch_s[d]) for d in (False, True)}

    def evaluate(self, commands=None, homogeneous=False):
        result = _simulate_plan(self, commands or {}, homogeneous, PlanContinuation(self))
        future = {key:times[self.seed_sizes[key]:] for key,times in result['arrivals'].items()}
        result['cv2'] = self.arrival_prefix.extend(future).cv_squared
        return result


def _simulate_plan(graph, commands, homogeneous, continuation):
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
    last_dispatch = dict(graph.initial_last_dispatch_s)
    eligibility = {p.trip:p.first_eligible_s for p in s.plans if p.first_eligible_s is not None}
    gap_wakes = set()
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
        first_eligible_s={trip:eligibility[trip] for trip in launches},
        public_delays={trip:max(0., time-graph.plan_by_trip[trip].public_s)
                       for trip,(time,_) in launches.items()},
        readiness_delays={trip:time-eligibility[trip] for trip,(time,_) in launches.items()},
        initial_queue=initial_queue, boardings=boardings, return_events=return_events, arrivals=arrivals,
        missing_commands=set(commands) - used, coverage=bool(coverage), failure=invalid,
        endpoint_queue=total_queue, endpoint_load=total_load)

