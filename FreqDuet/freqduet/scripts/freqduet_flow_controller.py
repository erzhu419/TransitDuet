"""V275: clearance-block forecasts with explicit baseline-fallback continuation."""

from collections import Counter
import math
import time

import numpy as np

import freqduet_causal_event_graph as graph
from freqduet_apc_flow_evidence import completed_flow
from freqduet_native_event_graph import NativeNoop, snapshot


def assemble(snapshot, visits, events):
    q = snapshot.now_s
    if q != snapshot.sampled_s:
        raise graph.Unsupported('stale query snapshot')
    vehicles = {v.bus:v for v in snapshot.vehicles}
    if not vehicles or len(vehicles) != len(snapshot.vehicles):
        raise ValueError('physical vehicle IDs must be unique')
    current = vehicles[snapshot.current_bus]
    if current.phase != 'prepare' or snapshot.native_action_s not in current.legal:
        raise ValueError('current query must be uncommitted and native action legal')
    if set(snapshot.routes) != {False,True}:
        raise graph.Unsupported('both physical directions are required')
    keys = {(d, s.station): s for d, route in snapshot.routes.items() for s in route}
    rows = {k: [] for k in keys}
    for v in visits:
        if v.arrival_s > v.completed_s or v.arrival_s > v.available_s:
            raise ValueError('invalid completed observation clocks')
        if v.completed_s < q and v.available_s < q and (v.direction, v.station) in rows:
            rows[v.direction, v.station].append(v)
    evidence = completed_flow(events, q)
    blocks = {k: [] for k in keys}
    for b in evidence['exact_blocks']:
        key = b['direction'], b['station']
        if key in blocks and b['end_s'] > b['start_s']:
            blocks[key].append(b)
    forecasts, queues = {}, {}
    observed = {k: [] for k in keys}
    for e in events:
        if e.key in observed and e.arrival_s <= q and e.available_s <= q:
            observed[e.key].append(e)
    for key, stop in keys.items():
        group = sorted(rows[key], key=lambda r: r.arrival_s)
        rows[key] = group
        if len(group) < graph.MIN_VISITS:
            raise graph.Unsupported('fewer than three completed visits at ' + str(key))
        if any(r.run_s <= 0 or r.gap_s < 0 or not 0 <= r.alighted <= r.incoming_load
               or not 0 <= r.incoming_load-r.alighted+r.boarded <= r.capacity for r in group):
            raise ValueError('invalid completed run or APC counts')
        recent = blocks[key][-graph.MIN_VISITS:]
        if not stop.terminal and len(recent) < graph.MIN_VISITS:
            raise graph.Unsupported('fewer than three closed clearance blocks at ' + str(key))
        seconds = sum(b['end_s']-b['start_s'] for b in recent)
        count = sum(b['arrivals'] for b in recent)
        incoming = sum(r.incoming_load for r in group)
        rate = 0. if stop.terminal else count/seconds
        forecasts[key] = dict(run_s=float(np.mean([r.run_s for r in group])), rate=rate,
            alight=sum(r.alighted for r in group)/incoming if incoming else 0.,
            visits=len(group), demand_blocks=len(recent), historical_arrivals=count,
            historical_seconds=seconds, future_rate_upper_bound_identified=False)
        if stop.terminal:
            queues[key] = 0.
            continue
        # The historical count is exact; its constant-rate extrapolation is not.
        # Reconcile subsequent visible APC, including unfinished current visits.
        anchor = recent[-1]
        amount, clock = 0., anchor['end_s']
        for e in observed[key]:
            if e.order <= anchor['last_order']:
                continue
            amount = max(0., amount + rate*(e.arrival_s-clock)-e.boarded)
            if e.cleared:
                amount = 0.
            clock = e.arrival_s
        queues[key] = amount + rate*(q-clock)
    result = graph.finish_assembly(snapshot, forecasts, rows)
    result.initial_queues = queues
    return result, dict(closed_blocks=sum(len(v) for v in blocks.values()),
        open_censored_blocks=len(evidence['open_blocks']), unavailable_visits=evidence['unavailable_visits'],
        used_blocks=sum(f['demand_blocks'] for k,f in forecasts.items() if not keys[k].terminal),
        max_completed_s=max(b['max_completed_s'] for bs in blocks.values() for b in bs),
        max_available_s=max(b['max_available_s'] for bs in blocks.values() for b in bs),
        forecast_assumption='last_three_nonoverlapping_closed_blocks_constant_rate',
        future_rate_upper_bound_identified=False)


class FallbackGraph:
    """Charge both simulations to the unchanged 128-evaluation query budget."""

    def __init__(self, base, homogeneous=False):
        self.base, self.snapshot, self.homogeneous = base, base.snapshot, homogeneous
        own = next(v for v in self.snapshot.vehicles if v.bus == self.snapshot.current_bus)
        self.key = own.bus, own.trip, self.snapshot.routes[own.direction][own.index].station
        self.reference, self.fallbacks, self.calls = None, {}, 0
        self.rejected = Counter()

    def evaluate(self, commands=None, homogeneous=False):
        assert homogeneous == self.homogeneous
        commands = commands or {}
        if self.reference is None:
            assert not commands
            self.reference = self.base.evaluate(homogeneous=homogeneous)
            self.calls += 1
            self.fallbacks[self.snapshot.native_action_s] = self.reference
            return self.reference
        if self.calls >= graph.MAX_EVALUATIONS:
            return dict(self.reference, failure='query_evaluation_budget')
        full = self.base.evaluate(commands, homogeneous)
        self.calls += 1
        h = commands.get(self.key, self.snapshot.native_action_s)
        if h not in self.fallbacks:
            if self.calls >= graph.MAX_EVALUATIONS:
                return dict(full, failure='query_evaluation_budget')
            self.fallbacks[h] = self.base.evaluate({self.key: h}, homogeneous)
            self.calls += 1
        fallback = self.fallbacks[h]
        acceptable = (graph.feasible(fallback, self.reference)
                      and fallback['cost'] <= self.reference['cost'] + 1e-9)
        if not acceptable:
            self.rejected[str(h)] += 1
        # Optimize the worse of the complete plan and actual baseline continuation.
        return dict(full, cost=max(full['cost'], fallback['cost']), full_cost=full['cost'],
            fallback=fallback, failure=full['failure'] or (None if acceptable else 'baseline_continuation_harm'))


def choose(model, homogeneous=False, proposals=None):
    wrapped = FallbackGraph(model, homogeneous)
    result = graph.solve(wrapped, homogeneous, proposals)
    full = result['selected']
    fallback = full.get('fallback', result['reference'])
    result.update(full_cost=full.get('full_cost', full['cost']), fallback=fallback,
        solver_proposals=result['evaluations'], evaluations=wrapped.calls,
        fallback_rejected=dict(wrapped.rejected))
    assert result['evaluations'] <= graph.MAX_EVALUATIONS
    return result


class FlowController(NativeNoop):
    def __init__(self, env, launches, credit, homogeneous=False):
        super().__init__(env, launches, credit)
        self.homogeneous, self.proposals = homogeneous, {}
        self.promise_records, self.pending_promises = [], {}
        self.execution_records = []

    def observe(self, bus, legal, native_action):
        start = time.perf_counter()
        key = int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id)
        now = float(self.env.current_time)
        self.masks[key] = tuple(legal)
        value, locks = snapshot(self.env, self.prefix, self.routes, key[0], legal, native_action,
                               self.credit.stream.pending_commands, self.masks)
        visits = self.prefix.strict(now)
        detail = {}
        try:
            model, detail = assemble(value, visits, self.prefix.flow_events)
            result = choose(model, self.homogeneous, self.proposals)
        except graph.Unsupported as error:
            result = dict(action_s=int(native_action), commands={}, evaluations=0,
                          status='unsupported_prefix', reason=str(error))
        action = int(result['action_s'])
        assert action in legal
        for promised in self.pending_promises.pop(key, []):
            promised.update(status='reached', actual_request_s=action, native_request_s=int(native_action),
                            reached_s=now, fulfilled=action == promised['planned_s'])
        if result['status'] == 'unsupported_prefix':
            for records in self.pending_promises.values():
                for record in records:
                    if 'cancelled_s' not in record:
                        record['cancelled_s'] = now
        self.proposals = {k:h for k,h in result['commands'].items() if k != key}
        reference, selected = result.get('reference'), result.get('selected')
        if reference is not None:
            for locked, departure_s in locks.items():
                assert locked not in reference['legal'] and locked not in result['commands']
                times = [t for t,kind,b in reference['nodes'] if kind == 'departure' and b == locked[0]]
                if times:
                    assert min(times) == departure_s
            for future,h in self.proposals.items():
                old = reference['actions'].get(future)
                if old is not None and h != old[1]:
                    record = dict(issued_s=now, key=future, planned_s=int(h), baseline_s=int(old[1]),
                                  predicted_query_s=selected['actions'][future][0], status='pending')
                    self.promise_records.append(record)
                    self.pending_promises.setdefault(future, []).append(record)
        self.execution_records.append(dict(time_s=now, bus=key[0], trip=key[1], station=key[2],
                                           action_s=action, native_s=int(native_action)))
        profile = dict(time_s=now, bus=key[0], trip=key[1], station=key[2],
            native_action_s=int(native_action), shadow_action_s=action, status=result['status'],
            reason=result.get('reason'), evaluations=result['evaluations'],
            fallback_rejected=result.get('fallback_rejected', {}), flow=detail,
            pending_locks=len(locks), total_s=time.perf_counter()-start)
        if reference is not None and math.isfinite(reference['cost']):
            fallback = result['fallback']
            profile['model_deltas'] = dict(full_cost=result['full_cost']-reference['cost'],
                fallback_cost=fallback['cost']-reference['cost'],
                full_cv2=selected['cv2']-reference['cv2'], fallback_cv2=fallback['cv2']-reference['cv2'])
        self.queries.append(profile)
        if len(self.queries) % 250 == 0:
            print('V275_QUERIES',len(self.queries),'t',now,'status',result['status'],flush=True)
        return profile

    def summary(self):
        times = [q['total_s'] for q in self.queries]
        supported = sum(q['evaluations'] > 0 for q in self.queries)
        for record in self.promise_records:
            if record['status'] == 'pending':
                record['status'] = 'not_reached_before_service_clearance'
        return dict(queries=len(times), supported_queries=supported,
            supported_fraction=supported/len(times), statuses=dict(Counter(q['status'] for q in self.queries)),
            unsupported_reasons=dict(Counter(q['reason'] for q in self.queries if q['reason'])),
            actual_overrides=sum(r['action_s'] != r['native_s'] for r in self.execution_records),
            signed_same_state_h_delta_s=sum(r['action_s']-r['native_s'] for r in self.execution_records),
            future_proposals=len(self.promise_records),
            future_statuses=dict(Counter(r['status'] for r in self.promise_records)),
            future_requests_matched=sum(r.get('fulfilled',False) for r in self.promise_records),
            future_requests_differed=sum(r['status'] == 'reached' and not r.get('fulfilled') for r in self.promise_records),
            future_proposals_cancelled=sum('cancelled_s' in r for r in self.promise_records),
            callback_latency_s=dict(mean=float(np.mean(times)),p95=float(np.quantile(times,.95)),max=max(times)),
            forecast_is_extrapolation=True, future_rate_upper_bound_identified=False)
