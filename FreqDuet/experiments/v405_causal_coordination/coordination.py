"""A detached dispatch and holding policy evaluated against one native continuation."""

from collections import Counter, defaultdict
from copy import copy
from dataclasses import replace
import math

from base_law import PERIOD_S, START_S, SlowManager
from coupled_model import CoupledCurrentController, CoupledPlanGraph
import freqduet_causal_event_graph as graph
from freqduet_apc_flow_evidence import completed_flow
from freqduet_flow_controller import FlowController
from freqduet_native_continuation import inputs
from plan_model import observe_plans


EPS = 1e-8


def partial_forecasts(snapshot, visits, events, priors):
    """Use exact closed blocks where available; expose censored stations as scenarios."""
    now = snapshot.now_s
    rows, observed, blocks = defaultdict(list), defaultdict(list), defaultdict(list)
    for v in visits:
        if v.completed_s < now and v.available_s < now:
            rows[v.direction, v.station].append(v)
    for e in events:
        if e.arrival_s <= now and e.available_s <= now:
            observed[e.key].append(e)
    for b in completed_flow(events, now)['exact_blocks']:
        if b['end_s'] > b['start_s']:
            blocks[b['direction'], b['station']].append(b)
    forecasts, stress, queues, stress_queues, censored = {}, {}, {}, {}, []
    for direction, route in snapshot.routes.items():
        for stop in route:
            key = direction, stop.station
            group = sorted(rows[key], key=lambda v: v.arrival_s)
            if len(group) < graph.MIN_VISITS:
                raise graph.Unsupported('insufficient completed running evidence at ' + str(key))
            rows[key] = group
            recent = blocks[key][-graph.MIN_VISITS:]
            if stop.terminal:
                rate = high = 0.
                queue = high_queue = 0.
            else:
                past = observed[key]
                if len(recent) == graph.MIN_VISITS:
                    rate = high = sum(b['arrivals'] for b in recent) / sum(
                        b['end_s'] - b['start_s'] for b in recent)
                    anchor = recent[-1]
                    clock, order = anchor['end_s'], anchor['last_order']
                    queue = high_queue = 0.
                else:
                    if key not in priors or not past:
                        raise graph.Unsupported('no historical APC prior at ' + str(key))
                    censored.append(key)
                    # A boarding total after an observed clearance constrains
                    # historical arrivals only. Neither scenario is an upper bound.
                    anchor_index = next((i for i in range(len(past)-1, -1, -1)
                                         if past[i].cleared), None)
                    if anchor_index is None:
                        clock, order = past[0].arrival_s, past[0].order - 1
                        floor = 0.
                    else:
                        anchor = past[anchor_index]
                        clock, order = anchor.arrival_s, anchor.order
                        span = past[-1].arrival_s - clock
                        floor = (sum(e.boarded for e in past[anchor_index+1:]) / span
                                 if span > 0 else 0.)
                    rate = max(priors[key], floor)
                    high = 1.25 * rate
                    queue = 0.
                    high_queue = 360. * high if anchor_index is None else 0.
                for event in past:
                    if event.order <= order:
                        continue
                    dt = event.arrival_s - clock
                    queue = 0. if event.cleared else max(0., queue + rate * dt - event.boarded)
                    high_queue = (0. if event.cleared else
                                  max(0., high_queue + high * dt - event.boarded))
                    clock = event.arrival_s
                queue += rate * (now - clock)
                high_queue += high * (now - clock)
            incoming = sum(v.incoming_load for v in group)
            row = dict(run_s=sum(v.run_s for v in group)/len(group), rate=rate,
                       alight=sum(v.alighted for v in group)/incoming if incoming else 0.)
            forecasts[key], stress[key] = row, dict(row, rate=high)
            queues[key], stress_queues[key] = queue, high_queue
    base = graph.finish_assembly(snapshot, forecasts, rows)
    base.initial_queues = queues
    high = graph.finish_assembly(snapshot, stress, rows)
    high.initial_queues = stress_queues
    # Compare identical time and service scopes under both demand assumptions.
    base.horizon_s = high.horizon_s = max(base.horizon_s, high.horizon_s)
    return (base, high), censored


def historical_rates(env, now):
    start, end = env.service_start_hour, env.service_end_hour
    hour = min(end, start + int(now // 3600))
    next_hour = min(end, hour + 1)
    fraction = (now % 3600) / 3600.
    def rate(od, h):
        return sum(float(x) for x in od.get(f'{h:02}:00:00', {}).values()
                   if float(x) > 0) / 60.
    return {(bool(s.direction), int(s.station_id)):
            (1-fraction) * rate(s.od, hour) + fraction * rate(s.od, next_hour)
            for s in env.stations if s.od is not None}


def ready(result):
    return (result['coverage'] and not result['failure'] and
            math.isfinite(result['cost']) and math.isfinite(result['cv2']))


def admissible(candidate, reference, public):
    """Protect delivered service, not unused seconds at a terminal."""
    if not ready(candidate) or candidate['missing_commands']:
        return False
    if candidate['launches'].keys() != reference['launches'].keys():
        return False
    if any(time < public[trip] for trip, (time, _) in candidate['launches'].items()):
        return False
    return (candidate['cost'] <= reference['cost'] + EPS and
            candidate['cv2'] <= reference['cv2'] + EPS and
            all(sum(candidate[k].values()) <= sum(reference[k].values()) + EPS
                for k in ('public_delays', 'readiness_delays')))


def release_options(plans, previous, now):
    """Change at most one future release; preserve the following public anchor."""
    for direction in (False, True):
        future = [p for p in plans if p.direction == direction]
        if direction not in previous or len(future) < 2:
            continue
        first, anchor = future[:2]
        if not (now < first.release_s < anchor.release_s <= now + PERIOD_S):
            continue
        if anchor.release_s != anchor.public_s or any(
                p.first_eligible_s is not None or p.min_gap_s is not None for p in (first, anchor)):
            continue
        for shift in (0, 15, 30, 60):
            when = first.public_s + shift
            if when == first.release_s or not (
                    max(now+1., previous[direction]+180.) <= when <= anchor.release_s-180.):
                continue
            yield (replace(first, release_s=when, target_s=when-previous[direction]),
                   replace(anchor, target_s=anchor.release_s-when))


def alternative(model, edit):
    variant = copy(model)
    variant.plan_by_trip = dict(model.plan_by_trip)
    variant.plan_by_trip.update((p.trip, p) for p in edit)
    variant.snapshot = replace(model.snapshot, plans=tuple(
        variant.plan_by_trip[p.trip] for p in model.snapshot.plans))
    return variant


def choose(models, key, native, legal, pending, edits):
    """Search paired transfers and terminal alternatives against unchanged native."""
    references = [m.evaluate() for m in models]
    count = len(models)
    if not all(map(ready, references)):
        return native, None, None, None, None, count, 'reference_incomplete', None
    public = {p.trip: p.public_s for p in models[0].snapshot.plans}
    candidates = [{key: h} for h in legal if h != native]
    if pending is not None:
        candidates = [{key: pending}] if pending != native and pending in legal else []
    else:
        reference = references[0]
        later = [(t, k, h) for k, (t, h) in reference['actions'].items()
                 if k[:2] == key[:2] and t > models[0].snapshot.now_s]
        if later:
            _, next_key, next_h = min(later)
            for h in legal:
                transferred = native + next_h - h
                if h != native and transferred in reference['legal'][next_key]:
                    candidates.append({key: h, next_key: transferred})
    best_score, best_action, best_pair, best_edit = 0., native, None, None
    best_model, best_fallback = references[0], references[0]
    unchanged = {native: references}
    for edit in (None, *edits):
        trial_models = models if edit is None else [alternative(m, edit) for m in models]
        roots = {}
        for commands in ([{}] if edit is not None else []) + candidates:
            h = commands.get(key, native)
            if h not in roots:
                root = {key: h} if h != native else {}
                rows = [m.evaluate(root) for m in trial_models]
                count += len(models)
                if edit is None:
                    unchanged[h] = rows
                roots[h] = rows if all(admissible(c, r, public) for c, r in zip(rows, references)) else None
            fallback = roots[h]
            if fallback is None:
                continue
            if edit is not None:
                trip = edit[0].trip
                # Do not attribute the effect of the current hold to the edit.
                if not all(ready(r) and trip in r['launches']
                           and c['launches'][trip][0] == edit[0].release_s
                           and c['launches'][trip][0] != r['launches'][trip][0]
                           for c, r in zip(fallback, unchanged[h])):
                    continue
            pair = next(((k, v) for k, v in commands.items() if k != key), None)
            rows = fallback
            if pair:
                rows = [m.evaluate(commands) for m in trial_models]
                count += len(models)
                if not all(admissible(c, r, public) for c, r in zip(rows, references)):
                    continue
            score = max(c['cost']-r['cost'] for c, r in zip(rows, references))
            if score < best_score - EPS:
                best_score, best_action, best_pair, best_edit = score, h, pair, edit
                best_model, best_fallback = rows[0], fallback[0]
    return (best_action, best_pair, best_edit, best_model, best_fallback, count,
            'coordination_improvement' if best_score < 0. else 'retain_native', references[0])


class Manager(SlowManager):
    def __init__(self, env, credit, plan_enabled):
        super().__init__('joint' if plan_enabled else 'fixed', env, credit, .5)
        self.plan_enabled, self.due = plan_enabled, False
        self.counts = Counter()

    def publish(self, env, bridge):
        self.counts['publications'] += 1
        self.due = self.plan_enabled

    def apply_plan(self, edit, now):
        roster = {int(t.launch_turn): t for t in self.env.timetables}
        for p in edit:
            t = roster[p.trip]
            assert not t.launched and not getattr(t, '_upper_queried', False)
            assert getattr(t, '_freqduet_first_eligible_time', None) is None
        for p in edit:
            t = roster[p.trip]
            t._freqduet_scheduled_launch = p.release_s
            t.target_headway = p.target_s
        self.counts['planned_changes'] += 1
        self.events.append(dict(kind='publish', time_s=now, budgets=[],
            frequency=dict(supported=False), plans=[dict(trip=edit[0].trip,
            public_s=edit[0].public_s, release_s=edit[0].release_s,
            shift_s=edit[0].release_s-edit[0].public_s)]))

    def summary(self):
        return dict(super().summary(), **self.counts)

    def on_return(self, bus, trip, time_s):
        super().on_return(bus, trip, time_s)
        if self.controller.pending and self.controller.pending[0][:2] == (bus, trip):
            self.controller.pending = None
            self.controller.activity['transfers_not_reached'] += 1


class Controller(CoupledCurrentController):
    def __init__(self, env, launches, credit, manager):
        super().__init__(env, launches, credit, manager)
        manager.controller = self
        self.pending, self.activity = None, Counter()

    def build_model(self, value, visits):
        try:
            base, info = FlowController.build_model(self, value, visits)
            bases = (base,)
        except graph.Unsupported as error:
            if not str(error).startswith('fewer than three closed clearance blocks'):
                raise
            bases, censored = partial_forecasts(value, visits, self.prefix.flow_events,
                                               historical_rates(self.env, value.now_s))
            info = dict(censored_keys=len(censored), forecast_scenarios=len(bases))
            self.activity['censored_queries'] += 1
        observed = inputs(self.env, self.prefix, self.evidence)
        plans, last = observe_plans(self.env)
        self.models = [CoupledPlanGraph(b, observed, plans, last, {}) for b in bases]
        moments = self.models[0].arrival_prefix.moments
        recorded = self.env.headway_events.headways_s
        assert (moments.count, moments.total, moments.square_total) == (
            len(recorded), sum(recorded), sum(h*h for h in recorded))
        return self.models[0], info

    def select_plan(self, model):
        s = model.snapshot
        own = next(v for v in s.vehicles if v.bus == s.current_bus)
        key = own.bus, own.trip, s.routes[own.direction][own.index].station
        pending = None
        if self.pending is not None:
            pair_key, h, issued = self.pending
            if key == pair_key:
                pending = h
            elif s.now_s <= issued + PERIOD_S:
                return dict(action_s=s.native_action_s, commands={}, evaluations=0,
                            status='native_until_paired_stop')
            else:
                self.pending = None
                self.activity['transfers_expired'] += 1
        edits = []
        if self.manager.due and pending is None:
            self.manager.due = False
            previous = {d:r['release_s'] for d,r in self.manager.last_actual.items() if r}
            edits = list(release_options(s.plans, previous, s.now_s))
            queried = {t.launch_turn for t in self.env.timetables if getattr(t, '_upper_queried', False)}
            edits = [edit for edit in edits if not any(p.trip in queried for p in edit)]
            self.activity['release_options_evaluated'] += len(edits)
        action, pair, edit, selected, fallback, evaluations, status, reference = choose(
            self.models, key, s.native_action_s, own.legal, pending, edits)
        if reference is None:
            return dict(action_s=action, commands={}, evaluations=evaluations, status=status)
        if pending is not None:
            self.activity['transfers_followed' if action == pending else 'transfers_abstained'] += 1
            self.pending = None
        elif pair is not None:
            self.pending = (*pair, s.now_s)
            self.activity['transfers_proposed'] += 1
        if edit is not None:
            self.manager.apply_plan(edit, s.now_s)
        if len(self.models) > 1 and action != s.native_action_s:
            self.activity['censored_overrides'] += 1
        return dict(action_s=action, commands={key: action} if action != s.native_action_s else {},
                    evaluations=evaluations, status=status, reference=reference, selected=selected,
                    fallback=fallback, full_cost=selected['cost'])

    def observe(self, bus, legal, native_action):
        key = int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id)
        profile = super().observe(bus, legal, native_action)
        if self.pending is not None and self.pending[0] == key:
            self.pending = None
            self.activity['transfers_abstained'] += 1
        return profile

    def summary(self):
        return dict(super().summary(), coordination=dict(self.activity),
                    transfer_outstanding=self.pending is not None)
