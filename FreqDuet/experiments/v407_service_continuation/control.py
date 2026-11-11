"""Single-action rollout with continuous published service inside its window."""

from coupled_model import _BudgetContinuation, project_native
from frozen_v406 import (Controller as FrozenController, ReferenceCommands,
                         ReferenceGraph, SimpleCap, EPS, ready)
from service_kernel import simulate_service


class ServiceGraph(ReferenceGraph):
    def evaluate(self, commands=None, homogeneous=False):
        submitted = ReferenceCommands(commands or {}, {})
        result = simulate_service(self, submitted, homogeneous,
                                  _BudgetContinuation(self, submitted))
        future = {key: times[self.seed_sizes[key]:]
                  for key, times in result['arrivals'].items()}
        result['cv2'] = self.arrival_prefix.extend(future).cv_squared
        return result


def admissible(candidate, reference, public):
    if not ready(candidate) or candidate['missing_commands']:
        return False
    if candidate['due_trips'] != reference['due_trips']:
        return False
    if not candidate['pending_trips'] <= reference['pending_trips']:
        return False
    if any(t < public[trip] for trip, (t, _) in candidate['launches'].items()):
        return False
    return (candidate['cost'] <= reference['cost'] + EPS
            and candidate['cv2'] <= reference['cv2'] + EPS
            and candidate['endpoint_queue'] + candidate['endpoint_load'] <=
                reference['endpoint_queue'] + reference['endpoint_load'] + EPS
            and all(sum(candidate[k].values()) <= sum(reference[k].values()) + EPS
                    for k in ('public_delays', 'readiness_delays')))


def choose_single(models):
    s = models[0].snapshot
    own = next(v for v in s.vehicles if v.bus == s.current_bus)
    key = own.bus, own.trip, s.routes[own.direction][own.index].station
    baseline = project_native(s.native_action_s, own.legal, 10)
    references = [m.evaluate() for m in models]
    count = len(models)
    if not all(map(ready, references)):
        return dict(action_s=baseline, commands={}, evaluations=count,
                    status='reference_incomplete')
    assert all(r['actions'][key][1] == baseline for r in references)
    public = {p.trip: p.public_s for p in s.plans}
    best_score, action, selected = 0., baseline, references[0]
    for h in own.legal:
        if h == baseline:
            continue
        candidates = [m.evaluate({key: h}) for m in models]
        count += len(models)
        if not all(admissible(c, r, public) for c, r in zip(candidates, references)):
            continue
        score = max(c['cost'] - r['cost'] for c, r in zip(candidates, references))
        if score < best_score - EPS:
            best_score, action, selected = score, h, candidates[0]
    return dict(action_s=action, commands={key: action} if action != baseline else {},
                evaluations=count, status='coordination_improvement' if action != baseline
                else 'retain_native', reference=references[0], selected=selected,
                fallback=selected, full_cost=selected['cost'])


class ServiceController(FrozenController):
    def build_model(self, value, visits):
        _, info = super().build_model(value, visits)
        for model in self.models:
            model.__class__ = ServiceGraph
            model.demand_end_s = float(self.env.protocol.demand_end_time_s)
            model.evaluation_end_s = float(self.env.protocol.evaluation_end_time_s)
        return self.models[0], info

    def select_plan(self, model):
        result = choose_single(self.models)
        if 'reference' in result:
            def receipt(row):
                return dict(due=len(row['due_trips']), launched=len(row['launches']),
                    pending=len(row['pending_trips']), unfinished=len(row['unfinished_trips']),
                    queue=row['endpoint_queue'], onboard=row['endpoint_load'],
                    public_delay=sum(row['public_delays'].values()),
                    readiness_delay=sum(row['readiness_delays'].values()),
                    balance_error=row['balance_error'])
            self.window_receipt = dict(end_s=result['reference']['window_end_s'],
                reference=receipt(result['reference']), selected=receipt(result['selected']))
        return result

    def observe(self, bus, legal, native_action):
        self.window_receipt = None
        profile = super().observe(bus, legal, native_action)
        if self.window_receipt is not None:
            profile['service_window'] = self.window_receipt
        return profile


class FixedCap(SimpleCap):
    def __init__(self, env, launches, credit, manager, cap):
        super().__init__(env, launches, credit, manager)
        self.cap = cap

    def observe(self, bus, legal, native_action):
        profile = super().observe(bus, legal, native_action)
        action = project_native(native_action, legal, self.cap)
        profile['shadow_action_s'] = action
        self.execution_records[-1]['action_s'] = action
        return profile
