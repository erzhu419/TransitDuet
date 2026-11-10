"""Co-decide new remaining-trip credit and the current legal command."""

import copy
import math
from collections import Counter

import freqduet_causal_event_graph as graph
from base_law import quota
from policies import Manager as FrozenManager, Controller as FrozenController
from coupled_model import choose_coupled


def public_delay(result):
    return sum(result['public_delays'].values())


def joint_choice(model, amount, homogeneous=False):
    baseline = choose_coupled(model, homogeneous)
    reference = baseline['selected']
    evaluations = baseline['evaluations']
    if reference['failure'] or not reference['coverage'] or not math.isfinite(reference['cv2']):
        return baseline, False, dict(reason='reference_unsupported', evaluations=evaluations)
    s = model.snapshot
    own = next(v for v in s.vehicles if v.bus == s.current_bus)
    cycle = own.bus, own.trip
    key = (*cycle, s.routes[own.direction][own.index].station)
    trial_model = copy.copy(model)
    trial_model.budget_remaining = dict(model.budget_remaining)
    trial_model.budget_remaining[cycle] = amount
    best = reference
    best_h = None
    rejected = Counter()
    for h in sorted(own.legal):
        if h > amount:
            continue
        try:
            candidate = trial_model.evaluate({key: int(h)}, homogeneous)
            evaluations += 1
        except graph.Unsupported:
            rejected['unsupported'] += 1
            continue
        if not graph.feasible(candidate, reference):
            rejected['common_reference_constraint'] += 1
            continue
        if candidate['cost'] > reference['cost'] + 1e-9:
            rejected['passenger_cost'] += 1
            continue
        better = candidate['cost'] < best['cost'] - 1e-9
        tied = abs(candidate['cost'] - best['cost']) <= 1e-9
        if better or (tied and public_delay(candidate) < public_delay(best) - 1e-9):
            best, best_h = candidate, int(h)
    receipt = dict(reason='admitted' if best_h is not None else 'no_joint_improvement',
                   evaluations=evaluations, rejected=dict(rejected), proposed_remaining_s=amount)
    if best_h is None:
        baseline['evaluations'] = evaluations
        baseline['solver_proposals'] = evaluations
        return baseline, False, receipt
    receipt.update(cost_delta=best['cost'] - reference['cost'],
                   cv2_delta=best['cv2'] - reference['cv2'],
                   public_delay_delta=public_delay(best) - public_delay(reference),
                   reference_action_s=baseline['action_s'], action_s=best_h)
    result = dict(action_s=best_h, commands={key: best_h}, evaluations=evaluations,
                  solver_proposals=evaluations, reference=reference, selected=best,
                  fallback=best, full_cost=best['cost'], fallback_rejected={},
                  status='joint_credit_improvement', optimality_claim=False,
                  decision_scope='one_cycle_credit_and_current_command_common_reference')
    return result, True, receipt


class JointManager(FrozenManager):
    def __init__(self, env, credit):
        super().__init__(env, credit, 'candidate', .5)
        self.pending = {}
        self.ledger = []
        self.admissions = []
        self.request_counts = Counter()

    def issue_budgets(self, estimates, matched):
        self.request_counts['expired_at_publication'] += len(self.pending)
        self.pending.clear()
        for target, match in matched.items():
            key = match['bus'], match['active_trip']
            if key not in estimates:
                continue
            item = estimates[key]
            normal = item['normal_command_s']
            pressure = max(0., item['ordinary_return_s'] + 1 - match['public_s'])
            if normal <= 0 or pressure <= 0 or not item['remaining_controls']:
                continue
            amount = quota(normal, pressure, self.retain_fraction)
            old = self.remaining.get(key)
            if old is not None and amount >= old:
                continue
            self.pending[key] = dict(item, target_trip=target, pressure_s=pressure,
                                     proposed_s=amount, spent_since_proposal_s=0,
                                     proposal_s=float(self.env.current_time))
        self.request_counts['proposed'] += len(self.pending)
        return []

    def request(self, key):
        row = self.pending.pop(key, None)
        if row is None:
            return None
        amount = row['proposed_s'] - row['spent_since_proposal_s']
        old = self.remaining.get(key)
        if amount < 0 or (old is not None and amount >= old):
            self.request_counts['spent_or_obsolete'] += 1
            return None
        return row, amount

    def admit(self, key, row, amount, receipt):
        old = self.remaining.get(key)
        assert amount >= 0 and (old is None or amount < old)
        self.remaining[key] = amount
        event = dict(kind='admit', bus=key[0], trip=key[1],
                     time_s=float(self.env.current_time), query_index=self.command_count,
                     old_remaining_s=old, remaining_s=amount, proposal_s=row['proposal_s'],
                     proposed_s=row['proposed_s'], spent_since_proposal_s=row['spent_since_proposal_s'],
                     **receipt)
        self.ledger.append(event)
        self.admissions.append(event)

    def commit(self, bus, trip, station, time_s, action):
        key = int(bus), int(trip)
        index = self.command_count
        if key in self.pending:
            self.pending[key]['spent_since_proposal_s'] += int(action)
        super().commit(bus, trip, station, time_s, action)
        self.ledger.append(dict(kind='command', bus=key[0], trip=key[1], station=int(station),
                                time_s=float(time_s), query_index=index, action_s=int(action)))

    def on_return(self, bus, trip, time_s):
        key = int(bus), int(trip)
        unused = self.remaining.get(key)
        super().on_return(bus, trip, time_s)
        self.pending.pop(key, None)
        if unused is not None:
            self.ledger.append(dict(kind='return', bus=key[0], trip=key[1],
                                    time_s=float(time_s), unused_s=unused))

    def summary(self):
        result = super().summary()
        result.update(budget_issues=sum(r['old_remaining_s'] is None for r in self.admissions),
                      budget_tightenings=sum(r['old_remaining_s'] is not None for r in self.admissions),
                      pending_requests=len(self.pending), request_counts=dict(self.request_counts))
        return result


class JointController(FrozenController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credit_decisions = []

    def select_plan(self, model):
        if self.manager.kind != 'candidate':
            return super().select_plan(model)
        s = model.snapshot
        own = next(v for v in s.vehicles if v.bus == s.current_bus)
        key = own.bus, own.trip
        request = self.manager.request(key)
        if request is None:
            return super().select_plan(model)
        row, amount = request
        result, accepted, receipt = joint_choice(model, amount, self.homogeneous)
        self.credit_decisions.append(dict(bus=key[0], trip=key[1], query_s=s.now_s,
                                          proposal_s=row['proposal_s'], **receipt))
        if accepted:
            self.manager.admit(key, row, amount, receipt)
            model.budget_remaining = dict(self.manager.remaining)
        self.selection = result
        return result
