"""Published terminal plans and persistent trip budgets for the frozen fast layer.

The unchanged V278 event kernel supplies physical timing and plan semantics.
The command mapping intercepts its one final-command submission per control
event. Each evaluation has its own budget ledger; pending root commands are
already represented as irreversible holds by the original snapshot builder.
"""

import copy
import math

import freqduet_causal_event_graph as graph
from controllers import FastArrivalController
from freqduet_arrival_metric_control import choose_current
from freqduet_flow_controller import FlowController
from freqduet_native_continuation import inputs
from plan_model import (PlanConsistentGraph, PlanContinuation, _simulate_plan,
                        observe_plans)


def project_native(native, legal, remaining=None):
    """Project a native request using only existing legal bins and current credit."""
    native = int(native)
    legal = tuple(int(h) for h in legal)
    if native not in legal:
        raise graph.Unsupported('native request outside the original legal mask')
    if remaining is None:
        return native
    if not math.isfinite(remaining) or remaining < 0:
        raise graph.Unsupported('invalid remaining trip holding budget')
    allowed = [h for h in legal if h <= native and h <= remaining]
    if not allowed:
        raise graph.Unsupported('no native-feasible command fits the remaining budget')
    return max(allowed)


class _BudgetCommands(dict):
    """Submit a final command once, after the native request and legal set exist."""

    def __init__(self, commands, remaining):
        super().__init__(commands)
        self.remaining = dict(remaining)
        self.spent = {key: 0 for key in remaining}
        self.closed_remaining = {}
        self.legal = {}

    def get(self, key, native=None):
        legal = self.legal[key]
        cycle = key[:2]
        remaining = self.remaining.get(cycle)
        if key in self:
            command = int(super().get(key))
            if command not in legal or (remaining is not None and command > remaining):
                raise graph.Unsupported('candidate command violates native mask or trip budget')
        else:
            command = project_native(native, legal, remaining)
        if remaining is not None:
            self.remaining[cycle] = remaining - command
            self.spent[cycle] += command
        return command

    def returned(self, bus, trip):
        cycle = bus, trip
        if cycle in self.remaining:
            self.closed_remaining[cycle] = self.remaining.pop(cycle)


class _BudgetContinuation(PlanContinuation):
    def __init__(self, model, commands):
        super().__init__(model)
        self.commands = commands

    def action(self, vehicle, command_s, legal):
        station = self.model.snapshot.routes[vehicle['d']][vehicle['i']].station
        self.commands.legal[vehicle['bus'], vehicle['trip'], station] = tuple(legal)
        # Keep this unprojected so the unchanged kernel still validates the
        # same-state native request. Projection occurs at final submission.
        return super().action(vehicle, command_s, legal)

    def event(self, kind, vehicle, now, service_s=None):
        if kind == 'return':
            self.commands.returned(vehicle['bus'], vehicle['trip'])
        return super().event(kind, vehicle, now, service_s)


class CoupledPlanGraph(PlanConsistentGraph):
    def __init__(self, base, observed, plans, last_dispatch_s, remaining_by_cycle):
        super().__init__(base, observed, plans, last_dispatch_s)
        self.budget_remaining = dict(remaining_by_cycle)
        if any(not math.isfinite(v) or v < 0 for v in self.budget_remaining.values()):
            raise graph.Unsupported('invalid remaining trip holding budget')

    def evaluate(self, commands=None, homogeneous=False):
        if not self.budget_remaining:
            return super().evaluate(commands, homogeneous)
        submitted = _BudgetCommands(commands or {}, self.budget_remaining)
        result = _simulate_plan(self, submitted, homogeneous, _BudgetContinuation(self, submitted))
        future = {key: times[self.seed_sizes[key]:] for key, times in result['arrivals'].items()}
        result['cv2'] = self.arrival_prefix.extend(future).cv_squared
        result.update(budget_remaining_s=dict(submitted.remaining),
                      budget_spent_s=dict(submitted.spent),
                      budget_closed_remaining_s=dict(submitted.closed_remaining))
        return result


def choose_coupled(model, homogeneous=False):
    """The same seven-bin search, with budgeted native as same-plan reference."""
    if not model.budget_remaining:
        return choose_current(model, homogeneous)
    reference = model.evaluate(homogeneous=homogeneous)
    s = model.snapshot
    own = next(v for v in s.vehicles if v.bus == s.current_bus)
    key = own.bus, own.trip, s.routes[own.direction][own.index].station
    remaining = model.budget_remaining.get((own.bus, own.trip))
    baseline = project_native(s.native_action_s, own.legal, remaining)
    assert reference['actions'][key][1] == baseline
    best, commands, evaluations = reference, {}, 1
    if not reference['failure'] and reference['coverage'] and math.isfinite(reference['cv2']):
        for h in sorted(own.legal):
            if h == baseline or (remaining is not None and h > remaining):
                continue
            trial = {key: int(h)}
            candidate = model.evaluate(trial, homogeneous)
            evaluations += 1
            if graph.feasible(candidate, reference) and candidate['cost'] < best['cost'] - 1e-9:
                best, commands = candidate, trial
    return dict(action_s=commands.get(key, baseline), commands=commands, evaluations=evaluations,
        solver_proposals=evaluations, reference=reference, selected=best, fallback=best,
        full_cost=best['cost'], fallback_rejected={},
        status=('model_improvement' if commands else
                'budgeted_native' if baseline != s.native_action_s else 'retain_native'),
        optimality_claim=False,
        decision_scope='all_budget_feasible_current_actions_with_budgeted_native_continuation')


class CoupledCurrentController(FastArrivalController):
    def __init__(self, env, launches, credit, manager):
        super().__init__(env, launches, credit)
        self.manager = manager
        self.quota_probed = set()

    def build_model(self, value, visits):
        base, info = FlowController.build_model(self, value, visits)
        plans, last_dispatch = observe_plans(self.env)
        model = CoupledPlanGraph(base, inputs(self.env, self.prefix, self.evidence),
                                 plans, last_dispatch, self.manager.remaining)
        recorded = self.env.headway_events.headways_s
        moments = model.arrival_prefix.moments
        assert moments.count == len(recorded)
        assert moments.total == sum(recorded)
        assert moments.square_total == sum(h*h for h in recorded)
        self.model = self.manager.last_model = model
        return model, info

    def select_plan(self, model):
        self.selection = choose_coupled(model, self.homogeneous)
        return self.selection

    def observe(self, bus, legal, native_action):
        profile = super().observe(bus, legal, native_action)
        cycle = int(bus.bus_id), int(bus.trip_id)
        remaining = self.manager.remaining.get(cycle)
        if profile['status'] == 'unsupported_prefix':
            action = project_native(native_action, legal, remaining)
            # FlowController already appended these same records. Its unsupported
            # path has no selected graph or future proposals, so keep all public
            # action receipts aligned before the caller submits and debits.
            if action != profile['shadow_action_s']:
                profile['shadow_action_s'] = action
                profile['budgeted_fallback'] = True
                self.execution_records[-1]['action_s'] = action
        elif (remaining is not None and native_action > remaining and cycle not in self.quota_probed
              and self.selection['reference']['coverage'] and not self.selection['reference']['failure']
              and math.isfinite(self.selection['reference']['cv2'])):
            # One diagnostic per physical trip. The actual selection is already
            # complete; only this cycle's quota is removed from the same-state
            # counterfactual. Other quotas and the published plan stay fixed.
            self.quota_probed.add(cycle)
            probe = copy.copy(self.model)
            probe.budget_remaining = dict(self.model.budget_remaining)
            del probe.budget_remaining[cycle]
            evaluate, evaluations = probe.evaluate, 0

            def counted(commands=None, homogeneous=False):
                nonlocal evaluations
                evaluations += 1
                return evaluate(commands, homogeneous)

            probe.evaluate = counted
            receipt = dict(budget_action_s=profile['shadow_action_s'], remaining_s=remaining)
            try:
                free = choose_coupled(probe, self.homogeneous)
                reference = free['reference']
                if reference['coverage'] and not reference['failure'] and math.isfinite(reference['cv2']):
                    receipt.update(supported=True, without_own_budget_s=free['action_s'],
                                   changed=free['action_s'] != profile['shadow_action_s'])
                else:
                    receipt.update(supported=False, without_own_budget_s=None, changed=None,
                                   reason='without-budget reference incomplete')
            except graph.Unsupported as error:
                receipt.update(supported=False, without_own_budget_s=None, changed=None, reason=str(error))
            receipt['evals'] = evaluations
            profile['quota_probe'] = receipt
        return profile
