"""Shared slow timetable with pressure, static remaining-trip, or stop limits."""

import math

from base_law import SlowManager, START_S, PERIOD_S
from provider import UpperSummaryProvider
from coupled_model import (CoupledCurrentController, CoupledPlanGraph, _BudgetCommands,
                           _BudgetContinuation, _simulate_plan, project_native)
import freqduet_causal_event_graph as graph


class Manager(SlowManager):
    def __init__(self, env, credit, kind, parameter):
        super().__init__('joint',env,credit,.5)
        self.kind,self.parameter=kind,float(parameter)
        self.capped_cycles=set()
        self.cap_issues=[];self.command_count=0
        self.provider=UpperSummaryProvider.from_environment(env,4)

    def publish(self, env, bridge):
        now=float(env.current_time)
        if now<START_S or int((now-START_S)//PERIOD_S)==self.last_cycle:return
        self.upper_summary=self.provider.refresh(now,env.frequency_tracker.total_updates,bridge.prefix.flow_events)
        super().publish(env,bridge)

    def issue_budgets(self, estimates, matched):
        if self.kind=='pressure':return super().issue_budgets(estimates,matched)
        issued=[]
        for key,item in estimates.items():
            if not item['remaining_controls']:continue
            if self.kind=='cap' and key not in self.capped_cycles:
                self.capped_cycles.add(key)
                self.cap_issues.append(dict(bus=key[0],trip=key[1],cap_s=self.parameter,
                    time_s=float(self.env.current_time),after_queries=self.command_count))
            elif self.kind=='static' and key not in self.remaining:
                value=int(math.ceil(self.parameter*item['remaining_controls']))
                self.remaining[key]=value
                issued.append(dict(item,target_trip=None,old_remaining_s=None,
                                   remaining_s=value,pressure_s=None))
        return issued

    def _match(self, captured, estimates, now_s, use_budget):
        if self.kind=='cap' and use_budget:
            estimates={key:dict(item,normal_command_s=min(item['normal_command_s'],
                self.parameter*item['remaining_controls'])) if key in self.capped_cycles else item
                for key,item in estimates.items()}
        return super()._match(captured,estimates,now_s,use_budget)

    def commit(self, bus, trip, station, time_s, action):
        if (int(bus),int(trip)) in self.capped_cycles:assert action<=self.parameter
        super().commit(bus,trip,station,time_s,action)
        self.command_count+=1

    def on_return(self, bus, trip, time_s):
        super().on_return(bus,trip,time_s)
        self.capped_cycles.discard((int(bus),int(trip)))


class CappedCommands(_BudgetCommands):
    def __init__(self, commands, remaining, caps):
        super().__init__(commands,remaining)
        self.caps=caps

    def get(self,key,native=None):
        cap=self.caps.get(key[:2])
        if cap is not None:
            if key in self:
                if self[key]>cap:raise graph.Unsupported('candidate exceeds published stop cap')
            else:
                native=project_native(native,self.legal[key],cap)
        return super().get(key,native)


class CappedGraph(CoupledPlanGraph):
    def evaluate(self,commands=None,homogeneous=False):
        submitted=CappedCommands(commands or {},self.budget_remaining,self.stop_caps)
        result=_simulate_plan(self,submitted,homogeneous,_BudgetContinuation(self,submitted))
        future={key:times[self.seed_sizes[key]:] for key,times in result['arrivals'].items()}
        result['cv2']=self.arrival_prefix.extend(future).cv_squared
        result.update(budget_remaining_s=dict(submitted.remaining),budget_spent_s=dict(submitted.spent),
                      budget_closed_remaining_s=dict(submitted.closed_remaining))
        return result


def choose_capped(model,homogeneous=False):
    reference=model.evaluate(homogeneous=homogeneous)
    s=model.snapshot;own=next(v for v in s.vehicles if v.bus==s.current_bus)
    key=own.bus,own.trip,s.routes[own.direction][own.index].station
    cap=model.stop_caps.get(key[:2])
    baseline=project_native(s.native_action_s,own.legal,cap)
    assert reference['actions'][key][1]==baseline
    best,commands,evaluations=reference,{},1
    if not reference['failure'] and reference['coverage'] and math.isfinite(reference['cv2']):
        for h in sorted(own.legal):
            if h==baseline or (cap is not None and h>cap):continue
            trial={key:int(h)};candidate=model.evaluate(trial,homogeneous);evaluations+=1
            if graph.feasible(candidate,reference) and candidate['cost']<best['cost']-1e-9:
                best,commands=candidate,trial
    return dict(action_s=commands.get(key,baseline),commands=commands,evaluations=evaluations,
        solver_proposals=evaluations,reference=reference,selected=best,fallback=best,
        full_cost=best['cost'],fallback_rejected={},status='model_improvement' if commands else
        'budgeted_native' if baseline!=s.native_action_s else 'retain_native',
        optimality_claim=False,decision_scope='published_stop_cap_current_actions')


class Controller(CoupledCurrentController):
    def build_model(self,value,visits):
        model,info=super().build_model(value,visits)
        if self.manager.kind=='cap':
            model.__class__=CappedGraph
            model.stop_caps={key:self.manager.parameter for key in self.manager.capped_cycles}
        return model,info

    def select_plan(self,model):
        if self.manager.kind!='cap':return super().select_plan(model)
        self.selection=choose_capped(model,self.homogeneous)
        return self.selection

    def observe(self,bus,legal,native_action):
        profile=super().observe(bus,legal,native_action)
        if self.manager.kind=='cap' and (int(bus.bus_id),int(bus.trip_id)) in self.manager.capped_cycles:
            if profile['status']=='unsupported_prefix':
                action=project_native(native_action,legal,self.manager.parameter)
                profile['shadow_action_s']=action
                self.execution_records[-1]['action_s']=action
        return profile
