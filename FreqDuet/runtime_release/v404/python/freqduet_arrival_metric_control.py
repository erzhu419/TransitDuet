"""V277: actual arrival-history CV2; current-action enumeration is a separate arm."""

import math

import numpy as np

import freqduet_causal_event_graph as graph
from freqduet_indexed_continuation import IndexedFlowController, IndexedNativeGraph
from freqduet_system_learner import ArrivalMoments


class ArrivalPrefix:
    """All observed nonterminal arrivals, not just completed service visits."""

    def __init__(self, routes, arrivals, now_s):
        self.keys = {(d,s.station) for d,route in routes.items() for s in route if not s.terminal}
        self.last, values = {}, []
        for direction,station,_,time_s,_ in arrivals:
            if time_s > now_s:
                raise ValueError('arrival history extends beyond the current query')
            key = direction,station
            if key not in self.keys:
                continue
            if key in self.last:
                if time_s < self.last[key]:
                    raise ValueError('arrival history is out of order within station')
                values.append(time_s-self.last[key])
            self.last[key] = time_s
        self.moments = ArrivalMoments()
        self.moments.observe(values)
        self.now_s = now_s

    def extend(self, future_arrivals):
        values = []
        for key in self.keys:
            last = self.last.get(key)
            for now in sorted(future_arrivals.get(key, ())):
                if now < self.now_s:
                    raise ValueError('projected arrival precedes the query')
                if last is not None:
                    values.append(now-last)
                last = now
        moments = ArrivalMoments()
        moments.count = self.moments.count + len(values)
        extra = np.asarray(values, dtype=float)
        moments.total = self.moments.total + float(extra.sum())
        moments.square_total = self.moments.square_total + float(extra @ extra)
        return moments


class ArrivalMetricGraph(IndexedNativeGraph):
    def __init__(self, base, observed):
        super().__init__(base, observed)
        self.arrival_prefix = ArrivalPrefix(self.snapshot.routes, observed['arrivals'], self.snapshot.now_s)
        # _simulate seeds its working arrays with completed[-2:] and each
        # current observed arrival. Only the appended suffix is a prediction.
        self.seed_sizes = {key:min(2,len(rows)) for key,rows in self.completed.items()}
        for v in self.snapshot.vehicles:
            if v.arrival_s is not None:
                key = v.direction,self.snapshot.routes[v.direction][v.index].station
                self.seed_sizes[key] += 1

    def evaluate(self, commands=None, homogeneous=False):
        result = super().evaluate(commands, homogeneous)
        future = {key:times[self.seed_sizes[key]:] for key,times in result['arrivals'].items()}
        result['cv2'] = self.arrival_prefix.extend(future).cv_squared
        return result


def choose_current(model, homogeneous=False):
    """Enumerate only the legal current command, then native continuation."""
    reference = model.evaluate(homogeneous=homogeneous)
    s = model.snapshot
    own = next(v for v in s.vehicles if v.bus == s.current_bus)
    key = own.bus,own.trip,s.routes[own.direction][own.index].station
    best, commands, evaluations = reference,{},1
    if not reference['failure'] and reference['coverage'] and math.isfinite(reference['cv2']):
        for h in sorted(own.legal):
            if h == s.native_action_s:
                continue
            trial = {key:int(h)}
            candidate = model.evaluate(trial, homogeneous)
            evaluations += 1
            if graph.feasible(candidate,reference) and candidate['cost'] < best['cost']-1e-9:
                best,commands = candidate,trial
    return dict(action_s=commands.get(key,s.native_action_s), commands=commands,
        evaluations=evaluations, solver_proposals=evaluations, reference=reference,
        selected=best, fallback=best, full_cost=best['cost'], fallback_rejected={},
        status='model_improvement' if commands else 'retain_native',
        optimality_claim=False, decision_scope='all_legal_current_actions_with_native_continuation')


class ArrivalMetricController(IndexedFlowController):
    def build_model(self, value, visits):
        base,info = super().build_model(value,visits)
        model = ArrivalMetricGraph(base,base.observed)
        recorded = self.env.headway_events.headways_s
        observed = model.arrival_prefix.moments
        assert observed.count == len(recorded), 'arrival history differs from native recorder'
        assert observed.total == sum(recorded), 'arrival headway totals differ from native recorder'
        assert observed.square_total == sum(h*h for h in recorded), 'arrival second moments differ'
        return model,info


class CurrentActionController(ArrivalMetricController):
    def select_plan(self, model):
        return choose_current(model,self.homogeneous)
