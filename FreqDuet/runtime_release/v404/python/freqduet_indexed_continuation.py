"""Output-equivalent indexing for the frozen V276 continuation model."""

from bisect import bisect_left, insort
from types import SimpleNamespace

import numpy as np

import freqduet_causal_event_graph as graph
from freqduet_consistent_flow import ConsistentFlowController
from freqduet_native_continuation import (Continuation, NativeGraph, estimate_follower_departure_gap,
    CausalDepartureRegularityCost)


class ReusedNativeRule:
    """Reuse only allocation; matching, ETA, context and bin selection stay native."""
    def __init__(self):
        from env.bus import Bus
        from run_freqduet_v106_service_tradeoff import rule_action
        self.Bus,self.select=Bus,rule_action
        self.context=CausalDepartureRegularityCost()
        self.departure=None
        self.proxy=SimpleNamespace(last_station=SimpleNamespace(station_id=0),
            _headway_recorder=SimpleNamespace(previous_departure_event=lambda *_:self.departure))
        self.masks={}

    def __call__(self, packet, legal):
        self.departure=packet['departure']
        self.proxy.forward_predecessor_trip_id=packet['predecessor']
        self.proxy.last_station.station_id=packet['station']
        self.proxy.direction=packet['direction']
        self.Bus._update_pre_action_forward_headway(self.proxy,packet['prepare_s'])
        follower=estimate_follower_departure_gap(current_bus_id=packet['bus'],
            current_direction=packet['direction'],current_progress_m=packet['progress_m'],
            current_time_s=packet['prepare_s'],service_dwell_proxy_s=packet['service_s'],vehicles=packet['avl'])
        context=self.context.capture(forward_headway_s=self.proxy.pre_action_forward_headway,
            target_headway_s=packet['target_s'],evidence_source=self.proxy.pre_action_forward_headway_source,
            follower_departure_gap_s=follower.departure_gap_s,follower_evidence_source=follower.source)
        key=tuple(legal)
        if key not in self.masks:
            self.masks[key]=np.isin(graph.ACTIONS,legal)
        action=int(graph.ACTIONS[self.select(context,'bilateral',.25,self.masks[key])])
        return action,context,follower


class IndexedNativeGraph(NativeGraph):
    def __init__(self, base, observed):
        super().__init__(base, observed)
        self.fleet_rank = {v.bus:i for i,v in enumerate(self.snapshot.vehicles)}
        self.originals = {v.bus:v for v in self.snapshot.vehicles}
        self.native_rule = ReusedNativeRule()
        self.coordinates = {d:tuple(sum(length for length,_ in links[:i+1])
            for i in range(len(links))) for d,links in observed['links'].items()}
        self.last_arrivals = {(d,station):trip for d,station,trip,_,_ in observed['arrivals']}
        indexed = {}
        for order,(d,station,trip,time,bus) in enumerate(observed['departures']):
            indexed.setdefault((d,station),[]).append((time,self.fleet_rank[bus],-order,trip,bus))
        self.departure_prefix = {key:tuple(sorted(rows)) for key,rows in indexed.items()}

    def evaluate(self, commands=None, homogeneous=False):
        return graph._simulate(self, commands or {}, homogeneous, IndexedContinuation(self))


class IndexedContinuation(Continuation):
    def __init__(self, model):
        self.model, self.data = model, model.observed
        self.rank = model.fleet_rank
        self.last_arrivals = model.last_arrivals.copy()
        self.future_departures = {}
        self.departure_order = len(self.data['departures'])
        self.history, self.predecessors, self.service = {}, {}, {}

    def station_progress(self, direction, index):
        return self.model.coordinates[direction][index]

    def event(self, kind, v, now, service_s=None):
        if kind not in ('arrival','departure'):
            return super().event(kind,v,now,service_s)
        bid,d,i = v['bus'],v['d'],v['i']
        launch = self.history[bid][-1]['launch']
        if kind == 'arrival':
            key = d,self.model.snapshot.routes[d][i].station
            self.predecessors[bid] = self.last_arrivals.get(key)
            self.last_arrivals[key] = v['trip']
            self.service[bid] = service_s
            self.save(v,now,self.station_progress(d,i),launch,0.,now+1)
        else:
            key = d,self.model.snapshot.routes[d][i-1].station
            row = now,self.rank[bid],-self.departure_order,v['trip'],bid
            self.departure_order += 1
            insort(self.future_departures.setdefault(key,[]),row)
            distance = self.data['links'][d][i][0]
            self.save(v,now,self.station_progress(d,i-1),launch,
                distance/max(1,v['frontier']-now),now+1)

    def departure_at(self, key, prepare, reader):
        # The short tuple excludes every same-tick event at/after the reader's
        # fleet rank. Negative order preserves max()'s original first-tie rule.
        cutoff = prepare,self.rank[reader]
        best = None
        for rows in (self.model.departure_prefix.get(key,()),self.future_departures.get(key,())):
            index = bisect_left(rows,cutoff)
            if index:
                row = rows[index-1]
                if best is None or row > best:
                    best = row
        return None if best is None else dict(trip_id=best[3],time_s=best[0])

    def action(self, v, command_s, legal):
        s = self.model.snapshot
        station = s.routes[v['d']][v['i']].station
        original = self.model.originals[v['bus']]
        if original.phase=='prepare' and original.trip==v['trip'] and original.index==v['i']:
            packet = self.data['packets'][v['bus'],v['trip'],station]
        else:
            prepare = command_s-1
            packet = dict(bus=v['bus'],trip=v['trip'],station=station,direction=v['d'],prepare_s=prepare,
                progress_m=self.station_progress(v['d'],v['i']),service_s=self.service[v['bus']],
                target_s=360.,predecessor=self.predecessors[v['bus']],avl=self.avl(prepare,v['bus']),
                departure=self.departure_at((v['d'],station),prepare,v['bus']))
        return self.model.native_rule(packet,legal)[0]


class IndexedFlowController(ConsistentFlowController):
    def build_model(self, value, visits):
        base,info = super().build_model(value,visits)
        return IndexedNativeGraph(base,base.observed),info
