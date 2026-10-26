"""V276: native bilateral semantics on observed or model-projected AVL state."""

import importlib
from types import SimpleNamespace

import numpy as np

import freqduet_causal_event_graph as graph
from lower.causal_follower_eta import AVLVehicleSnapshot, estimate_follower_departure_gap
from lower.causal_departure_regularity import CausalDepartureRegularityCost


def native_request(packet, legal):
    """Use the actual native forward matcher, follower estimator and bin selector."""
    from env.bus import Bus
    from run_freqduet_v106_service_tradeoff import rule_action

    recorder = SimpleNamespace(previous_departure_event=lambda *_: packet['departure'])
    proxy = SimpleNamespace(_headway_recorder=recorder,
        forward_predecessor_trip_id=packet['predecessor'],
        last_station=SimpleNamespace(station_id=packet['station']), direction=packet['direction'])
    Bus._update_pre_action_forward_headway(proxy, packet['prepare_s'])
    follower = estimate_follower_departure_gap(current_bus_id=packet['bus'],
        current_direction=packet['direction'], current_progress_m=packet['progress_m'],
        current_time_s=packet['prepare_s'], service_dwell_proxy_s=packet['service_s'],
        vehicles=packet['avl'])
    context = CausalDepartureRegularityCost().capture(
        forward_headway_s=proxy.pre_action_forward_headway,
        target_headway_s=packet['target_s'], evidence_source=proxy.pre_action_forward_headway_source,
        follower_departure_gap_s=follower.departure_gap_s, follower_evidence_source=follower.source)
    mask = np.isin(graph.ACTIONS, legal)
    action = int(graph.ACTIONS[rule_action(context, 'bilateral', .25, mask)])
    return action, context, follower


class PrepareEvidence:
    """Record precisely the inputs consumed at PREPARE, not later callback AVL."""
    def __init__(self):
        self.packets, self.checked, self.samples = {}, 0, {}

    def attach(self, Bus):
        self.Bus, self.original = Bus, Bus._update_pre_action_follower_eta
        self.module=importlib.import_module(Bus.__module__)
        self.original_estimator=self.module.estimate_follower_departure_gap
        def estimate(**kwargs):
            self.samples[kwargs['current_bus_id']]=dict(kwargs,vehicles=tuple(kwargs['vehicles']))
            return self.original_estimator(**kwargs)
        self.module.estimate_follower_departure_gap=estimate
        original = self.original
        def update(bus, now, buses):
            result = original(bus, now, buses)
            sample=self.samples[int(bus.bus_id)]
            recorder = bus._headway_recorder
            self.packets[int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id)] = dict(
                bus=int(bus.bus_id), trip=int(bus.trip_id), station=int(bus.last_station.station_id),
                direction=bool(bus.direction), prepare_s=float(now), progress_m=float(bus.travel_distance),
                service_s=float(bus.last_service_dwell_s), target_s=float(bus._target_headway),
                predecessor=bus.forward_predecessor_trip_id,
                departure=recorder.previous_departure_event(int(bus.last_station.station_id), bool(bus.direction)),
                avl=sample['vehicles'], expected_forward=bus.pre_action_forward_headway,
                expected_source=bus.pre_action_forward_headway_source,
                expected_follower=bus.pre_action_follower_departure_gap,
                expected_follower_id=bus.pre_action_follower_bus_id,
                expected_follower_source=bus.pre_action_follower_source)
            return result
        Bus._update_pre_action_follower_eta = update

    def detach(self):
        self.Bus._update_pre_action_follower_eta = self.original
        self.module.estimate_follower_departure_gap=self.original_estimator

    def check(self, bus, legal, native, query_s):
        packet = self.packets[int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id)]
        assert packet['prepare_s'] <= query_s
        action, context, follower = native_request(packet, legal)
        assert action == native, ('native bin mismatch', packet['bus'], action, native)
        assert context.forward_headway_s == packet['expected_forward']
        assert context.evidence_source == packet['expected_source']
        assert follower.departure_gap_s == packet['expected_follower']
        assert follower.follower_bus_id == packet['expected_follower_id']
        assert follower.source == packet['expected_follower_source']
        self.checked += 1
        return packet


def inputs(env, prefix, evidence):
    links = {d: tuple((float(r.distance), float(r.speed_limit)) for r in
        (env.routes[:len(env.routes)//2] if d else env.routes[len(env.routes)//2:])) for d in (False, True)}
    fleet = {int(b.bus_id): dict(progress=float(b.travel_distance) if b.on_route else 0.,
        launch=float(b.launch_time), speed=float(b.current_speed),
        service=float(b.last_service_dwell_s), predecessor=b.forward_predecessor_trip_id,
        target=float(b._target_headway)) for b in env.bus_all}
    return dict(links=links, fleet=fleet, packets=evidence.packets.copy(),
        arrivals=[(e.direction, e.station, e.trip, e.arrival_s, e.bus) for e in prefix.flow_events],
        departures=[(e.direction, e.station, e.trip, e.completed_s, e.bus) for e in prefix.flow_events
                    if e.completed_s is not None])


class NativeGraph(graph.EventGraph):
    def __init__(self, base, observed):
        super().__init__(base.snapshot, base.forecasts, base.completed, base.horizon_s, base.initial_queues)
        self.observed = observed

    def evaluate(self, commands=None, homogeneous=False):
        return graph._simulate(self, commands or {}, homogeneous, Continuation(self))


class Continuation:
    """Project existing running forecasts to AVL; never substitute arrival ETA for readiness."""
    def __init__(self, model):
        self.model, self.data = model, model.observed
        self.arrivals = list(self.data['arrivals'])
        self.departures = list(self.data['departures'])
        self.history, self.predecessors, self.service = {}, {}, {}
        self.rank = {v.bus:i for i,v in enumerate(model.snapshot.vehicles)}

    def initialize(self, vehicles):
        self.vehicles = vehicles
        now = self.model.snapshot.now_s
        for v in vehicles.values():
            row = self.data['fleet'][v['bus']]
            self.history[v['bus']] = []
            self.predecessors[v['bus']] = row['predecessor']
            self.service[v['bus']] = row['service']
            self.save(v, now, row['progress'], row['launch'], row['speed'], now, rank=-2)

    def station_progress(self, direction, index):
        return sum(distance for distance,_ in self.data['links'][direction][:index+1])

    def save(self, v, visible, progress, launch, speed, motion_s, rank=None):
        self.history[v['bus']].append(dict(visible=visible, direction=v['d'], on_route=v['phase']!='terminal',
            rank=self.rank[v['bus']] if rank is None else rank,
            trip=v['trip'], index=v['i'], progress=progress, launch=launch, speed=speed,
            travel=v['phase']=='travel', motion_s=motion_s, arrival_s=v['frontier']))

    def event(self, kind, v, now, service_s=None):
        bid, d, i = v['bus'], v['d'], v['i']
        launch = self.history[bid][-1]['launch']
        if kind == 'launch':
            self.service[bid], self.predecessors[bid] = 0., None
            self.save(v, now, 0., now, self.data['links'][d][0][0]/max(1,v['frontier']-now), now+1,rank=-1)
        elif kind == 'return':
            self.save(v, now, 0., launch, 0., now+1)
        elif kind == 'arrival':
            station = self.model.snapshot.routes[d][i].station
            prior = [a for a in self.arrivals if a[:2] == (d,station)]
            self.predecessors[bid] = prior[-1][2] if prior else None
            self.arrivals.append((d,station,v['trip'],now,bid))
            self.service[bid] = service_s
            self.save(v, now, self.station_progress(d,i), launch, 0., now+1)
        elif kind == 'departure':
            station = self.model.snapshot.routes[d][i-1].station
            self.departures.append((d,station,v['trip'],now,bid))
            distance = self.data['links'][d][i][0]
            self.save(v, now, self.station_progress(d,i-1), launch,
                distance/max(1,v['frontier']-now), now+1)

    def avl(self, now, reader=None):
        result = []
        reader_rank=float('inf') if reader is None else self.rank[reader]
        for bid, history in self.history.items():
            row = next(r for r in reversed(history) if r['visible'] < now
                       or r['visible']==now and r['rank']<reader_rank)
            progress = row['progress']
            if row['travel']:
                endpoint = self.station_progress(row['direction'],row['index'])
                span = max(1.,row['arrival_s']-row['motion_s']+1)
                progress += (endpoint-progress)*min(1.,max(0.,(now-row['motion_s'])/span))
            route_speed = self.data['links'][row['direction']][row['index']][1]
            result.append(AVLVehicleSnapshot(bid,row['direction'],row['on_route'],progress,
                row['launch'],row['speed'],route_speed))
        return tuple(result)

    def action(self, v, command_s, legal):
        s = self.model.snapshot
        station = s.routes[v['d']][v['i']].station
        original = next(o for o in s.vehicles if o.bus == v['bus'])
        if original.phase == 'prepare' and original.trip == v['trip'] and original.index == v['i']:
            packet = self.data['packets'][v['bus'],v['trip'],station]
        else:
            prepare = command_s-1
            # The frozen V275 runtime reads the inline AVL roster: earlier
            # vehicles have moved this tick; later vehicles have not.
            departures = [r for r in self.departures if r[:2] == (v['d'],station)
                and (r[3] < prepare or r[3] == prepare and self.rank[r[4]] < self.rank[v['bus']])]
            latest = max(departures,key=lambda r:(r[3],self.rank[r[4]])) if departures else None
            packet = dict(bus=v['bus'],trip=v['trip'],station=station,direction=v['d'],prepare_s=prepare,
                progress_m=self.station_progress(v['d'],v['i']), service_s=self.service[v['bus']],
                target_s=360., predecessor=self.predecessors[v['bus']], avl=self.avl(prepare,v['bus']),
                departure=None if latest is None else dict(trip_id=latest[2],time_s=latest[3]))
        return native_request(packet,legal)[0]
