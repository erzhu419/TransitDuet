"""Synthetic physical/clock tests and a bounded kernel workload measurement."""

from copy import copy
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS
from unittest.mock import patch

from control import FixedCap, ServiceGraph, admissible, choose_single
from frozen_v406 import ReferenceGraph, SimpleCap
from freqduet_causal_event_graph import ACTIONS, EventGraph, Snapshot, Stop, Vehicle, Visit
from freqduet_native_continuation import native_request
from plan_model import PublishedPlan
from service_kernel import simulate_service


def fixture(stops=3, buses=2, horizon=900, demand_end=100000, evaluation_end=100000):
    q = 1200.
    routes = {d: tuple(Stop(i, i != stops-1, i == stops-1) for i in range(stops))
              for d in (False, True)}
    vehicles = [Vehicle(0, 10, False, 0, 'prepare', 10., capacity=60.,
                        legal=tuple(ACTIONS.tolist()), arrival_s=q-10, boarded=4.)]
    vehicles += [Vehicle(i, -1, bool(i % 2), 0, 'terminal', 0., capacity=60.,
                         ready_s=q-100, legal=tuple(ACTIONS.tolist())) for i in range(1, buses)]
    plans = tuple(PublishedPlan(100+2*i+int(d), d, q+360*i, 360., q+360*i)
                  for i in range(1, int(horizon//360)+1) for d in (False, True))
    completed = {(d,i): [Visit(d,i,q-1000+360*j,q-990+360*j,q-1000+360*j,
                                10.,10.,2.,3.,360.) for j in range(3)]
                 for d in routes for i in range(stops)}
    forecasts = {(d,i): dict(run_s=10., rate=.02 if i != stops-1 else 0., alight=.2)
                 for d in routes for i in range(stops)}
    arrivals = [(d,i,90+j,v.arrival_s,0) for (d,i),rows in completed.items()
                for j,v in enumerate(rows)]
    departures = [(d,i,90+j,v.completed_s,0) for (d,i),rows in completed.items()
                  for j,v in enumerate(rows)]
    arrivals.append((False,0,10,q-10,0))
    packet = dict(bus=0,trip=10,station=0,direction=False,prepare_s=q-1,
                  progress_m=100.,service_s=12.,target_s=360.,predecessor=92,
                  departure=dict(trip_id=92,time_s=q-270),avl=())
    native = native_request(packet, vehicles[0].legal)[0]
    observed = dict(links={d:tuple((100.,10.) for _ in range(stops)) for d in routes},
        fleet={v.bus:dict(progress=100. if v.bus==0 else 0.,launch=q-60,speed=0.,
                         service=12. if v.bus==0 else 0.,predecessor=92,target=360.)
               for v in vehicles}, packets={(0,10,0):packet}, arrivals=arrivals,departures=departures)
    s = Snapshot(q,q,routes,tuple(vehicles),plans,0,native)
    base = EventGraph(s,forecasts,completed,q+horizon,{k:1. if not routes[k[0]][k[1]].terminal
                                                     else 0. for k in completed})
    last = {d:q-360 for d in routes}
    old = ReferenceGraph(base,observed,plans,last,{})
    new = ServiceGraph(base,observed,plans,last,{})
    new.demand_end_s, new.evaluation_end_s = demand_end, evaluation_end
    return old,new


class ZeroContinuation:
    def initialize(self, vehicles):
        pass

    def event(self, kind, vehicle, now, service_s=None):
        pass

    def action(self, vehicle, command_s, legal):
        return 0


def core_model(end, demand_end, plans=(), phase='terminal', remaining=0.):
    routes = {d:(Stop(0,True,False),Stop(1,False,True)) for d in (False,True)}
    v = Vehicle(0,10,False,0,phase,0.,capacity=60.,remaining_s=remaining,
                ready_s=0.,legal=tuple(ACTIONS.tolist()),
                arrival_s=None if phase == 'terminal' else 0.,boarded=0.)
    s = Snapshot(0.,0.,routes,(v,),tuple(plans),0,0)
    completed = {(d,i):[NS(arrival_s=-10.,completed_s=-9.)] for d in routes for i in (0,1)}
    forecasts = {(d,i):dict(run_s=10.,rate=1. if (d,i)==(False,0) else 0.,alight=0.)
                 for d in routes for i in (0,1)}
    return NS(snapshot=s,forecasts=forecasts,completed=completed,horizon_s=end,
              demand_end_s=demand_end,evaluation_end_s=end,
              initial_queues={k:0. for k in completed},initial_last_dispatch_s={False:-360.,True:-360.},
              plan_by_trip={p.trip:p for p in plans})


def checks():
    old,new = fixture()
    for h in ACTIONS.tolist():
        a,b = old.evaluate({(0,10,0):h}), new.evaluate({(0,10,0):h})
        for key in ('actions','arrivals','boardings','nodes','launches','returns','next_launch'):
            assert a[key] == b[key], key
        for key in ('cost','waiting','onboard','cv2'):
            assert abs(a[key]-b[key]) < 1e-7, (key,a[key],b[key])
        assert b['coverage'] and abs(b['balance_error']) < 1e-6

    old,new = fixture(horizon=1500)
    legacy,continuous = old.evaluate(),new.evaluate()
    cutoff = old.snapshot.now_s + 360 * len(old.snapshot.vehicles)
    added = {p.trip for p in new.snapshot.plans if cutoff < p.release_s <= new.horizon_s}
    assert added and added.isdisjoint(legacy['launches'])
    assert added <= continuous['due_trips']
    assert added & continuous['launches'].keys()
    assert added <= continuous['launches'].keys() | continuous['pending_trips']
    assert continuous['coverage'] and len(continuous['endpoint_vehicles']) == 2
    selected = choose_single([new])
    assert selected['action_s'] in new.snapshot.vehicles[0].legal
    assert len(selected['commands']) <= 1
    assert admissible(selected['selected'],selected['reference'],
                      {p.trip:p.public_s for p in new.snapshot.plans})

    old,new = fixture(horizon=1450)
    row = new.evaluate()
    assert row['coverage'] and row['unfinished_trips']
    assert any(v['frontier'] > row['window_end_s'] for v in row['endpoint_vehicles'].values()
               if v['phase'] != 'terminal')

    p = PublishedPlan(100,False,5.,360.,5.)
    model = core_model(20.,100.,(p,),phase='hold',remaining=100.)
    row = simulate_service(model,{},False,ZeroContinuation())
    assert row['coverage'] and row['pending_trips'] == {100} and not row['launches']
    assert row['public_delays'][100] == row['readiness_delays'][100] == 15.
    assert row['unfinished_trips'] == {10} and row['endpoint_vehicles'][0]['frontier'] == 99.
    assert row['endpoint_queue'] == 20. and row['waiting'] == 200.

    model = core_model(20.,10.)
    row = simulate_service(model,{},False,ZeroContinuation())
    assert row['coverage'] and row['generated'] == row['endpoint_queue'] == 10.
    assert row['waiting'] == 150. and row['delivered'] == 0.
    model.evaluation_end_s = 12.
    row = simulate_service(model,{},False,ZeroContinuation())
    assert row['window_end_s'] == 12. and row['waiting'] == 70.

    reference = continuous
    public = {p.trip:p.public_s for p in new.snapshot.plans}
    assert admissible(reference,reference,public)
    assert not admissible(dict(reference,endpoint_queue=reference['endpoint_queue']+1),reference,public)
    assert not admissible(dict(reference,pending_trips={min(reference['launches'])}),reference,public)
    assert not admissible(dict(reference,public_delays={1:sum(reference['public_delays'].values())+1}),
                          reference,public)

    def passive(self,bus,legal,native):
        self.execution_records.append(dict(action_s=10))
        return dict(shadow_action_s=10)
    cap = FixedCap.__new__(FixedCap)
    cap.cap,cap.execution_records = 15,[]
    with patch.object(SimpleCap,'observe',passive):
        p = cap.observe(None,(0,5,10,15,20,30,45),45)
        assert p['shadow_action_s'] == cap.execution_records[-1]['action_s'] == 15
        assert cap.observe(None,(0,5,10),10)['shadow_action_s'] == 10

    old,new = fixture(stops=20,buses=12,horizon=10000)
    workload = {}
    for label,model in (('frozen',old),('continuous',new)):
        start,nodes = perf_counter(),0
        for h in ACTIONS.tolist()*2:
            result = model.evaluate({(0,10,0):h})
            assert result['coverage'], (label,result['failure'])
            nodes += len(result['nodes'])
        workload[label] = dict(evaluations=14,seconds=perf_counter()-start,nodes=nodes)
    return dict(checks=['seven same-input native action continuations match before service cutoff',
        'published trips after old cutoff use the same physical fleet',
        'unfinished trips remain active across the finite window boundary',
        'unserved published trips retain censored delay and passenger stock',
        'known demand/evaluation clocks and passenger mass balance',
        'endpoint stock/delay protection and all native cap masks'],workload=workload)
