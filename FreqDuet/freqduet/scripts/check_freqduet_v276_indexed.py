#!/usr/bin/env python3
"""Zero-traffic equivalence checks and cProfile for V276's indexed kernel."""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time
import unittest

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_indexed_continuation import IndexedNativeGraph
from freqduet_native_continuation import AVLVehicleSnapshot, NativeGraph, native_request


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def stress_state(history_count=5200, now_s=50000., plans_per_direction=6):
    """Synthetic full-size state, not a new demand day or a traffic rollout."""
    now,prepare=float(now_s),float(now_s)-1.
    routes={d:tuple(graph.Stop(21*int(d)+i,control=i<20,terminal=i==20)
                    for i in range(21)) for d in (False,True)}
    positions=(10,8,6,4,2,0,10,8,6,4,2,0)
    vehicles,telemetry,avl=[],{},[]
    for bus,index in enumerate(positions):
        d=bus>=6
        own=bus==0
        progress=500.*(index+1-(0. if own else .5))
        launch=prepare-progress/2.5
        vehicles.append(graph.Vehicle(bus,bus,d,index,'prepare' if own else 'travel',
            36. if own else 20.,remaining_fraction=.5,arrival_s=prepare-40 if own else None,
            boarded=18. if own else None))
        telemetry[bus]=dict(progress=progress,launch=launch,speed=0. if own else 5.,
            service=40. if own else 0.,predecessor=90 if own else None,target=360.)
        avl.append(AVLVehicleSnapshot(bus,d,True,progress,launch,0. if own else 5.,8.))
    plans=tuple(graph.Plan(100+2*i+int(d),d,now+360*i)
                for d in (False,True) for i in range(plans_per_direction))
    plans=tuple(sorted(plans,key=lambda p:(p.release_s,p.trip)))
    station=routes[False][10].station
    packet=dict(bus=0,trip=0,station=station,direction=False,prepare_s=prepare,
        progress_m=telemetry[0]['progress'],service_s=40.,target_s=360.,predecessor=90,
        departure=dict(trip_id=90,time_s=prepare-300),avl=tuple(avl))
    native=native_request(packet,tuple(graph.ACTIONS))[0]
    snapshot=graph.Snapshot(now,now,routes,tuple(vehicles),plans,0,native)
    keys=[(d,s.station) for d,route in routes.items() for s in route]
    arrivals,departures=[],[]
    completed={key:[] for key in keys}
    forecasts={}
    for n in range(history_count):
        d,stop=keys[n%len(keys)]
        t=1000.+n*(now-3000.)/max(1,history_count)
        bus=(n//len(keys))%6+6*int(d)
        trip=10000+n//21
        arrivals.append((d,stop,trip,t,bus))
        departures.append((d,stop,trip,t+18,bus))
        completed[d,stop].append(graph.Visit(d,stop,t,t+18,t,100.,20.,4.,2.,360.))
    for d,route in routes.items():
        for stop in route:
            for j,ago in enumerate((1080.,720.,360.)):
                t=now-ago
                trip=20000+100*int(d)+3*stop.station+j
                departure=t+18
                if (d,stop.station)==(False,station) and j==2:
                    trip,t,departure=90,prepare-340,prepare-300
                bus=6*int(d)
                arrivals.append((d,stop.station,trip,t,bus))
                departures.append((d,stop.station,trip,departure,bus))
                completed[d,stop.station].append(graph.Visit(d,stop.station,t,departure,t,
                    100.,20.,4.,2.,360.))
            forecasts[d,stop.station]=dict(run_s=100.,rate=0. if stop.terminal else 4/360.,
                alight=.1,visits=len(completed[d,stop.station]),demand_blocks=3)
    arrivals.append((False,station,0,prepare-40,0))
    arrivals.sort(key=lambda r:(r[3],r[4]))
    # Native observations arrive in arrival order, not sorted departure order.
    departures.sort(key=lambda r:(r[3]-18,r[4]))
    base=graph.finish_assembly(snapshot,forecasts,completed)
    base.initial_queues={key:0. if routes[key[0]][key[1]-21*int(key[0])].terminal else 4.
                         for key in keys}
    observed=dict(links={d:((500.,8.),)*21 for d in routes},fleet=telemetry,
        packets={(0,0,station):packet},arrivals=arrivals,departures=departures)
    return NativeGraph(base,observed)


def check(a):
    import test_v276_native_continuation as legacy
    tests=Path(__file__).resolve().parents[1]/'tests'
    suite=unittest.TestSuite((
        unittest.defaultTestLoader.discover(str(tests),pattern='test_v276_indexed_continuation.py',
                                          top_level_dir=str(tests)),
        unittest.defaultTestLoader.discover(str(Path(legacy.__file__).parent),
            pattern='test_v276_native_continuation.py',top_level_dir=str(Path(legacy.__file__).parent))))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun>=16
    write(a.root/'checks/passed.json',dict(source=a.source,tests=result.testsRun,
        status='passed',physical_rollouts=0,training_updates=0,main_policy_changed=False))
    print('V276_INDEXED_CHECK_COMPLETE',result.testsRun,flush=True)


def profile_call(path,fn):
    profiler=cProfile.Profile()
    result=profiler.runcall(fn)
    path.parent.mkdir(parents=True,exist_ok=True)
    profiler.dump_stats(str(path))
    stats=pstats.Stats(profiler)
    rows=[]
    for (name,line,function),(primitive,calls,own,cumulative,_) in stats.stats.items():
        rows.append(dict(file=Path(name).name,line=line,function=function,primitive_calls=primitive,
            calls=calls,self_s=own,cumulative_s=cumulative))
    return result,dict(total_s=stats.total_tt,total_calls=stats.total_calls,
        top_self=sorted(rows,key=lambda r:r['self_s'],reverse=True)[:15],
        top_cumulative=sorted(rows,key=lambda r:r['cumulative_s'],reverse=True)[:15])


def benchmark(a):
    assert json.loads((a.root/'checks/passed.json').read_text())['source']==a.source
    reference=stress_state()
    begin=time.perf_counter()
    indexed=IndexedNativeGraph(reference,reference.observed)
    index_build_s=time.perf_counter()-begin
    # Warm imports and require a fully supported model, not an early fallback.
    baseline=reference.evaluate()
    assert baseline['coverage'] and not baseline['failure']
    assert baseline==indexed.evaluate()
    profiled_old,old_profile=profile_call(a.root/'profiles/frozen.prof',lambda:reference.evaluate())
    profiled_new,new_profile=profile_call(a.root/'profiles/indexed.prof',lambda:indexed.evaluate())
    assert profiled_old==profiled_new==baseline
    timings=[]
    for value in (0,15,45):
        plan={(0,0,10):value}
        row={}
        outputs=[]
        for name,model in (('frozen',reference),('indexed',indexed)):
            start=time.perf_counter()
            outputs.append(model.evaluate(plan))
            row[name+'_s']=time.perf_counter()-start
        assert outputs[0]==outputs[1]
        timings.append(dict(action_s=value,**row))
    solvers={}
    results=[]
    for name,model in (('frozen',reference),('indexed',indexed)):
        start=time.perf_counter()
        results.append(flow.choose(model))
        solvers[name+'_s']=time.perf_counter()-start
    assert results[0]==results[1]
    assert results[0]['evaluations']>1 and results[0]['evaluations']<=128
    write(a.root/'summary/results.json',dict(source=a.source,frozen_control_source='7023e2e784',
        status='exact_kernel_and_solver_equivalence_passed',synthetic_benchmark=True,
        physical_rollouts=0,training_updates=0,main_policy_changed=False,
        fleet=12,stops_per_direction=21,observed_departures=len(reference.observed['departures']),
        profiles=dict(frozen=old_profile,indexed=new_profile),fixed_plans=timings,solver=solvers,
        index_build_s=index_build_s,
        solver_speedup=solvers['frozen_s']/solvers['indexed_s'],
        solver_evaluations=results[0]['evaluations'],search_budget=128,beam_width=4,max_variables=6,
        limitations='Synthetic workload,not a measured full-day speedup. Existing traffic paths are untouched.'))
    print('V276_INDEXED_BENCHMARK_COMPLETE',solvers,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','benchmark','all'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    a=p.parse_args()
    if a.command=='all':
        check(a)
        benchmark(a)
    else:
        dict(check=check,benchmark=benchmark)[a.command](a)


if __name__=='__main__':
    main()
