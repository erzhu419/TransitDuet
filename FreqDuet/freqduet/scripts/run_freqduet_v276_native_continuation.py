#!/usr/bin/env python3
"""V276 native-rule equivalence followed by one bounded full-day comparison."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import unittest

from run_freqduet_v272_graph_pilot import read, write, selected_law, executed_chain, METRICS

DAYS = (2076001,2076002,2076003,2076004)
ARMS = ('fixed360_bilateral025','frozen_v275_apc','native_continuation_apc')
ITEMS = tuple((day,arm) for arm in ARMS for day in DAYS)
SHARDS = tuple(tuple(range(i,12,8)) for i in range(8))
SEED,SAMPLER = 12011,27011001


def inventory(root):
    # Restrict the inventory to experiment metadata, not event/log trees.
    def contains(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('day','days','demand_seed','demand_seeds','scenario_seed','scenario_seeds') or k.endswith('_days'):
                    if any(x in DAYS for x in (v if isinstance(v,list) else [v]) if isinstance(x,int)):
                        return True
                if isinstance(v,(dict,list)) and contains(v):
                    return True
        elif isinstance(value,list):
            return any(contains(v) for v in value if isinstance(v,(dict,list)))
        return False
    checked=0
    for experiment in root.parent.iterdir():
        if not experiment.is_dir() or experiment==root:
            continue
        paths=list(experiment.glob('*.json'))
        for name in ('budget','started','pairs','reservations','manifests','metadata'):
            paths.extend((experiment/name).rglob('*.json'))
        for p in paths:
            assert not contains(read(p)), ('development day already consumed',str(p))
            checked+=1
    return dict(files_checked=checked,collisions=0,raw_trace_reads=0)


def check(a):
    tests=Path(__file__).resolve().parents[1]/'tests'
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests),pattern=p) for p in
        ('test_v276_native_continuation.py','test_v275_flow_controller.py','test_v270_causal_event_graph.py',
         'test_v271_native_event_graph.py','test_v272_graph_pilot.py','test_v273_flow_contract.py','test_v274_passive_flow.py'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    metadata=inventory(a.root)
    write(a.root/'protocol.json',dict(source=a.source,days=DAYS,arms=ARMS,items=ITEMS,
        actor=SEED,sampler=SAMPLER,physical_start_cap=13,passive_cap=1,development_cap=12,
        automatic_retry=False,training_updates=0,main_policy_changed=False,
        frozen_v275_source='894ad14707',independent_future_replications=0))
    write(a.root/'checks/passed.json',dict(source=a.source,tests=result.testsRun,status='passed',metadata=metadata))
    print('V276_CHECK_COMPLETE',result.testsRun,flush=True)


def make_audit(fixed,arm,credit,passive):
    import numpy as np
    from freqduet_native_event_graph import native_query
    from freqduet_native_continuation import PrepareEvidence

    class Audit(fixed.BenchmarkAudit):
        def __init__(self):
            super().__init__(SEED,'fixed_bilateral025')
            self.bridge,self.evidence=None,None
            self.decisions,self.callback_wall=[],[]

        def attach(self,runner):
            super().attach(runner)
            from env.bus import Bus
            if passive:
                self.evidence=PrepareEvidence()
                self.evidence.attach(Bus)
            elif arm==ARMS[1]:
                from freqduet_flow_controller import FlowController
                self.bridge=FlowController(self.env,self.launches,credit)
                self.bridge.attach(Bus)
            elif arm==ARMS[2]:
                from freqduet_consistent_flow import ConsistentFlowController
                self.bridge=ConsistentFlowController(self.env,self.launches,credit)
                self.bridge.attach(Bus)

        def __call__(self,policy,states,features,theta):
            start=time.perf_counter()
            law=super().__call__(policy,states,features,theta)
            bus=self.runner._bus_for_agent(self.runner._pilot_agent)
            legal=tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
                policy.feasible_action_mask(states)[0].numpy()))
            native=int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
            query=native_query(self.env,bus,legal,native,credit.stream.pending_commands)
            if self.evidence is not None:
                self.evidence.check(bus,legal,native,float(self.env.current_time))
            action=native if self.bridge is None else self.bridge.observe(bus,legal,native)['shadow_action_s']
            query['executed_request_s']=int(action)
            self.decisions.append(query)
            self.callback_wall.append(time.perf_counter()-start)
            return law if self.bridge is None else selected_law(law,int(action),legal)

        def detach(self):
            if self.evidence is not None:
                self.evidence.detach()
            if self.bridge is not None:
                self.bridge.detach()
            super().detach()

        def finish(self,runner,records,holds):
            super().finish(runner,records,holds)
            self.chain=executed_chain(self.decisions,holds)
            assert self.execution['done_reason']=='service_cleared' and self.execution['unexecuted_s']==0

    return Audit()


def label_resources(bridge,dispatches,holds):
    terminal={d:bridge.routes[d][-1].station for d in bridge.routes}
    returns={(e.bus,e.trip):e.completed_s for e in bridge.prefix.flow_events
             if e.station==terminal[e.direction] and e.completed_s is not None}
    by_trip={}
    for h in holds:
        by_trip.setdefault((h['bus_id'],h['trip_id']),[]).append((h['time_s'],h['action_s'],h['station_id']))
    for prediction in bridge.predictions:
        for v in prediction['vehicles']:
            returned=returns.get((v['bus'],v['trip']))
            following=[r for r in dispatches if r['bus']==v['bus'] and returned is not None and r['actual_s']>returned]
            next_trip=min(following,key=lambda r:r['actual_s']) if following else None
            v['actual']=dict(return_s=returned,next_launch_s=None if next_trip is None else next_trip['actual_s'],
                next_trip=None if next_trip is None else next_trip['trip'],
                remaining_trip_h_s=sum(h for t,h,station in by_trip.get((v['bus'],v['trip']),[])
                    if t>=prediction['query_s'] and not (t==prediction['query_s'] and station==v['committed_station'])))


def collect(a,passive=False):
    import numpy as np
    assert read(a.root/'checks/passed.json')['source']==a.source
    if passive:
        name,day,arm='passive',2075001,ARMS[0]
    else:
        assert read(a.root/'passive/passed.json')['source']==a.source
        name='p'+str(a.index)
        day,arm=ITEMS[a.index]
    write(a.root/'started'/f'{name}.json',dict(source=a.source,day=day,arm=arm,automatic_retry=False))
    if arm==ARMS[1]:
        frozen=a.root.parent/'protocol_v6_v275_flow_controller/source_894ad14707/FreqDuet/freqduet/scripts'
        assert frozen.is_dir()
        sys.path.insert(0,str(frozen))
    import run_freqduet_v267_bilateral_factorial as fixed
    credit=fixed.pilot.CreditObserver()
    audit=make_audit(fixed,arm,credit,passive)
    begin=time.perf_counter()
    row,uniforms,_,holds=fixed.pilot.rollout(a.root,SEED,day,'stochastic',SAMPLER,[0.]*4,
        'v276_'+name,probability_rule=audit,trace_observer=audit,credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
        **audit.dispatch,**audit.delays)
    trace=dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,departures=audit.departures,
        queries=audit.decisions,uniforms=uniforms)
    if audit.bridge is not None:
        trace.update(graph_queries=audit.bridge.queries,promise_records=audit.bridge.promise_records)
        if arm==ARMS[2]:
            label_resources(audit.bridge,audit.dispatches,holds)
            trace['resource_predictions']=audit.bridge.predictions
    path=a.root/'traces'/f'{name}.pkl.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xb') as stream:
        pickle.dump(trace,stream,pickle.HIGHEST_PROTOCOL)
    measured=dict(source=a.source,day=day,arm=arm,results=row,contract=audit.contract,
        coverage=audit.coverage,execution=audit.execution,action_chain=audit.chain,
        graph=None if audit.bridge is None else audit.bridge.summary(),
        callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),p95=float(np.quantile(audit.callback_wall,.95)),
                                max=max(audit.callback_wall)),wall_s=time.perf_counter()-begin)
    write(a.root/'pairs'/f'{name}.json',measured)
    if passive:
        oldroot=a.root.parent/'protocol_v6_v275_flow_controller'
        cached=read(oldroot/'pairs/p0.json')
        for field in ('results','contract','coverage','execution'):
            fixed.pilot.joint.assert_same_state(measured[field],cached[field],'V276_passive_'+field)
        with gzip.open(oldroot/'traces/p0.pkl.gz','rb') as stream:
            oldtrace=pickle.load(stream)
        fixed.pilot.joint.assert_same_state(trace,oldtrace,'V276_passive_exact_original_trace')
        assert audit.evidence.checked==5200
        write(a.root/'passive/passed.json',dict(source=a.source,status='exact_equivalence_passed',
            same_state_checks=audit.evidence.checked,physical_paths=1,automatic_retry=False))
    print('V276_PATH_COMPLETE',name,flush=True)


def unit(a):
    returns={}
    write(a.root/'units_started'/f'{a.unit}.json',dict(source=a.source,indices=SHARDS[a.unit]))
    for i in SHARDS[a.unit]:
        returns[str(i)]=subprocess.run([sys.executable,'-u',__file__,'path','--root',str(a.root),
            '--source',a.source,'--index',str(i)],check=False).returncode
    write(a.root/'units'/f'{a.unit}.json',dict(source=a.source,returncodes=returns,
        status='complete' if not any(returns.values()) else 'failed_slot_no_retry'))
    assert not any(returns.values()),returns


def aggregate(a):
    import numpy as np
    missing=[i for i in range(12) if not (a.root/'pairs'/f'p{i}.json').exists()]
    failures=sorted(p.name for p in (a.root/'failures').glob('*.json'))
    if missing or failures:
        write(a.root/'summary/results.json',dict(status='incomplete_failed_slots',missing=missing,failures=failures,
            decision='stop_no_retry_no_confirmation',main_policy_changed=False))
        return
    rows=[read(a.root/'pairs'/f'p{i}.json') for i in range(12)]
    assert [(r['day'],r['arm']) for r in rows]==list(ITEMS)
    cube={(r['day'],r['arm']):r['results'] for r in rows}
    draws=np.random.default_rng(2760201).integers(0,4,(10000,4))
    effects={}
    for base in ARMS[:2]:
        delta=np.asarray([[cube[d,ARMS[2]][m]-cube[d,base][m] for m in METRICS] for d in DAYS])
        boot=delta[draws].mean(axis=1)
        stats={m:dict(mean=float(delta[:,j].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
            by_day={str(d):float(delta[i,j]) for i,d in enumerate(DAYS)}) for j,m in enumerate(METRICS)}
        strict=('passenger_person_s',) if base==ARMS[0] else ('readiness_delay_s','public_launch_delay_total_s')
        nonharm=('cv_squared','readiness_delay_s','public_launch_delay_total_s') if base==ARMS[0] else ('passenger_person_s','cv_squared')
        gate=all(stats[m]['ci95'][1]<0 for m in strict) and all(stats[m]['ci95'][1]<=0 for m in nonharm)
        gate=gate and all(max(stats[m]['by_day'].values())<=0 for m in ('trips_unlaunched','trips_incomplete'))
        effects['candidate_vs_'+base]=dict(metrics=stats,joint_gate=bool(gate))
    passed=all(x['joint_gate'] for x in effects.values())
    write(a.root/'summary/results.json',dict(source=a.source,status='twelve_development_paths_complete',
        effects=effects,joint_gate=passed,by_day=rows,independent_demand_days=4,independent_futures=0,
        main_policy_changed=False,training_updates=0,confirmation_started=False,
        decision='propose_independent_confirmation' if passed else 'close_candidate_no_expansion'))
    print('V276_AGGREGATE_COMPLETE',passed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','passive','path','unit','aggregate'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--index',type=int)
    p.add_argument('--unit',type=int)
    a=p.parse_args()
    try:
        if a.command=='passive':
            collect(a,True)
        elif a.command=='path':
            collect(a)
        else:
            dict(check=check,unit=unit,aggregate=aggregate)[a.command](a)
    except Exception as error:
        if a.command in ('passive','path'):
            name='passive' if a.command=='passive' else 'p'+str(a.index)
            write(a.root/'failures'/f'{name}.json',dict(error=repr(error),traceback=traceback.format_exc()[-6000:],
                slot_consumed=True,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
