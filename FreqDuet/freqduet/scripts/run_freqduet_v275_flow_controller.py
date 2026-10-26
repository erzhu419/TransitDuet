#!/usr/bin/env python3
"""One frozen controller version, twelve new-day paths, no training or retries."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
import traceback
import unittest

from run_freqduet_v272_graph_pilot import read, write, selected_law, executed_chain, METRICS


DAYS = (2075001, 2075002, 2075003, 2075004)
ARMS = ('fixed360_bilateral025', 'flow_apc_fallback', 'flow_homogeneous_fallback')
SEED, SAMPLER = 12011, 27011001
ITEMS = tuple((day,arm) for arm in ARMS for day in DAYS)
SHARDS = tuple(tuple(range(i,len(ITEMS),8)) for i in range(8))


def unused_metadata(root):
    """Read experiment indices, not the per-event/cell/log trees they describe."""
    pattern = re.compile(r'(?<![0-9.])(?:'+'|'.join(map(str,DAYS))+r')(?![0-9.])')
    hits, checked = [], 0
    def consumed(value):
        if isinstance(value,dict):
            for key,item in value.items():
                is_day=(key in ('day','days','scenario_seed','scenario_seeds','demand_seed','demand_seeds')
                        or key.endswith('_days'))
                if is_day:
                    values=item if isinstance(item,list) else [item]
                    if any(v in DAYS for v in values if isinstance(v,int)):
                        return True
                if isinstance(item,(dict,list)) and consumed(item):
                    return True
        elif isinstance(value,list):
            return any(consumed(item) for item in value if isinstance(item,(dict,list)))
        return False
    for experiment in root.parent.iterdir():
        if not experiment.is_dir() or experiment==root:
            continue
        files=list(experiment.glob('*.json'))
        for name in ('budget','started','pairs','reservations','manifests','metadata'):
            directory=experiment/name
            if directory.is_dir():
                files.extend(directory.rglob('*.json'))
        for path in files:
            if pattern.search(path.name) or consumed(read(path)):
                hits.append(str(path.relative_to(root.parent)))
            checked += 1
    assert not hits, ('requested development days already reserved or consumed',hits[:20])
    return dict(files_checked=checked,collisions=hits,raw_trace_reads=0,
                scope='Top-level protocol indices,budget,started,pairs,reservations,manifests,metadata;not cell/event/log trees.')


def check(a):
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'),pattern=pattern) for pattern in
        ('test_v275_flow_controller.py','test_v270_causal_event_graph.py','test_v271_native_event_graph.py',
         'test_v272_graph_pilot.py','test_v273_flow_contract.py','test_v274_passive_flow.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun > 0
    assert 'runner_v3' not in sys.modules
    metadata = unused_metadata(a.root)
    write(a.root/'protocol.json',dict(source=a.source,days=DAYS,arms=ARMS,unique_paths=12,
        actor_seed=SEED,sampler_seed=SAMPLER,max_concurrent=8,shards=SHARDS,
        tests=result.testsRun,metadata=metadata,automatic_retry=False,
        demand='Last three positive-duration nonoverlapping strictly completed clearance blocks, '
            'including full intermediate visits;constant-rate future point extrapolation,not a future bound.',
        continuation='Require full-plan feasibility AND current-only/bilateral-fallback feasibility '
            'and passenger nonharm;minimize worst passenger cost of those two continuations.',
        model_budget=128,beam_width=4,max_variables=6,main_policy_changed=False,training_updates=0,
        independent_future_replications=0,confirmation_started=False))
    for index,(day,arm) in enumerate(ITEMS):
        write(a.root/'budget'/f'p{index}.json',dict(source=a.source,index=index,day=day,arm=arm,
            automatic_retry=False,physical_start_cap=1))
    write(a.root/'checks/status.json',dict(source=a.source,tests=result.testsRun,status='passed',
        check_entry=__file__,metadata=metadata,physical_starts=0,training_updates=0))
    print('V275_CHECK_AND_SEAL_COMPLETE',result.testsRun,flush=True)


def make_audit(fixed,arm,credit):
    import numpy as np
    from freqduet_native_event_graph import native_query
    from freqduet_flow_controller import FlowController

    class Audit(fixed.BenchmarkAudit):
        def __init__(self):
            super().__init__(SEED,'fixed_bilateral025')
            self.bridge, self.decisions, self.callback_wall = None,[],[]

        def attach(self,runner):
            super().attach(runner)
            if arm != ARMS[0]:
                from env.bus import Bus
                self.bridge = FlowController(runner.env,self.launches,credit,homogeneous=arm==ARMS[2])
                self.bridge.attach(Bus)

        def __call__(self,policy,states,features,theta):
            start=time.perf_counter()
            law=super().__call__(policy,states,features,theta)
            bus=self.runner._bus_for_agent(self.runner._pilot_agent)
            legal=tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
                policy.feasible_action_mask(states)[0].numpy()))
            native=int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
            query=native_query(self.env,bus,legal,native,credit.stream.pending_commands)
            action=native if self.bridge is None else self.bridge.observe(bus,legal,native)['shadow_action_s']
            query['executed_request_s']=int(action)
            self.decisions.append(query)
            self.callback_wall.append(time.perf_counter()-start)
            return law if self.bridge is None else selected_law(law,int(action),legal)

        def detach(self):
            if self.bridge is not None:
                self.bridge.detach()
            super().detach()

        def finish(self,runner,records,holds):
            super().finish(runner,records,holds)
            self.chain=executed_chain(self.decisions,holds)
            assert self.execution['done_reason']=='service_cleared'
            assert self.execution['unexecuted_s']==0 and self.execution['physical_vehicles']==12
            if self.bridge is not None:
                receipts={(r['time_s'],r['bus_id'],r['trip_id'],r['station_id']):r['action_s'] for r in holds}
                for promise in self.bridge.promise_records:
                    if promise['status']=='reached':
                        promise['actual_command_verified']=receipts[(promise['reached_s'],*promise['key'])]==promise['actual_request_s']
                        assert promise['actual_command_verified']

    return Audit()


def collect(a):
    import numpy as np
    assert a.index in range(12)
    day,arm=ITEMS[a.index]
    sealed=read(a.root/'protocol.json')
    assert sealed['source']==a.source and sealed['unique_paths']==12
    assert read(a.root/'checks/status.json')['status']=='passed'
    budget=read(a.root/'budget'/f'p{a.index}.json')
    assert budget['source']==a.source and (budget['day'],budget['arm'])==(day,arm)
    write(a.root/'started'/f'p{a.index}.json',dict(source=a.source,day=day,arm=arm,automatic_retry=False))
    import run_freqduet_v267_bilateral_factorial as fixed
    credit=fixed.pilot.CreditObserver()
    audit=make_audit(fixed,arm,credit)
    start=time.perf_counter()
    row,uniforms,_,holds=fixed.pilot.rollout(a.root,SEED,day,'stochastic',SAMPLER,[0.]*4,
        f'v275_p{a.index}',probability_rule=audit,trace_observer=audit,credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
        **audit.dispatch,**audit.delays)
    trace=dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,
        departures=audit.departures,queries=audit.decisions,uniforms=uniforms)
    if audit.bridge is not None:
        trace.update(graph_queries=audit.bridge.queries,flow_events=audit.bridge.prefix.flow_events,
            completed_visits=audit.bridge.prefix.visits,promise_records=audit.bridge.promise_records)
    graph_summary=None if audit.bridge is None else audit.bridge.summary()
    path=a.root/'traces'/f'p{a.index}.pkl.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xb') as stream:
        pickle.dump(trace,stream,protocol=pickle.HIGHEST_PROTOCOL)
    write(a.root/'pairs'/f'p{a.index}.json',dict(source=a.source,index=a.index,day=day,arm=arm,
        results=row,contract=audit.contract,coverage=audit.coverage,execution=audit.execution,
        action_chain=audit.chain,graph=graph_summary,
        callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
            p95=float(np.quantile(audit.callback_wall,.95)),max=float(max(audit.callback_wall))),
        wall_s=time.perf_counter()-start,training_updates=0,main_policy_changed=False))
    print('V275_PATH_COMPLETE',a.index,day,arm,flush=True)


def unit(a):
    assert a.unit in range(8)
    write(a.root/'units_started'/f'{a.unit}.json',dict(source=a.source,items=SHARDS[a.unit]))
    returns={}
    for index in SHARDS[a.unit]:
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'path','--root',str(a.root),
             '--source',a.source,'--index',str(index)]
        returns[str(index)]=subprocess.run(cmd,check=False).returncode
    write(a.root/'units'/f'{a.unit}.json',dict(source=a.source,returncodes=returns,
        status='complete' if not any(returns.values()) else 'contains_failed_path',automatic_retry=False))
    assert not any(returns.values()),returns


def summarize(cube):
    import numpy as np
    draws=np.random.default_rng(2750201).integers(0,4,size=(10000,4))
    effects={}
    for base in (ARMS[0],ARMS[2]):
        delta=np.asarray([[cube[d,ARMS[1]][m]-cube[d,base][m] for m in METRICS] for d in DAYS])
        boot=delta[draws].mean(axis=1)
        stats={m:dict(mean=float(delta[:,i].mean()),ci95=np.quantile(boot[:,i],[.025,.975]).tolist(),
            by_day={str(d):float(delta[j,i]) for j,d in enumerate(DAYS)}) for i,m in enumerate(METRICS)}
        passed=(stats['passenger_person_s']['ci95'][1]<0 and all(stats[m]['ci95'][1]<=0
            for m in ('cv_squared','readiness_delay_s','public_launch_delay_total_s'))
            and all(delta[:,METRICS.index(m)].max()<=0 for m in ('trips_unlaunched','trips_incomplete')))
        effects['apc_vs_'+base]=dict(metrics=stats,joint_development_gate=bool(passed))
    return effects


def aggregate(a):
    import run_freqduet_v267_bilateral_factorial as fixed
    failures=sorted(p.name for p in (a.root/'failures').glob('*.json'))
    missing=[i for i in range(12) if not (a.root/'pairs'/f'p{i}.json').exists()]
    if missing or failures:
        write(a.root/'summary/results.json',dict(source=a.source,status='incomplete_failed_slots',
            missing=missing,failures=failures,decision='stop_no_retry_no_confirmation',main_policy_changed=False))
        return
    rows=[read(a.root/'pairs'/f'p{i}.json') for i in range(12)]
    cube={}
    alignment={}
    for i,row in enumerate(rows):
        assert row['source']==a.source and (row['day'],row['arm'])==ITEMS[i]
        cube[row['day'],row['arm']]=row['results']
    for day in DAYS:
        cell=[r for r in rows if r['day']==day]
        for r in cell[1:]:
            fixed.pilot.joint.assert_same_state(r['contract'],cell[0]['contract'],'V275_matched_contract')
        uniforms=[]
        for r in cell:
            with gzip.open(a.root/'traces'/f"p{r['index']}.pkl.gz",'rb') as stream:
                uniforms.append(pickle.load(stream)['uniforms'])
        for r,other in zip(cell[1:],uniforms[1:]):
            alignment[str(day)+'_'+r['arm']]=fixed.pilot.event.alignment(uniforms[0],other)
    effects=summarize(cube)
    passed=all(r['joint_development_gate'] for r in effects.values())
    write(a.root/'summary/results.json',dict(source=a.source,status='twelve_development_paths_complete',
        effects=effects,joint_development_gate=passed,
        decision='propose_separate_independent_confirmation' if passed else 'close_frozen_candidate_no_expansion',
        by_day=[dict(day=r['day'],arm=r['arm'],results={m:r['results'][m] for m in METRICS},
            graph=r['graph'],action_chain=r['action_chain'],callback_latency_s=r['callback_latency_s']) for r in rows],
        event_uniform_alignment=alignment,independent_demand_days=4,independent_future_replications=0,
        physical_paths=12,confirmation_started=False,training_updates=0,main_policy_changed=False,
        controller_admitted=False,raw_traces_remain_server_side=True))
    print('V275_AGGREGATE_COMPLETE',passed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','path','unit','aggregate'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--index',type=int)
    p.add_argument('--unit',type=int)
    a=p.parse_args()
    try:
        dict(check=check,path=collect,unit=unit,aggregate=aggregate)[a.command](a)
    except Exception as error:
        if a.command=='path':
            write(a.root/'failures'/f'p{a.index}.json',dict(source=a.source,error=repr(error),
                traceback=traceback.format_exc()[-6000:],slot_consumed=True,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
