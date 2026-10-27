#!/usr/bin/env python3
"""Three explicitly authorized, same-policy V276 efficiency replacements."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

import run_freqduet_v276_native_continuation as frozen
from freqduet_indexed_continuation import IndexedFlowController, IndexedNativeGraph
from freqduet_native_continuation import NativeGraph

REMAINING=(8,10,11)


class CheckedIndexedGraph(IndexedNativeGraph):
    def __init__(self,base):
        super().__init__(base,base.observed)
        self.matches,self.reference_wall=0,0.

    def evaluate(self,commands=None,homogeneous=False):
        optimized=super().evaluate(commands,homogeneous)
        begin=time.perf_counter()
        original=NativeGraph.evaluate(self,commands,homogeneous)
        self.reference_wall+=time.perf_counter()-begin
        assert original==optimized, 'frozen/indexed real-state kernel mismatch'
        self.matches+=1
        return optimized


class CheckedIndexedFlowController(IndexedFlowController):
    def __init__(self,env,launches,credit):
        super().__init__(env,launches,credit)
        self.pending_check=None
        self.kernel_audit=dict(verified=False,queries=0,matching_model_calls=0,extra_reference_wall_s=0.)

    def build_model(self,value,visits):
        model,info=super().build_model(value,visits)
        if not self.kernel_audit['verified']:
            model=CheckedIndexedGraph(model)
            self.pending_check=model
        return model,info

    def record_prediction(self,result,value):
        if self.pending_check is not None:
            audit=self.kernel_audit
            audit['queries']+=1
            audit['matching_model_calls']+=self.pending_check.matches
            audit['extra_reference_wall_s']+=self.pending_check.reference_wall
            if result['evaluations']>1:
                audit['verified']=True
            self.pending_check=None
        super().record_prediction(result,value)

    def summary(self):
        return dict(super().summary(),kernel_equivalence_audit=dict(self.kernel_audit))


def make_audit(fixed,credit):
    Base=type(frozen.make_audit(fixed,frozen.ARMS[2],credit,False))
    class Audit(Base):
        def attach(self,runner):
            fixed.BenchmarkAudit.attach(self,runner)
            from env.bus import Bus
            self.bridge=CheckedIndexedFlowController(self.env,self.launches,credit)
            self.bridge.attach(Bus)
    return Audit()


def collect(a):
    auth=frozen.read(a.root/'authorization.json')
    assert auth['source']==a.source and a.index in auth['indices'] and a.index in REMAINING
    assert auth['approved_extra_starts']<=3 and auth['automatic_retry'] is False
    assert not (a.original_root/'pairs'/f'p{a.index}.json').exists(), 'completed original must be reused'
    assert frozen.read(a.original_root/'protocol.json')['source']=='7023e2e784'
    frozen.write(a.root/'started'/f'p{a.index}.json',dict(source=a.source,index=a.index,
        replaces_original_slot=True,automatic_retry=False))
    import run_freqduet_v267_bilateral_factorial as fixed
    credit=fixed.pilot.CreditObserver()
    audit=make_audit(fixed,credit)
    day,arm=frozen.ITEMS[a.index]
    begin=time.perf_counter()
    row,uniforms,_,holds=fixed.pilot.rollout(a.root,frozen.SEED,day,'stochastic',frozen.SAMPLER,
        [0.]*4,'v276_indexed_p'+str(a.index),probability_rule=audit,trace_observer=audit,credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
        **audit.dispatch,**audit.delays)
    frozen.label_resources(audit.bridge,audit.dispatches,holds)
    trace=dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,departures=audit.departures,
        queries=audit.decisions,uniforms=uniforms,graph_queries=audit.bridge.queries,
        promise_records=audit.bridge.promise_records,resource_predictions=audit.bridge.predictions)
    path=a.root/'traces'/f'p{a.index}.pkl.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xb') as stream:
        pickle.dump(trace,stream,pickle.HIGHEST_PROTOCOL)
    result=dict(source=a.source,original_control_source='7023e2e784',index=a.index,day=day,arm=arm,
        results=row,contract=audit.contract,coverage=audit.coverage,execution=audit.execution,
        action_chain=audit.chain,graph=audit.bridge.summary(),wall_s=time.perf_counter()-begin,
        callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
            p95=float(np.quantile(audit.callback_wall,.95)),max=max(audit.callback_wall)))
    frozen.write(a.root/'pairs'/f'p{a.index}.json',result)
    frozen.write(a.root/'receipts'/f'p{a.index}'/'finished.json',dict(source=a.source,
        index=a.index,status='complete',kernel_audit=audit.bridge.kernel_audit,automatic_retry=False))
    print('V276_INDEXED_REPLACEMENT_COMPLETE',a.index,flush=True)


def paired_effects(cube):
    draws=np.random.default_rng(2760201).integers(0,4,(10000,4))
    effects={}
    for base in frozen.ARMS[:2]:
        delta=np.asarray([[cube[d,frozen.ARMS[2]][m]-cube[d,base][m] for m in frozen.METRICS]
                          for d in frozen.DAYS])
        boot=delta[draws].mean(axis=1)
        stats={m:dict(mean=float(delta[:,j].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
            by_day={str(d):float(delta[i,j]) for i,d in enumerate(frozen.DAYS)})
            for j,m in enumerate(frozen.METRICS)}
        strict=('passenger_person_s',) if base==frozen.ARMS[0] else ('readiness_delay_s','public_launch_delay_total_s')
        nonharm=('cv_squared','readiness_delay_s','public_launch_delay_total_s') if base==frozen.ARMS[0] else ('passenger_person_s','cv_squared')
        gate=all(stats[m]['ci95'][1]<0 for m in strict) and all(stats[m]['ci95'][1]<=0 for m in nonharm)
        gate=gate and all(max(stats[m]['by_day'].values())<=0 for m in ('trips_unlaunched','trips_incomplete'))
        effects['candidate_vs_'+base]=dict(metrics=stats,joint_gate=bool(gate))
    return effects


def aggregate(a):
    import run_freqduet_v267_bilateral_factorial as fixed
    auth=frozen.read(a.root/'authorization.json')
    rows,origins,missing=[],{},[]
    for index in range(12):
        root=a.root if index in auth['indices'] else a.original_root
        path=root/'pairs'/f'p{index}.json'
        if not path.exists():
            missing.append(index)
            continue
        row=frozen.read(path)
        assert (row['day'],row['arm'])==frozen.ITEMS[index]
        assert row['source']==(a.source if root==a.root else '7023e2e784')
        rows.append(row)
        origins[str(index)]=dict(root=str(root),source=row['source'])
    failures=[p.name for p in (a.root/'failures').glob('*.json')]
    if missing or failures:
        frozen.write(a.root/'summary/results.json',dict(status='incomplete_failed_slots',
            missing=missing,failures=failures,decision='stop_no_retry_no_confirmation',main_policy_changed=False))
        return
    alignment={}
    for day in frozen.DAYS:
        cell=[r for r in rows if r['day']==day]
        uniforms=[]
        for r in cell:
            fixed.pilot.joint.assert_same_state(r['contract'],cell[0]['contract'],'V276_matched_contract')
            i=frozen.ITEMS.index((day,r['arm']))
            with gzip.open(Path(origins[str(i)]['root'])/'traces'/f'p{i}.pkl.gz','rb') as stream:
                uniforms.append(pickle.load(stream)['uniforms'])
        for r,other in zip(cell[1:],uniforms[1:]):
            aligned=fixed.pilot.event.alignment(uniforms[0],other)
            assert aligned['uniform_mismatches']==0
            alignment[str(day)+'_'+r['arm']]=aligned
    cube={(r['day'],r['arm']):r['results'] for r in rows}
    effects=paired_effects(cube)
    passed=all(r['joint_gate'] for r in effects.values())
    frozen.write(a.root/'summary/results.json',dict(source=a.source,original_control_source='7023e2e784',
        status='twelve_development_paths_complete',effects=effects,joint_gate=passed,
        path_origins=origins,by_day=rows,event_uniform_alignment=alignment,
        independent_demand_days=4,independent_futures=0,extra_start_allowance=auth['approved_extra_starts'],
        original_start_cap=13,total_authorized_start_cap=16,main_policy_changed=False,
        training_updates=0,confirmation_started=False,
        decision='propose_independent_confirmation' if passed else 'close_candidate_no_expansion'))
    print('V276_INDEXED_AGGREGATE_COMPLETE',passed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('path','aggregate'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--index',type=int,choices=REMAINING)
    a=p.parse_args()
    try:
        dict(path=collect,aggregate=aggregate)[a.command](a)
    except Exception as error:
        if a.command=='path':
            frozen.write(a.root/'failures'/f'p{a.index}.json',dict(source=a.source,
                error=repr(error),traceback=traceback.format_exc()[-6000:],slot_consumed=True,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
