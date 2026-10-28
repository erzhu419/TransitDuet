#!/usr/bin/env python3
"""Twenty authorized development paths; frozen historical arms run in separate interpreters."""

import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback


DAYS = (2077001,2077002,2077003,2077004)
ARMS = ('fixed360_bilateral025','frozen_v275_apc','frozen_v276_apc',
        'arrival_cv_beam','arrival_cv_current')
ITEMS = tuple((day,arm) for arm in ARMS for day in DAYS)
METRICS = ('passenger_person_s','journey_min','cv_squared','readiness_delay_s',
    'public_launch_delay_total_s','holding_s_per_trip','waiting_person_s','onboard_person_s',
    'trips_unlaunched','trips_incomplete')


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def arm_env(a,arm):
    result = dict(os.environ,**read(a.env_file))
    src = Path(__file__).resolve().parents[1]
    if arm == ARMS[1]:
        frozen = a.results_root/'protocol_v6_v275_flow_controller/source_894ad14707/FreqDuet/freqduet'
        result['PYTHONPATH'] = str(frozen/'scripts')+':'+str(frozen/'tests')+':'+result['PYTHONPATH']
    elif arm in ARMS[3:]:
        result['PYTHONPATH'] = str(src/'scripts')+':'+str(src/'tests')+':'+result['PYTHONPATH']
    return result


def child_args(a,command):
    return [sys.executable,'-u',str(Path(__file__).resolve()),command,'--root',str(a.root),
        '--results-root',str(a.results_root),'--env-file',str(a.env_file),'--source',a.source]


def rollout_root(a):
    # The frozen evaluator locates its shared runtime proof in root.parent.
    # Matrix records stay nested; runtime logs/ready markers use the outer root.
    return a.root.parent


def controller(arm):
    # Python inserts the entry script directory before PYTHONPATH. Remove that
    # insertion for historical arms so their frozen module resolution is real.
    own = Path(__file__).resolve().parent
    if arm in ARMS[:3]:
        sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != own]
    if arm == ARMS[0]:
        return None,{}
    if arm == ARMS[1]:
        import freqduet_flow_controller as flow
        cls = flow.FlowController
        assert 'source_894ad14707' in flow.__file__
    elif arm == ARMS[2]:
        import freqduet_indexed_continuation as model
        cls = model.IndexedFlowController
        assert 'source_4a216af00c' in model.__file__
    else:
        import freqduet_arrival_metric_control as model
        cls = model.ArrivalMetricController if arm == ARMS[3] else model.CurrentActionController
        assert own == Path(model.__file__).resolve().parent
    import freqduet_flow_controller as flow
    return cls,dict(controller=sys.modules[cls.__module__].__file__,flow=flow.__file__)


def inventory(a):
    def contains(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ('day','days','demand_seed','demand_seeds','scenario_seed','scenario_seeds') or key.endswith('_days'):
                    if any(v in DAYS for v in (value if isinstance(value,list) else [value]) if isinstance(v,int)):
                        return True
                if isinstance(value,(dict,list)) and contains(value):
                    return True
        elif isinstance(obj,list):
            return any(contains(v) for v in obj if isinstance(v,(dict,list)))
        return False
    files = 0
    for experiment in a.results_root.iterdir():
        if not experiment.is_dir() or experiment == a.root.parent:
            continue
        paths = list(experiment.glob('*.json'))
        for name in ('budget','started','pairs','reservations','manifests','metadata'):
            paths.extend((experiment/name).rglob('*.json'))
        for path in paths:
            assert not contains(read(path)), ('new development day already consumed',str(path))
            files += 1
    return dict(files_checked=files,collisions=0,raw_trace_reads=0)


def prepare(a):
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('check_freqduet_v277_arrival_metric.py')),
        '--out',str(a.root/'preflight/synthetic.json'),'--source',a.source],check=True)
    metadata = inventory(a)
    failed = a.root.parent/'matrix'
    assert sorted(p.name for p in (failed/'failures').glob('*.json')) == [f'p{i}.json' for i in range(8)]
    assert not list((failed/'pairs').glob('*.json'))
    for i in range(8):
        assert 'runtime_optimization_20261014/benchmark_summary.json' in read(failed/'failures'/f'p{i}.json')['traceback']
    modules = {}
    for arm in ARMS:
        subprocess.run(child_args(a,'probe')+['--arm',arm],env=arm_env(a,arm),check=True)
        modules[arm] = read(a.root/'preflight/probes'/(arm+'.json'))
    write(a.root/'protocol.json',dict(source=a.source,days=DAYS,arms=ARMS,items=ITEMS,
        physical_start_cap=28,unique_development_paths=20,failed_prior_starts=8,
        authorization='user_approved_20_then_8_identical_startup_replacements',
        max_concurrent_paths=8,actor=12011,sampler=27011001,automatic_retry=False,
        independent_days=4,independent_futures=0,training_updates=0,main_policy_changed=False,
        bootstrap_draws=10000,bootstrap_seed=2770201,
        gates=dict(vs_fixed='passenger upper95<0; CV2/readiness/public-delay upper95<=0',
            vs_v275='readiness/public-delay upper95<0; passenger/CV2 upper95<=0',
            vs_v276='passenger/CV2/readiness/public-delay upper95<=0',
            trips='no extra unlaunched/incomplete trips on any day',
            eligibility='each corrected arm must pass all three historical comparator gates',
            enum_vs_beam='paired description, not selection by model score or automatic promotion'),
        scope='Development only; no independent confirmation, APC attribution or frequency claim.'))
    write(a.root/'preflight/passed.json',dict(source=a.source,status='passed',metadata=metadata,
        module_resolution=modules,physical_starts=0,training_updates=0))
    print('V277_PREPARE_COMPLETE',flush=True)


def path(a):
    assert read(a.root/'protocol.json')['source'] == a.source
    day,arm = ITEMS[a.index]
    write(a.root/'started'/f'p{a.index}.json',dict(source=a.source,day=day,arm=arm,
        slot_consumed=True,automatic_retry=False))
    cls,modules = controller(arm)
    import numpy as np
    import run_freqduet_v267_bilateral_factorial as fixed
    import run_freqduet_v276_native_continuation as prior
    credit = fixed.pilot.CreditObserver()
    Base = type(prior.make_audit(fixed,prior.ARMS[0],credit,False))
    class Audit(Base):
        def attach(self,runner):
            fixed.BenchmarkAudit.attach(self,runner)
            if cls is not None:
                from env.bus import Bus
                self.bridge = cls(self.env,self.launches,credit)
                self.bridge.attach(Bus)
    audit = Audit()
    begin = time.perf_counter()
    row,uniforms,_,holds = fixed.pilot.rollout(rollout_root(a),12011,day,'stochastic',27011001,
        [0.]*4,'v277_p'+str(a.index),probability_rule=audit,trace_observer=audit,credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
        **audit.dispatch,**audit.delays)
    bridge = audit.bridge
    if arm in ARMS[2:]:
        prior.label_resources(bridge,audit.dispatches,holds)
    trace = dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,
        departures=audit.departures,queries=audit.decisions,uniforms=uniforms,
        graph_queries=[] if bridge is None else bridge.queries,
        resource_predictions=bridge.predictions if arm in ARMS[2:] else [])
    dest = a.root/'traces'/f'p{a.index}.pkl.gz'
    dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest,'xb') as stream:
        pickle.dump(trace,stream,pickle.HIGHEST_PROTOCOL)
    write(a.root/'pairs'/f'p{a.index}.json',dict(source=a.source,index=a.index,day=day,arm=arm,
        results=row,contract=audit.contract,coverage=audit.coverage,execution=audit.execution,
        action_chain=audit.chain,graph=None if bridge is None else bridge.summary(),
        module_resolution=modules,wall_s=time.perf_counter()-begin,
        callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
            p95=float(np.quantile(audit.callback_wall,.95)),max=max(audit.callback_wall))))
    print('V277_PATH_COMPLETE',a.index,flush=True)


def shard(a):
    assert read(a.root/'preflight/passed.json')['source'] == a.source
    completed,failed = [],[]
    for index in range(a.shard,20,8):
        process = subprocess.run(child_args(a,'path')+['--index',str(index)],
                                 env=arm_env(a,ITEMS[index][1]))
        if process.returncode:
            failed.append(index)
            break
        completed.append(index)
    write(a.root/'receipts'/f'shard_{a.shard}'/'finished.json',dict(source=a.source,
        status='failed' if failed else 'complete',completed=completed,failed=failed,automatic_retry=False))
    if failed:
        raise RuntimeError('physical slot failed; no automatic retry')
    print('V277_SHARD_COMPLETE',a.shard,flush=True)


def paired_gate(stats,base):
    strict = ('passenger_person_s',) if base == ARMS[0] else (
        ('readiness_delay_s','public_launch_delay_total_s') if base == ARMS[1] else ())
    nonharm = tuple(m for m in ('passenger_person_s','cv_squared','readiness_delay_s',
                                'public_launch_delay_total_s') if m not in strict)
    gate = all(stats[m]['ci95'][1]<0 for m in strict) and all(stats[m]['ci95'][1]<=0 for m in nonharm)
    return gate and all(max(stats[m]['by_day'].values())<=0 for m in ('trips_unlaunched','trips_incomplete'))


def aggregate(a):
    import numpy as np
    import run_freqduet_v267_bilateral_factorial as fixed
    rows,missing = [],[]
    for i in range(20):
        p = a.root/'pairs'/f'p{i}.json'
        if not p.exists():
            missing.append(i)
        else:
            r = read(p)
            assert r['source'] == a.source and (r['day'],r['arm']) == ITEMS[i]
            rows.append(r)
    if missing:
        write(a.root/'summary/results.json',dict(status='incomplete_failed_slots',missing=missing,
            main_policy_changed=False,decision='stop_no_retry_no_confirmation'))
        return
    cube = {(r['day'],r['arm']):r for r in rows}
    alignment = {}
    for day in DAYS:
        cell = [cube[day,arm] for arm in ARMS]
        uniforms = []
        for r in cell:
            fixed.pilot.joint.assert_same_state(r['contract'],cell[0]['contract'],'V277_matched_contract')
            with gzip.open(a.root/'traces'/f"p{r['index']}.pkl.gz",'rb') as stream:
                uniforms.append(pickle.load(stream)['uniforms'])
        for arm,other in zip(ARMS[1:],uniforms[1:]):
            value = fixed.pilot.event.alignment(uniforms[0],other)
            assert value['uniform_mismatches'] == 0
            alignment[str(day)+'_'+arm] = value
    draws = np.random.default_rng(2770201).integers(0,4,(10000,4))
    effects,eligible = {},{}
    for arm in ARMS[3:]:
        gates = []
        for base in ARMS[:3]:
            delta = np.array([[cube[d,arm]['results'][m]-cube[d,base]['results'][m]
                               for m in METRICS] for d in DAYS])
            boot = delta[draws].mean(axis=1)
            stats = {m:dict(mean=float(delta[:,j].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
                by_day={str(d):float(delta[i,j]) for i,d in enumerate(DAYS)}) for j,m in enumerate(METRICS)}
            gate = paired_gate(stats,base)
            effects[arm+'_vs_'+base] = dict(metrics=stats,joint_gate=bool(gate))
            gates.append(gate)
        eligible[arm] = bool(all(gates))
    enum_delta = np.array([[cube[d,ARMS[4]]['results'][m]-cube[d,ARMS[3]]['results'][m]
                           for m in METRICS] for d in DAYS])
    boot = enum_delta[draws].mean(axis=1)
    comparison = {m:dict(mean=float(enum_delta[:,j].mean()),
        ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
        by_day={str(d):float(enum_delta[i,j]) for i,d in enumerate(DAYS)}) for j,m in enumerate(METRICS)}
    write(a.root/'summary/results.json',dict(source=a.source,status='twenty_development_paths_complete',
        effects=effects,eligible=eligible,current_vs_beam=comparison,by_day=rows,
        event_uniform_alignment=alignment,independent_demand_days=4,independent_futures=0,
        main_policy_changed=False,training_updates=0,confirmation_started=False,
        failed_prior_starts=8,total_authorized_start_cap=28,
        decision='review_eligible_candidate_before_confirmation' if any(eligible.values()) else 'close_no_expansion'))
    print('V277_AGGREGATE_COMPLETE',eligible,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','probe','path','shard','aggregate'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--index',type=int,choices=range(20))
    parser.add_argument('--shard',type=int,choices=range(8))
    a = parser.parse_args()
    if a.command == 'probe':
        _,modules = controller(a.arm)
        import run_freqduet_v267_bilateral_factorial as fixed
        _,runner,_,_ = fixed.pilot.joint.optimized_runtime(rollout_root(a))
        modules['runtime_runner'] = sys.modules[runner.__module__].__file__
        modules['runtime_proof'] = str(rollout_root(a).parent/'runtime_optimization_20261014')
        write(a.root/'preflight/probes'/(a.arm+'.json'),modules)
        return
    try:
        dict(prepare=prepare,path=path,shard=shard,aggregate=aggregate)[a.command](a)
    except Exception as error:
        if a.command == 'path':
            write(a.root/'failures'/f'p{a.index}.json',dict(source=a.source,error=repr(error),
                traceback=traceback.format_exc()[-6000:],slot_consumed=True,automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
