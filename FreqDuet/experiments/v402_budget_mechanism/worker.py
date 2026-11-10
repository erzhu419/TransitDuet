"""One charged full-day case, using the frozen optimized native runtime."""

import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,separators=(',',':'),allow_nan=False,default=lambda v:v.item())
        stream.write('\n')


def helper(assets,name):
    spec=importlib.util.spec_from_file_location('v402_'+name,assets/'support'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def bind(assets):
    import basis_worker as frozen
    joint=frozen.base.fixed.pilot.joint
    joint.MODEL_ROOT=assets/'runtime/model';joint.TRAIN_ROOT=assets/'runtime/native_training'
    joint.optimized_runtime.__globals__['OPTIMIZED_ROOT']=Path(
        '/home/zhengliang01/scheduleurm_work/FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet')
    import runner_v3
    runner_v3.SCRIPT_DIR=joint.MODEL_ROOT
    return frozen


def policy_arm(kind,value):return kind+'_'+str(value).replace('.','p')


def grid(plan):
    return [dict(arm=policy_arm(kind,v),kind=kind,parameter=v) for kind,values in
        (('static',plan['static_rates_s_per_remaining_control']),('cap',plan['stop_caps_s'])) for v in values]


def selected_cases(root,plan):
    selection=read(root/'selection/summary/results.json');assert selection['passed']
    result=[dict(arm='pressure',kind='pressure',parameter=.5)]
    for family in ('static','cap'):
        for item in selection['families'][family]['components']:
            if item['arm'] not in {r['arm'] for r in result}:result.append(item)
    return result


def make_audit(frozen,credit,kind,parameter):
    from policies import Manager,Controller
    class Audit(frozen.base.Audit):
        def attach(self,runner):
            super().attach(runner)
            self.bridge.detach()
            self.manager=Manager(self.env,self.credit,kind,parameter)
            self.bridge=Controller(self.env,self.launches,self.credit,self.manager)
            from env.bus import Bus
            self.bridge.attach(Bus)
    return Audit('joint',credit)


def run(args):
    plan=read(HERE/'protocol.json')
    assert read(args.root/'checks/summary/ready.json')['passed']
    assert read(args.root/'freshness/summary/ready.json')['passed']
    if args.phase!='engineering':assert read(args.root/'engineering_gate/summary/ready.json')['passed']
    if args.phase=='calibration':
        assert args.day in plan['calibration_days'] and args.arm in {r['arm'] for r in grid(plan)}
        case=next(r for r in grid(plan) if r['arm']==args.arm)
    elif args.phase=='confirmation':
        assert args.day in plan['confirmation_days']
        case=next(r for r in selected_cases(args.root,plan) if r['arm']==args.arm)
    else:
        assert args.day==2410001 and args.arm in ('pressure','native_plan','cap_45')
        case=dict(arm=args.arm,kind='native_plan' if args.arm=='native_plan' else
                  'pressure' if args.arm=='pressure' else 'cap',parameter=.5 if args.arm=='pressure' else 45)
    out=args.root/args.phase/args.arm/f'd{args.day}'
    out.mkdir(parents=True,exist_ok=False)
    identity=dict(protocol_id=plan['protocol_id'],phase=args.phase,day=args.day,**case,
        physical_starts=1,training_updates=0,main_policy_changed=False,automatic_retry=False)
    write(out/'started.json',identity);begin=time.perf_counter()
    try:
        frozen=bind(args.assets)
        credit=frozen.base.fixed.pilot.CreditObserver()
        native=case['kind']=='native_plan'
        audit=frozen.base.Audit('plan',credit) if native else make_audit(frozen,credit,case['kind'],case['parameter'])
        runtime=args.root/'runtime'/f'{args.phase}_{args.arm}_d{args.day}'
        runtime.mkdir(exist_ok=False)
        label=f'v314_d{args.day}_'+('plan' if native else 'joint')
        row,uniforms,_,holds=frozen.base.fixed.pilot.rollout(runtime,12011,args.day,'stochastic',27011001,
            [0.]*4,label,probability_rule=audit,trace_observer=audit,credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,**audit.dispatch,**audit.delays)
        frozen.base.previous.label_resources(audit.bridge,audit.dispatches,holds)
        trace=dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,departures=audit.departures,
            queries=audit.decisions,uniforms=uniforms,returns=credit.stream.returns,
            graph_queries=audit.bridge.queries,resource_predictions=audit.bridge.predictions,
            slow_events=audit.manager.events,publish_ticks=audit.publish_ticks,public_clocks=audit.public_clocks)
        issues=getattr(audit.manager,'cap_issues',[])
        trace['cap_issues']=issues;caps={(r['bus'],r['trip']):r for r in issues};checked=0
        for index,q in enumerate(audit.decisions):
            issue=caps.get((q['bus'],q['trip']))
            if issue and index>=issue['after_queries']:
                assert q['executed_request_s']<=issue['cap_s'];checked+=1
        result=dict(identity,completed=True,status='complete_development_path',results=row,contract=audit.contract,
            actor=12011,sampler=27011001,confirmation_expanded=False,
            coverage=audit.coverage,execution=audit.execution,action_chain=audit.chain,
            graph=audit.bridge.summary(),slow=audit.manager.summary(),cap_account=dict(passed=True,
                issued_cycles=len(issues),checked_commands=checked))
        probes=[q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result['quota_probes']=dict(attempted=len(probes),supported=sum(p['supported'] is True for p in probes),
            changed=sum(p['changed'] is True for p in probes),extra_evaluations=sum(p['evals'] for p in probes))
        physical=helper(args.assets,'physical')
        result['physical_account']=helper(args.assets,'v326').physical_account(physical,result,trace)
        assert result['physical_account']['passed']
        if case['kind']=='static':assert result['slow']['budget_tightenings']==0
        if case['kind']=='cap':assert checked>0 and not getattr(audit.manager,'capped_cycles')
        units=read(args.assets/'K4/config.json')['reward_units']
        result['endpoints']=helper(args.assets,'v324').endpoints(result,262,units)
        result['wall_s']=time.perf_counter()-begin
        frozen.dump(out/'raw/path.pkl.gz',trace)
        write(out/'summary/results.json',result)
        print('V402_CASE_COMPLETE',args.phase,args.arm,args.day,flush=True)
    except Exception:
        write(out/'summary/failure.json',dict(identity,error=traceback.format_exc()[-6500:]));raise
    finally:write(out/'summary/finished.json',dict(identity,completed=(out/'summary/results.json').exists()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--phase',choices=('engineering','calibration','confirmation'),required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--day',type=int,required=True)
    run(parser.parse_args())
