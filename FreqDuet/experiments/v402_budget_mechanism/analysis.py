"""Server-only engineering gate, calibration seal and paired confirmation."""

import argparse
from pathlib import Path
import statistics

import numpy as np

from worker import HERE,read,write,helper,grid,selected_cases

METRICS=('journey_s_per_passenger','waiting_s_per_passenger','onboard_s_per_passenger',
         'public_delay_s_per_day','readiness_s_per_trip','cv_squared','holding_s_per_trip','J0')


def folder(root,phase,arm,day):return root/phase/arm/f'd{day}'


def pressure_reference(root,day):
    return root.parent.parent/'FreqDuet-v401-budget-conditioned-upper/run01/development/manual_joint/r050'/f'd{day}'


def paired(args,paths):
    gate=helper(args.assets,'gate_default');physical=helper(args.assets,'physical')
    rows=[read(p/'summary/results.json') for p in paths]
    reference=gate.raw(paths[0]/'raw/path.pkl.gz')
    for row,path in zip(rows,paths):
        assert row['completed'] and row['physical_account']['passed']
        trace=gate.raw(path/'raw/path.pkl.gz')
        pair=physical.crn_pair(reference['uniforms'],trace['uniforms'])
        assert pair['exact'] and pair['common_events']==5200 and pair['left_only']==pair['right_only']==0
        assert row['contract']==rows[0]['contract'] and trace['public_clocks']==reference['public_clocks']
    return rows


def engineering(args,plan):
    gate=helper(args.assets,'gate_default');checks={};roundoff=[]
    comparisons=[('pressure',pressure_reference(args.root,2410001),
                  folder(args.root,'engineering','pressure',2410001)),
                 ('cap45_noop',folder(args.root,'engineering','native_plan',2410001),
                  folder(args.root,'engineering','cap_45',2410001))]
    for name,left,right in comparisons:
        a,b=paired(args,[left,right]);ta=gate.raw(left/'raw/path.pkl.gz');tb=gate.raw(right/'raw/path.pkl.gz')
        differences={}
        for key in ('results','contract','coverage','execution','action_chain'):
            diff=gate.difference(a[key],b[key],key,roundoff)
            if diff is not None:differences[key]=diff
        for key in ('holds','dispatches','arrivals','departures','queries','uniforms','returns','publish_ticks','public_clocks'):
            diff=gate.difference(ta[key],tb[key],key,roundoff)
            if diff is not None:differences[key]=diff
        if name=='pressure':
            for key in ('slow','quota_probes'):
                diff=gate.difference(a[key],b[key],key,roundoff)
                if diff is not None:differences[key]=diff
            diff=gate.difference(ta['slow_events'],tb['slow_events'],'slow_events',roundoff)
            if diff is not None:differences['slow_events']=diff
        checks[name]=dict(passed=not differences,differences=differences)
    result=dict(completed=True,passed=all(v['passed'] for v in checks.values()),checks=checks,
                diagnostic_roundoff=len(roundoff),physical_starts=0,training_updates=0)
    write(args.root/'engineering_gate/summary/results.json',result)
    if result['passed']:write(args.root/'engineering_gate/summary/ready.json',result)
    print('V402_ENGINEERING_GATE',result['passed'],flush=True)


def selection(args,plan):
    data={};paths=[]
    for day in plan['calibration_days']:
        ordered=[pressure_reference(args.root,day)]+[folder(args.root,'calibration',c['arm'],day) for c in grid(plan)]
        rows=paired(args,ordered)
        data[day,'pressure']=rows[0]
        for case,row in zip(grid(plan),rows[1:]):data[day,case['arm']]=row
    exposure=lambda arm:statistics.mean(data[d,arm]['execution']['executed_s'] for d in plan['calibration_days'])
    target=exposure('pressure');families={}
    for kind in ('static','cap'):
        cases=[dict(c,mean_holding_s=exposure(c['arm'])) for c in grid(plan) if c['kind']==kind]
        below=[c for c in cases if c['mean_holding_s']<=target]
        above=[c for c in cases if c['mean_holding_s']>=target]
        if not below or not above:
            families[kind]=dict(passed=False,target=target,cases=cases,reason='outside_calibration_support');continue
        low=min(below,key=lambda c:(target-c['mean_holding_s'],c['parameter']))
        high=min(above,key=lambda c:(c['mean_holding_s']-target,c['parameter']))
        if high['mean_holding_s']==low['mean_holding_s']:
            components=[dict(low,probability=1.)]
        else:
            probability=(target-low['mean_holding_s'])/(high['mean_holding_s']-low['mean_holding_s'])
            components=[dict(low,probability=1-probability),dict(high,probability=probability)]
        matched=sum(c['probability']*c['mean_holding_s'] for c in components)
        assert abs(matched-target)<1e-7 and all(0<=c['probability']<=1 for c in components)
        families[kind]=dict(passed=True,components=components,matched_holding_s=matched,cases=cases)
    result=dict(completed=True,passed=all(v['passed'] for v in families.values()),
        pressure_mean_holding_s=target,families=families,
        calibration={str(d):{arm:{m:r['endpoints'][m] for m in METRICS} for (day,arm),r in data.items() if day==d}
                     for d in plan['calibration_days']},
        physical_starts=0,training_updates=0,test_outcomes_used=False)
    write(args.root/'selection/summary/results.json',result)
    if result['passed']:write(args.root/'selection/summary/ready.json',dict(passed=True))
    print('V402_CALIBRATION_SEALED',result['passed'],families.keys(),flush=True)


def aggregate(args,plan):
    chosen=selected_cases(args.root,plan);sealed=read(args.root/'selection/summary/results.json')
    data={};execution={};times=[]
    for day in plan['confirmation_days']:
        rows=paired(args,[folder(args.root,'confirmation',c['arm'],day) for c in chosen])
        for case,row in zip(chosen,rows):
            data[day,case['arm']]=row['endpoints'];times.append(row['wall_s'])
            execution[f'{case["arm"]}_d{day}']=dict(physical=True,cap=row['cap_account'],
                supported=row['graph']['supported_queries'],overrides=row['graph']['actual_overrides'],
                executed_holding_s=row['execution']['executed_s'],planned_changes=row['slow']['planned_changes'])
        print('V402_CONFIRMED_DAY',day,flush=True)
    n=len(plan['confirmation_days']);rng=np.random.default_rng(plan['bootstrap_seed'])
    draws=rng.integers(0,n,size=(plan['bootstrap_draws'],n));effects={};decisions={}
    for kind in ('static','cap'):
        components=sealed['families'][kind]['components'];effects[kind]={}
        for metric in METRICS:
            values=np.asarray([data[d,'pressure'][metric]-sum(c['probability']*data[d,c['arm']][metric]
                               for c in components) for d in plan['confirmation_days']])
            ci=np.quantile(values[draws].mean(1),[.025,.975]).tolist()
            effects[kind][metric]=dict(mean=float(values.mean()),ci95=ci,
                by_day=dict(zip(map(str,plan['confirmation_days']),values.tolist())))
        decisions[kind]=dict(dispatch_improves=effects[kind]['public_delay_s_per_day']['ci95'][1]<0,
            cv_nonharm=effects[kind]['cv_squared']['ci95'][1]<=0)
        decisions[kind]['passed']=all(decisions[kind].values())
    starts=sum(1 for phase in ('engineering','calibration','confirmation')
               for _ in (args.root/phase).glob('*/*/started.json'))
    assert starts==3+64+n*len(chosen)<=plan['physical_start_cap']
    result=dict(completed=True,passed=all(v['passed'] for v in decisions.values()),decisions=decisions,
        effects=effects,sealed_components={k:sealed['families'][k]['components'] for k in ('static','cap')},
        by_day={str(d):{c['arm']:data[d,c['arm']] for c in chosen} for d in plan['confirmation_days']},
        execution=execution,physical_starts_charged=starts,training_updates=0,main_policy_changed=False,
        statistical_unit='16 independent simulation demand days; fixed daily-mixture expectations',
        wall_s=dict(median=statistics.median(times),max=max(times)),
        limitations='One operating retention and one frozen fast actor; no real deployment or learned-frequency superiority claim.')
    write(args.root/'aggregate/summary/results.json',result)
    print('V402_FINAL',result['passed'],decisions,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('engineering','selection','aggregate'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a,read(HERE/'protocol.json'))
