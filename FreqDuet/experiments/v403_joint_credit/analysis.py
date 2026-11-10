"""Server-side exact engineering gates and the sealed four-day screen."""

import argparse
from collections import Counter
from pathlib import Path
import statistics

import numpy as np

from worker import HERE,read,write,helper

METRICS=('journey_s_per_passenger','waiting_s_per_passenger','onboard_s_per_passenger',
         'public_delay_s_per_day','readiness_s_per_trip','cv_squared','holding_s_per_trip','J0')


def folder(root,phase,arm,day):return root/phase/arm/f'd{day}'


def paired(args,paths):
    gate=helper(args.assets,'gate_default');physical=helper(args.assets,'physical')
    rows=[read(p/'summary/results.json') for p in paths]
    reference=gate.raw(paths[0]/'raw/path.pkl.gz')
    for row,path in zip(rows,paths):
        assert row['completed'] and row['physical_account']['passed']
        trace=gate.raw(path/'raw/path.pkl.gz')
        match=physical.crn_pair(reference['uniforms'],trace['uniforms'])
        assert match['exact'] and match['common_events']==5200
        assert match['left_only']==match['right_only']==0
        assert row['contract']==rows[0]['contract'] and trace['public_clocks']==reference['public_clocks']
    return rows


def equivalence(args,plan):
    old=args.root.parent.parent/'FreqDuet-v402-budget-mechanism/run01/engineering/pressure/d2410001'
    gate=helper(args.assets,'gate_default');checks={};roundoff=[]
    for arm in ('pressure','candidate_noop'):
        new=folder(args.root,'engineering',arm,plan['engineering_day'])
        a,b=paired(args,[old,new]);left=gate.raw(old/'raw/path.pkl.gz');right=gate.raw(new/'raw/path.pkl.gz')
        errors={}
        for key in ('results','contract','coverage','execution','action_chain','slow','quota_probes'):
            difference=gate.difference(a[key],b[key],key,roundoff)
            if difference is not None:errors[key]=difference
        for key in ('holds','dispatches','arrivals','departures','queries','uniforms','returns',
                    'publish_ticks','public_clocks','slow_events'):
            difference=gate.difference(left[key],right[key],key,roundoff)
            if difference is not None:errors[key]=difference
        checks[arm]=dict(passed=not errors,differences=errors)
    output=dict(completed=True,passed=all(r['passed'] for r in checks.values()),
                checks=checks,diagnostic_roundoff=len(roundoff),physical_starts=0,training_updates=0)
    write(args.root/'equivalence/summary/results.json',output)
    if output['passed']:write(args.root/'equivalence/summary/ready.json',output)
    print('V403_EQUIVALENCE',output['passed'],flush=True)


def engineering(args,plan):
    equivalent=read(args.root/'equivalence/summary/results.json')
    row=read(folder(args.root,'engineering','candidate',plan['engineering_day'])/'summary/results.json')
    passed=equivalent['passed'] and row['physical_account']['passed'] and row['credit_account']['passed']
    passed=passed and row['credit_account']['admissions']>0
    output=dict(completed=True,passed=bool(passed),credit_account=row['credit_account'],
                request_counts=row['slow']['request_counts'],physical_starts=0,training_updates=0,
                performance_outcomes_used=False)
    write(args.root/'engineering_gate/summary/results.json',output)
    if passed:write(args.root/'engineering_gate/summary/ready.json',output)
    print('V403_ENGINEERING',bool(passed),row['credit_account']['admissions'],flush=True)


def aggregate(args,plan):
    data={};execution={};times=[];reasons=Counter()
    for day in plan['development_days']:
        rows=paired(args,[folder(args.root,'development',a,day) for a in plan['arms']])
        for arm,row in zip(plan['arms'],rows):
            data[day,arm]=row['endpoints'];times.append(row['wall_s'])
            execution[f'{arm}_d{day}']=dict(physical=row['physical_account']['passed'],
                credit=row['credit_account'],holding_s=row['execution']['executed_s'],
                supported_queries=row['graph']['supported_queries'],overrides=row['graph']['actual_overrides'],
                planned_changes=row['slow']['planned_changes'],request_counts=row['slow'].get('request_counts',{}))
        candidate=folder(args.root,'development','candidate',day)
        trace=helper(args.assets,'gate_default').raw(candidate/'raw/path.pkl.gz')
        reasons.update(r['reason'] for r in trace['credit_decisions'])
        print('V403_PAIRED_DAY',day,flush=True)
    n=len(plan['development_days']);rng=np.random.default_rng(plan['bootstrap_seed'])
    draws=rng.integers(0,n,size=(plan['bootstrap_draws'],n));effects={};decisions={}
    comparisons={'static':[('static_7p5',1-plan['static_high_probability']),('static_10',plan['static_high_probability'])],
                 'cap':[('cap_10',1-plan['cap_high_probability']),('cap_15',plan['cap_high_probability'])],
                 'pressure':[('pressure',1.)]}
    for comparator,components in comparisons.items():
        effects[comparator]={}
        for metric in METRICS:
            values=np.asarray([data[d,'candidate'][metric]-sum(w*data[d,a][metric] for a,w in components)
                               for d in plan['development_days']])
            effects[comparator][metric]=dict(mean=float(values.mean()),
                descriptive_ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
                by_day=dict(zip(map(str,plan['development_days']),values.tolist())))
        if comparator!='pressure':
            decisions[comparator]=dict(mean_delay_improves=effects[comparator]['public_delay_s_per_day']['mean']<0,
                                      mean_cv_nonharm=effects[comparator]['cv_squared']['mean']<=0)
            decisions[comparator]['passed']=all(decisions[comparator].values())
    starts=sum(1 for phase in ('engineering','development') for _ in (args.root/phase).glob('*/*/started.json'))
    assert starts==plan['physical_start_cap']==27
    output=dict(completed=True,development_screen_passed=all(v['passed'] for v in decisions.values()),
        decisions=decisions,effects=effects,credit_decision_reasons=dict(reasons),execution=execution,
        by_day={str(d):{a:data[d,a] for a in plan['arms']} for d in plan['development_days']},
        means={a:{m:statistics.mean(data[d,a][m] for d in plan['development_days']) for m in METRICS}
               for a in plan['arms']},physical_starts_charged=starts,training_updates=0,
        main_policy_changed=False,independent_confirmation=False,
        wall_s=dict(median=statistics.median(times),max=max(times)))
    write(args.root/'aggregate/summary/results.json',output)
    print('V403_FINAL',output['development_screen_passed'],decisions,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('equivalence','engineering','aggregate'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a,read(HERE/'protocol.json'))
