"""Server-only tail, early-departure and predeclared vehicle-chain readouts."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study import read, write
from worker import helper
from archive_figures import style, save


def circulation_readout(args,plan,gate):
    """Evaluation-only trip labels; never used to select a control action."""
    from collections import defaultdict
    output={};plan_binding={}
    for day in plan['days']['nominal']:
        arms={}
        for arm in plan['arms']:
            trace=gate.raw(args.root/'cases/nominal'/str(day)/arm/'raw/path.pkl.gz')
            holds=defaultdict(float)
            for q in trace['queries']:
                holds[q['trip']]+=q['executed_request_s']
            returned={int(r['trip_id']):r['physical_time_s'] for r in trace['returns']}
            launches=trace['dispatches']
            budgeted={int(b['trip']) for e in trace['slow_events'] if e['kind']=='publish' for b in e['budgets']}
            rows={}
            for d in launches:
                later=[v for v in launches if v['bus']==d['bus'] and v['actual_s']>d['actual_s']]
                following=min(later,key=lambda v:v['actual_s']) if later else None
                rows[d['trip']]=dict(holding_s=holds[d['trip']],actual_s=d['actual_s'],bus=d['bus'],
                    scheduled_s=d['scheduled_s'],eligible_s=d['eligible_s'],
                    return_s=returned[d['trip']],public_s=d['public_s'],following=following,
                    budgeted=d['trip'] in budgeted)
            arms[arm]=rows
        output[str(day)]={}
        plan_binding[str(day)]={}
        for arm,reference in (('plan','fixed_model'),('pressure','budget')):
            changed=[r for r in arms[arm].values() if r['scheduled_s']!=r['public_s']]
            plan_binding[str(day)][arm]=dict(
                revised_final_releases=len(changed),
                advanced=sum(r['scheduled_s']<r['public_s'] for r in changed),
                departed_at_revised_release=sum(r['actual_s']==r['scheduled_s'] for r in changed),
                later_than_revised_release=sum(r['actual_s']>r['scheduled_s'] for r in changed),
                conditional_comparator=reference,
                actual_departures_changed=sum(r['actual_s']!=arms[reference][t]['actual_s'] for t,r in arms[arm].items()),
                assigned_vehicles_changed=sum(r['bus']!=arms[reference][t]['bus'] for t,r in arms[arm].items()))
        for arm in (v for v in plan['arms'] if v!='pressure'):
            groups={}
            assert arms[arm].keys()==arms['pressure'].keys()
            for trip,c in arms['pressure'].items():
                b=arms[arm][trip];hour=int(c['public_s']//3600)
                group=f"{hour}:"+('pressure_issued' if c['budgeted'] else 'no_pressure_issue')
                row=groups.setdefault(group,dict(trips=0,holding_delta_s=0.,return_delta_s=0.,
                    departure_delta_s=0.,readiness_delta_s=0.,public_delay_delta_s=0.,
                    earlier_returns=0,earlier_departures=0,
                    same_following_trip=0,following_departure_delta_s=0.))
                row['trips']+=1
                row['holding_delta_s']+=c['holding_s']-b['holding_s']
                row['return_delta_s']+=c['return_s']-b['return_s']
                row['departure_delta_s']+=c['actual_s']-b['actual_s']
                row['readiness_delta_s']+=(c['actual_s']-c['eligible_s'])-(b['actual_s']-b['eligible_s'])
                row['public_delay_delta_s']+=max(0.,c['actual_s']-c['public_s'])-max(0.,b['actual_s']-b['public_s'])
                row['earlier_returns']+=c['return_s']<b['return_s']
                row['earlier_departures']+=c['actual_s']<b['actual_s']
                if c['following'] and b['following'] and c['following']['trip']==b['following']['trip']:
                    row['same_following_trip']+=1
                    row['following_departure_delta_s']+=c['following']['actual_s']-b['following']['actual_s']
            output[str(day)][arm]=groups
    totals={}
    pooled_hours={}
    for day,comparators in output.items():
        totals[day]={}
        for arm,groups in comparators.items():
            fields=next(iter(groups.values())).keys()
            totals[day][arm]={k:sum(r[k] for r in groups.values()) for k in fields}
            for group,row in groups.items():
                target=pooled_hours.setdefault(arm,{}).setdefault(group,{k:0. for k in fields})
                for k,v in row.items():target[k]+=v
    return dict(by_day=output,day_totals=totals,pooled_hour_sums=pooled_hours,
        plan_binding=plan_binding,contrast='pressure minus named comparator',
        units='sums and counts by immutable-public-release hour and whether pressure issued credit',
        statistical_unit='paired demand day; trips are not independent replications',
        following_filter='same next trip label in each branch; physical identity is evaluation-only',
        interpretation='Realized trajectory differences and policy-dependent budget groups, not isolated causal effects of an individual hold.')


def main(args):
    data = read(args.root / 'aggregate/summary/results.json')
    plan = read(Path(__file__).parent / 'protocol.json')
    formal_starts=list((args.root/'cases').glob('*/*/*/started.json'))
    engineering_starts=list((args.root/'engineering').glob('*/*/*/started.json'))
    charged=sum(read(p)['physical_starts'] for p in formal_starts+engineering_starts)
    failed_captures=sum((p.parent/'summary/failure.json').is_file() for p in engineering_starts)
    assert len(formal_starts)==144 and len(engineering_starts)==2 and charged==146 and failed_captures==1
    out = args.root / 'extra/summary'
    out.mkdir(parents=True, exist_ok=True)
    effects, means, fast_contrast, clock_audit, cohorts, control_readout = {}, {}, {}, {}, {}, {}
    tail_vs_mixtures={}
    tail_table=[r'\begin{center}\small',r'\begin{tabular}{lrr}\toprule',
        r'Condition & Journey P95 vs static (s) & Journey P95 vs cap (s) \\ \midrule']
    table = [r'\begin{center}\small', r'\begin{tabular}{lrrr}\toprule',
        r'Condition & Journey P95 difference (s) & Early trips/day & Unserved/day \\ \midrule']
    for regime, arms in data['details'].items():
        days = plan['days'][regime]
        assert all(len({arms[a][str(d)]['passengers']['generated'] for a in plan['arms']})==1
                   for d in days), 'Generated populations differ within a paired day.'
        clock_audit[regime]={}
        cohorts[regime]={}
        control_readout[regime]={}
        cohort_rows={}
        for d in days:
            offsets={}
            control_readout[regime][str(d)]={}
            for a in plan['arms']:
                path=read(args.root/'cases'/regime/str(d)/a/'summary/results.json')
                cohort_rows[d,a]=path['passenger_detail']['station_hour']
                n=path['passenger_detail']['generated']
                offsets[a]=sum(v['timestamp_minus_interval'] for v in path['clock_accounting'].values())/n
                control_readout[regime][str(d)][a]=dict(
                    executed_holding_s=path['execution']['executed_s'],
                    own_state_request_H_delta_s=path['graph']['signed_same_state_h_delta_s'],
                    supported_fraction=path['graph']['supported_fraction'],statuses=path['graph']['statuses'],
                    action_counts=path['results']['action_counts'],plan_changes=path['slow']['planned_changes'],
                    budget_issues=path['slow']['budget_issues'])
            clock_audit[regime][str(d)]=dict(timestamp_minus_integral_s_per_person=offsets,
                max_paired_total_offset_difference_s=max(abs(v-offsets['pressure']) for v in offsets.values()))
        comparators={'bilateral':[('fixed_bilateral',1.)],
            'static':[('static_7p5',1-plan['static_high_probability']),('static_10',plan['static_high_probability'])],
            'cap':[('cap_10',1-plan['cap_high_probability']),('cap_15',plan['cap_high_probability'])]}
        for family,components in comparators.items():
            by_station,by_hour={},{}
            for d in days:
                current=cohort_rows[d,'pressure']
                for key,(n,w,o) in current.items():
                    station,hour=map(int,key.split(':'))
                    baseline=sum(weight*sum(cohort_rows[d,arm][key][1:]) for arm,weight in components)
                    for collection,group in ((by_station,station),(by_hour,hour)):
                        row=collection.setdefault(group,[0,0.])
                        row[0]+=n;row[1]+=w+o-baseline
            cohorts[regime][family]={axis:{str(k):dict(passengers=v[0],
                timestamp_journey_difference_s_per_person=v[1]/v[0]) for k,v in values.items()}
                for axis,values in (('origin_station',by_station),('generation_hour',by_hour))}
        rng = np.random.default_rng(plan['bootstrap_seed'])
        draws = rng.integers(0, len(days), (plan['bootstrap_draws'], len(days)))
        def get(arm, day, metric):
            r = arms[arm][str(day)]
            if metric in ('early_departures', 'early_total_s', 'early_max_s'):
                return r['operating'][metric]
            if metric == 'journey_p95_s':
                return r['passengers']['journey_p50_p90_p95_s'][2]
            if metric == 'wait_p95_s':
                return r['passengers']['wait_p50_p90_p95_s'][2]
            return r['passengers'][metric]
        metrics = ('journey_p95_s','wait_p95_s','early_departures','early_total_s',
                   'early_max_s','generated','unserved','incomplete')
        means[regime] = {a: {m: float(np.mean([get(a,d,m) for d in days])) for m in metrics}
                         for a in plan['arms']}
        effects[regime] = {}
        fast_contrast[regime] = {}
        for metric, joint_native in data['effects'][regime]['joint_vs_bilateral'].items():
            joint_model = data['effects'][regime]['joint_vs_fixed_model'][metric]
            v = np.array([joint_native['by_day'][str(d)]-joint_model['by_day'][str(d)] for d in days])
            fast_contrast[regime][metric] = dict(mean=float(v.mean()),
                descriptive_ci95=np.quantile(v[draws].mean(1),[.025,.975]).tolist(),
                by_day=dict(zip(map(str,days),v.tolist())))
        for m in metrics:
            v = np.array([get('pressure',d,m)-get('fixed_bilateral',d,m) for d in days])
            effects[regime][m] = dict(mean=float(v.mean()),
                descriptive_ci95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
        tail_vs_mixtures[regime]={}
        for family in ('static','cap'):
            tail_vs_mixtures[regime][family]={}
            for m in metrics:
                v=np.array([get('pressure',d,m)-sum(w*get(a,d,m) for a,w in comparators[family]) for d in days])
                tail_vs_mixtures[regime][family][m]=dict(mean=float(v.mean()),
                    descriptive_ci95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
        cells=[]
        for family in ('static','cap'):
            row=tail_vs_mixtures[regime][family]['journey_p95_s'];lo,hi=row['descriptive_ci95']
            cells.append(f"${row['mean']:+.2f}\;[{lo:+.2f},{hi:+.2f}]$")
        tail_table.append(regime+' & '+' & '.join(cells)+r' \\')
        e=effects[regime]['journey_p95_s'];lo,hi=e['descriptive_ci95']
        table.append(regime + ' & ' + f"${e['mean']:+.2f}\;[{lo:+.2f},{hi:+.2f}]$" +
            ' & ' + f"{means[regime]['pressure']['early_departures']:.2f}" +
            ' & ' + f"{means[regime]['pressure']['unserved']:.2f}" + r' \\')
    table += [r'\bottomrule\end{tabular}\end{center}',
        'P95 is calculated within each complete demand day before taking paired day differences. '
        'It is not a percentile pooled over policies or days. Intervals are descriptive paired '
        'day-bootstrap intervals. Early departures are counted against immutable public releases; '
        'they are not offset against positive lateness. Unserved passengers retain fixed-horizon '
        'waiting exposure. The full source JSON includes the native baseline and all model arms.']
    tail_table += [r'\bottomrule\end{tabular}\end{center}',
        'Simple-control P95 values are expectations of each component policy\'s daily percentile '
        'under the original whole-day randomization, not percentiles of pooled passengers.']
    (out/'service_readout.tex').write_text('\n'.join(table+tail_table)+'\n')
    day=plan['days']['nominal'][0]
    gate=helper(args.assets,'gate_default')
    circulation=circulation_readout(args,plan,gate)
    def trace(arm):
        return gate.raw(args.root/'cases/nominal'/str(day)/arm/'raw/path.pkl.gz')
    pressure=trace('pressure')
    publication=next(event for event in pressure['slow_events'] if event['kind']=='publish' and event['budgets'])
    first=publication['budgets'][0]
    target=first['trip'];chain={}
    for arm in ('fixed_model','plan','budget','pressure'):
        t=pressure if arm=='pressure' else trace(arm)
        launch=next(d for d in t['dispatches'] if d['trip']==target)
        bus=launch['bus']
        returned=next(r for r in t['returns'] if int(r['bus_id'])==bus and int(r['trip_id'])==target)
        later=[d for d in t['dispatches'] if d['bus']==bus and d['actual_s']>launch['actual_s']]
        following=min(later,key=lambda d:d['actual_s']) if later else None
        commands=[[q['time_s'],q['station'],q['executed_request_s']] for q in t['queries']
                  if q['bus']==bus and q['trip']==target]
        chain[arm]=dict(bus=bus,trip=target,launch=launch,return_s=returned['physical_time_s'],
                        following=following,commands=commands)
    style()
    fig,axes=plt.subplots(2,1,figsize=(7.15,3.7),gridspec_kw={'height_ratios':[1.1,1]},sharex=True)
    colors=('#777777','#B77942','#668A73','#287C8E')
    origin=min(c['launch']['actual_s'] for c in chain.values())
    source={}
    for i,((arm,c),color) in enumerate(zip(chain.items(),colors)):
        start=(c['launch']['actual_s']-origin)/60;end=(c['return_s']-origin)/60
        axes[0].plot([start,end],[i,i],color=color,lw=2)
        axes[0].scatter([start,end],[i,i],color=color,marker='|',s=80)
        if c['following']:
            depart=(c['following']['actual_s']-origin)/60
            axes[0].plot([end,depart],[i,i],color=color,lw=1,ls=':')
            axes[0].scatter([depart],[i],color=color,marker='>',s=25)
        q=np.array(c['commands'])
        axes[0].scatter((q[:,0]-origin)/60,np.full(len(q),i),s=4+q[:,2],
                        facecolors='none',edgecolors=color,lw=.6)
        c['plot']=dict(query_minutes=((q[:,0]-origin)/60).tolist(),
            cumulative_commands_s=np.cumsum(q[:,2]).tolist(),start_min=start,return_min=end,
            next_departure_min=None if c['following'] is None else (c['following']['actual_s']-origin)/60)
        axes[1].step(c['plot']['query_minutes'],c['plot']['cumulative_commands_s'],where='post',
            label=arm.replace('_model',''),color=color,ls=('-','--',':','-.')[i])
    for ax in axes:
        ax.axvline((publication['time_s']-origin)/60,color='.5',ls=':',lw=.7)
    axes[0].set(yticks=range(4),yticklabels=[f"{a.replace('_model','')} (bus {c['bus']})" for a,c in chain.items()])
    axes[0].set_title(f'First nominal day; first issued-credit trip {target}',fontsize=8)
    axes[1].set(xlabel='Minutes from the earliest matched trip departure',ylabel='Cumulative command H (s)')
    axes[0].text(-.12,1.03,'a',transform=axes[0].transAxes,weight='bold')
    axes[1].text(-.12,1.03,'b',transform=axes[1].transAxes,weight='bold')
    axes[1].legend(ncol=4,fontsize=6,loc='upper left')
    fig.tight_layout()
    save(fig,out,'vehicle_time_chain')
    write(out/'source_data.json',dict(day=day,first_issue=first,chain=chain,
        issue_minutes=(publication['time_s']-origin)/60,
        note='Circles mark submission times and size commanded H, not assumed physical dwell intervals; all commands executed.'))
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.6))
    for ax,axis,label in zip(axes,('generation_hour','origin_station'),
                            ('Generation hour since service start','Recorded origin station ID')):
        for family,color,marker in (('bilateral','#287C8E','o'),('static','#59616A','s'),('cap','#B77942','^')):
            entries=cohorts['nominal'][family][axis];keys=sorted(map(int,entries))
            ax.plot(keys,[entries[str(k)]['timestamp_journey_difference_s_per_person'] for k in keys],
                    color=color,marker=marker,ms=2.5,lw=.8,label='vs '+family)
        ax.axhline(0,color='.7',lw=.7);ax.set_xlabel(label)
        ax.set_ylabel('Timestamp journey difference (s/person)')
    axes[0].legend(fontsize=6)
    fig.tight_layout();save(fig,out,'cohort_exposure')
    write(out/'cohort_source_data.json',dict(cohorts=cohorts,
        estimand='Passenger-weighted mean within each generation cohort across four nominal days; not the primary equal-day mean.'))
    rows=[v for arms in data['details'].values() for days in arms.values() for v in days.values()]
    walls=[v['wall_s'] for v in rows]
    callback_p95=[v['operating']['callback_p50_p95_max_s'][1] for v in rows]
    computation=dict(formal_paths=len(rows),rollout_wall_s_min_median_p95_max=np.quantile(walls,[0,.5,.95,1]).tolist(),
        path_specific_callback_p95_range_s=[min(callback_p95),max(callback_p95)],
        per_arm_wall_s_median_max={arm:np.quantile([v['wall_s'] for r in data['details'].values()
            for v in r[arm].values()],[.5,1]).tolist() for arm in plan['arms']},
        native_baseline_includes_passive_graph=True,queue_and_result_sync_excluded=True,
        realtime_guarantee=False)
    support={r:dict(mean_supported_fraction=float(np.mean([v['pressure']['supported_fraction'] for v in days.values()])),
        total_overrides=sum(data['details'][r]['pressure'][d]['overrides'] for d in days),
        mean_plan_changes=float(np.mean([v['pressure']['plan_changes'] for v in days.values()])),
        mean_budget_issues=float(np.mean([v['pressure']['budget_issues'] for v in days.values()])))
        for r,days in control_readout.items()}
    write(out/'results.json',dict(completed=True,means=means,effects=effects,
        fixed_model_minus_native_bilateral=fast_contrast,
        tail_vs_mixtures=tail_vs_mixtures,
        clock_audit=clock_audit,
        control_readout=control_readout,
        circulation=circulation,
        computation=computation,pressure_control_support=support,
        charged_start_account=dict(formal=len(formal_starts),engineering=len(engineering_starts),
            total=charged,failed_observation_capture_retained=failed_captures==1),
        physical_starts=0,training_updates=0,mechanism_day=day,mechanism_trip=target,
        all_incomplete_zero=all(v['incomplete']==0 for r in means.values() for v in r.values()),
        primary_gates_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);main(p.parse_args())
