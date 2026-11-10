"""Read V401 completed JSON receipts; no trajectories, checkpoints or simulation."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main(args):
    modes={};batches={};rows=[];durations={};data={}
    for arm in ('explicit','implicit'):
        modes[arm]=Counter();batches[arm]=[];durations['train_'+arm]=[]
        for batch in range(8):
            counts=Counter()
            for day in range(1780001+4*batch,1780005+4*batch):
                row=read(args.root/'train'/arm/f'b{batch}'/f'd{day}'/'summary/results.json')
                assert row['completed'] and row['physical_account']['passed'] and row['quota_account']['passed']
                counts.update(row['upper']['mode_counts'])
                durations['train_'+arm].append(row['wall_s'])
            modes[arm].update(counts)
            batches[arm].append(dict(batch=batch,mode_counts=dict(counts)))
    for day,rho,arm in itertools.product(range(2410001,2410005),(.25,.5),('manual_joint','implicit','explicit')):
        row=read(args.root/'development'/arm/f'r{round(rho*100):03d}'/f'd{day}'/'summary/results.json')
        result=row['results'];end=row['endpoints'];n=row['contract']['passengers_generated']
        assert row['completed'] and row['physical_account']['passed'] and row['quota_account']['passed']
        assert end['journey_s_per_passenger']==result['passenger_person_s']/n>0
        assert end['cv_squared']==result['cv_squared']>0
        assert end['public_delay_s_per_day']==result['public_launch_delay_total_s']
        upper=row.get('upper')
        if upper:assert row['action_semantics']['passed'] and upper['actions']==upper['transitions']==47
        data[day,rho,arm]=row
        rows.append(dict(day=day,rho=rho,arm=arm,endpoints=end,
            mode_counts=upper['mode_counts'] if upper else dict(joint=row['slow']['publications']),
            publications=row['slow']['publications'],planned_changes=row['slow']['planned_changes'],
            supported_queries=row['graph']['supported_queries'],actual_overrides=row['graph']['actual_overrides'],
            executed_holding_s=row['execution']['executed_s'],quota_passed=row['quota_account']['passed'],wall_s=row['wall_s']))
        durations.setdefault('development_'+arm,[]).append(row['wall_s'])
    equality=[];budget_effect=[]
    for day,rho in itertools.product(range(2410001,2410005),(.25,.5)):
        reference=data[day,rho,'manual_joint']
        for arm in ('explicit','implicit'):
            row=data[day,rho,arm]
            equality.append(dict(day=day,rho=rho,arm=arm,
                endpoints_equal=row['endpoints']==reference['endpoints'],
                action_histogram_equal=row['results']['action_counts']==reference['results']['action_counts'],
                slow_summary_equal=row['slow']==reference['slow'],
                learned_mode_counts=row['upper']['mode_counts']))
    for arm in ('manual_joint','implicit','explicit'):
        budget_effect.append(dict(arm=arm,lower_minus_higher_budget={metric:
            statistics.mean(data[day,.25,arm]['endpoints'][metric]-data[day,.5,arm]['endpoints'][metric]
                            for day in range(2410001,2410005))
            for metric in ('journey_s_per_passenger','cv_squared','readiness_s_per_trip','public_delay_s_per_day')}))
    result=dict(completed=True,training_mode_counts={k:dict(v) for k,v in modes.items()},
        training_by_batch=batches,development=rows,comparisons_with_manual=equality,
        budget_effect=budget_effect,duration_s={k:dict(min=min(v),median=statistics.median(v),max=max(v))
                                               for k,v in durations.items()},
        all_development_learned_actions_joint=all(c['learned_mode_counts'].get('joint')==47 for c in equality),
        all_development_endpoints_equal_manual=all(c['endpoints_equal'] for c in equality),
        physical_starts=0,optimizer_updates=0,main_policy_changed=False)
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'results.json').open('x') as stream:
        json.dump(result,stream,separators=(',',':'),allow_nan=False);stream.write('\n')
    print('V401_READOUT_COMPLETE',json.dumps({k:result[k] for k in
        ('all_development_learned_actions_joint','all_development_endpoints_equal_manual','training_mode_counts')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
