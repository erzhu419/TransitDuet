"""Eight sequential development shards, then paired day-level screening."""

import argparse
import itertools
from pathlib import Path
import subprocess
import sys

import numpy as np

from worker import read,write,helper

HERE=Path(__file__).parent
METRICS=('waiting_s_per_passenger','onboard_s_per_passenger','journey_s_per_passenger',
         'readiness_s_per_trip','public_delay_s_per_day','cv_squared','holding_s_per_trip','J0')


def location(root,arm,rho,day):
    return root/'development'/arm/f'r{round(rho*100):03d}'/f'd{day}'


def shard(args,plan):
    cells=list(itertools.product(plan['development_days_proposed'],plan['retentions'],plan['development_arms']))
    completed=[];rc=0
    try:
        for day,rho,arm in cells[args.shard::8]:
            command=[sys.executable,'-u',str(HERE/'worker.py'),'path','--root',str(args.root),
                '--assets',str(args.assets),'--arm',arm,'--split','development','--day',str(day),'--rho',str(rho)]
            if arm!='manual_joint':command+=['--checkpoint',str(args.root/'checkpoints'/arm/'b8')]
            rc=subprocess.run(command).returncode
            if rc:break
            completed.append(dict(day=day,rho=rho,arm=arm))
    except Exception:
        rc=1;raise
    finally:
        write(args.root/f'development_shard_{args.shard}/summary/finished.json',
              dict(completed=rc==0 and len(completed)==3,cases=completed,returncode=rc,automatic_retry=False))
    raise SystemExit(rc)


def aggregate(args,plan):
    gate=helper(args.assets,'gate_default');physical=helper(args.assets,'physical')
    data={};missing=[]
    for day,rho,arm in itertools.product(plan['development_days_proposed'],plan['retentions'],plan['development_arms']):
        p=location(args.root,arm,rho,day)
        if not (p/'summary/results.json').exists():missing.append(str(p));continue
        row=read(p/'summary/results.json')
        assert row['completed'] and row['physical_account']['passed'] and row['quota_account']['passed']
        data[day,rho,arm]=row
    if missing:
        write(args.root/'summary/results.json',dict(completed=False,missing=missing));return
    results={};passed=[]
    days=plan['development_days_proposed']
    draws=np.asarray(list(itertools.product(range(4),repeat=4)))
    for rho in plan['retentions']:
        effects={}
        for day in days:
            traces=[gate.raw(location(args.root,a,rho,day)/'raw/path.pkl.gz') for a in plan['development_arms']]
            for trace in traces[1:]:
                pair=physical.crn_pair(traces[0]['uniforms'],trace['uniforms'])
                assert pair['exact'] and pair['common_events']==5200 and pair['left_only']==pair['right_only']==0
                assert trace['public_clocks']==traces[0]['public_clocks']
        for other in ('manual_joint','implicit'):
            effects[other]={}
            for metric in METRICS:
                delta=np.asarray([data[d,rho,'explicit']['endpoints'][metric]-data[d,rho,other]['endpoints'][metric] for d in days])
                effects[other][metric]=dict(mean=float(delta.mean()),
                    descriptive_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                    by_day=dict(zip(map(str,days),delta.tolist())))
            screen=effects[other]
            passed.append(screen['journey_s_per_passenger']['mean']<0 and all(screen[m]['mean']<=0 for m in
                ('readiness_s_per_trip','public_delay_s_per_day','cv_squared')))
        results[str(rho)]=effects
    counts={}
    for arm in plan['arms']:
        checkpoint=read(args.root/'checkpoints'/arm/'b8/summary/results.json')
        assert checkpoint['completed_batches']==8 and checkpoint['successful_updates']==1536
        counts[arm]=checkpoint['successful_updates']
    starts=sum(1 for _ in args.root.glob('engineering/*/started.json'))
    starts+=sum(1 for _ in args.root.glob('engineering_retry01/*/started.json'))
    starts+=sum(1 for _ in args.root.glob('train/*/*/*/started.json'))
    starts+=sum(1 for _ in args.root.glob('development/*/*/*/started.json'))
    amendment=read(HERE/'engineering_restart_authorization.json')
    assert starts==amendment['effective_physical_start_cap']==92
    write(args.root/'summary/results.json',dict(completed=True,status='development_screen_complete',
        effects=results,screen_passed=all(passed),candidate_training_updates=counts,
        physical_starts_charged=starts,original_failed_engineering_starts=2,
        statistical_unit='four development demand days',
        independent_confirmation=False,main_policy_changed=False,automatic_retry=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('shard','aggregate'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=range(8))
    args=parser.parse_args();plan=read(HERE/'protocol.json')
    (shard if args.command=='shard' else aggregate)(args,plan)
