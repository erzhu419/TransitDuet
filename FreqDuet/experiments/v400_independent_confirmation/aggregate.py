"""Read all fixed confirmation cells on the server; report without selection."""

import argparse
from pathlib import Path
import traceback

import numpy as np

from evaluation import read, write, helper, HERE

CONTRASTS = {
    'K4_minus_manual_r050':{'K4_r050':1,'manual_r050':-1},
    'K4_minus_manual_r025':{'K4_r025':1,'manual_r025':-1},
    'K4_tight_minus_default':{'K4_r025':1,'K4_r050':-1},
    'manual_tight_minus_default':{'manual_r025':1,'manual_r050':-1},
    'budget_interaction':{'K4_r025':1,'manual_r025':-1,'K4_r050':-1,'manual_r050':1},
    'K4_minus_K0_r050':{'K4_r050':1,'K0_r050':-1}}
METRICS=('waiting_s_per_passenger','onboard_s_per_passenger','journey_s_per_passenger',
         'readiness_s_per_trip','public_delay_s_per_day','cv_squared','holding_s_per_trip','J0')


def main(args):
    plan=read(HERE/'manifest.json')
    days=plan['days']
    missing=[f'{a["id"]}_d{d}' for d in days for a in plan['arms']
             if not (args.root/'cases'/f'{a["id"]}_d{d}'/'summary/results.json').is_file()]
    if missing:
        write(args.root/'aggregate/summary/results.json',dict(completed=False,
            status='confirmation_incomplete',missing=missing,training_updates=0,main_policy_changed=False))
        return
    physical=helper(args.assets,'physical')
    gate=helper(args.assets,'gate_default')
    by_day,validation={},{}
    for day in days:
        by_day[str(day)]={}
        base=args.root/'cases'/f'K4_r050_d{day}'
        reference=read(base/'summary/results.json')
        trace=gate.raw(base/'raw/path.pkl.gz')
        for arm in plan['arms']:
            folder=args.root/'cases'/f'{arm["id"]}_d{day}'
            row=read(folder/'summary/results.json')
            assert row['day']==day and row['arm']==arm['id'] and row['physical_account']['passed']
            assert row['quota_account']['passed'] and row['budget_retain_fraction']==arm['rho']
            other=gate.raw(folder/'raw/path.pkl.gz')
            pair=physical.crn_pair(trace['uniforms'],other['uniforms'])
            assert pair['exact'] and pair['left_only']==pair['right_only']==0 and pair['common_events']==5200
            assert row['contract']==reference['contract'] and other['public_clocks']==trace['public_clocks']
            by_day[str(day)][arm['id']]={m:row['endpoints'][m] for m in METRICS}
            validation[f'{arm["id"]}_d{day}']={'physical_and_quota':True,'common_events':5200}
        print('V400_PAIRED_DAY',day,flush=True)
    rng=np.random.default_rng(plan['bootstrap']['seed'])
    draws=rng.integers(0,len(days),size=(plan['bootstrap']['draws'],len(days)))
    effects={}
    for name,weights in CONTRASTS.items():
        effects[name]={}
        for metric in METRICS:
            values=np.array([sum(w*by_day[str(d)][a][metric] for a,w in weights.items()) for d in days])
            assert np.isfinite(values).all()
            effects[name][metric]={'mean':float(values.mean()),
                'ci95':np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                'by_day':dict(zip(map(str,days),values.tolist()))}
    budget=effects['K4_tight_minus_default']
    harmonic=effects['K4_minus_K0_r050']
    decisions={'budget_service_improvement':budget['journey_s_per_passenger']['ci95'][1]<0
                   and budget['readiness_s_per_trip']['ci95'][1]<=0,
               'harmonic_journey_improvement':harmonic['journey_s_per_passenger']['ci95'][1]<0}
    starts=sum((args.root/'cases'/f'{a["id"]}_d{d}'/'started.json').is_file()
               for d in days for a in plan['arms'])
    starts+=sum((args.root/'engineering'/f'{a}_d2077002'/'started.json').is_file()
                for a in ('K4_r050','manual_r050'))
    assert starts==82
    write(args.root/'aggregate/summary/results.json',dict(protocol_id=plan['protocol_id'],
        completed=True,status='frozen_independent_day_confirmation_complete',days=days,
        by_day=by_day,effects=effects,validation=validation,primary_decisions=decisions,
        statistical_unit='independent simulation demand day',independent_demand_days=16,
        learner_seeds=[314011],learner_seed_count=1,independent_futures=0,physical_starts_charged=starts,
        training_updates=0,main_policy_changed=False,automatic_retry=False,
        interpretation='Report both claims and all manual/regularity tradeoffs; no automatic promotion or retuning.'))
    print('V400_CONFIRMATION_COMPLETE',decisions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write(args.root/'aggregate/summary/failure.json',dict(completed=False,
            error=traceback.format_exc()[-5000:],training_updates=0,main_policy_changed=False))
        raise
