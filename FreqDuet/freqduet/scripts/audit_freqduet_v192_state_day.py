#!/usr/bin/env python3
"""Common-key state composition and day-level residual covariance; moments only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v191_common_keys as parent


prior = parent.prior
read,write,require,close = parent.read,parent.write,parent.require,parent.close
N,SIZE,DAYS,TARGET = parent.N,parent.SIZE,parent.DAYS,parent.TARGET
SCOPES = ('clearance','hour13','hour13_moving_launch')
GROUPS = dict(clock_geometry=('query_offset_min','remaining_stops','remaining_km','first_fraction'),
    forecast_bounds=('horizon_min','running_min','service_min','lower_radius_min','upper_radius_min'),
    leg_support=('native_leg_mean_min','native_n','native_latest_age_min','matched_n',
        'matched_latest_age_min','matched_mean_age_min','matched_entry_age_min','predicted_entry'),
    completed_service=('service_prefix_under_min','service_prefix_latest_age_min','service_prefix_n'),
    current_stage=('stage_service','stage_command','stage_ready'),other_innovation=('signed_remaining_s',))


def registration():
    return dict(stage='v192_common_key_state_day_decomposition',source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS),moment_source='a7d126966a',parent_analysis='e9efbe9bae',
        frozen=dict(v151=True,v146=True,v172=True,main=True,window_s=3600.,tails=[.1,.1]),
        inputs='V190 moment cells only; original64 days; no query/residual/record archive reads',
        selection='same V191 common actor/mode/scope/road/entry-hour keys with >=19 complete-input days in EACH half',
        weights='V191 fixed support-only per-actor/key two-half mean mass; no outcome-based weights or day selection',
        signal='innovation_rms_s',target='frozen V172 full-ETA upper miss',controls=GROUPS,
        states='all original query controls plus both innovations and upper event; report weighted mean shifts '
            'and shifts divided by average-half within-key SD, not nonlinear distribution matching',
        reference='one residualizer from (C_first+C_last)/2,each C within actor/road/entry-hour; '
            'one common reference residual-SD denominator for both halves; recompute for each bootstrap draw',
        partition='total=within-day+between-day; adjusted covariance=raw-signal*predicted-target '
            '-predicted-signal*raw-target+predicted-signal*predicted-target; '
            'allocate cross-control products symmetrically to fixed control groups',
        inference=dict(draws=prior.DRAWS,seed=prior.SEED,actor='shared actors',
            day='independently sampled days in each half; no arbitrary pairing',
            intervals='diagnostic95%; original V190 replication/family gates unchanged',
            fixed='common masks and support balancing factors; statistical residualizer re-estimated per draw'),
        limitations='Moment-based mean/covariance diagnostics cannot identify nonlinear state composition or causal risk drift. '
            'Control-group shares depend on the fixed coordinate blocks and are not causal contributions. '
            'Residuals here are statistically adjusted innovation/upper-event variables,not physical ETA component errors.',
        interval_fit=False,predictor_fit=False,confirmation_started=False,new_replays=0,new_training=0)


def day_matrices(m):
    sums,mass = m[:,:,1:1+N],m[:,:,:1]
    second = prior.second(m.sum(axis=1))
    mu = np.divide(sums.sum(axis=0),mass.sum(axis=0),out=np.zeros_like(sums[0]),where=mass.sum(axis=0)>0.)
    centered = sums-mass*mu
    between = np.einsum('dki,dkj->dij',centered,np.divide(centered,mass,
        out=np.zeros_like(centered),where=mass>0.))
    within = second-np.einsum('dki,dkj->dij',sums,np.divide(sums,mass,
        out=np.zeros_like(sums),where=mass>0.))
    return within,between


def bilinear(a,c,b):
    return np.einsum('...i,...ij,...j->...',a,c,b)


def decomposition(c,measurement):
    _,_,den,x,y = measurement
    signal,target = np.zeros_like(x),np.zeros_like(y)
    signal[...,1],target[...,TARGET] = 1.,1.
    fitted_x,fitted_y = signal-x,target-y
    def normalized(a,b):
        value = bilinear(a,c,b)
        return np.divide(value,den,out=np.full_like(value,np.nan),where=den>1e-12)
    four = dict(raw=normalized(signal,target),target_adjustment=-normalized(signal,fitted_y),
        signal_adjustment=-normalized(fitted_x,target),joint_adjustment=normalized(fitted_x,fitted_y))
    groups = {}
    for name,fields in GROUPS.items():
        ix = [prior.VARS.index(f) for f in fields]
        a,b = np.zeros_like(x),np.zeros_like(y)
        a[...,ix],b[...,ix] = fitted_x[...,ix],fitted_y[...,ix]
        groups[name] = -normalized(signal,b)-normalized(a,target)+.5*(normalized(a,fitted_y)+normalized(fitted_x,b))
    adjusted = normalized(x,y)
    require(np.allclose(sum(four.values()),adjusted,atol=1e-9,equal_nan=True),'four-term covariance partition differs')
    require(np.allclose(four['raw']+sum(groups.values()),adjusted,atol=1e-9,equal_nan=True),'control-group partition differs')
    return dict(adjusted=adjusted,**four,groups=groups)


def state_shifts(cov,sums):
    mass = sums[...,0]
    mu = np.divide(sums[...,1:],mass[...,None],out=np.full_like(sums[...,1:],np.nan),where=mass[...,None]>0.)
    delta = mu[1]-mu[0]
    scale = np.sqrt(np.maximum(np.diagonal(cov.mean(axis=0),axis1=-2,axis2=-1),0.)/mass.mean(axis=0)[...,None])
    standardized = np.divide(delta,scale,out=np.full_like(delta,np.nan),where=scale>1e-12)
    return mu,delta,standardized


def resample(first,second,within,weights):
    cov,_ = prior.actor_covariance(first,second,weights)
    within = (weights@within.reshape(len(first),-1)).reshape(-1,N,N)
    sums = weights@first.sum(axis=1)
    return cov,within,sums


def reference_correlation(c,measurement):
    x,y = measurement[3:]
    numerator = bilinear(x,c,y)
    den = np.sqrt(np.maximum(bilinear(x,c,x),0.)*np.maximum(bilinear(y,c,y),0.))
    return np.divide(numerator,den,out=np.full_like(den,np.nan),where=den>1e-12)


def block_pairs(c,measurement):
    x,y = measurement[3:]
    fitted_x,fitted_y = -x.copy(),-y.copy()
    fitted_x[1] += 1.
    fitted_y[TARGET] += 1.
    vectors = {}
    for g,fields in GROUPS.items():
        ix = [prior.VARS.index(f) for f in fields]
        a,b = np.zeros(N),np.zeros(N)
        a[ix],b[ix] = fitted_x[ix],fitted_y[ix]
        vectors[g] = a,b
    return {g:{h:parent.number(bilinear(a,c,b)/measurement[2]) for h,(_,b) in vectors.items()}
        for g,(a,_) in vectors.items()}


def pair_summary(points,draws):
    return dict(first=prior.interval(points[0],draws[0]),last=prior.interval(points[1],draws[1]),
        delta=prior.interval(points[1]-points[0],draws[1]-draws[0]))


def reproduction(points,masses,count,expected):
    require(count == expected['common_keys'],'V191 common-key count differs')
    for h,name in enumerate(('first','last')):
        close(masses[h],expected['support']['balanced'][name])
        for si,signal in enumerate(prior.SIGNALS):
            for block,unique in (('baseline',False),('unique',True)):
                value = parent.measure(points[h],si,unique)[0]
                saved = expected['stats'][signal+'|'+block]['arms']['balanced'][name]['mean']
                require(saved is None if not np.isfinite(value) else saved is not None,'V191 missing point differs')
                if saved is not None:
                    require(abs(float(value)-saved)<1e-8,'V191 balanced correlation differs')


def run_case(args,actors,mode,scope):
    aw,dw = parent.weights(len(actors))
    pc,pw,ps = np.zeros((2,N,N)),np.zeros((2,N,N)),np.zeros((2,1+N))
    bc,bw,bs = np.zeros((2,prior.DRAWS,N,N)),np.zeros((2,prior.DRAWS,N,N)),np.zeros((2,prior.DRAWS,1+N))
    dc,dwithin = np.zeros((2,32,N,N)),np.zeros((2,32,N,N))
    key_support,actor_details = [],[]
    for ai,actor in enumerate(actors):
        cell = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
        require(cell['source']=='a7d126966a' and all((r['actor'],r['mode'])==(actor,mode) for r in cell['rows']),
            'original moment source differs')
        m,keys = parent.actor_matrix(cell['rows'],scope)
        keep = parent.masks(m)[2]
        key_support.extend(dict(actor=actor,key=list(k),first_days=int(np.count_nonzero(m[:32,i,0])),
            last_days=int(np.count_nonzero(m[32:,i,0]))) for i,k in enumerate(keys) if keep[i])
        actor_cov,actor_within = [],[]
        for h,values in enumerate(parent.balance(m[:32,keep],m[32:,keep])):
            first,second = values[:,:,:1+N],prior.second(values.sum(axis=1))
            cov,mass = prior.actor_covariance(first,second,np.full((1,32),1./32.))
            within,between = day_matrices(values)
            require(np.allclose((within+between).mean(axis=0),cov[0],rtol=1e-9,atol=1e-7),'day partition differs')
            pc[h] += cov[0]/len(actors)
            pw[h] += within.mean(axis=0)/len(actors)
            ps[h] += first.sum(axis=1).mean(axis=0)/len(actors)
            dc[h] += between/len(actors)
            dwithin[h] += within/len(actors)
            actor_cov.append(cov[0])
            actor_within.append(within.mean(axis=0))
            for start in range(0,prior.DRAWS,64):
                stop = min(start+64,prior.DRAWS)
                cov,wd,sums = resample(first,second,within,dw[h,start:stop])
                weight = aw[start:stop,ai]
                bc[h,start:stop] += weight[:,None,None]*cov
                bw[h,start:stop] += weight[:,None,None]*wd
                bs[h,start:stop] += weight[:,None]*sums
        actor_details.append(dict(actor=actor,covariances=np.array(actor_cov),within=np.array(actor_within)))
        print('V192_COMMON_ACTOR',mode,scope,actor,int(keep.sum()),flush=True)
    old = read(args.v191_root/'details'/f'{mode}_{scope}.json')
    saved_keys = old['common_key_support']
    require(key_support==saved_keys,'V191 key identities/day support differ')
    reproduction(pc,ps[:,0],len(key_support),old['result'])
    print('V192_V191_BALANCED_POINTS_REPRODUCED',mode,scope,flush=True)
    pm,bm = parent.measure(pc.mean(axis=0),1,True),parent.measure(bc.mean(axis=0),1,True)
    native = [parent.measure(pc[h],1,True)[0] for h in range(2)]
    bnative = [parent.measure(bc[h],1,True)[0] for h in range(2)]
    reference = [reference_correlation(pc[h],pm) for h in range(2)]
    breference = [reference_correlation(bc[h],bm) for h in range(2)]
    bridge = dict(v191=pair_summary(native,bnative),shared_coefficients=pair_summary(reference,breference),
        shared_minus_v191=pair_summary([reference[h]-native[h] for h in range(2)],
            [breference[h]-bnative[h] for h in range(2)]))
    part,boot_part = {},{}
    for name,p,b in (('total',pc,bc),('within_day',pw,bw),('between_day',pc-pw,bc-bw)):
        part[name] = [decomposition(p[h],pm) for h in range(2)]
        boot_part[name] = [decomposition(b[h],bm) for h in range(2)]
    require(np.allclose(np.mean([p['adjusted'] for p in part['total']]),pm[0],equal_nan=True),
        'reference adjusted mean differs')
    partitions = {}
    for name in part:
        partitions[name] = {k:pair_summary([p[k] for p in part[name]],[b[k] for b in boot_part[name]])
            for k in ('adjusted','raw','target_adjustment','signal_adjustment','joint_adjustment')}
        partitions[name]['control_groups'] = {g:pair_summary([p['groups'][g] for p in part[name]],
            [b['groups'][g] for b in boot_part[name]]) for g in GROUPS}
    for h in range(2):
        close(part['total'][h]['adjusted'],part['within_day'][h]['adjusted']+part['between_day'][h]['adjusted'])
    means,delta,smd = state_shifts(pc,ps)
    _,bdelta,bsmd = state_shifts(bc,bs)
    fields = (*prior.SIGNALS,'above',*prior.CONTROLS)
    states = {f:dict(first=parent.number(means[0,prior.VARS.index(f)]),last=parent.number(means[1,prior.VARS.index(f)]),
        delta=prior.interval(delta[prior.VARS.index(f)],bdelta[:,prior.VARS.index(f)]),
        within_key_sd_scaled_delta=prior.interval(smd[prior.VARS.index(f)],bsmd[:,prior.VARS.index(f)])) for f in fields}
    days = []
    compact_days = []
    for h in range(2):
        parts = decomposition(dc[h],pm)
        values = parts['adjusted']/32.
        close(float(values.sum()),part['between_day'][h]['adjusted'])
        days.append([dict(day=day,between_contribution=parent.number(values[i]),
            raw_contribution=parent.number(parts['raw'][i]/32.),
            control_groups={g:parent.number(parts['groups'][g][i]/32.) for g in GROUPS},
            within_contribution=parent.number(parent.contribution(dwithin[h,i],pm)/32.))
            for i,day in enumerate(DAYS[:32] if h==0 else DAYS[32:])])
        compact_days.append(dict(positive_days=int(np.count_nonzero(values>0.)),negative_days=int(np.count_nonzero(values<0.)),
            positive_sum=float(values[values>0.].sum()),negative_sum=float(values[values<0.].sum())))
    actors_out = []
    for a in actor_details:
        actors_out.append(dict(actor=a['actor'],total=[parent.number(parent.contribution(c,pm)/len(actors)) for c in a['covariances']],
            between_day=[parent.number(parent.contribution(c,pm)/len(actors)) for c in a['covariances']-a['within']]))
    result = dict(mode=mode,scope=scope,common_keys=len(key_support),balanced_mass=ps[:,0].tolist(),
        v191_points_reproduced=True,reference_r=parent.number(pm[0]),partitions=partitions,
        state_shifts=states,between_day_signs=compact_days,correlation_bridge=bridge)
    write(args.out_root/'details'/f'{mode}_{scope}.json',dict(result=result,common_key_support=key_support,
        day_contributions=days,actor_contributions=actors_out,
        between_day_control_pair_matrix=[block_pairs(pc[h]-pw[h],pm) for h in range(2)],
        reference_vectors=dict(x=pm[3].tolist(),y=pm[4].tolist(),den=float(pm[2]))))
    print('V192_STATE_DAY_CASE_COMPLETE',mode,scope,flush=True)
    return result


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v192_*.py','-v'],check=True)
    original = read(args.v191_root/'summary/status.json')
    require(original['source']=='e9efbe9bae' and original['v190_points_reproduced'] and
        original['protocol']['calibration_days']==list(DAYS) and not original['map_updates'] and
        not original['interval_fit'] and not original['main_policy_changed'],'requires completed frozen V191')
    fields = [f for fs in GROUPS.values() for f in fs]
    require(len(fields)==len(set(fields)) and set(fields)==set(prior.CONTROLS)|{'signed_remaining_s'},'control groups differ')
    first_detail = read(args.v191_root/'details/old_clearance.json')
    actors = sorted({k['actor'] for k in first_detail['common_key_support']})
    require(len(actors)==12,'actor grid differs')
    plan = registration()
    require(not (args.out_root/'protocol.json').exists(),'do not overwrite registered analysis')
    write(args.out_root/'protocol.json',plan)
    print('V192_READONLY_PLAN_SEALED',flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(run_case,args,actors,m,s) for m,s in product(('old','handoff'),SCOPES)]),1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"readonly_mode_scope"}'%i,flush=True)
    write(args.out_root/'summary/status.json',dict(status='v192_common_key_state_day_complete',source=plan['source'],
        protocol=plan,results=sorted(results,key=lambda r:(r['mode'],r['scope'])),source_paths=1536,
        v191_points_reproduced=True,raw_capture_files_opened=0,interval_fit=False,predictor_fit=False,
        confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,
        map_updates=0,main_policy_changed=False))
    print('V192_STATE_DAY_COMPLETE_NO_FIT',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v190-root','v191-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
