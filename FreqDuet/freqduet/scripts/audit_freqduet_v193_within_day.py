#!/usr/bin/env python3
"""Read common-key calibration moments: within-day variance, event mass and signs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v192_state_day as parent


prior,common = parent.prior,parent.parent
read,write,require,close = parent.read,parent.write,parent.require,parent.close
N,SIZE,DAYS,TARGET = parent.N,parent.SIZE,parent.DAYS,parent.TARGET
FIELDS = ('mass','upper_mass','mixed_mass','mixed_upper_mass','innovation_variance_mixed',
    'innovation_variance_no_upper','innovation_variance_all_upper')


def registration():
    return dict(stage='v193_within_day_variance_event_concentration',source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS),moment_source='a7d126966a',parent_analysis='9a11be5ab6',
        frozen=dict(v151=True,v146=True,v172=True,main=True,window_s=3600.,tails=[.1,.1]),
        selection='exact V191/V192 common keys with >=19 complete-input days in EACH half; fixed support-mass balancing',
        inputs='V190 moments only; no raw queries/residuals/records,confirmations or new rollouts',
        signal='innovation_rms_s',target='frozen V172 upper event',
        reference='V192 average-half within-key covariance residualizer; same reference in both halves; recompute per draw',
        variance='raw and reference-residual within actor/road/entry-hour/day variance; divide by original balanced analysis mass',
        concentration='upper-event mass across32 days versus exposure mass; effective days=(sum mass)^2/sum(mass^2); '
            'weight concentration,not independent observations or statistical effective sample size',
        mixed_cells='0<weighted upper mass<cell mass per actor/key/day; outcome-defined description only,not a deployable condition',
        directions='daily within-key covariance pooled across actors,raw and reference-residual correlations; '
            'zero variance/missing exposure gives unknown,not zero correlation; no per-day significance tests or date selection',
        inference=dict(draws=prior.DRAWS,seed=prior.SEED,actor='shared actors',day='independent32-day draws in each half',
            intervals='diagnostic95%,direct half differences; V190 gates unchanged',
            concentration_bootstrap='each sampled day occurrence is one position; multiplicity enters linearly,not squared',
            fixed='key masks and support balancing; refit only diagnostic covariance projections each draw'),
        limitations='Weighted binary moments do not supply independent event/episode counts. '
            'This is linear calibration diagnosis,not causal drift,held-future skill or a policy result.',
        interval_fit=False,predictor_fit=False,confirmation_started=False,new_replays=0,new_training=0)


def ratio(a,b):
    a,b = np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>1e-12)


def cell_masses(m):
    mass,upper = m[:,:,0],m[:,:,1+TARGET]
    require(np.all(upper>=0.) and np.all(upper<=mass+1e-10),'invalid binary event mass')
    second = prior.second(m)
    require(np.allclose(second[:,:,TARGET,TARGET],upper,atol=1e-10),'upper event is not binary')
    variance = second[:,:,1,1]-np.divide(m[:,:,2]**2,mass,out=np.zeros_like(mass),where=mass>0.)
    require(np.all(variance>=-1e-7),'negative innovation variance')
    variance = np.maximum(variance,0.)
    mixed,no_upper,all_upper = (upper>0.)&(upper<mass),upper==0.,(upper==mass)&(mass>0.)
    return np.stack((mass.sum(axis=1),upper.sum(axis=1),(mass*mixed).sum(axis=1),
        (upper*mixed).sum(axis=1),(variance*mixed).sum(axis=1),(variance*no_upper).sum(axis=1),
        (variance*all_upper).sum(axis=1)),axis=-1)


def concentration(mass,weights):
    mean = (mass*weights).sum(axis=-1)
    square = (mass*mass*weights).sum(axis=-1)
    effective = 32.*ratio(mean*mean,square)
    maximum = np.max(np.where(weights>0.,mass,0.),axis=-1)
    return effective,ratio(maximum,32.*mean)


def correlations(c,x,y):
    vx,vy = parent.bilinear(x,c,x),parent.bilinear(y,c,y)
    return ratio(parent.bilinear(x,c,y),np.sqrt(np.maximum(vx,0.)*np.maximum(vy,0.)))


def metrics(cov,daily,masses,weights,measurement):
    x,y = measurement[3:]
    within = np.einsum('bd,bdij->bij',weights,daily)
    totals = np.einsum('bd,bdf->bf',weights,masses)
    mass,upper = totals[:,0],totals[:,1]
    raw_x,raw_y = np.zeros(N),np.zeros(N)
    raw_x[1],raw_y[TARGET] = 1.,1.
    residual_x_var = parent.bilinear(x,within,x)
    total_x_var = parent.bilinear(x,cov,x)
    raw_daily = correlations(daily,raw_x,raw_y)
    adjusted_daily = correlations(daily,x[...,None,:],y[...,None,:])
    event_effective,event_max = concentration(masses[:,:,1],weights)
    exposure_effective,_ = concentration(masses[:,:,0],weights)
    result = dict(raw_within_innovation_variance_s2=ratio(within[:,1,1],mass),
        adjusted_within_innovation_variance_s2=ratio(residual_x_var,mass),
        raw_within_innovation_variance_fraction=ratio(within[:,1,1],cov[:,1,1]),
        adjusted_within_innovation_variance_fraction=ratio(residual_x_var,total_x_var),
        raw_within_upper_variance=ratio(within[:,TARGET,TARGET],mass),
        adjusted_within_upper_variance=ratio(parent.bilinear(y,within,y),mass),
        raw_within_correlation=correlations(within,raw_x,raw_y),
        adjusted_within_correlation=correlations(within,x,y),
        v192_within_contribution=common.contribution(within,measurement),
        upper_rate=ratio(upper,mass),effective_upper_mass_days=event_effective,
        effective_exposure_mass_days=exposure_effective,largest_upper_day_share=event_max,
        mixed_cell_mass_fraction=ratio(totals[:,2],mass),mixed_cell_upper_mass_fraction=ratio(totals[:,3],upper),
        innovation_variance_in_mixed_cells_fraction=ratio(totals[:,4],totals[:,4:].sum(axis=-1)))
    for name,values in (('raw',raw_daily),('adjusted',adjusted_daily)):
        valid = np.isfinite(values)
        for label,mask in (('positive',valid&(values>0.)),('negative',valid&(values<0.)),
                ('zero',valid&(values==0.)),('unknown',~valid)):
            result[name+'_'+label+'_day_fraction'] = (weights*mask).sum(axis=-1)
    return result,dict(raw=raw_daily,adjusted=adjusted_daily)


def run_case(args,actors,mode,scope):
    aw,dw = common.weights(len(actors))
    pc,bc = np.zeros((2,N,N)),np.zeros((2,prior.DRAWS,N,N))
    days = np.zeros((2,len(actors),32,N,N))
    masses = np.zeros((2,len(actors),32,len(FIELDS)))
    keys_out = []
    for ai,actor in enumerate(actors):
        cell = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
        require(cell['source']=='a7d126966a' and all((r['actor'],r['mode'])==(actor,mode) for r in cell['rows']),
            'moment actor/mode/source differs')
        m,keys = common.actor_matrix(cell['rows'],scope)
        keep = common.masks(m)[2]
        keys_out.extend(dict(actor=actor,key=list(k),first_days=int(np.count_nonzero(m[:32,i,0])),
            last_days=int(np.count_nonzero(m[32:,i,0]))) for i,k in enumerate(keys) if keep[i])
        for h,v in enumerate(common.balance(m[:32,keep],m[32:,keep])):
            first,seconds = v[:,:,:1+N],prior.second(v.sum(axis=1))
            within,between = parent.day_matrices(v)
            days[h,ai],masses[h,ai] = within,cell_masses(v)
            c,_ = prior.actor_covariance(first,seconds,np.full((1,32),1./32.))
            require(np.allclose((within+between).mean(axis=0),c[0],atol=1e-7),'within covariance partition differs')
            pc[h] += c[0]/len(actors)
            for start in range(0,prior.DRAWS,64):
                stop = min(start+64,prior.DRAWS)
                c,_ = prior.actor_covariance(first,seconds,dw[h,start:stop])
                bc[h,start:stop] += aw[start:stop,ai,None,None]*c
        print('V193_COMMON_ACTOR',mode,scope,actor,int(keep.sum()),flush=True)
    saved = read(args.v192_root/'details'/f'{mode}_{scope}.json')
    require(keys_out==saved['common_key_support'],'V192 common keys/support differ')
    point_ref,boot_ref = common.measure(pc.mean(axis=0),1,True),common.measure(bc.mean(axis=0),1,True)
    close(float(point_ref[0]),saved['result']['reference_r'])
    for name,actual in (('x',point_ref[3]),('y',point_ref[4])):
        require(np.allclose(actual,saved['reference_vectors'][name],rtol=1e-8,atol=1e-8),'V192 residualizer differs')
    point,draws,detail_days = [],[],[]
    for h in range(2):
        wd,mm = days[h].mean(axis=0)[None],masses[h].mean(axis=0)[None]
        p,dd = metrics(pc[h,None],wd,mm,np.full((1,32),1./32.),point_ref)
        close(float(mm[:,:,0].mean()),saved['result']['balanced_mass'][h])
        close(float(p['v192_within_contribution'][0]),saved['result']['partitions']['within_day']['adjusted'][('first','last')[h]]['mean'])
        point.append({k:float(v[0]) for k,v in p.items()})
        details = []
        for i,day in enumerate(DAYS[:32] if h==0 else DAYS[32:]):
            details.append(dict(day=day,exposure_mass=float(mm[0,i,0]),upper_mass=float(mm[0,i,1]),
                upper_rate=common.number(ratio(mm[0,i,1],mm[0,i,0])),
                raw_within_r=common.number(dd['raw'][0,i]),adjusted_within_r=common.number(dd['adjusted'][0,i]),
                raw_innovation_variance_s2=common.number(ratio(wd[0,i,1,1],mm[0,i,0]))))
        detail_days.append(details)
        bvalues = {k:np.empty(prior.DRAWS) for k in p}
        for start in range(0,prior.DRAWS,64):
            stop = min(start+64,prior.DRAWS)
            bd = (aw[start:stop]@days[h].reshape(len(actors),-1)).reshape(-1,32,N,N)
            bm = (aw[start:stop]@masses[h].reshape(len(actors),-1)).reshape(-1,32,len(FIELDS))
            ref = tuple(v[start:stop] for v in boot_ref)
            values,_ = metrics(bc[h,start:stop],bd,bm,dw[h,start:stop],ref)
            for k,v in values.items():
                bvalues[k][start:stop] = v
        draws.append(bvalues)
        print('V193_WITHIN_DAY_HALF_COMPLETE',mode,scope,h,flush=True)
    estimates = {k:parent.pair_summary([p[k] for p in point],[d[k] for d in draws]) for k in point[0]}
    result = dict(mode=mode,scope=scope,common_keys=len(keys_out),v192_points_reproduced=True,estimates=estimates)
    write(args.out_root/'details'/f'{mode}_{scope}.json',dict(result=result,common_key_support=keys_out,days=detail_days))
    return result


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v193_*.py','-v'],check=True)
    original = read(args.v192_root/'summary/status.json')
    require(original['source']=='9a11be5ab6' and original['v191_points_reproduced'] and
        original['protocol']['calibration_days']==list(DAYS) and not original['map_updates'] and
        not original['interval_fit'] and not original['main_policy_changed'],'requires frozen completed V192')
    detail = read(args.v192_root/'details/old_clearance.json')
    actors = sorted({k['actor'] for k in detail['common_key_support']})
    require(len(actors)==12,'actor grid differs')
    require(not (args.out_root/'protocol.json').exists(),'do not overwrite diagnostic registration')
    plan = registration()
    write(args.out_root/'protocol.json',plan)
    print('V193_READONLY_PLAN_SEALED',flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(run_case,args,actors,m,s)
                for m,s in product(('old','handoff'),parent.SCOPES)]),1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"readonly_mode_scope"}'%i,flush=True)
    write(args.out_root/'summary/status.json',dict(status='v193_within_day_diagnostic_complete',source=plan['source'],
        protocol=plan,results=sorted(results,key=lambda r:(r['mode'],r['scope'])),source_paths=1536,
        v192_points_reproduced=True,raw_capture_files_opened=0,interval_fit=False,predictor_fit=False,
        confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,
        map_updates=0,main_policy_changed=False))
    print('V193_WITHIN_DAY_COMPLETE_NO_FIT',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v190-root','v192-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
