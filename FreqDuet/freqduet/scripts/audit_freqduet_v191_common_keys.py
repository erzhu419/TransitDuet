#!/usr/bin/env python3
"""Read V190 moments only: common19-day keys, composition and day-risk changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v190_leg_risk as prior


read,write,require,close = prior.read,prior.write,prior.require,prior.close
DAYS,N,SIZE = prior.DAYS,prior.N,prior.SIZE
TARGET = prior.VARS.index('above')
ARMS = ('legacy','common','balanced')
BLOCKS = ('baseline','unique')


def registration():
    return dict(stage='v191_common_supported_keys',source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS),capture_source='20d914af03',moment_source='a7d126966a',
        frozen=dict(v151=True,v146=True,v172=True,main=True,window_s=3600.,tails=[.1,.1]),
        scope='V190 moments only; no original query/residual/record archive reads',
        common='same actor/mode/scope/direction/road/entry-hour key has >=19 complete-input days in EACH half',
        arms=dict(legacy='original V190 split-specific >=19-day masks, for reproduction only',
            common='intersection of the two masks, original episode/day mass unchanged',
            balanced='same intersection; per-actor/key reference mass=(first mass+last mass)/2; '
                'rescale each half by reference/original key mass, using support mass only, not outcomes'),
        decomposition='delta_legacy=(delta_legacy-delta_common)+(delta_common-delta_balanced)+delta_balanced; '
            'fixed sequential standardization, not unique causal shares',
        day_partition='within-key covariance=mean within-day covariance+between-day mean covariance; '
            'apply each half full-covariance residualizer to BOTH components and use the full residual denominator',
        normalization='delta correlation splits exactly into covariance and inverse-SD contributions by symmetric product identity',
        inference=dict(draws=prior.DRAWS,seed=prior.SEED,actor='same actor draws for both halves/arms/modes',
            day='independent32-day multinomial draws in first/last halves; no arbitrary day-i/day-(i+32) pairing',
            support='original common masks and support-only balancing factors fixed before resampling',
            intervals='diagnostic95%; directly test half differences, never infer difference from differing significance'),
        focus=['clearance','hour13','hour13_moving_launch'],diagnostic='hour2',
        controls=list(prior.CONTROLS),signals=list(prior.SIGNALS),target='frozen V172 full-ETA upper miss',
        invalid='missing support/variance is unknown, never a zero effect; no key pooling or threshold reduction',
        limitations='Within-key state mix can still change. Statistical partitions do not identify a causal risk-process change. '
            'Fixed-weight bootstrap intervals condition on the support-derived balancing factors. V190 gates unchanged.',
        interval_fit=False,predictor_fit=False,confirmation_started=False,new_replays=0,new_training=0)


def masks(m):
    first,last = prior.supported_keys(m[:32,:,0]),prior.supported_keys(m[32:,:,0])
    return first,last,first & last


def balance(first,last):
    a,b = first[:,:,0].mean(axis=0),last[:,:,0].mean(axis=0)
    require(np.all(a > 0.) and np.all(b > 0.),'common key missing in a half')
    reference = (a+b)/2.
    return first*(reference/a)[None,:,None],last*(reference/b)[None,:,None]


def day_partition(m):
    first = m[:,:,:1+N]
    seconds = prior.second(m.sum(axis=1))
    cov,mass = prior.actor_covariance(first,seconds,np.full((1,len(m)),1./len(m)))
    sums = first[:,:,1:]
    scaled = np.divide(sums,np.sqrt(np.maximum(first[:,:,0],0.))[...,None],
        out=np.zeros_like(sums),where=first[:,:,:1] > 0.)
    within = seconds-np.einsum('dki,dkj->dij',scaled,scaled)
    pooled = first.mean(axis=0)
    mu = np.divide(pooled[:,1:],pooled[:,:1],out=np.zeros_like(pooled[:,1:]),where=pooled[:,:1] > 0.)
    cross = np.einsum('dki,kj->dij',sums,mu)
    centered_days = seconds-cross-np.swapaxes(cross,-1,-2)+np.einsum('dk,ki,kj->dij',first[:,:,0],mu,mu)
    require(np.allclose(centered_days.mean(axis=0),cov[0],rtol=1e-9,atol=1e-7),'day covariance partition differs')
    return cov[0],float(mass[0]),within,centered_days


def measure(cov,signal,unique):
    indices = [prior.VARS.index(c) for c in prior.CONTROLS]+([1-signal] if unique else [])
    cc = cov[...,indices,:][...,:,indices]
    sd = np.sqrt(np.maximum(np.diagonal(cc,axis1=-2,axis2=-1),0.))
    scale = sd[...,None,:]*sd[...,:,None]
    corr = np.divide(cc,scale,out=np.zeros_like(cc),where=scale > 1e-12)
    cross = np.divide(cov[..., [signal,TARGET],:][...,indices],sd[...,None,:],
        out=np.zeros_like(cov[..., [signal,TARGET],:][...,indices]),where=sd[...,None,:] > 1e-12)
    beta = cross@np.linalg.pinv(corr,rcond=1e-10,hermitian=True)
    beta = np.divide(beta,sd[...,None,:],out=np.zeros_like(beta),where=sd[...,None,:] > 1e-12)
    x,y = np.zeros((*cov.shape[:-2],N)),np.zeros((*cov.shape[:-2],N))
    x[...,signal],y[...,TARGET] = 1.,1.
    x[...,indices],y[...,indices] = -beta[...,0,:],-beta[...,1,:]
    def quadratic(a,b):
        return np.einsum('...i,...ij,...j->...',a,cov,b)
    numerator = quadratic(x,y)
    den = np.sqrt(np.maximum(quadratic(x,x),0.)*np.maximum(quadratic(y,y),0.))
    rho = np.divide(numerator,den,out=np.full_like(den,np.nan),where=den > 1e-12)
    return rho,numerator,den,x,y


def contribution(cov,measurement):
    _,_,den,x,y = measurement
    value = np.einsum('...i,...ij,...j->...',x,cov,y)
    return np.divide(value,den,out=np.full_like(value,np.nan),where=den > 1e-12)


def normalization(first,last):
    _,a,da,*_ = first
    _,b,db,*_ = last
    ia = np.divide(1.,da,out=np.full_like(da,np.nan),where=da > 1e-12)
    ib = np.divide(1.,db,out=np.full_like(db,np.nan),where=db > 1e-12)
    return (b-a)*(ia+ib)/2.,(a+b)*(ib-ia)/2.


def number(value):
    return float(value) if np.isfinite(value) else None


def weights(actors):
    rng = np.random.default_rng(prior.SEED)
    a = rng.multinomial(actors,np.full(actors,1./actors),size=prior.DRAWS)/actors
    d = np.stack([rng.multinomial(32,np.full(32,1./32.),size=prior.DRAWS)/32. for _ in range(2)])
    return a,d


def actor_matrix(rows,scope):
    require(len(rows) == 64 and {r['day'] for r in rows} == set(DAYS),'incomplete/duplicate original64 grid')
    cells = {r['day']:next(c for c in r['scopes'] if c['scope'] == scope) for r in rows}
    keys = sorted({tuple(g['key']) for c in cells.values() for g in c['groups']})
    locations = {k:i for i,k in enumerate(keys)}
    m = np.zeros((64,len(keys),SIZE))
    for i,day in enumerate(DAYS):
        seen = set()
        for g in cells[day]['groups']:
            k = tuple(g['key'])
            require(k not in seen,'duplicate road key')
            seen.add(k)
            m[i,locations[k]] = g['moments']
    return m,keys


def reproduce(covs,masses,counts,expected,mode,scope):
    for split,cov,mass,count in zip(('all64','first32','last32'),covs,masses,counts):
        old = next(r for r in expected['results'] if (r['mode'],r['scope'],r['split']) == (mode,scope,split))
        require(count == old['retained_keys'],'V190 support-key count differs')
        close(mass,old['analysis_mass'])
        for si,signal in enumerate(prior.SIGNALS):
            for block,unique in (('partial',True),('base_adjusted_diagnostic',False)):
                actual = measure(cov,si,unique)[0]
                want = old['estimates'][signal]['above'][block]['mean']
                require(want is None if not np.isfinite(actual) else want is not None,'V190 missing estimate differs')
                if want is not None:
                    require(abs(float(actual)-want) < 1e-8,'V190 point correlation differs')


def run_case(args,model,mode,scope):
    actors = sorted(model['actors'])
    aw,dw = weights(len(actors))
    pc,bc = np.zeros((3,2,N,N)),np.zeros((3,2,prior.DRAWS,N,N))
    pm,bm = np.zeros((3,2)),np.zeros((3,2,prior.DRAWS))
    within,bwithin = np.zeros((2,N,N)),np.zeros((2,prior.DRAWS,N,N))
    day_cov,mean_values = np.zeros((2,32,N,N)),np.zeros((3,2,1+N))
    original_cov,original_mass,original_count = np.zeros((N,N)),0.,0
    common_keys,actor_support,actor_cov = [],[],[]
    for ai,actor in enumerate(actors):
        b = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
        require(b['source'] == 'a7d126966a' and all((r['actor'],r['mode']) == (actor,mode) for r in b['rows']),
            'moment source/actor/mode differs')
        m,keys = actor_matrix(b['rows'],scope)
        a,z,common = masks(m)
        common_keys.extend(dict(actor=actor,key=list(k),first_days=int(np.count_nonzero(m[:32,i,0])),
            last_days=int(np.count_nonzero(m[32:,i,0]))) for i,k in enumerate(keys) if common[i])
        full = m[:,prior.supported_keys(m[:,:,0]),:]
        cov,mass = prior.actor_covariance(full[:,:,:1+N],prior.second(full.sum(axis=1)),np.full((1,64),1./64.))
        original_cov += cov[0]/len(actors)
        original_mass += mass[0]/len(actors)
        original_count += full.shape[1]
        pairs = [(m[:32,a,:],m[32:,z,:]),(m[:32,common,:],m[32:,common,:])]
        pairs.append(balance(*pairs[1]))
        per_actor = []
        for arm,pair in enumerate(pairs):
            for half,values in enumerate(pair):
                first,seconds = values[:,:,:1+N],prior.second(values.sum(axis=1))
                c,mass = prior.actor_covariance(first,seconds,np.full((1,32),1./32.))
                pc[arm,half] += c[0]/len(actors)
                pm[arm,half] += mass[0]/len(actors)
                mean_values[arm,half] += first.sum(axis=1).mean(axis=0)/len(actors)
                if arm == 2:
                    _,_,wd,dc = day_partition(values)
                    within[half] += wd.mean(axis=0)/len(actors)
                    day_cov[half] += dc/len(actors)
                    per_actor.append(c[0].tolist())
                for start in range(0,prior.DRAWS,64):
                    stop = min(start+64,prior.DRAWS)
                    c,mm = prior.actor_covariance(first,seconds,dw[half,start:stop])
                    factor = aw[start:stop,ai]
                    bc[arm,half,start:stop] += factor[:,None,None]*c
                    bm[arm,half,start:stop] += factor*mm
                    if arm == 2:
                        bwithin[half,start:stop] += factor[:,None,None]*(dw[half,start:stop]@wd.reshape(32,-1)).reshape(-1,N,N)
        close(pm[2,0],pm[2,1])
        actor_cov.append(dict(actor=actor,balanced_covariances=per_actor))
        actor_support.append(dict(actor=actor,first_keys=int(a.sum()),last_keys=int(z.sum()),common_keys=int(common.sum())))
        print('V191_COMMON_ACTOR',mode,scope,actor,int(common.sum()),flush=True)
    reference = read(args.v190_readout_root/'summary/status.json')
    reproduce([original_cov,*pc[0]],[original_mass,*pm[0]],
        [original_count,sum(a['first_keys'] for a in actor_support),sum(a['last_keys'] for a in actor_support)],
        reference,mode,scope)
    print('V191_V190_POINTS_REPRODUCED',mode,scope,flush=True)
    stats,details = {},{}
    for si,signal in enumerate(prior.SIGNALS):
        for block,unique in (('baseline',False),('unique',True)):
            point = [[measure(pc[arm,h],si,unique) for h in range(2)] for arm in range(3)]
            boot = [[measure(bc[arm,h],si,unique) for h in range(2)] for arm in range(3)]
            dr = [p[1][0]-p[0][0] for p in point]
            db = [p[1][0]-p[0][0] for p in boot]
            terms = (dr[0]-dr[1],dr[1]-dr[2],dr[2])
            draws = (db[0]-db[1],db[1]-db[2],db[2])
            require(np.allclose(sum(terms),dr[0],equal_nan=True),'composition identity differs')
            wc = [contribution(within[h],point[2][h]) for h in range(2)]
            bwc = [contribution(bwithin[h],boot[2][h]) for h in range(2)]
            norm,nscale = normalization(*point[2])
            bnorm,bscale = normalization(*boot[2])
            require(np.allclose(norm+nscale,dr[2],equal_nan=True),'normalization identity differs')
            value = dict(arms={arm:dict(first=prior.interval(point[i][0][0],boot[i][0][0]),
                last=prior.interval(point[i][1][0],boot[i][1][0]),delta=prior.interval(dr[i],db[i])) for i,arm in enumerate(ARMS)},
                delta_partition={k:prior.interval(v,d) for k,v,d in zip(
                    ('key_membership','key_mass_mix','standardized_within_key'),terms,draws)},
                balanced_day_partition=dict(first_within=number(wc[0]),first_between=number(point[2][0][0]-wc[0]),
                    last_within=number(wc[1]),last_between=number(point[2][1][0]-wc[1]),
                    delta_within=prior.interval(wc[1]-wc[0],bwc[1]-bwc[0]),
                    delta_between=prior.interval(dr[2]-(wc[1]-wc[0]),db[2]-(bwc[1]-bwc[0]))),
                balanced_delta_normalization=dict(covariance=prior.interval(norm,bnorm),
                    inverse_sd=prior.interval(nscale,bscale)))
            contributions = []
            for h in range(2):
                v = contribution(day_cov[h],point[2][h])/32.
                require(np.allclose(v.sum(),point[2][h][0],equal_nan=True),'day attribution does not sum to correlation')
                days = DAYS[:32] if h == 0 else DAYS[32:]
                contributions.append(None if not np.isfinite(v).all() else
                    [dict(day=day,correlation_contribution=float(x)) for day,x in zip(days,v)])
            value['top_abs_day_contributions'] = [None if c is None else [c[i] for i in np.argsort(
                [-abs(v['correlation_contribution']) for v in c])[:3]] for c in contributions]
            stats[signal+'|'+block] = value
            details[signal+'|'+block] = contributions
    descriptions = []
    for arm in range(3):
        for h in range(2):
            v = mean_values[arm,h]
            descriptions.append(dict(arm=ARMS[arm],half=('first32','last32')[h],mass=float(pm[arm,h]),
                means=None if v[0] == 0. else dict(zip(prior.VARS,(v[1:]/v[0]).tolist()))))
    result = dict(mode=mode,scope=scope,common_keys=len(common_keys),actor_support=actor_support,
        support={arm:dict(first=float(pm[i,0]),last=float(pm[i,1])) for i,arm in enumerate(ARMS)},
        stats=stats,descriptions=descriptions,v190_points_reproduced=True)
    write(args.out_root/'details'/f'{mode}_{scope}.json',dict(result=result,
        common_key_support=common_keys,day_contributions=details,actor_covariances=actor_cov))
    compact = {k:v for k,v in result.items() if k not in ('actor_support','descriptions')}
    write(args.out_root/'cells'/f'{mode}_{scope}.json',compact)
    print('V191_COMMON_CASE_COMPLETE',mode,scope,flush=True)
    return compact


def preflight(args,model):
    actors,mode,scope = sorted(model['actors']),'old','clearance'
    cov,mass,counts = np.zeros((3,N,N)),np.zeros(3),np.zeros(3,dtype=int)
    for actor in actors:
        b = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
        require(b['source'] == 'a7d126966a','preflight source differs')
        m,_ = actor_matrix(b['rows'],scope)
        a,z,common = masks(m)
        require(not np.any(common & ~(a & z)),'common support threshold weakened')
        for j,v in enumerate((m[:,prior.supported_keys(m[:,:,0]),:],m[:32,a,:],m[32:,z,:])):
            pc,pm = prior.actor_covariance(v[:,:,:1+N],prior.second(v.sum(axis=1)),np.full((1,len(v)),1./len(v)))
            cov[j] += pc[0]/len(actors)
            mass[j] += pm[0]/len(actors)
            counts[j] += v.shape[1]
    reproduce(cov,mass,counts,read(args.v190_readout_root/'summary/status.json'),mode,scope)
    print('V191_REAL_MOMENT_PREFLIGHT_PASSED',flush=True)


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v191_*.py','-v'],check=True)
    parent = read(args.v190_root/'protocol.json')
    complete = read(args.v190_readout_root/'summary/status.json')
    require(parent['source'] == 'a7d126966a' and complete['source'] == '30829987d4' and complete['paths'] == 1536 and
        parent['calibration_days'] == list(DAYS) and parent['support']['minimum_distinct_days'] == 19 and
        not complete['interval_fit'] and not complete['main_policy_changed'],'requires frozen completed V190 original64')
    model = read(args.v172_root/'model.json')
    require(model['execution_source'] == 'a5b45df520' and model['calibration_days'] == list(DAYS) and
        len(model['actors']) == 12,'frozen actor/day grid differs')
    plan = registration()
    require(not (args.out_root/'protocol.json').exists(),'do not overwrite analysis plan')
    write(args.out_root/'protocol.json',plan)
    print('V191_READONLY_PLAN_SEALED',flush=True)
    preflight(args,model)
    results = []
    with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(run_case,args,model,m,s) for m,s in
                product(('old','handoff'),prior.SCOPES)]),1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":8,"unit":"readonly_mode_scope"}'%i,flush=True)
    write(args.out_root/'summary/status.json',dict(status='v191_common_keys_diagnostic_complete',source=plan['source'],
        protocol=plan,results=sorted(results,key=lambda r:(r['mode'],r['scope'])),source_paths=1536,
        v190_points_reproduced=True,raw_capture_files_opened=0,interval_fit=False,predictor_fit=False,
        confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,
        map_updates=0,main_policy_changed=False))
    print('V191_COMMON_KEYS_COMPLETE_NO_FIT',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v190-root','v190-readout-root','v172-root'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
