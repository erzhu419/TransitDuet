#!/usr/bin/env python3
"""Preregister original64 supported leg innovations versus frozen full-ETA tails."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v189_leg_innovation as support


read, write, require, close = support.read, support.write, support.require, support.close
clocks, running = support.prior, support.prior.prior
DAYS, MIN_DAYS = support.DAYS, 19
SCOPES = running.SCOPES
PRIMARY = ('clearance','hour13_moving_launch')
SIGNALS = ('signed_remaining_s','innovation_rms_s')
TARGETS = ('below','above','lower_excess_s','upper_excess_s','score_s')
CONTROLS = ('query_offset_min','remaining_stops','remaining_km','first_fraction',
    'horizon_min','running_min','service_min','lower_radius_min','upper_radius_min',
    'native_leg_mean_min','native_n','native_latest_age_min','matched_n',
    'matched_latest_age_min','matched_mean_age_min','matched_entry_age_min',
    'service_prefix_under_min','service_prefix_latest_age_min','service_prefix_n',
    'predicted_entry','stage_service','stage_command','stage_ready')
VARS = (*SIGNALS,*TARGETS,*CONTROLS)
N = len(VARS)
TRI = np.triu_indices(N)
SIZE = 1+N+len(TRI[0])
DRAWS, SEED, FAMILY = 10000, 20260803, 16


def registration():
    return dict(stage='v190_supported_leg_tail_information',source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS),parent_capture='20d914af03',parent_map='a5b45df520',
        parent_support='92c59ee242',corrected_support='3333f6729c',
        frozen=dict(v151=True,v146=True,v172=True,main=True,window_s=3600.,tails=[.1,.1]),
        inputs='V189 physical road, strict pre-entry baseline, arrival availability and visible-entry clock unchanged',
        signals=dict(signed_remaining_s='first-leg remaining fraction * (matched duration mean-current native mean)',
            innovation_rms_s='first-leg remaining fraction * sqrt(mean((duration-pre-entry native mean)^2)); '
                'RMS surprise, not sample standard deviation; defined for one completed traversal'),
        support=dict(minimum_completed_traversals=1,minimum_distinct_days=MIN_DAYS,
            key=['actor','mode','scope','direction','first_leg_destination','reference_entry_hour'],
            prerequisite='unchanged native/V172 support AND matching completed leg AND >=9 strictly completed '
                'global service-prefix episodes; never fill unavailable controls with zero',
            gate='count distinct days with complete query-time inputs, separately in all64/first32/last32; '
                'retain only keys with >=19 days in that split; masks frozen before resampling',
            nine='V189 nine-traversal count is diagnostic, not a weakened or replaced admission gate',
            missing='no key pooling, no fallback signal, no lower threshold, no zero-valued missing effects'),
        outcomes='Frozen V172 full ETA: truth<L,truth>U,lower/upper excess,interval score with .1/.1 tails',
        controls=list(CONTROLS),fixed_effects='within actor/mode/road/reference-entry-hour; retain between-day variation',
        adjustment='standardized covariance projection, pinv rcond=1e-10; each signal additionally controls the other',
        weights='original V189 equal query within eligible episode, equal episode/path, nominal equal actor/day; '
            'do not renormalize a selected episode or silently replace missing paths',
        primary_scopes=list(PRIMARY),diagnostic_scopes=['hour2','hour13'],
        inference=dict(draws=DRAWS,seed=SEED,clusters='independent actor and shared-day multinomial draws',
            family=FAMILY,primary='2 modes x 2 primary scopes x 2 signals x 2 tail indicators; '
                'Bonferroni percentile intervals at 1-.05/16; diagnostic intervals95%',
            directions='signed:lower negative,upper positive; RMS:both positive',
            replication='all64 family interval has the expected sign in BOTH modes, and95% intervals '
                'in BOTH predetermined halves have that sign in BOTH modes; undefined is inconclusive',
            scope='calibration-only linear incremental association, not conditional independence, '
                'held-future prediction skill, conformal coverage or controller efficacy'),
        interval_fit=False,predictor_fit=False,confirmation_started=False,new_replays=0,new_training=0)


class RiskPrefix:
    def __init__(self, rows, history):
        groups = defaultdict(list)
        for (direction,station),values in rows.items():
            for end,start,duration in values:
                baseline = history.estimate(('running',direction,station),start)
                if baseline is None:
                    continue
                require(baseline['latest'] < start <= end,'unfinished pre-entry input')
                error = duration-baseline['mean']
                groups[direction,station,int(start//3600)].append((end,duration,baseline['mean'],error*error,start,end))
        self.groups = {}
        for k,values in groups.items():
            a = np.array(sorted(values))
            self.groups[k] = a[:,0].tolist(),np.vstack((np.zeros(5),np.cumsum(a[:,1:],axis=0)))

    def query(self, key, now, mean):
        if key not in self.groups:
            return None
        times,sums = self.groups[key]
        left,right = bisect_left(times,now-3600.),bisect_left(times,now)
        if left == right:
            return None
        duration,baseline,square,entry,end = (sums[right]-sums[left])/(right-left)
        require(now-3600. <= times[left] <= times[right-1] < now,'unfinished innovation label')
        close((duration-baseline)-(mean-baseline),duration-mean)
        return dict(n=right-left,latest=times[right-1],remaining=duration-mean,
            rms=float(np.sqrt(max(square,0.))),mean_age_min=(now-end)/60.,entry_age_min=(now-entry)/60.)


def moment(v):
    v = np.asarray(v)
    return np.r_[1.,v,np.outer(v,v)[TRI]]


def second(m):
    a = np.zeros((*m.shape[:-1],N,N))
    a[...,TRI[0],TRI[1]] = m[...,1+N:]
    a[...,TRI[1],TRI[0]] = m[...,1+N:]
    return a


def leg(q, geometry, entries, history, prefix):
    p,now = q['observable']['position'],q['query_time_s']
    if p is None or p['origin_index'] == p['target_index']:
        return None
    direction = bool(q['condition_values']['direction'])
    route = geometry[direction]
    i = p['origin_index']+1
    station = route['ids'][i]
    native = history.estimate(('running',direction,station),now)
    saved = next(s['components']['running'] for s in q['support']['completed_components'] if s['station'] == station)
    require(native is not None and native['latest'] < now and all(native[c] == saved[c]
        for c in ('count','latest','historical_fallback')),'native input support differs')
    close(native['mean'],saved['mean'])
    entry,end = entries[int(q['episode'][0]),station,direction]
    require(end >= now,'current remaining leg already finished')
    clock,kind = support.visible_entry(now,p['initial_residence_prediction_s'],entry)
    k = int(direction),station,int(clock//3600)
    value = prefix.query(k,now,native['mean'])
    v = np.zeros(len(support.FIELDS))
    v[0],v[support.FIELDS.index(kind)] = 1.,1.
    if value is not None:
        require(value['n'] <= native['count'],'innovation cohort exceeds native cohort')
        v[1:3],v[-1] = (1.,float(value['n'] >= 9)),value['n']
        if value['n'] == native['count']:
            close(value['remaining'],0.)
            v[support.FIELDS.index('same_native_cohort')] = 1.
    fraction = (route['positions'][i]-p['progress_m'])/(route['positions'][i]-route['positions'][i-1])
    require(-1e-8 <= fraction <= 1.+1e-8,'invalid first-leg fraction')
    return k,v,value,native,float(fraction),kind


def features(q, r, bounds, item, service):
    k,_,x,native,fraction,kind = item
    p,now = q['observable']['position'],q['query_time_s']
    lo,hi = bounds
    require(r['label']['status'] == 'resolved' and r['residual_s'] is not None and
        r['label']['available_time_s'] == r['label']['target_ready_time_s'] >= now,'full-ETA label clock differs')
    truth,point = r['label']['full_eta_s'],q['point_s']
    close(sum(r['residual_s'].values()),point-truth)
    lower,upper = max(lo-truth,0.),max(truth-hi,0.)
    stage = q['native_stage']['kind']
    require(stage in ('service','known-command','ready-residence','moving-launch'),'unknown supported stage')
    v = [fraction*x['remaining'],fraction*x['rms'],float(truth < lo),float(truth > hi),
        lower,upper,hi-lo+10.*(lower+upper),
        (now-3600*k[2])/60.,p['remaining_stops'],p['remaining_distance_m']/1000.,fraction,
        point/60.,q['prediction_coarse']['running']/60.,q['prediction_coarse']['service']/60.,
        (point-lo)/60.,(hi-point)/60.,native['mean']/60.,native['count'],(now-native['latest'])/60.,
        x['n'],(now-x['latest'])/60.,x['mean_age_min'],x['entry_age_min'],
        -service['mean'][1]/60.,(now-service['latest'])/60.,service['n'],float(kind == 'predicted_entry'),
        float(stage == 'service'),float(stage == 'known-command'),float(stage == 'ready-residence')]
    require(len(v) == N and np.isfinite(v).all(),'missing/nonfinite query inputs')
    return v


def scan(args,actor,day,mode,table,old,geometry):
    require(day in DAYS,'only original64 calibration days')
    root,name = args.v166_root/'calibration',f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'],receipt['seed'],receipt['day'],receipt['mode']) ==
        ('20d914af03',actor,day,mode) and not receipt['interval_fit'] and
        not receipt['main_policy_changed'],'capture differs')
    _,rows,history = clocks.events(root,name,geometry)
    entries,prefix = support.entry_lookup(root,name,geometry),RiskPrefix(rows,history)
    mapping = {tuple(e['key']):e for e in table['candidate']}
    atoms,selected,count = {},[],0
    for q,r in zip_longest(*(running.previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries','residuals'))):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and
            q['episode'] == r['episode'] and q['query_time_s'] == r['query_time_s'] and
            (q['actor'],q['day'],q['mode']) == (actor,day,mode),'query/residual join differs')
        count += 1
        running.previous.add_first(atoms,q,r)
        if not q['denominator_eligible']:
            continue
        now = q['query_time_s']
        scopes = [s for s in SCOPES if (s == 'clearance' and q['condition_values']['query_phase'] == 'clearance') or
            (s == 'hour2' and q['observable']['hour'] == 2) or (s == 'hour13' and q['observable']['hour'] == 13) or
            (s == 'hour13_moving_launch' and q['observable']['hour'] == 13 and q['native_stage']['kind'] == 'moving-launch')]
        if not scopes:
            continue
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'],'late load backfill')
        period = running.temporal.STRATA[int(running.temporal.strata(np.array([now]))[0])]
        entry = mapping.get((*q['candidate_key'],period)) if q['native_supported'] else None
        bounds = running.previous.saved.fit.interval(q['point_s'],entry,q['native_supported'])
        require(bounds is None or entry['day_count'] >= MIN_DAYS,'V172 support weakened')
        selected.append((q,r,scopes,bounds))
    require(count == receipt['query_record_count'],'incomplete query path')
    service = running.previous.Prefix(atoms)
    base,raw,groups = ({s:{} for s in SCOPES},{s:defaultdict(dict) for s in SCOPES},{s:defaultdict(dict) for s in SCOPES})
    for q,r,scopes,bounds in sorted(selected,key=lambda v:(v[0]['query_time_s'],v[0]['clocks']['query_ready_order'])):
        ep,now = tuple(q['episode']),q['query_time_s']
        h = service.query(now,None)['global']
        item = None if bounds is None else leg(q,geometry,entries,history,prefix)
        vector = None if item is None or item[2] is None or h['mean'] is None else moment(features(q,r,bounds,item,h))
        for scope in scopes:
            base[scope].setdefault(ep,np.zeros(2))[:] += (1.,float(bounds is not None))
            if item is not None:
                k,v = item[:2]
                raw[scope][ep].setdefault(k,np.zeros(len(support.FIELDS)))[:] += v
                if vector is not None:
                    groups[scope][ep].setdefault(k,np.zeros(SIZE))[:] += vector
    def collapse(source,scope):
        result = {}
        for ep,values in source[scope].items():
            for k,v in values.items():
                result.setdefault(k,np.zeros_like(v))[:] += v/(base[scope][ep][0]*len(base[scope]))
        return result
    results = []
    for scope in SCOPES:
        expected = next(c for c in old['scopes'] if c['scope'] == scope)
        require(len(base[scope]) == expected['eligible_episodes'],'original episode denominator differs')
        fraction = np.mean([v[1]/v[0] for v in base[scope].values()]) if base[scope] else None
        require(fraction is not None and expected['finite_fraction'] is not None,'missing original scope')
        close(fraction,expected['finite_fraction'])
        before = {tuple(g['key']):np.array(g['values']) for g in expected['groups']}
        after = collapse(raw,scope)
        require(set(before) == set(after),'V189 raw support keys differ')
        for k in before:
            require(np.allclose(before[k],after[k],rtol=0.,atol=1e-10),'V189 raw support weights differ')
        results.append(dict(scope=scope,eligible_episodes=len(base[scope]),baseline_fraction=float(fraction),
            raw_one_mass=float(sum(v[1] for v in after.values())),
            groups=[dict(key=list(k),moments=v.tolist()) for k,v in collapse(groups,scope).items()]))
    return dict(actor=actor,day=day,mode=mode,scopes=results)


def bucket(args,model,chunk):
    actor,mode = chunk['actor'],chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source'] == 'a5b45df520','V172 source differs')
    old = read(args.v189_root/'cells'/f'{actor}_{mode}.json')
    require(old['source'] == '92c59ee242','V189 extraction differs')
    lookup = {r['day']:r for r in old['rows']}
    require(set(lookup) == set(DAYS),'V189 day grid differs')
    geometry,rows = clocks.native.load_geometry(args.runtime),[]
    for i,day in enumerate(DAYS,1):
        rows.append(scan(args,actor,day,mode,table,lookup[day],geometry))
        if i % 8 == 0:
            print('V190_CALIBRATION_DAYS',actor,mode,i,'/64',flush=True)
    write(args.out_root/'cells'/f'{actor}_{mode}.json',dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],rows=rows))


def actor_covariance(first,seconds,weights):
    # Only per-key mass/first moments need resampling; second moments sum linearly.
    if first.shape[1] == 0:
        return np.zeros((len(weights),N,N)),np.zeros(len(weights))
    f = (weights@first.reshape(len(first),-1)).reshape(-1,first.shape[1],1+N)
    mass,sums = f[:,:,0],f[:,:,1:]
    scaled = np.divide(sums,np.sqrt(np.maximum(mass,0.))[...,None],
        out=np.zeros_like(sums),where=mass[...,None] > 0.)
    return (weights@seconds.reshape(len(seconds),-1)).reshape(-1,N,N)-np.einsum('bki,bkj->bij',scaled,scaled),mass.sum(axis=1)


def partial(cov,signal,controls=True):
    indices = ([VARS.index(c) for c in CONTROLS]+[1-signal]) if controls else []
    if indices:
        cc = cov[...,indices,:][...,:,indices]
        sd = np.sqrt(np.maximum(np.diagonal(cc,axis1=-2,axis2=-1),0.))
        scale = sd[...,None,:]*sd[...,:,None]
        corr = np.divide(cc,scale,out=np.zeros_like(cc),where=scale > 1e-12)
        cross = np.divide(cov[..., :,indices],sd[...,None,:],
            out=np.zeros_like(cov[..., :,indices]),where=sd[...,None,:] > 1e-12)
        cov = cov-cross@np.linalg.pinv(corr,rcond=1e-10,hermitian=True)@np.swapaxes(cross,-1,-2)
    result = []
    for target in TARGETS:
        j = VARS.index(target)
        den = np.sqrt(np.maximum(cov[...,signal,signal],0.)*np.maximum(cov[...,j,j],0.))
        result.append(np.divide(cov[...,signal,j],den,out=np.full_like(den,np.nan),where=den > 1e-12))
    return np.stack(result,axis=-1)


def supported_keys(mass):
    return np.count_nonzero(mass > 0.,axis=0) >= MIN_DAYS


def interval(value,draws,alpha=.05):
    valid = np.isfinite(draws)
    return dict(mean=float(value) if np.isfinite(value) else None,valid_draws=int(valid.sum()),
        ci=None if not valid.all() else np.percentile(draws,[100*alpha/2,100*(1-alpha/2)]).tolist())


def summarize(rows,actors,days,scope):
    index = {(r['actor'],r['day']):next(c for c in r['scopes'] if c['scope'] == scope) for r in rows}
    require(set(index) == set(product(actors,days)) and len(index) == len(rows),'incomplete crossed grid')
    rng = np.random.default_rng(SEED)
    aw = rng.multinomial(len(actors),np.full(len(actors),1./len(actors)),size=DRAWS)/len(actors)
    dw = rng.multinomial(len(days),np.full(len(days),1./len(days)),size=DRAWS)/len(days)
    point,draw_cov = np.zeros((N,N)),np.zeros((DRAWS,N,N))
    mass,draw_mass,raw_mass,blocked,keys = 0.,np.zeros(DRAWS),0.,0.,0
    actor_results,day_mass = [],np.zeros(len(days))
    for ai,actor in enumerate(actors):
        cells = [index[actor,day] for day in days]
        identities = sorted({tuple(g['key']) for c in cells for g in c['groups']})
        positions = {k:i for i,k in enumerate(identities)}
        m = np.zeros((len(days),len(identities),SIZE))
        for di,c in enumerate(cells):
            raw_mass += c['raw_one_mass']/(len(actors)*len(days))
            for g in c['groups']:
                m[di,positions[tuple(g['key'])]] = g['moments']
        keep = supported_keys(m[:,:,0])
        blocked += float(m[:,~keep,0].sum()/(len(actors)*len(days)))
        keys += int(keep.sum())
        m = m[:,keep,:]
        first,seconds = m[:,:,:1+N],second(m.sum(axis=1))
        pc,pm = actor_covariance(first,seconds,np.full((1,len(days)),1./len(days)))
        point += pc[0]/len(actors)
        mass += pm[0]/len(actors)
        day_mass += first[:,:,0].sum(axis=1)/len(actors)
        actor_results.append(dict(actor=actor,mass=float(pm[0]),retained_keys=int(keep.sum()),
            partial={s:dict(zip(TARGETS,[float(x) if np.isfinite(x) else None for x in partial(pc[0],si)]))
                for si,s in enumerate(SIGNALS)}))
        for start in range(0,DRAWS,64):
            stop = min(start+64,DRAWS)
            cov,mm = actor_covariance(first,seconds,dw[start:stop])
            draw_cov[start:stop] += aw[start:stop,ai,None,None]*cov
            draw_mass[start:stop] += aw[start:stop,ai]*mm
    estimates = {}
    for si,s in enumerate(SIGNALS):
        p,boot = partial(point,si),partial(draw_cov,si)
        raw = partial(point,si,False)
        estimates[s] = {t:dict(partial=interval(p[j],boot[:,j]),
            family=interval(p[j],boot[:,j],.05/FAMILY) if scope in PRIMARY and t in ('below','above') else None,
            unadjusted=float(raw[j]) if np.isfinite(raw[j]) else None) for j,t in enumerate(TARGETS)}
    return dict(scope=scope,retained_keys=keys,raw_one_mass=raw_mass,analysis_mass=mass,
        below19_complete_input_mass=blocked,analysis_mass_ci95=np.percentile(draw_mass,[2.5,97.5]).tolist(),
        estimates=estimates,actor_margins=actor_results,
        day_margins=[dict(day=day,mass=float(day_mass[i])) for i,day in enumerate(days)])


def reduce_scope(args,model,mode,scope):
    rows = []
    for actor in model['actors']:
        b = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(b['source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],'cell source differs')
        rows.extend(b['rows'])
    result = []
    for split,days in (('all64',DAYS),('first32',DAYS[:32]),('last32',DAYS[32:])):
        result.append(dict(mode=mode,split=split,**summarize([r for r in rows if r['day'] in days],
            sorted(model['actors']),days,scope)))
        print('V190_CROSSED_RISK_SUMMARY',mode,scope,split,flush=True)
    write(args.out_root/'details'/f'{mode}_{scope}.json',result)
    return result


def evidence(results):
    decisions = []
    for scope,signal,target in product(PRIMARY,SIGNALS,('below','above')):
        sign = -1 if signal == 'signed_remaining_s' and target == 'below' else 1
        parts = []
        for mode,split in product(('old','handoff'),('all64','first32','last32')):
            r = next(r for r in results if (r['mode'],r['scope'],r['split']) == (mode,scope,split))
            e = r['estimates'][signal][target]['family' if split == 'all64' else 'partial']
            ci = e['ci']
            verdict = None if ci is None or e['mean'] is None else bool(ci[0] > 0. if sign > 0 else ci[1] < 0.)
            parts.append(dict(mode=mode,split=split,expected_sign=sign,passed=verdict))
        decisions.append(dict(scope=scope,signal=signal,target=target,
            decision='inconclusive' if any(p['passed'] is None for p in parts) else
                'replicated_calibration_association' if all(p['passed'] for p in parts) else 'not_replicated',parts=parts))
    return decisions


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v190_*.py','-v'],check=True)
    model = running.parents(args)
    original = read(args.v189_root/'summary/status.json')
    corrected = read(args.v189_readout_root/'summary/status.json')
    require(original['source'] == '92c59ee242' and original['paths'] == 1536 and
        corrected['source'] == '3333f6729c' and corrected['observed_key_count_corrected'] and
        corrected['strict_pre_entry_and_arrival_clocks'],'requires completed corrected V189 support')
    plan = registration()
    require(not (args.out_root/'protocol.json').exists(),'do not overwrite preregistration')
    write(args.out_root/'protocol.json',plan)
    print('V190_PROTOCOL_SEALED_BEFORE_ANY_RISK_ASSOCIATION',flush=True)
    chunk = model['chunks'][0]
    old = read(args.v189_root/'cells'/f"{chunk['actor']}_{chunk['mode']}.json")
    scan(args,chunk['actor'],DAYS[0],chunk['mode'],read(args.v172_root/chunk['map_file']),
        next(r for r in old['rows'] if r['day'] == DAYS[0]),clocks.native.load_geometry(args.runtime))
    print('V190_REAL_PATH_SUPPORT_REPLICATION_PASSED',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(bucket,args,model,c) for c in model['chunks']]),1):
            job.result()
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}'%i,flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn')) as pool:
        for job in as_completed([pool.submit(reduce_scope,args,model,m,s) for m,s in product(('old','handoff'),SCOPES)]):
            results.extend(job.result())
    require(running.parents(args) == model and read(args.out_root/'protocol.json') == plan,'frozen parent changed')
    # Detailed actor/day diagnostics remain on the server, not in the small summary.
    compact = [{k:v for k,v in r.items() if k not in ('actor_margins','day_margins')} for r in results]
    write(args.out_root/'summary/status.json',dict(status='v190_supported_leg_tail_information_complete',source=plan['source'],
        protocol=plan,paths=1536,original_v189_support_reproduced=True,results=sorted(compact,key=lambda r:(r['mode'],r['scope'],r['split'])),
        evidence=evidence(results),interval_fit=False,predictor_fit=False,map_updates=0,confirmation_outcomes_read=False,
        confirmation_started=False,new_replays=0,new_training=0,main_policy_changed=False))
    print('V190_SUPPORTED_LEG_TAIL_INFORMATION_COMPLETE_NO_FIT',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v166-root','v166-audit-root','v172-root','v189-root','v189-readout-root','runtime'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    run(p.parse_args())
