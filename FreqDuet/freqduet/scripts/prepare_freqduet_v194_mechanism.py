#!/usr/bin/env python3
"""Register component hypotheses and admit clocked original64 mechanism moments."""

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

import audit_freqduet_v191_common_keys as common
import audit_freqduet_v152_component_truth as accounting


prior = common.prior
read,write,require,close = common.read,common.write,common.require,common.close
DAYS,SCOPES = prior.DAYS,('clearance','hour13','hour13_moving_launch')
OUTCOMES = ('first_running_under_s','later_running_under_s','service_under_s',
    'holding_outside_joint_under_s','execution_wait_under_s','current_joint_under_s','full_under_s')
VARS = (*prior.VARS,'innovation_total_s','innovation_absorbed_s',*OUTCOMES)
N = len(VARS)
TRI = np.triu_indices(N)
SIZE = 1+N+len(TRI[0])
FACTS = ('source_latest_age_s','source_mean_age_s','first_running_label_lead_s',
    'all_running_label_lead_s','service_label_lead_s','holding_label_lead_s',
    'actual_entry_hour_mismatch','current_joint_present')


def registration():
    return dict(stage='v194_signed_innovation_component_timing',source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS),capture_source='20d914af03',moment_source='a7d126966a',common_source='e9efbe9bae',
        frozen=dict(v151=True,v146=True,v172=True,main=True,window_s=3600.,tails=[.1,.1]),
        hypotheses=dict(H1='Signed unabsorbed same-road innovation predicts first remaining-leg running underestimation within day.',
            H2='A same-sign running relationship can attenuate on later legs rather than persist over the whole remaining path.',
            H3='Service and post-ready component covariance can offset running covariance in full ETA.'),
        signals=dict(J='remaining fraction*(matched completed duration mean-current native road mean),same signed signal as V190',
            I='remaining fraction*mean(duration-pre-entry native mean)',A='remaining fraction*(current native mean-mean pre-entry native mean)',
            identity='I=A+J; never put all three in one regression or treat absorbed I as unabsorbed news',
            RMS='unchanged V190 other-signal control only,no RMS threshold,candidate gate or interval overlay'),
        targets=list(OUTCOMES),sign='truth-prediction; full=sum(first running,later running,service,holding outside joint,execution wait,current joint)',
        holding='Current joint-ready residence is separate,never relabeled pure holding; holding outside joint excludes that segment.',
        clocks='Source arrivals strictly before query,within unchanged3600s; pre-entry native inputs strictly before source entry. '
            'First running label at first arrival,all running/holding at target arrival,service/full at target ready. '
            'Truth clips intervals to [query,target ready]; actual future entry clocks are outcomes only,not feature keys.',
        support='Exact V191 common actor/mode/scope/road/entry-hour keys with>=19 complete-input days in EACH half; no extra subgroup admission.',
        weights='Reproduce V190 moment cells exactly; equal query within ORIGINAL eligible episode,equal episode,path,actor/day; '
            'then V191 fixed support-only key-mass balance. No renormalization after component selection.',
        controls=[*prior.CONTROLS,'innovation_rms_s'],
        next_analysis=dict(primary_scopes=['clearance','hour13_moving_launch'],diagnostic='hour13 overlaps clearance,not independent replication',
            contrasts=['cov(J,first_running_under_s)>0',
                'cov(J,service_under_s+holding_outside_joint_under_s+execution_wait_under_s+current_joint_under_s)<0'],
            family='8 contrasts:2 modes x2 primary scopes x2 contrasts; all64 Bonferroni percentile alpha=.05/8',
            replication='expected family sign in both modes on all64 AND expected95% sign in both fixed32-day halves/both modes',
            moments='Within actor/road/entry-hour/day covariance is primary; between-day covariance is separate descriptive evidence.',
            adjustment='One linear projection on fixed controls from average half within-day covariance; apply same projection to '
                'J and every additive outcome. Compare unstandardized covariance on the same mass/units; correlations are secondary.',
            inference='10000 crossed shared-actor/independent-half-day draws,seed20260803; recompute projection each draw,keep masks/balance fixed',
            H2='Summed later-road covariance is diagnostic,not a per-road effect or equivalence test;no distance pooling/post-hoc horizon bins',
            full_contrast='Full-minus-first covariance also contains later running;never attribute it alone to service/holding cancellation',
            stops='No replicated first-leg sign:do not build a propagation correction. No offset contrast:do not claim cancellation. '
                'No effect is not inferred from a missing estimate. Positive association is not held-future skill or causal holding effect.'),
        execution='This stage seals hypotheses and prepares/admit clocks and component moments only; no new association estimates yet.',
        preflight='first sorted actor,first and last ORIGINAL calibration days,both modes; pass before full1536-path preparation',
        limitations='Original64 reused development days,not independent confirmation. Actual future component labels only verify associations later. '
            'Outcome timing summaries are not deployable conditions. Component covariance cannot identify causal control mediation.',
        interval_fit=False,predictor_fit=False,mechanism_associations_estimated=False,
        confirmation_started=False,new_replays=0,new_training=0)


def moment(v):
    v = np.asarray(v)
    return np.r_[1.,v,np.outer(v,v)[TRI]]


def old_moment(m):
    second = np.zeros((N,N))
    second[TRI],second[(TRI[1],TRI[0])] = m[1+N:],m[1+N:]
    return np.r_[m[0],m[1:1+prior.N],second[prior.TRI]]


def innovation_parts(prefix,key,now,mean,fraction):
    times,sums = prefix.groups[key]
    lo,hi = bisect_left(times,now-3600.),bisect_left(times,now)
    require(lo<hi and times[hi-1]<now,'unfinished innovation source')
    duration,baseline = ((sums[hi]-sums[lo])/(hi-lo))[:2]
    i,a,j = fraction*(duration-baseline),fraction*(mean-baseline),fraction*(duration-mean)
    close(i-a,j)
    return i,a,j


def load_timeline(root,name,geometry):
    with np.load(root/'records'/(name+'.npz')) as z:
        fields,launch_fields = z['event_fields'].tolist(),z['launch_fields'].tolist()
        arrivals = {prior.clocks.key(dict(zip(fields,r.tolist()))):dict(zip(fields,r.tolist())) for r in z['arrivals']}
        departures = {prior.clocks.key(dict(zip(fields,r.tolist()))):dict(zip(fields,r.tolist())) for r in z['departures']}
        launches = {int(r[0]):dict(zip(launch_fields,r.tolist())) for r in z['launches']}
    ready = {prior.clocks.key(r):r for r in prior.running.previous.stream(root/'ready_labels'/(name+'.jsonl.gz'))}
    path = dict(arrivals=arrivals,departures=departures,ready=ready,
        dispatch={trip:dict(actual_s=row['time_s']) for trip,row in launches.items()})
    rows,entries = defaultdict(list),{}
    for k,a in arrivals.items():
        trip,station,direction = k
        route = geometry[direction]
        ix = route['index'][station]
        require(ix>0,'road arrival at terminal')
        before = departures.get((trip,route['ids'][ix-1],direction))
        require(before is not None or ix==1,'missing adjacent road departure')
        start = launches[trip]['time_s'] if before is None else before['time_s']
        require(start<=a['time_s'],'negative recorded traversal')
        rows[direction,station].append((a['time_s'],start,a['time_s']-start))
        entries[k] = start,a['time_s']
    rows = {k:sorted(v) for k,v in rows.items()}
    history = prior.clocks.native.CompletedMeans([(('running',d,s),end,duration)
        for (d,s),v in rows.items() for end,start,duration in v])
    return path,rows,entries,history


def component_labels(q,r,item,path,geometry,entries):
    now,ep = q['query_time_s'],q['episode']
    trip,bus,target,direction = ep
    k = trip,target,bool(direction)
    end = path['ready'][k]['ready_time_s']
    require(path['ready'][k]['bus_id']==bus and end==r['label']['available_time_s']>=now,
        'same physical follower/label clock differs')
    row = dict(prediction_time_s=now,station_id=target,direction=bool(direction),
        evidence=dict(predicted_follower_trip_id=trip,predicted_follower_bus_id=bus))
    joint = q['inseparable_current_ready_key']
    actual = accounting.parts.actual_parts(path,row,geometry[bool(direction)],None if joint is None else tuple(joint))
    predicted = q['prediction_parts']
    for name,value in accounting.coarse(actual).items():
        close(value,r['truth_coarse'][name])
    close(sum(predicted.values()),q['point_s'])
    close(sum(actual.values()),r['label']['full_eta_s'])
    road = item[0][1]
    start,first_arrival = entries[trip,road,bool(direction)]
    target_arrival = path['arrivals'][k]['time_s']
    require(now<=first_arrival<=target_arrival<=end,'component label order differs')
    first_truth = accounting.parts.overlap(start,first_arrival,now,end)
    first_prediction = item[4]*item[3]['mean']
    values = [first_truth-first_prediction,
        (actual['running']-first_truth)-(predicted['running']-first_prediction),
        actual['current_service']+actual['downstream_service']-predicted['current_service']-predicted['downstream_service'],
        actual['holding']-predicted['holding'],actual['execution_wait']-predicted['execution_wait'],
        actual['current_ready_joint']-predicted['current_ready_joint'],r['label']['full_eta_s']-q['point_s']]
    close(sum(values[:-1]),values[-1])
    close(values[0]+values[1],-r['residual_s']['running'])
    close(values[2],-r['residual_s']['service'])
    close(sum(values[3:6]),-r['residual_s']['residence'])
    x = item[2]
    facts = [now-x['latest'],60*x['mean_age_min'],first_arrival-now,target_arrival-now,end-now,
        target_arrival-now,float(int(start//3600)!=item[0][2]),float(joint is not None)]
    return values,facts


def collapse(source,base):
    result = {}
    for ep,groups in source.items():
        for k,v in groups.items():
            result.setdefault(k,np.zeros_like(v))[:] += v/(base[ep][0]*len(base))
    return result


def scan(args,actor,day,mode,table,geometry,reference,admitted):
    require(day in DAYS,'only original64 calibration days')
    root,name = args.v166_root/'calibration',f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'],receipt['seed'],receipt['day'],receipt['mode'])==
        ('20d914af03',actor,day,mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'],'capture differs')
    path,rows,entries,history = load_timeline(root,name,geometry)
    prefix,mapping = prior.RiskPrefix(rows,history),{tuple(e['key']):e for e in table['candidate']}
    atoms,selected,count = {},[],0
    for q,r in zip_longest(*(prior.running.previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries','residuals'))):
        require(q is not None and r is not None and q['query_key']==r['query_key'] and q['episode']==r['episode'] and
            q['query_time_s']==r['query_time_s'] and (q['actor'],q['day'],q['mode'])==(actor,day,mode),'query/label join differs')
        count += 1
        prior.running.previous.add_first(atoms,q,r)
        if not q['denominator_eligible']:
            continue
        scopes = [s for s in SCOPES if (s=='clearance' and q['condition_values']['query_phase']=='clearance') or
            (s=='hour13' and q['observable']['hour']==13) or
            (s=='hour13_moving_launch' and q['observable']['hour']==13 and q['native_stage']['kind']=='moving-launch')]
        if not scopes:
            continue
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'],'late load backfill')
        now = q['query_time_s']
        period = prior.running.temporal.STRATA[int(prior.running.temporal.strata(np.array([now]))[0])]
        entry = mapping.get((*q['candidate_key'],period)) if q['native_supported'] else None
        bounds = prior.running.previous.saved.fit.interval(q['point_s'],entry,q['native_supported'])
        require(bounds is None or entry['day_count']>=19,'frozen interval support weakened')
        selected.append((q,r,scopes,bounds))
    require(count==receipt['query_record_count'],'query path incomplete')
    service = prior.running.previous.Prefix(atoms)
    base,old,new,facts = ({s:{} for s in SCOPES},*[{s:defaultdict(dict) for s in SCOPES} for _ in range(3)])
    clock_queries = 0
    for q,r,scopes,bounds in sorted(selected,key=lambda v:(v[0]['query_time_s'],v[0]['clocks']['query_ready_order'])):
        ep,now = tuple(q['episode']),q['query_time_s']
        h = service.query(now,None)['global']
        item = None if bounds is None else prior.leg(q,geometry,entries,history,prefix)
        vector = None if item is None or item[2] is None or h['mean'] is None else prior.features(q,r,bounds,item,h)
        included = [] if vector is None else [s for s in scopes if item[0] in admitted[s]]
        if included:
            i,a,j = innovation_parts(prefix,item[0],now,item[3]['mean'],item[4])
            close(j,vector[0])
            components,timing = component_labels(q,r,item,path,geometry,entries)
            extended = moment([*vector,i,a,*components])
            require(np.allclose(old_moment(extended),prior.moment(vector),rtol=0.,atol=1e-8),'extended query moments differ')
            clock_queries += 1
        for s in scopes:
            base[s].setdefault(ep,np.zeros(2))[:] += (1.,float(bounds is not None))
            if vector is not None:
                k = item[0]
                old[s][ep].setdefault(k,np.zeros(prior.SIZE))[:] += prior.moment(vector)
                if s in included:
                    new[s][ep].setdefault(k,np.zeros(SIZE))[:] += extended
                    facts[s][ep].setdefault(k,np.zeros(1+len(FACTS)))[:] += np.r_[1.,timing]
    result = []
    for s in SCOPES:
        expected = next(c for c in reference['scopes'] if c['scope']==s)
        require(len(base[s])==expected['eligible_episodes'],'original episode denominator differs')
        original = {tuple(g['key']):np.array(g['moments']) for g in expected['groups']}
        reproduced = collapse(old[s],base[s])
        require(set(original)==set(reproduced),'V190 complete-input keys differ')
        for k in original:
            require(np.allclose(original[k],reproduced[k],rtol=0.,atol=1e-8),'V190 moments/weights differ')
        values,meta = collapse(new[s],base[s]),collapse(facts[s],base[s])
        require(set(values)==set(original)&admitted[s],'admitted common-key cohort differs')
        for k,v in values.items():
            require(np.allclose(old_moment(v),original[k],rtol=0.,atol=1e-8),'common-key moment mass changed')
        result.append(dict(scope=s,eligible_episodes=len(base[s]),groups=[dict(key=list(k),moments=v.tolist(),
            facts=meta[k].tolist()) for k,v in values.items()]))
    return dict(actor=actor,day=day,mode=mode,clock_queries=clock_queries,v190_moments_reproduced=True,scopes=result)


def bucket(args,chunk):
    actor,mode = chunk['actor'],chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source']=='a5b45df520','V172 source differs')
    geometry = prior.clocks.native.load_geometry(args.runtime)
    saved = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
    require(saved['source']=='a7d126966a','V190 source differs')
    lookup = {r['day']:r for r in saved['rows']}
    admitted = {s:{tuple(k['key']) for k in read(args.v191_root/'details'/f'{mode}_{s}.json')['common_key_support']
        if k['actor']==actor} for s in SCOPES}
    rows = []
    for i,day in enumerate(DAYS,1):
        rows.append(scan(args,actor,day,mode,table,geometry,lookup[day],admitted))
        if i%8==0:
            print('V194_CLOCK_DAYS',actor,mode,i,'/64',flush=True)
    for s in SCOPES:
        for k in admitted[s]:
            present = [day for day in DAYS if any(tuple(g['key'])==k for g in
                next(c for c in rows[DAYS.index(day)]['scopes'] if c['scope']==s)['groups'])]
            require(sum(d in DAYS[:32] for d in present)>=19 and sum(d in DAYS[32:] for d in present)>=19,
                'component admission lost19-day support in a half')
    write(args.out_root/'cells'/f'{actor}_{mode}.json',dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],rows=rows))
    return dict(actor=actor,mode=mode,paths=64,clock_queries=sum(r['clock_queries'] for r in rows),
        admitted_keys={s:len(v) for s,v in admitted.items()},v190_moments_reproduced=True)


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v194_*.py','-v'],check=True)
    parent = read(args.v191_root/'summary/status.json')
    require(parent['source']=='e9efbe9bae' and parent['protocol']['calibration_days']==list(DAYS) and
        parent['v190_points_reproduced'] and not parent['map_updates'],'requires frozen V191 calibration cohort')
    model = read(args.v172_root/'model.json')
    require(model['execution_source']=='a5b45df520' and model['calibration_days']==list(DAYS) and
        len(model['actors'])==12,'V172 actor/day grid differs')
    require(not (args.out_root/'protocol.json').exists(),'do not overwrite mechanism registration')
    plan = registration()
    write(args.out_root/'protocol.json',plan)
    print('V194_MECHANISM_PROTOCOL_SEALED_NO_ASSOCIATION_READOUT',flush=True)
    actor = min(model['actors'])
    preflight = []
    for mode in ('old','handoff'):
        chunk = next(c for c in model['chunks'] if (c['actor'],c['mode'])==(actor,mode))
        table = read(args.v172_root/chunk['map_file'])
        old = read(args.v190_root/'cells'/f'{actor}_{mode}.json')
        admitted = {s:{tuple(k['key']) for k in read(args.v191_root/'details'/f'{mode}_{s}.json')['common_key_support']
            if k['actor']==actor} for s in SCOPES}
        for day in (DAYS[0],DAYS[-1]):
            preflight.append(scan(args,actor,day,mode,table,prior.clocks.native.load_geometry(args.runtime),
                next(r for r in old['rows'] if r['day']==day),admitted))
    require(sum(r['clock_queries'] for r in preflight)>0,'no preflight clock evidence')
    write(args.out_root/'preflight/status.json',dict(status='v194_temporal_preflight_passed',paths=4,
        clock_queries=sum(r['clock_queries'] for r in preflight),v190_moments_reproduced=True))
    print('V194_TEMPORAL_PREFLIGHT_PASSED',flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(bucket,args,c) for c in model['chunks']]),1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}'%i,flush=True)
    write(args.out_root/'summary/status.json',dict(status='v194_mechanism_temporal_admission_complete',source=plan['source'],
        protocol=plan,paths=sum(r['paths'] for r in results),clock_queries=sum(r['clock_queries'] for r in results),
        results=sorted(results,key=lambda r:(r['actor'],r['mode'])),preflight_paths=4,
        v190_moments_reproduced=True,common_support_19_each_half_preserved=True,component_additivity_checked=True,
        source_and_label_clocks_checked=True,interval_fit=False,predictor_fit=False,mechanism_associations_estimated=False,
        confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,map_updates=0,main_policy_changed=False))
    print('V194_TEMPORAL_ADMISSION_COMPLETE_NO_ASSOCIATIONS_OR_FIT',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','runtime','v166-root','v172-root','v190-root','v191-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    run(parser.parse_args())
