#!/usr/bin/env python3
"""Preregister and audit road/entry-hour innovation support on original64 only."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v188_hour2_clocks as prior


read, write, require, close = prior.read, prior.write, prior.require, prior.close
DAYS, WINDOW, MIN_DAYS = prior.DAYS, 3600., 19
SCOPES = prior.prior.temporal.SCOPES
FIELDS = ('first_leg', 'at_least_one', 'at_least_nine', 'observed_entry',
    'predicted_entry', 'same_native_cohort', 'completed_legs')


def registration():
    return dict(stage='v189_leg_innovation_support', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), parents=dict(v188='80cdc761f7', v172='a5b45df520', capture='20d914af03'),
        frozen=dict(v151=True, v146=True, v172=True, main=True, window_s=WINDOW, tails=[.1,.1]),
        unit='one physical trip/direction/destination-stop road traversal, not repeated ETA queries',
        innovation='I_e=duration_e-mu_road(entry_e); mu uses only arrivals strictly before entry_e',
        availability='arrival_e, not target ready; no entry baseline means unavailable, never zero',
        query_key=['actor', 'mode', 'direction', 'first_remaining_leg_destination', 'reference_entry_hour'],
        entry_clock='observed physical entry only if strictly before query; otherwise query+recorded initial residence',
        selection='same road/direction/reference-entry-hour, arrival in [query-3600,query); no pooled/fallback signal',
        absorption='J(q)=mean(I_e)-mean(mu_road(q)-mu_road(entry_e))=mean(duration_e)-mu_road(q)',
        identity='if the matched cohort equals the entire native cohort, J(q)=0; do not count I as new information',
        support=dict(distinct_days=MIN_DAYS, counts=[1,9],
            nine='diagnostic stricter support count, not a new native minimum or a released estimator',
            abstention='missing entry baseline, no matched completed leg, or <19 contributing days remain unavailable'),
        weights='within each scope: equal query within original physical follower/target episode, '
            'equal episode per path, equal actor and day; support days counted separately for each actor/mode/key',
        inference='support metadata only; no residual-target correlations, outcome scoring, bootstrap, '
            'bias correction, interval construction or candidate ranking in V189',
        next='Only if support is adequate, preregister a separate incremental-information audit; '
            'do not promote a first-leg support result to a full-ETA result',
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0)


class Innovations:
    def __init__(self, rows, history):
        groups = defaultdict(list)
        self.total = self.with_baseline = 0
        for (direction, station), values in rows.items():
            for end, start, duration in values:
                self.total += 1
                baseline = history.estimate(('running', direction, station), start)
                if baseline is None:
                    continue
                require(baseline['latest'] < start <= end, 'pre-entry forecast includes unfinished input')
                self.with_baseline += 1
                groups[direction, station, int(start//3600)].append((end,duration,baseline['mean']))
        self.groups = {}
        for k, values in groups.items():
            values.sort()
            a = np.array(values)
            self.groups[k] = (a[:,0].tolist(),np.vstack((np.zeros(2),np.cumsum(a[:,1:],axis=0))))

    def query(self, key, now, current_mean):
        if key not in self.groups:
            return None
        times, sums = self.groups[key]
        left, right = bisect_left(times,now-WINDOW), bisect_left(times,now)
        if left == right:
            return None
        duration, baseline = (sums[right]-sums[left])/(right-left)
        require(now-WINDOW <= times[left] <= times[right-1] < now, 'innovation availability clock differs')
        innovation, absorbed = duration-baseline, current_mean-baseline
        remaining = duration-current_mean
        close(innovation-absorbed,remaining)
        return dict(n=right-left, latest=times[right-1], innovation=innovation,
            absorbed=absorbed, remaining=remaining)


def visible_entry(now, initial_residence, recorded_entry):
    require(initial_residence >= 0., 'negative query residence')
    if recorded_entry is not None and recorded_entry < now:
        return recorded_entry, 'observed_entry'
    return now+initial_residence, 'predicted_entry'


def entry_lookup(root, name, geometry):
    with np.load(root/'records'/(name+'.npz')) as z:
        fields, launch_fields = z['event_fields'].tolist(),z['launch_fields'].tolist()
        arrivals = [dict(zip(fields,r.tolist())) for r in z['arrivals']]
        departures = {prior.key(dict(zip(fields,r.tolist()))):dict(zip(fields,r.tolist())) for r in z['departures']}
        launches = {int(r[0]):dict(zip(launch_fields,r.tolist())) for r in z['launches']}
    entries = {}
    for a in arrivals:
        trip,station,direction = k = prior.key(a)
        i,route = geometry[direction]['index'][station],geometry[direction]
        require(i > 0 and k not in entries, 'duplicate/terminal road label')
        before = departures.get((trip,route['ids'][i-1],direction))
        require(before is not None or i == 1, 'missing adjacent road entry')
        start = launches[trip]['time_s'] if before is None else before['time_s']
        require(start <= a['time_s'], 'negative road duration')
        entries[k] = (start,a['time_s'])
    return entries


def scan(args, actor, day, mode, table, geometry):
    require(day in DAYS, 'only original64 calibration days')
    root,name = args.v166_root/'calibration',f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'],receipt['seed'],receipt['day'],receipt['mode']) ==
        ('20d914af03',actor,day,mode) and not receipt['interval_fit'] and
        not receipt['main_policy_changed'], 'capture differs')
    _,rows,history = prior.events(root,name,geometry)
    entries,innovation = entry_lookup(root,name,geometry),Innovations(rows,history)
    mapping = {tuple(e['key']):e for e in table['candidate']}
    base,groups = {s:{} for s in SCOPES},{s:defaultdict(dict) for s in SCOPES}
    records,identities = 0,set()
    for q in prior.prior.previous.stream(root/'queries'/(name+'.jsonl.gz')):
        records += 1
        require((q['actor'],q['day'],q['mode']) == (actor,day,mode) and
            tuple(q['query_key']) not in identities, 'duplicate/wrong query identity')
        identities.add(tuple(q['query_key']))
        if not q['denominator_eligible']:
            continue
        now,ep = q['query_time_s'],tuple(q['episode'])
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
        scopes = ['all',q['condition_values']['query_phase']]
        if q['observable']['hour'] in (2,13):
            scopes.append('hour'+str(q['observable']['hour']))
        if q['observable']['hour'] == 13 and q['native_stage']['kind'] == 'moving-launch':
            scopes.append('hour13_moving_launch')
        period = prior.prior.temporal.STRATA[int(prior.prior.temporal.strata(np.array([now]))[0])]
        entry = mapping.get((*q['candidate_key'],period)) if q['native_supported'] else None
        finite = bool(entry is not None and entry['status'] == 'finite')
        if finite:
            require(entry['day_count'] >= MIN_DAYS, 'original V172 support weakened')
        vector,k = np.zeros(len(FIELDS)),None
        p = q['observable']['position']
        if finite and p is not None and p['origin_index'] < p['target_index']:
            direction = bool(q['condition_values']['direction'])
            station = geometry[direction]['ids'][p['origin_index']+1]
            current = history.estimate(('running',direction,station),now)
            saved = next(s['components']['running'] for s in q['support']['completed_components'] if s['station'] == station)
            require(current is not None and current['latest'] < now and
                all(current[c] == saved[c] for c in ('count','latest','historical_fallback')), 'native support differs')
            close(current['mean'],saved['mean'])
            physical = entries.get((ep[0],station,direction))
            require(physical is not None and physical[1] >= now, 'first remaining leg already completed/missing')
            clock,kind = visible_entry(now,p['initial_residence_prediction_s'],physical[0])
            k = int(direction),station,int(clock//3600)
            value = innovation.query(k,now,current['mean'])
            vector[0],vector[FIELDS.index(kind)] = 1.,1.
            if value is not None:
                vector[1:3],vector[-1] = (1.,float(value['n'] >= 9)),value['n']
                require(value['n'] <= current['count'], 'innovation cohort exceeds native raw cohort')
                if value['n'] == current['count']:
                    close(value['remaining'],0.)
                    vector[FIELDS.index('same_native_cohort')] = 1.
        for scope in scopes:
            base[scope].setdefault(ep,np.zeros(3))[:] += (1.,float(q['native_supported']),float(finite))
            if k is not None:
                groups[scope][ep].setdefault(k,np.zeros(len(FIELDS)))[:] += vector
    require(records == receipt['query_record_count'], 'query capture incomplete')
    expected = {(r['day'],r['scope']):r for r in table['cells']}
    results = []
    for scope in SCOPES:
        episodes = base[scope]
        finite_fraction = None if not episodes else float(np.mean([v[2]/v[0] for v in episodes.values()]))
        old = expected[day,scope]
        require(len(episodes) == old['eligible_episodes'] and
            sum(int(v[0]) for v in episodes.values()) == old['eligible_queries'], 'original denominator differs')
        if finite_fraction is None:
            require(old['finite_fraction'] is None, 'empty scope gained support')
        else:
            close(finite_fraction,old['finite_fraction'])
        totals = {}
        for ep,values in groups[scope].items():
            for k,v in values.items():
                totals.setdefault(k,np.zeros(len(FIELDS)))[:] += v/(episodes[ep][0]*len(episodes))
        results.append(dict(scope=scope,eligible_episodes=len(episodes),finite_fraction=finite_fraction,
            groups=[dict(key=list(k),values=v.tolist()) for k,v in totals.items()]))
    return dict(actor=actor,day=day,mode=mode,queries=records,physical_legs=innovation.total,
        pre_entry_supported_legs=innovation.with_baseline,scopes=results)


def bucket(args, model, chunk):
    actor,mode = chunk['actor'],chunk['mode']
    geometry = prior.native.load_geometry(args.runtime)
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source'] == 'a5b45df520', 'V172 source differs')
    rows = []
    for i,day in enumerate(DAYS,1):
        rows.append(scan(args,actor,day,mode,table,geometry))
        if i % 8 == 0:
            print('V189_SUPPORT_DAYS',actor,mode,i,'/64',flush=True)
    write(args.out_root/'cells'/f'{actor}_{mode}.json',dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],rows=rows))


def summarize(rows, actors, mode, scope):
    require({(r['actor'],r['day']) for r in rows} == set(product(actors,DAYS)) and
        len(rows) == len(actors)*len(DAYS), 'incomplete/duplicate actor/day support grid')
    days,total,blocked = defaultdict(set),np.zeros(len(FIELDS)),np.zeros(2)
    rates,denominators,index = [],[],{(r['actor'],r['day']):r for r in rows}
    for row in rows:
        cell = next(c for c in row['scopes'] if c['scope'] == scope)
        if not cell['eligible_episodes']:
            continue
        v = np.zeros(len(FIELDS))
        for g in cell['groups']:
            v += g['values']
            for j in (1,2):
                if g['values'][j] > 0.:
                    days[row['actor'],tuple(g['key']),j].add(row['day'])
        rates.append((row['actor'],row['day'],v))
        denominators.append(cell['finite_fraction'])
        total += v/(len(actors)*len(DAYS))
    for actor,day,v in rates:
        row = index[actor,day]
        cell = next(c for c in row['scopes'] if c['scope'] == scope)
        for g in cell['groups']:
            for i,j in enumerate((1,2)):
                if len(days.get((actor,tuple(g['key']),j),())) < MIN_DAYS:
                    blocked[i] += g['values'][j]/(len(actors)*len(DAYS))
    def marginal(axis):
        ids = actors if axis == 0 else DAYS
        return [dict(id=k,nonempty_paths=sum(r[axis] == k for r in rates),
            rates=dict(zip(FIELDS,np.mean([r[2] for r in rates if r[axis] == k],axis=0).tolist())))
            for k in ids if any(r[axis] == k for r in rates)]
    complete = len(rates) == len(rows)
    return dict(mode=mode,scope=scope,nonempty_paths=len(rates),complete_crossed_grid=complete,
        rates=dict(zip(FIELDS,total.tolist())) if complete else None,
        observed_nonempty_path_rates=None if not rates else
            dict(zip(FIELDS,np.mean([r[2] for r in rates],axis=0).tolist())),
        baseline_finite_fraction=None if not denominators else float(np.mean(denominators)),
        below19_supported_mass=dict(at_least_one=float(blocked[0]),at_least_nine=float(blocked[1])) if complete else None,
        keys_by_count={str(n):dict(observed=sum(k[2] == j and bool(d) for k,d in days.items()),
            at_least19=sum(k[2] == j and len(d) >= MIN_DAYS for k,d in days.items())) for j,n in ((1,1),(2,9))},
        actor_margins=marginal(0),day_margins=marginal(1))


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v189_*.py','-v'],check=True)
    model = prior.prior.parents(args)
    parent = read(args.v188_root/'summary/status.json')
    require(parent['status'] == 'v188_hour2_clocks_complete' and parent['source'] == '80cdc761f7' and
        parent['paths'] == 1536 and parent['native_running_predictions_and_support_reproduced'], 'V188 not complete')
    plan = registration()
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    write(args.out_root/'protocol.json',plan)
    print('V189_PROTOCOL_SEALED_BEFORE_SUPPORT_COUNTS',flush=True)
    c = model['chunks'][0]
    scan(args,c['actor'],DAYS[0],c['mode'],read(args.v172_root/c['map_file']),prior.native.load_geometry(args.runtime))
    print('V189_REAL_ARCHIVE_PREFLIGHT_PASSED',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(bucket,args,model,c) for c in model['chunks']]),1):
            job.result()
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}'%i,flush=True)
    rows = []
    for c in model['chunks']:
        b = read(args.out_root/'cells'/f"{c['actor']}_{c['mode']}.json")
        require(b['source'] == plan['source'], 'bucket source differs')
        rows.extend(b['rows'])
    require(len(rows) == 1536 and prior.prior.parents(args) == model, 'parent or path count changed')
    results = [summarize([r for r in rows if r['mode'] == mode],model['actors'],mode,scope)
        for mode,scope in product(('old','handoff'),SCOPES)]
    write(args.out_root/'summary/status.json',dict(status='v189_leg_innovation_support_complete',source=plan['source'],
        protocol=plan,paths=len(rows),queries=sum(r['queries'] for r in rows),
        physical_legs=sum(r['physical_legs'] for r in rows),
        pre_entry_supported_legs=sum(r['pre_entry_supported_legs'] for r in rows),results=results,
        original_v172_support_reproduced=True,strict_pre_entry_and_arrival_clocks=True,absorption_identity_verified=True,
        interval_fit=False,predictor_fit=False,residual_files_opened=0,confirmation_outcomes_read=False,
        confirmation_started=False,new_replays=0,new_training=0,main_policy_changed=False,
        decision='Support audit only; no estimator or full-ETA mapping released.'))
    print('V189_SUPPORT_COMPLETE_NO_FIT',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v166-root','v166-audit-root','v172-root','v188-root','runtime'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    run(p.parse_args())
