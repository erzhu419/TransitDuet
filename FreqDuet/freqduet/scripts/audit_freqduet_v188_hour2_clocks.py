#!/usr/bin/env python3
"""Read original64 archives: hour2 predictor absorption and completion-clock lag."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix

import audit_freqduet_v187_running_increment as prior
import audit_freqduet_v120_component_eta as native


read, write, require, close = prior.read, prior.write, prior.require, prior.close
DAYS = prior.DAYS
VARS = ('prefix_running', 'prefix_service', *prior.TARGETS, 'predicted_running_s', 'truth_running_s',
    'query_offset_min', 'remaining_km', 'prediction_horizon_min', 'prefix_query_age_min',
    'prefix_label_delay_min', 'prefix_before_hour', 'prefix_same_stage', 'native_latest_age_min', 'native_before_hour')
N = len(VARS)
TRI = np.triu_indices(N)
SIZE = 1+N+len(TRI[0])
FACTS = ('prefix_completion_age_s', 'prefix_query_age_s', 'prefix_label_delay_s',
    'prefix_ready_after_arrival_s', 'prefix_before_hour', 'prefix_same_stage',
    'native_latest_age_s', 'native_mean_completion_age_s', 'native_before_hour',
    'native_fallback_fraction', 'prediction_horizon_s', 'predicted_running_s', 'truth_running_s')
CONTROLS = dict(service_only=('prefix_service',),
    query_state=('prefix_service', 'query_offset_min', 'remaining_km', 'prediction_horizon_min'),
    completed_clocks=('prefix_service', 'query_offset_min', 'remaining_km', 'prediction_horizon_min',
        'prefix_query_age_min', 'prefix_label_delay_min', 'prefix_before_hour', 'prefix_same_stage',
        'native_latest_age_min', 'native_before_hour'),
    forecast_level=('prefix_service', 'predicted_running_s'))
TARGETS = ('underestimate_s', 'running_underestimate_s', 'above', 'below', 'score_s')


def moment(v):
    v = np.asarray(v)
    return np.r_[1., v, (v[:, None]*v[None, :])[TRI]]


def unpack(m):
    matrix = np.zeros((*m.shape[:-1], N, N))
    matrix[..., TRI[0], TRI[1]] = m[..., 1+N:]
    matrix[..., TRI[1], TRI[0]] = m[..., 1+N:]
    return matrix


def centered(m):
    mass, sums = m[..., 0], m[..., 1:1+N]
    outer = sums[..., :, None]*sums[..., None, :]
    correction = np.divide(outer, mass[..., None, None], out=np.zeros_like(outer), where=mass[..., None, None] > 0.)
    return unpack(m)-correction


def adjusted(cov, controls):
    indices = [VARS.index(k) for k in controls]
    cc = cov[..., indices, :][..., :, indices]
    sd = np.sqrt(np.maximum(np.diagonal(cc, axis1=-2, axis2=-1), 0.))
    scale = sd[..., :, None]*sd[..., None, :]
    corr = np.divide(cc, scale, out=np.zeros_like(cc), where=scale > 1e-12)
    cross = np.divide(cov[..., :, indices], sd[..., None, :],
        out=np.zeros_like(cov[..., :, indices]), where=sd[..., None, :] > 1e-12)
    residual = cov-cross@np.linalg.pinv(corr, rcond=1e-10)@np.swapaxes(cross, -1, -2)
    return residual


def correlation(cov, target):
    i, j = VARS.index('prefix_running'), VARS.index(target)
    den = np.sqrt(np.maximum(cov[..., i, i], 0.)*np.maximum(cov[..., j, j], 0.))
    return np.divide(cov[..., i, j], den, out=np.full_like(den, np.nan), where=den > 1e-12)


class PrefixClocks:
    def __init__(self, atoms, extras):
        self.events = sorted([(a[1], a[0], a[3], extras[ep]) for ep, a in atoms.items()], key=lambda a:a[:2])
        self.index, self.pool, self.total = 0, deque(), np.zeros(10)
        self.stages = Counter()

    def query(self, now, stage):
        while self.index < len(self.events) and self.events[self.index][0] < now:
            available, order, errors, extra = self.events[self.index]
            p, t, arrival, first_stage = extra
            v = np.r_[errors, p, t, order[0], available, available-arrival, float(order[0] < 7200.), available-order[0]]
            self.pool.append((available, v, first_stage))
            self.total += v
            self.stages[first_stage] += 1
            self.index += 1
        while self.pool and self.pool[0][0] < now-3600.:
            _, v, first_stage = self.pool.popleft()
            self.total -= v
            self.stages[first_stage] -= 1
        n = len(self.pool)
        require(not n or now-3600. <= self.pool[0][0] <= self.pool[-1][0] < now, 'unfinished/out-of-window prefix')
        if n < 9:
            return dict(n=n, latest=None if not n else self.pool[-1][0], mean=None)
        m = self.total/n
        age, query_age, delay = now-m[6], now-m[5], m[9]
        close(query_age, age+delay)
        return dict(n=n, latest=self.pool[-1][0], mean=m[:3], predicted_running_s=m[3], truth_running_s=m[4],
            prefix_completion_age_s=age, prefix_query_age_s=query_age, prefix_label_delay_s=delay,
            prefix_ready_after_arrival_s=m[7], prefix_before_hour=m[8], prefix_same_stage=self.stages[stage]/n)


def events(root, name, geometry):
    with np.load(root/'records'/(name+'.npz')) as z:
        fields, launch_fields = z['event_fields'].tolist(), z['launch_fields'].tolist()
        arrivals = [dict(zip(fields, r.tolist())) for r in z['arrivals']]
        departures = {key(dict(zip(fields, r.tolist()))): dict(zip(fields, r.tolist())) for r in z['departures']}
        launches = {int(r[0]): dict(zip(launch_fields, r.tolist())) for r in z['launches']}
    rows = defaultdict(list)
    for a in arrivals:
        trip, station, direction = key(a)
        route = geometry[direction]
        i = route['index'][station]
        require(i > 0, 'ordinary running label at terminal')
        before = departures.get((trip, route['ids'][i-1], direction))
        require(before is not None or i == 1, 'missing upstream departure')
        start = launches[trip]['time_s'] if before is None else before['time_s']
        rows[direction, station].append((a['time_s'], start, a['time_s']-start))
    history = native.CompletedMeans([(('running', d, s), end, value)
        for (d, s), values in rows.items() for end, start, value in values])
    return {key(a): a['time_s'] for a in arrivals}, {k:sorted(v) for k,v in rows.items()}, history


def key(row):
    return int(row['trip_id']), int(row['station_id']), bool(row['direction'])


def absorbed(q, geometry, rows, history):
    p, now = q['observable']['position'], q['query_time_s']
    route = geometry[bool(q['condition_values']['direction'])]
    supports = {r['station']:r['components']['running'] for r in q['support']['completed_components']}
    predicted, metadata = 0., []
    for i in range(p['origin_index']+1, p['target_index']+1):
        station, direction = route['ids'][i], bool(q['condition_values']['direction'])
        value = history.estimate(('running', direction, station), now)
        require(value is not None and value['latest'] < now, 'unfinished native running input')
        saved = supports[station]
        require(value['count'] == saved['count'] and value['latest'] == saved['latest'] and
            value['historical_fallback'] == saved['historical_fallback'], 'native support clocks/counts differ')
        close(value['mean'], saved['mean'])
        fraction = (route['positions'][i]-p['progress_m'])/(route['positions'][i]-route['positions'][i-1]) if i == p['origin_index']+1 else 1.
        require(-1e-8 <= fraction <= 1.+1e-8, 'remaining running geometry differs')
        predicted += fraction*value['mean']
        samples = rows[direction, station]
        times = [r[0] for r in samples]
        left, right = bisect_left(times, now-3600.), bisect_left(times, now)
        if left == right:
            left = 0
        selected = samples[left:right]
        require(len(selected) == value['count'], 'native duration cohort differs')
        metadata.append((now-value['latest'], np.mean([now-r[0] for r in selected]),
            np.mean([r[1] < 7200. for r in selected]), float(value['historical_fallback'])))
    close(predicted, q['prediction_coarse']['running'])
    if not metadata:
        return None
    means = np.mean(metadata, axis=0)
    return dict(zip(('native_latest_age_s', 'native_mean_completion_age_s', 'native_before_hour',
        'native_fallback_fraction'), means.tolist()))


def scan(args, actor, day, mode, table, geometry, old):
    require(day in DAYS, 'only original64 calibration days')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
        ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'], 'capture differs')
    arrivals, running, history = events(root, name, geometry)
    entries = {tuple(e['key']):e for e in table['candidate']}
    atoms, extras, selected, records = {}, {}, [], 0
    for q, r in zip_longest(*(prior.previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and q['episode'] == r['episode'] and
            q['query_time_s'] == r['query_time_s'] and (q['actor'],q['day'],q['mode']) == (actor,day,mode), 'stream join differs')
        records += 1
        ep = tuple(q['episode'])
        before = atoms.get(ep)
        prior.previous.add_first(atoms, q, r)
        if ep in atoms and atoms[ep] is not before:
            arrival = arrivals[ep[0], ep[2], bool(ep[3])]
            require(arrival <= r['label']['available_time_s'], 'running arrival follows ready label')
            extras[ep] = (q['prediction_coarse']['running'], r['truth_coarse']['running'], arrival, q['native_stage']['kind'])
        if not q['denominator_eligible'] or q['observable']['hour'] != 2:
            continue
        period = prior.temporal.STRATA[int(prior.temporal.strata(np.array([q['query_time_s']]))[0])]
        entry = entries.get((*q['candidate_key'],period)) if q['native_supported'] else None
        bounds = prior.previous.saved.fit.interval(q['point_s'],entry,q['native_supported'])
        y = None
        if bounds is not None:
            require(r['label']['status'] == 'resolved' and r['label']['available_time_s'] ==
                r['label']['target_ready_time_s'] >= q['query_time_s'] and r['residual_s'] is not None, 'label clock differs')
            close(sum(r['residual_s'].values()), q['point_s']-r['label']['full_eta_s'])
            y = prior.outcomes(q['point_s'],r['label']['full_eta_s'],r['residual_s']['running'],bounds)
        selected.append((q,r,y,prior.previous.prior.features(q,entry)))
    require(records == receipt['query_record_count'], 'query cohort incomplete')
    prefix, reference = PrefixClocks(atoms,extras), prior.previous.Prefix(atoms)
    base, original, groups, facts = {}, defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for q,r,y,f in sorted(selected,key=lambda v:(v[0]['query_time_s'],v[0]['clocks']['query_ready_order'])):
        now, ep = q['query_time_s'],tuple(q['episode'])
        h = prefix.query(now,q['native_stage']['kind'])
        ref = reference.query(now,None)['global']
        require(h['n'] == ref['n'] and h['latest'] == ref['latest'], 'completed-prefix cohort/clock differs')
        if h['mean'] is not None:
            require(np.allclose(h['mean'],ref['mean'],rtol=0.,atol=1e-8), 'completed-prefix vector differs')
        row = base.setdefault(ep,np.zeros(2+len(prior.TARGETS)))
        row[:2] += (1.,float(y is not None))
        if y is None:
            continue
        row[2:] += y
        if h['mean'] is None:
            continue
        x = -h['mean'][:2]
        old_vector = prior.moment(x,y)
        for level, fields in prior.PARTITIONS.items():
            identity = level,tuple(f[k] for k in fields)
            original[ep].setdefault(identity,np.zeros(prior.SIZE))[:] += old_vector
        meta = absorbed(q,geometry,running,history)
        if meta is None:
            continue
        v = dict(prefix_running=x[0],prefix_service=x[1],**dict(zip(prior.TARGETS,y)),
            predicted_running_s=q['prediction_coarse']['running'],truth_running_s=r['truth_coarse']['running'],
            query_offset_min=(now-7200.)/60., remaining_km=q['observable']['position']['remaining_distance_m']/1000.,
            prediction_horizon_min=q['point_s']/60., prefix_query_age_min=h['prefix_query_age_s']/60.,
            prefix_label_delay_min=h['prefix_label_delay_s']/60.,prefix_before_hour=h['prefix_before_hour'],
            prefix_same_stage=h['prefix_same_stage'],native_latest_age_min=meta['native_latest_age_s']/60.,
            native_before_hour=meta['native_before_hour'])
        vector = moment([v[k] for k in VARS])
        anatomy = {**h,**meta,'prediction_horizon_s':q['point_s'],
            'predicted_running_s':v['predicted_running_s'],'truth_running_s':v['truth_running_s']}
        for level, fields in (('geometry_time',prior.states.STATE),('geometry_quarter',(*prior.states.STATE,'quarter'))):
            f['quarter'] = int((now-7200.)//900.)
            identity = level,tuple(f[k] for k in fields)
            groups[ep].setdefault(identity,np.zeros(SIZE))[:] += vector
        for axis,value in (('all','all'),('stage',f['stage']),('quarter',str(f['quarter']))):
            facts[ep].setdefault((axis,value),np.zeros(1+len(FACTS)))[:] += np.r_[1.,[anatomy[k] for k in FACTS]]
    nf = sum(v[1] > 0 for v in base.values())
    require(nf > 0 and len(base) > 0, 'empty original hour2 cell')
    def totals(source):
        result = {}
        for ep,groups in source.items():
            for k,v in groups.items():
                result.setdefault(k,np.zeros_like(v))[:] += v/(base[ep][1]*nf)
        return result
    original = totals(original)
    expected = next(row for row in old['rows'] if row['day'] == day and row['scope'] == 'hour2')
    want = {(g['level'],tuple(g['key'])):g['moments'] for g in expected['groups']}
    require(set(original) == set(want), 'V187 fixed-state cohort differs')
    for k,v in original.items():
        require(np.allclose(v,want[k],rtol=1e-9,atol=1e-7), 'V187 weighted moments differ')
    endpoint = np.mean([v[2:]/v[1] for v in base.values() if v[1] > 0],axis=0)
    require(np.allclose(endpoint,expected['endpoint'],rtol=0.,atol=1e-8), 'V187 endpoint differs')
    close(np.mean([v[1]/v[0] for v in base.values()]),expected['finite_fraction'])
    return dict(actor=actor,day=day,mode=mode,
        groups=[dict(level=l,key=list(k),moments=v.tolist()) for (l,k),v in totals(groups).items()],
        facts=[dict(axis=a,value=k,values=v.tolist()) for (a,k),v in totals(facts).items()])


def bucket(args,model,chunk):
    actor,mode = chunk['actor'],chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source'] == 'a5b45df520', 'V172 source differs')
    old = read(args.v187_cells_root/'cells'/f'{actor}_{mode}.json')
    require(old['source'] == '920866629e','V187 extraction source differs')
    geometry = native.load_geometry(args.runtime)
    rows = []
    for i,day in enumerate(DAYS,1):
        rows.append(scan(args,actor,day,mode,table,geometry,old))
        print('V188_HOUR2_DAY_READ',actor,mode,i,'/64',flush=True)
    require(read(args.v172_root/chunk['map_file']) == table,'frozen V172 changed')
    write(args.out_root/'cells'/f'{actor}_{mode}.json',dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],rows=rows))


def summarize(rows,actors,days,level):
    index = {(r['actor'],r['day']):r for r in rows}
    require(len(index) == len(rows) and set(index) == set(product(actors,days)),'actor/day grid incomplete')
    keys = sorted({tuple(g['key']) for r in rows for g in r['groups'] if g['level'] == level},key=str)
    require(bool(keys),'no native running support; do not substitute zero')
    positions = {k:i for i,k in enumerate(keys)}
    matrix = np.zeros((len(rows),len(keys),SIZE))
    for i,k in enumerate(product(actors,days)):
        for g in index[k]['groups']:
            if g['level'] == level:
                matrix[i,positions[tuple(g['key'])]] = g['moments']
    weights = prior.previous.prior.bootstrap_weights(len(actors),len(days))
    mean, flat = matrix.mean(axis=0),csr_matrix(matrix.reshape(len(rows),-1))
    cov = centered(mean).sum(axis=0)
    boot = {k:[] for k in CONTROLS}
    for start in range(0,len(weights),32):
        draws = (flat.T@weights[start:start+32].T).T.reshape(-1,len(keys),SIZE)
        c = centered(draws).sum(axis=1)
        for block,controls in CONTROLS.items():
            adjusted_cov = adjusted(c,controls)
            boot[block].append(np.stack([correlation(adjusted_cov,t) for t in TARGETS],axis=-1))
    result = {}
    for block,controls in CONTROLS.items():
        c,draws = adjusted(cov,controls),np.concatenate(boot[block])
        x,e,t,p = map(VARS.index,('prefix_running','running_underestimate_s','truth_running_s','predicted_running_s'))
        close(c[x,e],c[x,t]-c[x,p])
        result[block] = dict(controls=list(controls),estimates={target:prior.estimate(correlation(c,target),draws[:,i])
            for i,target in enumerate(TARGETS)},covariance_identity=dict(error=float(c[x,e]),
                truth=float(c[x,t]),prediction=float(c[x,p])))
    support = matrix[:,:,0].reshape(len(actors),len(days),len(keys)) > 0.
    counts = support.any(axis=0).sum(axis=0)
    return dict(level=level,strata=len(keys),running_clock_mass=float(mean[:,0].sum()),
        below19_day_mass=float(mean[counts < 19,0].sum()),blocks=result)


def facts_summary(rows,actors,days):
    index = {(r['actor'],r['day']):r for r in rows}
    identities = sorted({(v['axis'],v['value']) for r in rows for v in r['facts']})
    weights = prior.previous.prior.bootstrap_weights(len(actors),len(days))
    result = []
    for identity in identities:
        matrix = np.zeros((len(rows),1+len(FACTS)))
        for i,k in enumerate(product(actors,days)):
            found = [v for v in index[k]['facts'] if (v['axis'],v['value']) == identity]
            if found:
                matrix[i] = found[0]['values']
        mean,draws = matrix.mean(axis=0),weights@matrix
        valid = (draws[:,0] > 0.).all()
        result.append(dict(axis=identity[0],value=identity[1],score_mass=float(mean[0]),
            means={k:None if mean[0] == 0. else float(mean[i+1]/mean[0]) for i,k in enumerate(FACTS)},
            ci95={k:None if not valid else np.percentile(draws[:,i+1]/draws[:,0],[2.5,97.5]).tolist()
                for i,k in enumerate(FACTS)}))
    return result


def reduce_mode(args,model,mode):
    actors,rows = sorted(model['actors']),[]
    for actor in actors:
        b = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(b['source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],'bucket source differs')
        rows.extend(b['rows'])
    results = []
    for split,days in (('all64',DAYS),('first32',DAYS[:32]),('last32',DAYS[32:])):
        selected = [r for r in rows if r['day'] in days]
        for level in ('geometry_time','geometry_quarter'):
            results.append(dict(mode=mode,split=split,**summarize(selected,actors,days,level)))
        print('V188_CROSSED_SUMMARY',mode,split,flush=True)
    anatomy = facts_summary(rows,actors,DAYS)
    write(args.out_root/'details'/f'{mode}.json',dict(results=results,anatomy=anatomy))
    return dict(mode=mode,results=results,anatomy=anatomy)


def run(args):
    require(args.out_root not in (args.v166_root,args.v172_root,args.v187_cells_root),'output must be separate')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v188_*.py','-v'],check=True)
    model = prior.parents(args)
    # Validate a real archive join before launching all24 extraction buckets.
    chunk = model['chunks'][0]
    geometry = native.load_geometry(args.runtime)
    scan(args,chunk['actor'],DAYS[0],chunk['mode'],read(args.v172_root/chunk['map_file']),geometry,
        read(args.v187_cells_root/'cells'/f"{chunk['actor']}_{chunk['mode']}.json"))
    print('V188_READONLY_PREFLIGHT_PASSED',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for i,job in enumerate(as_completed([pool.submit(bucket,args,model,c) for c in model['chunks']]),1):
            job.result()
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}'%i,flush=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        results = [job.result() for job in [pool.submit(reduce_mode,args,model,m) for m in ('old','handoff')]]
    require(prior.parents(args) == model,'frozen parent changed')
    write(args.out_root/'summary/status.json',dict(status='v188_hour2_clocks_complete',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],paths=1536,calibration_days=list(DAYS),
        v187_hour2_endpoints_and_moments_reproduced=True,native_running_predictions_and_support_reproduced=True,
        bootstrap_draws=5000,bootstrap_seed=20260803,results=results,interval_fit=False,predictor_fit=False,
        map_updates=0,confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,
        main_policy_changed=False,interpretation='Read-only association decomposition,not causal effects of changing a predictor.'))
    print('V188_HOUR2_CLOCKS_COMPLETE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root','v166-root','v166-audit-root','v172-root','v187-cells-root','runtime'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    run(p.parse_args())
