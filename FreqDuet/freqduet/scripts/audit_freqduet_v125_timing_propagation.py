#!/usr/bin/env python3
"""Read the 32 consumed V124 factorial paths; locate timing propagation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_freqduet_v124_eta_factorial as frozen
import run_freqduet_v92_elapsed_review as elapsed


COMPONENTS = elapsed.COMPONENTS
STAGES = ('launch', 'arrival', 'departure')
FOCUS = (frozen.PRIMARY+'_vs_'+frozen.prior.BILATERAL,
    frozen.prior.PRIMARY+'_vs_'+frozen.prior.BILATERAL)
PROFILE_FIELDS = ('cv_squared', 'raw_variance_s2', 'mean_normalization',
    *(n+'_allocated' for n in COMPONENTS))


def edge_rows(points):
    previous, rows = {}, []
    for key, point in points.items():
        group = key[1:]
        if group in previous:
            before, old = previous[group]
            h = point['time_s']-old['time_s']
            if h < 0:
                raise AssertionError('timing events are not chronological within a stop/direction')
            rows.append(dict(key=key, predecessor=before, headway=h,
                coordinates=np.asarray(point['coordinates'])-old['coordinates']))
        previous[group] = key, point
    return rows


def moment(rows):
    if not rows or sum(r['headway'] for r in rows) == 0:
        return None
    return elapsed.covariance_moments([r['headway'] for r in rows], [r['coordinates'] for r in rows])


def effect(base, candidate):
    delta = (candidate['covariance']-base['covariance'])/base['mean']**2
    normalization = candidate['variance']*(1/candidate['mean']**2-1/base['mean']**2)
    result = {n+'_allocated': float(delta[i].sum()) for i, n in enumerate(COMPONENTS)}
    result.update(cv_squared=candidate['cv_squared']-base['cv_squared'],
        raw_variance_s2=candidate['variance']-base['variance'],
        mean_headway_s=candidate['mean']-base['mean'], mean_normalization=normalization,
        release_diagonal=float(delta[0, 0]), holding_diagonal=float(delta[1, 1]),
        release_holding_cross=float(2*delta[0, 1]))
    np.testing.assert_allclose(sum(result[n+'_allocated'] for n in COMPONENTS)+normalization,
        result['cv_squared'], atol=1e-10, rtol=0)
    return result


def contribution(rows, total):
    return sum((r['headway']-total['mean'])**2 for r in rows)/total['samples']/total['mean']**2


def profiles(base, candidate, grouping):
    tables = []
    for rows in (base, candidate):
        table = {}
        for row in rows:
            table.setdefault(grouping(row), []).append(row)
        tables.append(table)
    totals = moment(base), moment(candidate)
    result = []
    for group in sorted(tables[0].keys() | tables[1].keys()):
        b, c = tables[0].get(group, []), tables[1].get(group, [])
        bm, cm = moment(b), moment(c)
        local = effect(bm, cm) if bm is not None and cm is not None else None
        local = {k: local[k] for k in PROFILE_FIELDS} if local is not None else None
        result.append(dict(group=group, base_n=len(b), candidate_n=len(c), effect=local,
            pooled_cv2_delta=contribution(c, totals[1])-contribution(b, totals[0])))
    np.testing.assert_allclose(sum(r['pooled_cv2_delta'] for r in result),
        totals[1]['cv_squared']-totals[0]['cv_squared'], atol=1e-10, rtol=0)
    return result


def first_change(base, candidate, field):
    changed = [k for k in base if base[k][field] != candidate[k][field]]
    if not changed:
        return None
    k = min(changed, key=lambda k: min(base[k]['time_s'], candidate[k]['time_s']))
    b, c = base[k], candidate[k]
    delta = np.asarray(c['coordinates'])-b['coordinates']
    return dict(trip_id=k[0], station_id=k[1], direction=k[2],
        base_time_s=b['time_s'], candidate_time_s=c['time_s'],
        delta={n: float(delta[i]) for i, n in enumerate(COMPONENTS)},
        time_delta_s=c['time_s']-b['time_s'])


def read_path(root, seed, day, method, route):
    pair = json.loads((root/'pairs'/f'{seed}_{day}_{method}.json').read_text())
    with np.load(root/'records'/f'{seed}_{day}_{method}_0.npz') as record:
        fields = record['event_fields'].tolist()
        events = {k: [dict(zip(fields, r.tolist())) for r in record[k]] for k in ('arrivals', 'departures')}
        launches = [dict(zip(record['launch_fields'].tolist(), r.tolist())) for r in record['launches']]
    if (pair['seed'], pair['day'], pair['method']) != (seed, day, method):
        raise AssertionError('consumed timing path identity differs from V124')
    split = elapsed.split_stop_time(events)
    points = {stage: {} for stage in STAGES}
    directions, departure = {}, {elapsed.key(r): r for r in events['departures']}
    holding = 0.
    for row in events['arrivals']:
        k = elapsed.key(row)
        directions[k[0]] = k[2]
        holding += departure[k]['realized_holding']-row['realized_holding']
    for stage in ('arrival', 'departure'):
        for row in events[stage+'s']:
            k = elapsed.key(row)
            if (k, stage) in split:
                points[stage][k] = dict(time_s=row['time_s'], holding_s=row['realized_holding'],
                    coordinates=np.asarray(split[k, stage]))
    vehicle, launch_times, readiness = {}, {}, []
    for row in launches:
        trip = int(row['trip_id'])
        direction = directions[trip]
        k = trip, route[direction]['ids'][0], direction
        points['launch'][k] = dict(time_s=row['time_s'], holding_s=0.,
            coordinates=np.asarray([row['time_s'], 0., 0., 0.]))
        vehicle[trip], launch_times[trip] = int(row['bus_id']), row['time_s']
        readiness.append(row['readiness_s'])
    rows = {stage: edge_rows(p) for stage, p in points.items()}
    moments = {stage: moment(r) for stage, r in rows.items()}
    result = pair['results']
    np.testing.assert_allclose([moments['arrival']['cv_squared'], moments['launch']['cv_squared'],
        moments['launch']['mean'], moments['launch']['variance'], holding/len(launches), np.mean(readiness)],
        [result['cv_squared'], result['dispatch_cv_squared'], result['dispatch_gap_mean_s'],
         result['dispatch_gap_variance_s2'], result['holding_s_per_trip'], result['readiness_delay_s']],
        atol=1e-6, rtol=0)
    return dict(pair=pair, points=points, rows=rows, moments=moments,
        vehicle=vehicle, launch_times=launch_times)


def analyze_cell(v123, v124, out, runtime, index):
    seed, day = frozen.ITEMS[index]
    route = frozen.prior.model.load_geometry(runtime)
    paths = {m: read_path(v124 if m == frozen.PRIMARY else v123, seed, day, m, route) for m in frozen.METHODS}
    reference = paths[frozen.prior.BILATERAL]
    for path in paths.values():
        for stage in STAGES:
            if path['points'][stage].keys() != reference['points'][stage].keys():
                raise AssertionError('paired complete trips do not share timing-event support')
    effects, first, spatial, temporal = {}, {}, {}, {}
    for name, (a, b) in frozen.CONTRASTS.items():
        base, candidate = paths[b], paths[a]
        effects[name], first[name], spatial[name], temporal[name] = {}, {}, {}, {}
        for stage in STAGES:
            bm, cm = base['moments'][stage], candidate['moments'][stage]
            for k, v in effect(bm, cm).items():
                effects[name][stage+'_'+k] = v
            bp, cp = base['points'][stage], candidate['points'][stage]
            first[name][stage] = first_change(bp, cp, 'time_s')
            br, cr = base['rows'][stage], candidate['rows'][stage]
            neighbours = {r['key']: r['predecessor'] for r in br}
            effects[name][stage+'_predecessor_changes'] = sum(
                r['key'] not in neighbours or neighbours[r['key']] != r['predecessor'] for r in cr)
            spatial[name][stage] = profiles(br, cr, lambda r: (int(r['key'][2]),
                route[r['key'][2]]['index'][r['key'][1]], r['key'][1]))
            temporal[name][stage] = profiles(br, cr, lambda r: int(reference['launch_times'][r['key'][0]]//14400))
        changed = [t for t in base['vehicle'] if base['vehicle'][t] != candidate['vehicle'][t]]
        first[name]['vehicle_assignment_changes'] = len(changed)
        first[name]['first_holding_difference'] = first_change(base['points']['departure'], candidate['points']['departure'], 'holding_s')
        expected = candidate['pair']['results']['cv_squared']-base['pair']['results']['cv_squared']
        np.testing.assert_allclose(effects[name]['arrival_cv_squared'], expected, atol=1e-10, rtol=0)
    detail = dict(seed=seed, day=day, effects=effects, first_changes=first, spatial=spatial, temporal=temporal)
    frozen.pilot.write_json(out/'details'/f'p{index}.json', detail)
    # Full profiles remain remote; return the failing day's two forecast contrasts only.
    focus = {name: {stage: spatial[name][stage] for stage in ('launch', 'arrival')}
        for name in FOCUS} if day == frozen.prior.DAYS[-1] else {}
    times = {name: temporal[name] for name in FOCUS}
    return dict(seed=seed, day=day, effects=effects, first_changes=first,
        focus_spatial=focus, temporal=times)


def run(v123, v124, out, runtime, workers):
    parent = json.loads((v124/'summary/status.json').read_text())
    if parent['status'] != 'v124_consumed_eta_factorial_complete' or parent['missing_arm_rollouts'] != 8:
        raise AssertionError('V125 requires the complete V124 consumed factorial')
    start, cells = time.monotonic(), []
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=0, total=len(frozen.ITEMS), unit='cell')), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(analyze_cell, v123, v124, out, runtime, i) for i in range(len(frozen.ITEMS))]
        for future in as_completed(futures):
            cells.append(future.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(cells), total=len(frozen.ITEMS),
                unit='cell', elapsed_seconds=time.monotonic()-start)), flush=True)
    cells.sort(key=lambda r: (r['seed'], r['day']))
    frozen.pilot.write_json(out/'summary/status.json', dict(status='v125_consumed_timing_propagation_complete',
        effects=frozen.prior.inference.describe(cells, 'effects'), actor_day=cells,
        axes=frozen.prior.axes.axis_summary(cells, 'effects', list(cells[0]['effects'][FOCUS[0]])),
        consumed_paths=32, exact_parent_timing_holding_readiness=True,
        spatial_and_cohort_pooled_cv2_reconcile=True, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        cohort='Four-hour bins of raw-bilateral actual launch, anchored by current trip; incoming gaps retained.',
        limitations='Consumed timing/covariance localization, not executed interventions or causal mediation. '
            'Actual neighbours are retained and predecessor changes reported. '
            'Unobserved terminal return timestamps are not reconstructed from last stop departures. '
            'Nonholding stop time includes service and execution ticks; remaining elapsed is not pure driving. '
            'Empty/zero-mean local CV2 is unavailable, never a zero effect. Full profiles stay server-side.'))
    print('V125_TIMING_PROPAGATION_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v123-root', type=Path, required=True)
    p.add_argument('--v124-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v125_*.py', '-v'], check=True)
    run(a.v123_root, a.v124_root, a.out_root, a.runtime_root, a.workers)
