#!/usr/bin/env python3
"""Read only V213 trajectories: plan realization and holding propagation."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np


EPS = 1e-8
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
METRICS = ('journey_s', 'cv_squared', 'readiness_delay_s', 'inclusive_launch_delay_s',
    'passenger_person_s', 'waiting_person_s', 'onboard_person_s')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def key(r):
    return r['trip_id'], r['station_id'], r['direction']


def sign(value):
    return 'unresolved' if value is None else 'positive' if value > EPS else 'negative' if value < -EPS else 'zero'


def moments(events):
    previous, gaps, parts = {}, [], []
    for r in events:
        k = r['station_id'], r['direction']
        if k in previous:
            before = previous[k]
            gaps.append(r['time_s']-before['time_s'])
            parts.append([r[f]-before[f] for f in ('release', 'realized_holding', 'other_elapsed')])
        previous[k] = r
    h, x = np.asarray(gaps), np.asarray(parts)
    np.testing.assert_allclose(x.sum(1), h, atol=1e-8, rtol=0)
    mean = float(h.mean())
    covariance = (x-x.mean(0)).T @ (x-x.mean(0))/len(x)
    variance = float(h.var())
    np.testing.assert_allclose(covariance.sum(), variance, atol=1e-7, rtol=1e-12)
    return dict(mean=mean, covariance=covariance, variance=variance, cv2=variance/mean**2)


def cv_parts(b, c):
    d = (c['covariance']-b['covariance'])/b['mean']**2
    values = dict(release=d[0, 0], holding=d[1, 1], other=d[2, 2],
        release_holding=2*d[0, 1], release_other=2*d[0, 2], holding_other=2*d[1, 2],
        normalization=c['variance']*(1/c['mean']**2-1/b['mean']**2))
    np.testing.assert_allclose(sum(values.values()), c['cv2']-b['cv2'], atol=1e-10, rtol=0)
    return {k: float(v) for k, v in values.items()}


def prepare(data, cached):
    r, t = data['result'], data['trace']
    require(data['archived_endpoints_and_saved_actions_exact'], 'original replay equivalence flag missing')
    require({k: r[k] for k in cached} == cached, 'saved V213 result differs from V211 selection archive')
    histogram = Counter(str(int(q['command_s'])) for q in t['lower_queries'])
    require({str(a): histogram[str(a)] for a in ACTIONS} == r['results']['action_counts'], 'full query action histogram differs')
    require(set(histogram) <= {str(a) for a in ACTIONS}, 'non-native holding command')
    require(sum(q['command_s'] for q in t['lower_queries']) == r['ledger']['executed_holding_s'], 'executed holding does not close')
    queries = {key(q): q for q in t['lower_queries']}
    require(len(queries) == len(t['lower_queries']), 'duplicate lower query identity')
    for change in r['lower']:
        q = queries[tuple(change['identity'][:3])]
        require((q['time_s'], q['bus_id'], q['command_s']) ==
            (change['time_s'], change['identity'][3], change['actual_command_s']), 'registered command not found in full trace')
    m = moments(t['arrivals'])
    np.testing.assert_allclose(m['cv2'], r['results']['cv_squared'], atol=1e-10, rtol=0)
    launch = {v['trip']: v for v in t['actual_launches']}
    directions = {v['trip_id']: v['direction'] for v in t['upper_callbacks']}
    gaps, previous = [], {}
    for v in t['actual_launches']:
        direction = directions[v['trip']]
        if direction in previous:
            gaps.append(v['actual_s']-previous[direction])
        previous[direction] = v['actual_s']
    for actual, expected in ((np.mean(gaps), r['results']['dispatch_gap_mean_s']),
        (np.var(gaps), r['results']['dispatch_gap_variance_s2']),
        (np.mean([v['actual_s']-v['eligible_s'] for v in launch.values()]), r['results']['readiness_delay_s'])):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    latest, by_stop, dep_neighbours, ready_neighbours = {}, defaultdict(list), {}, {}
    for call in t['upper_callbacks']:
        for change in call['changes']:
            trip = change['trip_id']
            if trip in launch and call['time_s'] <= launch[trip]['actual_s']:
                latest[trip] = change['after']['planned_s']
    for q in t['lower_queries']:
        by_stop[q['station_id'], q['direction']].append(q)
    for rows in by_stop.values():
        rows.sort(key=lambda q: (q['time_s'], q['ordinal']))
        for a, b in zip(rows, rows[1:]):
            if a['bus_id'] != b['bus_id']:
                ready_neighbours[key(a)] = b
    by_stop.clear()
    for dep in t['departures']:
        by_stop[dep['station_id'], dep['direction']].append(dep)
    for rows in by_stop.values():
        for i in range(1, len(rows)-1):
            a, own, b = rows[i-1:i+2]
            if launch[own['trip_id']]['bus'] != launch[b['trip_id']]['bus']:
                dep_neighbours[key(own)] = a, own, b
    return dict(data=data, queries=queries, moments=m, launches=launch, latest_plan=latest,
        ready_neighbours=ready_neighbours, dep_neighbours=dep_neighbours)


def anchor_call(prepared):
    r, t = prepared['data']['result'], prepared['data']['trace']
    calls = [v for v in t['upper_callbacks'] if v['trip_id'] == r['upper']['trip']
        and v['time_s'] == r['upper']['query_s'] and v['fresh_commands']]
    require(len(calls) == 1, 'fresh anchor callback is not unique')
    return calls[0]


def plan_realization(b, c):
    bc, cc = anchor_call(b), anchor_call(c)
    bm, cm = ({v['trip_id']: v for v in call['changes']} for call in (bc, cc))
    counts, totals = Counter(), Counter()
    for trip in bm.keys() | cm.keys():
        before = (bm.get(trip) or cm[trip])['before']
        if trip in bm and trip in cm:
            require(bm[trip]['before'] == cm[trip]['before'], 'pre-anchor plan differs')
        bp, cp = (m[trip]['after']['planned_s'] if trip in m else before['planned_s'] for m in (bm, cm))
        delta = cp-bp
        if abs(delta) <= EPS:
            continue
        counts['anchor_plan_changed_trips'] += 1
        totals['absolute_anchor_plan_delta_s'] += abs(delta)
        require(trip in b['launches'] and trip in c['launches'], 'anchor-affected trip not launched')
        actual = c['launches'][trip]['actual_s']-b['launches'][trip]['actual_s']
        totals['absolute_affected_launch_delta_s'] += abs(actual)
        counts['affected_launch_changed'] += abs(actual) > EPS
        counts['affected_launch_same_sign'] += actual*delta > EPS
        counts['affected_launch_opposite_sign'] += actual*delta < -EPS
        if trip in b['latest_plan'] and trip in c['latest_plan']:
            final = c['latest_plan'][trip]-b['latest_plan'][trip]
            counts['latest_plan_observed'] += 1
            counts['plan_difference_erased_before_launch'] += abs(final) <= EPS
            counts['plan_difference_persisted_but_actual_unchanged'] += abs(final) > EPS and abs(actual) <= EPS
        else:
            counts['latest_plan_unobserved'] += 1
    counts['paths_with_anchor_plan_change'] += counts['anchor_plan_changed_trips'] > 0
    counts['paths_with_any_actual_launch_change'] += any(abs(c['launches'][k]['actual_s']-v['actual_s']) > EPS
        for k, v in b['launches'].items())
    return dict(counts), dict(totals)


def ready_marginal(context, gap, baseline, candidate):
    if not context['evidence_valid'] or gap is None or max(baseline, candidate) > gap:
        return None
    f, target = context['forward_headway_s'], context['target_headway_s']
    if target is None or target <= 0 or f is None or min(f, gap) < 0:
        return None
    loss = lambda h: ((f+h-target)**2+(gap-h-target)**2)/(2*target**2)
    return loss(baseline)-loss(candidate)


def departure_marginal(neighbours, extra, target):
    if neighbours is None or target is None or target <= 0:
        return None
    front, own, behind = neighbours
    f, g = own['time_s']-front['time_s'], behind['time_s']-own['time_s']
    if extra > f:
        return None
    return ((f-extra-target)**2+(g+extra-target)**2-(f-target)**2-(g-target)**2)/(2*target**2)


def holding_labels(b, c):
    br, cr = b['data']['result'], c['data']['result']
    cutoff = max(v['time_s'] for v in (*br['lower'], *cr['lower']))
    registered = {tuple(r['identity'][:3]) for r in cr['lower']}
    counts, seconds = Counter(), Counter()
    for k, cq in c['queries'].items():
        bq = b['queries'].get(k)
        if bq is None or k in registered or min(bq['time_s'], cq['time_s']) <= cutoff:
            continue
        extra = cq['command_s']-bq['command_s']
        if extra <= EPS:
            continue
        ctx, ready = cq['context'], c['ready_neighbours'].get(k)
        gap = ready['time_s']-cq['time_s'] if ready else None
        estimate = ready_marginal(ctx, ctx['follower_departure_gap_s'] if ctx['follower_evidence_valid'] else None,
            bq['command_s'], cq['command_s'])
        truth = ready_marginal(ctx, gap, bq['command_s'], cq['command_s'])
        dep = departure_marginal(c['dep_neighbours'].get(k), extra, ctx['target_headway_s'])
        for label, score in (('query_estimate', estimate), ('completed_ready', truth), ('completed_departure', dep)):
            name = label+'/'+sign(score)
            counts[name] += 1
            seconds[name] += extra
        cross = 'ready_'+sign(truth)+'/departure_'+sign(dep)
        counts[cross] += 1
        seconds[cross] += extra
        counts['positive_changes'] += 1
        seconds['positive_changes_s'] += extra
    return dict(counts), dict(seconds)


def audit_group(root, old, index):
    ci, future = divmod(index, 16)
    archived = read(old/'select'/f'c{ci}_f{future}.json')['rows']
    paths = [prepare(read(root/'records'/f'c{ci}_f{future}_a{a}.json.gz'), archived[a]) for a in range(15)]
    group, results = read(root/'groups'/f'c{ci}_f{future}.json'), {}
    for a in range(1, 15):
        u, l = divmod(a, 5)
        control = 5*u if l else 0
        name = f'lower{l}_at_upper{u}' if l else f'upper{u}_only'
        b, c = paths[control], paths[a]
        pc, ps = plan_realization(b, c)
        hc, hs = holding_labels(b, c)
        original = group['contrasts'][name]
        require(hc.get('positive_changes', 0) == original['counts'].get('post_second_increases', 0), 'positive holding event count differs')
        np.testing.assert_allclose(hs.get('positive_changes_s', 0.),
            original['seconds'].get('post_second_positive_command_delta_s', 0.), atol=EPS, rtol=0)
        results[name] = dict(plan_counts=pc, plan_seconds=ps, label_counts=hc, label_seconds=hs,
            cv_parts=cv_parts(b['moments'], c['moments']),
            effects=dict(journey_s=60*original['effects']['journey_min'], **{k: original['effects'][k] for k in METRICS if k != 'journey_s'}))
    print('V213_REVIEW_GROUP', index, flush=True)
    return dict(context=ci, future=future, checked=15, contrasts=results)


def intervals(values):
    a = np.asarray(values).reshape(4, 16)
    rng = np.random.default_rng(213)
    ai, di = rng.integers(0, 2, (5000, 2)), rng.integers(0, 2, (5000, 2))
    means = a.mean(1).reshape(2, 2)
    crossed = means[ai[:, :, None], di[:, None, :]].mean((1, 2))
    weights = rng.multinomial(16, [1/16]*16, size=(5000, 4))
    conditional = np.einsum('bcf,cf->b', weights, a)/64
    return dict(mean=float(a.mean()), crossed_ci95=np.percentile(crossed, [2.5, 97.5]).tolist(),
        conditional_ci95=np.percentile(conditional, [2.5, 97.5]).tolist(), context_means=a.mean(1).tolist())


def run(root, old, out, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        rows = list(pool.map(audit_group, itertools.repeat(root), itertools.repeat(old), range(64)))
    summary = {}
    for contrast in rows[0]['contrasts']:
        selected = [r['contrasts'][contrast] for r in rows]
        combined = {}
        for field in ('plan_counts', 'plan_seconds', 'label_counts', 'label_seconds'):
            counter = Counter()
            for r in selected:
                counter.update(r[field])
            combined[field] = dict(counter)
        combined['effects'] = {k: intervals([r['effects'][k] for r in selected]) for k in METRICS}
        combined['mean_cv_parts'] = {k: float(np.mean([r['cv_parts'][k] for r in selected])) for k in selected[0]['cv_parts']}
        summary[contrast] = combined
    result = dict(status='v213_readonly_coordination_review_complete', checked_paths=sum(r['checked'] for r in rows),
        all_archived_fields_exact=True, all_full_query_histograms_exact=True, all_registered_commands_exact=True,
        actual_arrival_cv2_dispatch_gaps_and_readiness_reconstructed=True, contrasts=summary,
        smoke=read(root/'smoke/done.json'), new_rollouts=0, training_updates=0, main_policy_changed=False,
        limitations='Consumed selection paths,two actors/two demand days;descriptive intervals,not a promotion test. '
            'Mechanical completed-neighbour comparisons hold neighbours fixed;no individual action necessity or harm is identified. '
            'Old complete trajectories were not saved;full off/on equivalence is established by four smoke pairs,not 960 old trace comparisons.')
    path = out/'summary/status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('V213_READONLY_REVIEW_COMPLETE', result['checked_paths'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v213-root', type=Path, required=True)
    p.add_argument('--v211-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v213_coordination_review.py', '-v'], check=True)
    run(a.v213_root, a.v211_root, a.out_root, a.workers)


if __name__ == '__main__':
    main()
