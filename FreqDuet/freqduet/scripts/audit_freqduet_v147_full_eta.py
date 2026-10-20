#!/usr/bin/env python3
"""Read saved full follower ETAs; account for identity/source changes and holding."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.run_freqduet_protocol_v2_matrix import hierarchical_bootstrap


VERSIONS = ('native', 'v140', 'v141')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def key(row):
    return int(row['trip_id']), int(row['station_id']), bool(row['direction'])


def trace(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def close(a, b):
    if not np.isclose(a, b, rtol=0, atol=1e-7):
        raise AssertionError(f'saved-clock/accounting identity failed: {a} != {b}')


def follower(row):
    e = row['evidence']
    return e['predicted_follower_trip_id'], e['predicted_follower_bus_id']


def ready_time(ready, identity, stop):
    label = ready.get((identity[0], *stop))
    if label is None or label['bus_id'] != identity[1]:
        return None
    return label['ready_time_s']


def target_label(row, ready):
    if not row['context']['follower_evidence_valid']:
        return 'no_valid_follower_evidence', None
    identity = follower(row)
    if None in identity:
        return 'missing_follower_identity', None
    label = ready.get((identity[0], *key(row)[1:]))
    if label is None:
        return 'missing_same_trip_ready', None
    if label['bus_id'] != identity[1]:
        return 'same_trip_bus_mismatch', None
    gap = label['ready_time_s']-row['prediction_time_s']
    return ('same_follower_already_ready', None) if gap < 0 else ('resolved', gap)


def successors(rows):
    previous, result = {}, {}
    for row in rows:
        k = key(row)
        if k[1:] in previous:
            result[previous[k[1:]]] = k
        previous[k[1:]] = k
    return result


def describe(rows):
    fields = set().union(*(set(r)-{'episode'} for r in rows)) if rows else set()
    metrics = {}
    for field in sorted(fields):
        episodes = defaultdict(list)
        for r in rows:
            if field in r:
                episodes[tuple(r['episode'])].append(r[field])
        metrics[field] = mean([mean(v) for v in episodes.values()])
    return dict(queries=len(rows), episodes=len({tuple(r['episode']) for r in rows}), metrics=metrics)


def load_path(root, name, cutoff):
    rows = {key(r): r for r in trace(root/'traces'/(name+'.jsonl.gz'))}
    ready = {key(r): r for r in trace(root/'ready_labels'/(name+'.jsonl.gz'))}
    saved = {tuple(r['query']): r for r in trace(root/'shadows'/(name+'.jsonl.gz'))}
    stages = {tuple(r['query']): r for r in trace(root/'intervals'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        following = successors([dict(zip(fields, r.tolist())) for r in record['departures']])
    if rows.keys() != saved.keys():
        raise AssertionError('saved shadows and decision support differ')
    groups, counts, labelled = defaultdict(list), Counter(), {}
    for k, row in rows.items():
        forecast, now = saved[k], row['prediction_time_s']
        close(now, row['evidence']['ready_time_s'])
        close(now, forecast['time_s'])
        if forecast['native_s'] != row['online']['gap_s']:
            raise AssertionError('saved native point changed')
        status, truth = target_label(row, ready)
        if status == 'resolved' and any(forecast[v+'_s'] is None for v in VERSIONS):
            status = 'incomplete_prediction_triplet'
        counts[status] += 1
        if status != 'resolved':
            continue
        errors = {v: forecast[v+'_s']-truth for v in VERSIONS}
        values = dict(episode=(*follower(row), *k[1:]),
            **{v+'_mae_s': abs(e) for v, e in errors.items()}, v141_bias_s=errors['v141'],
            v141_minus_native_mae_s=abs(errors['v141'])-abs(errors['native']),
            v141_minus_v140_mae_s=abs(errors['v141'])-abs(errors['v140']))
        labels = ['all', 'clearance' if now >= cutoff else 'active']
        if int(now//3600) == 13:
            labels.append('hour13')
        if row['original_prediction']['status'] == 'not_causally_moving':
            labels.append('fallback')
        if k in stages:
            s = stages[k]
            labels.append(s['kind']+('_finite' if s['rank']['finite'] else '_abstain'))
        for label in labels:
            groups[label].append(values)
        following_key = following.get(k)
        next_label = ready.get(following_key)
        if next_label is not None and next_label['ready_time_s'] >= now:
            next_truth = next_label['ready_time_s']-now
            groups['next_departure_target'].append(dict(episode=k,
                identity_matches_next=float((follower(row)[0], *k[1:]) == following_key),
                identity_offset_s=truth-next_truth, absolute_identity_offset_s=abs(truth-next_truth),
                native_next_mae_s=abs(forecast['native_s']-next_truth),
                v141_next_mae_s=abs(forecast['v141_s']-next_truth)))
        else:
            counts['missing_or_past_next_departure_ready'] += 1
        labelled[k] = dict(row=row, forecast=forecast, truth=truth)
    return dict(rows=rows, ready=ready, labelled=labelled,
        stats={g: describe(v) for g, v in groups.items()}, counts=dict(counts))


def symmetric_identity(b, c, ready_b, ready_c):
    fb, fc, stop = follower(b), follower(c), key(b)[1:]
    bb, bc, cb, cc = [ready_time(r, f, stop) for r, f in
        ((ready_b, fb), (ready_b, fc), (ready_c, fb), (ready_c, fc))]
    if any(t is None for t in (bb, bc, cb, cc)):
        return None
    physical = .5*((cb-bb)+(cc-bc))-(c['prediction_time_s']-b['prediction_time_s'])
    replacement = .5*((bc-bb)+(cc-cb))
    close(physical+replacement, cc-c['prediction_time_s']-(bb-b['prediction_time_s']))
    return physical, replacement


def request(row):
    context = row['context']
    if not (context['evidence_valid'] and context['follower_evidence_valid']):
        return None
    prediction = row['online']['gap_s'] if row['online']['component_used'] else context['follower_departure_gap_s']
    u = (prediction-context['forward_headway_s'])/8.
    close(u, .25*row['feedback_error_s'])
    return prediction, u


def holding_parts(b, c, truth_b, truth_c, physical, replacement):
    rb, rc = request(b), request(c)
    if rb is None or rc is None:
        return None
    pb, ub = rb
    pc, uc = rc
    clipped_b, clipped_c = np.clip([ub, uc], 0., 45.)
    slope = (clipped_c-clipped_b)/(uc-ub) if uc != ub else float(0. < ub < 45.)
    forward = c['context']['forward_headway_s']-b['context']['forward_headway_s']
    error = (pc-truth_c)-(pb-truth_b)
    holding = c['executed_s']-b['executed_s']
    parts = dict(physical_ready_s=slope*physical/8., identity_s=slope*replacement/8.,
        forecast_error_s=slope*error/8., forward_gap_s=-slope*forward/8.,
        quantization_or_execution_s=holding-(clipped_c-clipped_b))
    close(sum(parts.values()), holding)
    return dict(parts, holding_delta_s=holding,
        absolute_forecast_error_contribution_s=abs(parts['forecast_error_s']),
        absolute_identity_contribution_s=abs(parts['identity_s']))


def source(row, forecast):
    return (row['evidence']['follower_source'], row['original_prediction']['status'],
        row['online']['status'], forecast['v140_status'], forecast['v141_status'])


def pair_case(root, out, seed, day):
    target = out/'cells'/f'{seed}_{day}.json'
    if target.exists():
        if read(target)['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
            raise AssertionError('cached audit cell has a different source')
        return
    paths = {}
    for mode in ('old', 'handoff'):
        name = f'{seed}_{day}_{mode}'
        receipt = read(root/'paths'/(name+'.json'))
        paths[mode] = load_path(root, name, receipt['contract']['clock']['demand_end_time_s'])
    bpath, cpath = paths['old'], paths['handoff']
    counts, groups = Counter(), defaultdict(list)
    counts['unpaired_decision_keys'] = len(bpath['rows'].keys() ^ cpath['rows'].keys())
    for k in bpath['rows'].keys() & cpath['rows'].keys():
        if k not in bpath['labelled'] or k not in cpath['labelled']:
            counts['missing_paired_label_or_forecast'] += 1
            continue
        b, c = bpath['labelled'][k], cpath['labelled'][k]
        br, cr = b['row'], c['row']
        counts['own_bus_identity_changed'] += br['evidence']['bus_id'] != cr['evidence']['bus_id']
        identity_change = follower(br) != follower(cr)
        source_change = source(br, b['forecast']) != source(cr, c['forecast'])
        group = ('identity_changed' if identity_change else 'identity_same')+'/'+(
            'source_changed' if source_change else 'source_same')
        counts[group] += 1
        delta_truth = c['truth']-b['truth']
        errors = {v: c['forecast'][v+'_s']-b['forecast'][v+'_s']-delta_truth for v in VERSIONS}
        values = dict(episode=k, native_delta_mae_s=abs(errors['native']),
            v141_delta_mae_s=abs(errors['v141']), native_error_change_s=errors['native'],
            v141_error_change_s=errors['v141'],
            v141_minus_native_delta_mae_s=abs(errors['v141'])-abs(errors['native']))
        for label in ('all', group):
            groups['paired/'+label].append(values)
        decomposition = symmetric_identity(br, cr, bpath['ready'], cpath['ready'])
        if decomposition is None:
            counts['missing_symmetric_identity_anchor'] += 1
            continue
        physical, replacement = decomposition
        parts = holding_parts(br, cr, b['truth'], c['truth'], physical, replacement)
        for label in ('all', group):
            groups['identity/'+label].append(dict(episode=k, physical_ready_delta_s=physical,
                identity_replacement_s=replacement, absolute_identity_replacement_s=abs(replacement)))
            if parts is not None:
                groups['holding/'+label].append(dict(episode=k, **parts))
    write(target, dict(seed=seed, day=day, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        modes={m: dict(stats=p['stats'], counts=p['counts']) for m, p in paths.items()},
        paired={g: describe(v) for g, v in groups.items()}, counts=dict(counts)))


def inference(cells, stats, group, field):
    values = [stats(c).get(group, {}).get('metrics', {}).get(field) for c in cells]
    finite = [v for v in values if v is not None]
    result = dict(contributing_cells=len(finite), mean=mean(finite) if finite else None, ci95=None)
    if len(finite) != len(cells):
        result['decision'] = 'inconclusive_crossed_support'
        return result
    frame = pd.DataFrame([dict(train_seed=c['seed'], eval_seed=c['day'], value=v) for c, v in zip(cells, values)])
    result.update(ci95=np.percentile(hierarchical_bootstrap(frame, 'value', draws=5000), [2.5, 97.5]).tolist())
    for axis in ('seed', 'day'):
        margins = frame.groupby('train_seed' if axis == 'seed' else 'eval_seed')['value'].mean()
        result[axis] = dict(groups=len(margins), minimum=float(margins.min()), maximum=float(margins.max()),
            negative_n=int((margins < 0).sum()), positive_n=int((margins > 0).sum()))
    return result


def aggregate(cells, stats):
    groups = sorted(set().union(*(stats(c).keys() for c in cells)))
    result = {}
    for group in groups:
        fields = sorted(set().union(*(stats(c).get(group, {}).get('metrics', {}).keys() for c in cells)))
        result[group] = dict(queries=sum(stats(c).get(group, {}).get('queries', 0) for c in cells),
            metrics={f: inference(cells, stats, group, f) for f in fields})
    return result


def run(args):
    parent = read(args.v146_root/'summary/status.json')
    model = read(args.v146_root/'model.json')
    if not parent['primary_joint_pass'] or (model['lower_extension_s'], model['upper_extension_s']) != (3., 0.):
        raise AssertionError('requires frozen successful V146 calibration')
    reg = parent['registration']
    cases = list(itertools.product(reg['actors'], reg['days']['confirmation']))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v147_full_eta.py', '-v'], check=True)
    root = args.v146_root/'confirmation'
    pair_case(root, args.out_root, *cases[0])
    print('V147_PREFLIGHT_SAVED_PAIR_COMPLETE', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        tasks = [pool.submit(pair_case, root, args.out_root, seed, day) for seed, day in cases]
        for n, task in enumerate(as_completed(tasks), 1):
            task.result()
            if n % 32 == 0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=len(cases), unit='saved_pair')), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    results = {m: aggregate(cells, lambda c: c['modes'][m]['stats']) for m in ('old', 'handoff')}
    paired = aggregate(cells, lambda c: c['paired'])
    counts = {m: dict(sum((Counter(c['modes'][m]['counts']) for c in cells), Counter())) for m in ('old', 'handoff')}
    write(args.out_root/'summary/status.json', dict(status='v147_readonly_full_eta_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], saved_paths=2*len(cases),
        actors=reg['actors'], days=reg['days']['confirmation'], results=results, paired=paired,
        label_counts=counts, pair_counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        saved_points_recomputed=False, calibration_refit=False, main_policy_changed=False,
        limitations='Consumed confirmation paths; diagnostic comparisons, not new policy confirmation. '
            'Full ETA truth is same-trip/bus target-ready time. Next-departure targets and symmetric '
            'cross-identity anchors are retrospective labels only. Missing crossed groups have no CI. '
            'Holding parts are exact path accounting, not action-specific mediation. V146 intervals '
            'cover only current-stage duration, not full ETA; points/window/tails/controls unchanged.'))
    print('V147_READONLY_FULL_ETA_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v146-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
