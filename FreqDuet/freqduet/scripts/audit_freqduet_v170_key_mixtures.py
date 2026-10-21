#!/usr/bin/env python3
"""Read frozen key mixtures and saved hour13 forecast contexts; never refit."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import audit_freqduet_v169_residual_transfer as prior


saved = prior.saved
read, write, require, close = saved.read, saved.write, saved.require, saved.fit.close
KEY_FIELDS = ('actor', 'mode', 'query_phase', 'predicted_target_phase', 'direction', 'target_stop',
    'position_status', 'native_stage_kind', 'native_support_class', 'predicted_horizon_band')
EVENTS = ('all', 'new', 'repaired', 'both', 'neither')
FEATURES = ('point_s', 'running_bias_s', 'service_bias_s', 'residence_bias_s', 'joint_bias_s',
    'upper_delta_s', 'calibration_days', 'candidate_qlow_s', 'reference_qlow_s')


def day_weights(days, episodes):
    _, den = saved.fit.hierarchy(days, episodes)
    return 1./den


def mixtures(entries, indices, denominator, hour, errors, above):
    n, width = len(entries), int(hour.max())+1
    w = 1./denominator
    flat = indices*width+hour
    mass = np.bincount(flat, weights=w, minlength=n*width).reshape(n, width)
    bias = np.bincount(flat, weights=w*errors.sum(axis=1), minlength=n*width).reshape(n, width)
    tail = np.bincount(flat, weights=w*above, minlength=n*width).reshape(n, width)
    output = [None]*n
    for k, entry in enumerate(entries):
        if entry['status'] != 'finite':
            continue
        total, h2 = mass[k].sum(), mass[k, 2]
        close(total, 1.)
        output[k] = dict(key=entry['key'], days=entry['day_count'], qlow_s=entry['qlow_s'],
            hour_mass=(mass[k]/total).tolist(), pooled_bias_s=float(bias[k].sum()/total),
            pooled_upper_rate=float(tail[k].sum()/total), hour2_mass=float(h2/total),
            hour2_bias_s=None if h2 == 0. else float(bias[k, 2]/h2),
            hour2_upper_rate=None if h2 == 0. else float(tail[k, 2]/h2),
            other_hours_bias_s=None if h2 == total else float((bias[k].sum()-bias[k, 2])/(total-h2)))
    return output


def hour2_cells(actor, mode, table, z, candidate, reference):
    ri, keys, refs = z['atom_request'], z['candidate_key'], z['reference_key']
    requests, errors = z['requests'][ri], z['residual_rsd']
    mask = (requests[:, 4]//3600 == 2) & z['finite_candidate'][ri]
    day, episode = requests[mask, 0].astype(int), requests[mask, 1].astype(int)
    names = ['joint_bias_s', 'above', 'reference_above']
    bounds = z['bounds'][ri][mask]
    truth = requests[mask, 3]-errors[mask].sum(axis=1)
    columns = [errors[mask].sum(axis=1), truth > bounds[:, 0, 1], truth > bounds[:, 1, 1]]
    for label, records, indices in (('candidate', candidate, keys[mask]), ('reference', reference, refs[mask])):
        selected = [records[k] for k in indices]
        for field in ('hour2_mass', 'pooled_bias_s', 'hour2_bias_s',
                'pooled_upper_rate', 'hour2_upper_rate', 'qlow_s', 'days'):
            require(all(r is not None and r[field] is not None for r in selected), 'hour2 key mixture incomplete')
            names.append(label+'_'+field)
            columns.append(np.array([r[field] for r in selected]))
        width = len(selected[0]['hour_mass'])
        for hour in range(width):
            names.append(label+'_mix_hour_'+str(hour))
            columns.append(np.array([r['hour_mass'][hour] for r in selected]))
    values, w = np.column_stack(columns), day_weights(day, episode)
    return [dict(actor=actor, day=int(d), mode=mode, queries=int((day == d).sum()),
        **dict(zip(names, np.sum(values[day == d]*w[day == d, None], axis=0).tolist()))) for d in np.unique(day)]


def hour13(actor, mode, cohort, days, episodes, keys, point, errors, bounds, table, reference, expected_days):
    require(set(days) == set(expected_days), 'hour13 day coverage missing')
    require(np.isfinite(errors).all() and np.isfinite(bounds).all(), 'hour13 label or matched bounds missing')
    truth = point-errors.sum(axis=1)
    a, b = truth > bounds[:, 0, 1], truth > bounds[:, 1, 1]
    flags = (np.ones(len(days), dtype=bool), a & ~b, b & ~a, a & b, ~a & ~b)
    candidates = [table['candidate'][k] for k in keys]
    refs = {tuple(r['key']): r for r in reference}
    coarse = [refs[tuple(r['key'][:3])] for r in candidates]
    features = np.column_stack((point, errors, errors.sum(axis=1), bounds[:, 0, 1]-bounds[:, 1, 1],
        [r['day_count'] for r in candidates], [r['qlow_s'] for r in candidates], [r['qlow_s'] for r in coarse]))
    w = day_weights(days, episodes)
    cells = []
    for d in expected_days:
        pick = days == d
        rates = {e: float(np.sum(w[pick]*f[pick])) for e, f in zip(EVENTS, flags)}
        close(rates['all'], 1.)
        cells.append(dict(actor=actor, day=int(d), mode=mode, cohort=cohort, queries=int(pick.sum()), **rates,
            gap=rates['new']-rates['repaired'], above=rates['new']+rates['both'],
            reference_above=rates['repaired']+rates['both']))
    key_rows = []
    for event, flag in zip(EVENTS, flags):
        weights = w*flag/len(expected_days)
        mass = np.bincount(keys, weights=weights, minlength=len(table['candidate']))
        sums = np.array([np.bincount(keys, weights=weights*v, minlength=len(mass)) for v in features.T]).T
        for k in np.flatnonzero(mass):
            key_rows.append(dict(actor=actor, mode=mode, cohort=cohort, event=event,
                key=table['candidate'][k]['key'], mass=float(mass[k]), sums=sums[k].tolist()))
    return cells, key_rows


def confirmation(args, actor, mode, table, reservation):
    root = args.v168_root/'confirmation'
    indices = {tuple(e['key']): i for i, e in enumerate(table['candidate'])}
    days, episodes, keys, points, errors, bounds, ep_index = [], [], [], [], [], [], {}
    for day in reservation['confirmation_days']:
        name = f'{actor}_{day}_{mode}.jsonl.gz'
        streams = [saved.stream(root/d/name) for d in ('queries', 'interval_predictions', 'residuals')]
        for q, p, r in zip_longest(*streams):
            require(q is not None and p is not None and r is not None and q['query_key'] == p['query_key'] == r['query_key']
                and q['query_time_s'] == p['query_time_s'] == r['query_time_s'], 'saved context/bound/label join differs')
            if q['observable']['hour'] != 13 or not p['finite']:
                continue
            require((q['actor'], q['day'], q['mode'], p['map_source']) == (actor, day, mode, saved.MAP_SOURCE) and
                p['reference_finite'] and q['denominator_eligible'] and q['native_supported'] and
                r['label']['status'] == 'resolved' and q['episode'] == r['episode'], 'hour13 source or target differs')
            k = indices[tuple(q['candidate_key'])]
            require(table['candidate'][k]['status'] == 'finite', 'finite request has unsupported key')
            error = [r['residual_s'][c] for c in prior.CHANNELS]
            close(sum(error), q['point_s']-r['label']['full_eta_s'])
            days.append(day)
            episodes.append(ep_index.setdefault((day, *q['episode']), len(ep_index)))
            keys.append(k)
            points.append(q['point_s'])
            errors.append(error)
            bounds.append((p['bounds'], p['reference_bounds']))
    return hour13(actor, mode, 'confirmation32', *(np.asarray(v) for v in
        (days, episodes, keys, points, errors, bounds)), table, table['reference'], reservation['confirmation_days'])


def bucket(args, actor, mode, reservation):
    table = read(args.v167_root/'maps'/f'{actor}_{mode}.json')
    require(table['execution_source'] == saved.MAP_SOURCE, 'wrong frozen map source')
    with np.load(args.v167_root/table['atom_file']) as z:
        ri, request, error = z['atom_request'], z['requests'][z['atom_request']], z['residual_rsd']
        hour = (request[:, 4]//3600).astype(int)
        truth, bounds = request[:, 3]-error.sum(axis=1), z['bounds'][ri]
        mixtures_by_kind = [mixtures(table[kind], z[index], z['mass_denominator'][:, column], hour, error,
            truth > bounds[:, column, 1]) for kind, index, column in
            (('candidate', 'candidate_key', 0), ('reference', 'reference_key', 1))]
        h2 = hour2_cells(actor, mode, table, z, *mixtures_by_kind)
        mask = (hour == 13) & z['finite_candidate'][ri]
        cal = hour13(actor, mode, 'calibration64', request[mask, 0].astype(int), request[mask, 1].astype(int),
            z['candidate_key'][mask], request[mask, 3], error[mask], bounds[mask], table, table['reference'],
            reservation['calibration_days'])
    confirm = confirmation(args, actor, mode, table, reservation)
    result = dict(actor=actor, mode=mode, hour2=h2, hour13=cal[0]+confirm[0], keys=cal[1]+confirm[1])
    write(args.out_root/'mixtures'/f'{actor}_{mode}.json', dict(zip(('candidate', 'reference'), mixtures_by_kind)))
    write(args.out_root/'cells'/f'{actor}_{mode}.json', result)
    print('V170_ACTOR_MODE_COMPLETE', actor, mode, flush=True)
    return result


def composition(rows, actors):
    groups = defaultdict(lambda: np.zeros(1+len(FEATURES)))
    for r in rows:
        dimensions = [(f, r['key'][i]) for i, f in enumerate(KEY_FIELDS) if i >= 2]
        dimensions.append(('key_signature', r['key'][2:]))
        for field, value in dimensions:
            group = (r['cohort'], r['mode'], r['event'], field, json.dumps(value, separators=(',', ':')))
            groups[group] += np.array([r['mass'], *r['sums']])/len(actors)
    # Event denominators come from complete key_signature partitions, not nonempty day subsets.
    totals = defaultdict(float)
    for identity, value in groups.items():
        if identity[3] == 'key_signature':
            totals[identity[:3]] += value[0]
    output = []
    for identity, v in sorted(groups.items()):
        cohort, mode, event, field, value = identity
        output.append(dict(cohort=cohort, mode=mode, event=event, field=field, value=json.loads(value),
            mass=float(v[0]), event_share=float(v[0]/totals[identity[:3]]),
            **dict(zip(FEATURES, (v[1:]/v[0]).tolist()))))
    return output


def crossed(rows, metrics):
    result = dict(cells=len(rows), queries=sum(r['queries'] for r in rows), mean={}, ci95={})
    frame = pd.DataFrame(rows).rename(columns={'actor': 'train_seed', 'day': 'eval_seed'})
    for name in metrics:
        result['mean'][name] = float(frame[name].mean())
        draws = saved.hierarchical_bootstrap(frame, name, draws=5000)
        result['ci95'][name] = np.percentile(draws, [2.5, 97.5]).tolist()
    return result


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v170_*.py', '-v'], check=True)
    model = read(args.v167_root/'model.json')
    registration = read(args.v168_root/'registration.json')
    v169 = read(args.v169_root/'summary/status.json')
    require(model == registration['sealed_model'] and model['execution_source'] == saved.MAP_SOURCE and
        v169['status'] == 'v169_readonly_complete' and v169['v168_metrics_reproduced'], 'frozen parent missing')
    require(model['frozen_protocol']['conditioning']['candidate_key'] == list(KEY_FIELDS), 'key definition changed')
    r = registration['reservation']
    results = []
    if args.cells_root is not None:
        results = [read(args.cells_root/f'{a}_{m}.json') for a in r['actors'] for m in ('old', 'handoff')]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            jobs = [pool.submit(bucket, args, a, m, r) for a in r['actors'] for m in ('old', 'handoff')]
            for done, job in enumerate(as_completed(jobs), 1):
                results.append(job.result())
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=24, unit='actor_mode')), flush=True)
    h2, h13 = [c for b in results for c in b['hour2']], [c for b in results for c in b['hour13']]
    compositions = composition([k for b in results for k in b['keys']], r['actors'])
    diagnostics = []
    for cohort, days in (('calibration64', r['calibration_days']), ('confirmation32', r['confirmation_days'])):
        for mode in ('old', 'handoff'):
            selected = [c for c in h13 if (c['cohort'], c['mode']) == (cohort, mode)]
            require(len(selected) == len(r['actors'])*len(days) and {(c['actor'], c['day']) for c in selected} ==
                {(a, d) for a in r['actors'] for d in days}, 'hour13 crossed grid incomplete')
            summary = crossed(selected, ('new', 'repaired', 'gap', 'above', 'reference_above'))
            old = next(v for v in v169['summaries'] if v['cohort'].startswith(cohort) and
                (v['mode'], v['scope'], v['mask']) == (mode, 'hour13', 'matched'))
            for metric, previous in (('new', 'candidate_only_upper'), ('repaired', 'reference_only_upper'),
                    ('above', 'above'), ('reference_above', 'reference_above')):
                close(summary['mean'][metric], old[previous])
            groups = [c for c in compositions if (c['cohort'], c['mode']) == (cohort, mode)]
            for event in ('new', 'repaired'):
                for field in KEY_FIELDS[2:]:
                    close(sum(c['mass'] for c in groups if (c['event'], c['field']) == (event, field)),
                        summary['mean'][event])
            diagnostics.append(dict(cohort=cohort, mode=mode, hour13=summary,
                composition=[c for c in groups if c['event'] in ('new', 'repaired') and
                    c['field'] in ('predicted_horizon_band', 'position_status', 'native_stage_kind', 'native_support_class')],
                top_signatures={e: sorted([c for c in groups if (c['event'], c['field']) == (e, 'key_signature')],
                    key=lambda c: c['mass'], reverse=True)[:4] for e in ('new', 'repaired')}))
    h2_summaries = []
    for mode in ('old', 'handoff'):
        selected = [c for c in h2 if c['mode'] == mode]
        require(len(selected) == len(r['actors'])*len(r['calibration_days']) and
            {(c['actor'], c['day']) for c in selected} == {(a, d) for a in r['actors'] for d in r['calibration_days']},
            'hour2 crossed grid incomplete')
        hours = {k for c in selected for k in c if '_mix_hour_' in k}
        for c in selected:
            for k in hours:
                c.setdefault(k, 0.)
        summary = crossed(selected, [k for k in selected[0] if k not in ('actor', 'day', 'mode', 'queries')])
        old = next(v for v in v169['summaries'] if v['cohort'].startswith('calibration64') and
            (v['mode'], v['scope'], v['mask']) == (mode, 'hour2', 'matched'))
        for metric in ('joint_bias_s', 'above', 'reference_above'):
            close(summary['mean'][metric], old[metric])
        h2_summaries.append(dict(mode=mode, **summary))
    write(args.out_root/'details/compositions.json', compositions)
    write(args.out_root/'details/actor_day_cells.json', dict(hour2=h2, hour13=h13))
    require(read(args.v167_root/'model.json') == model, 'frozen model changed')
    write(args.out_root/'summary/status.json', dict(status='v170_readonly_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], map_source=saved.MAP_SOURCE,
        saved_cells_root=None if args.cells_root is None else str(args.cells_root), hour2=h2_summaries, hour13=diagnostics,
        event_weights='equal days and actors; equal matched episodes then queries; category mass partitions event rate',
        mixture_weights='original V167 mass_denominator; no refit, reweighting or new quantiles',
        key_signature_fields=list(KEY_FIELDS[2:]), prior_metrics_reproduced=True, frozen_model_equal=True,
        v168_gate=v169['frozen_v168_gate'], new_rollouts=0, new_training=0, interval_fit=False, map_updates=0,
        main_policy_changed=False, limitations='calibration is in-sample; diagnostics do not select gates; '
            'residence includes holding, execution waits and inseparable current-ready remainder'))
    print('V170_READONLY_COMPLETE_PRIOR_METRICS_REPRODUCED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v167-root', 'v168-root', 'v169-root', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cells-root', type=Path)
    run(parser.parse_args())
