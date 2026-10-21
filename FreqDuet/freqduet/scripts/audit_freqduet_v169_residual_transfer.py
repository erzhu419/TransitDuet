#!/usr/bin/env python3
"""Read saved calibration/confirmation residuals and frozen bounds; never refit."""

import argparse
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

import run_freqduet_v168_confirmation as saved


read, write, require = saved.read, saved.write, saved.require
CHANNELS = ('running', 'service', 'residence')
COMPONENTS = tuple(c+'_'+m+'_s' for c in (*CHANNELS, 'joint') for m in ('bias', 'mae'))
BOUND_METRICS = (*saved.METRICS, 'upper_delta_s', 'lower_delta_s', 'upper_contracted',
    'upper_contraction_s', 'upper_expansion_s', 'candidate_only_upper', 'reference_only_upper',
    'both_upper', 'neither_upper', 'new_upper_reference_covered', 'removed_upper_candidate_covered',
    'candidate_upper_excess_s', 'reference_upper_excess_s', 'calibration_days')
STRATA = ('upper_contracted', 'upper_equal', 'upper_expanded', 'candidate_only_upper',
    'reference_only_upper', 'both_upper', 'neither_upper')


def residual_values(errors):
    total = errors.sum(axis=1)
    joined = np.column_stack((errors, total))
    values = np.stack((joined, np.abs(joined)), axis=2).reshape(len(errors), 8)
    return np.column_stack((values, (np.abs(errors).sum(axis=1)-np.abs(total))/2.))


def bound_values(truth, bounds, days):
    columns = []
    for lo, hi in (bounds[:, 0].T, bounds[:, 1].T):
        width = hi-lo
        columns.extend((lo <= truth, truth < lo, truth > hi, width,
            width+10.*np.maximum(lo-truth, 0.)+10.*np.maximum(truth-hi, 0.)))
    columns[0] = columns[0] & ~columns[2]
    columns[5] = columns[5] & ~columns[7]
    du, dl = bounds[:, 0, 1]-bounds[:, 1, 1], bounds[:, 0, 0]-bounds[:, 1, 0]
    a, b = columns[2], columns[7]
    columns.extend((columns[4]-columns[9], columns[3]-columns[8], du, dl, du < 0.,
        np.maximum(-du, 0.), np.maximum(du, 0.), a & ~b, b & ~a, a & b, ~a & ~b,
        a & columns[5], b & columns[0], np.maximum(truth-bounds[:, 0, 1], 0.),
        np.maximum(truth-bounds[:, 1, 1], 0.), days))
    return np.column_stack(columns)


def episode_mean(values, episodes):
    if not len(episodes):
        return None
    _, inv, n = np.unique(episodes, return_inverse=True, return_counts=True)
    return np.array([np.mean(np.bincount(inv, weights=v)/n) for v in values.T])


def cells(actor, mode, cohort, data):
    day, ep, hour, native, finite, errors, point, bounds, support = data
    require(not np.any(finite & ~native), 'finite request without native support')
    require(np.isfinite(errors[native]).all(), 'native residual missing; cannot silently drop')
    require(np.isfinite(bounds[finite]).all(), 'finite candidate without matched reference')
    component = np.full((len(day), 9), np.nan)
    component[native] = residual_values(errors[native])
    metrics = (*COMPONENTS, 'cancellation_s')
    all_values = np.full((len(day), len(metrics)+len(BOUND_METRICS)), np.nan)
    all_values[:, :len(metrics)] = component
    truth = point[finite]-errors[finite].sum(axis=1)
    all_values[finite, len(metrics):] = bound_values(truth, bounds[finite], support[finite])
    positions = {k: len(metrics)+i for i, k in enumerate(BOUND_METRICS)}
    output = []
    for d in np.unique(day):
        for scope in ('all', 'hour2', 'hour13'):
            eligible = (day == d) & (True if scope == 'all' else hour == int(scope[4:]))
            masks = [('native', eligible & native, metrics),
                ('matched', eligible & finite, (*metrics, *BOUND_METRICS))]
            if scope == 'hour13':
                masks.extend((s, eligible & finite &
                    ((all_values[:, positions['upper_delta_s']] == 0.) if s == 'upper_equal' else
                     (all_values[:, positions['upper_delta_s']] > 0.) if s == 'upper_expanded' else
                     (all_values[:, positions[s]] == 1.)), (*metrics, *BOUND_METRICS)) for s in STRATA)
            for mask_name, mask, names in masks:
                v = episode_mean(all_values[mask, :len(names)], ep[mask])
                fraction = episode_mean(np.column_stack((native[eligible], finite[eligible])), ep[eligible])
                output.append(dict(actor=actor, day=int(d), mode=mode, cohort=cohort, scope=scope,
                    mask=mask_name, eligible_queries=int(eligible.sum()), selected_queries=int(mask.sum()),
                    selected_episodes=len(np.unique(ep[mask])), native_fraction=None if fraction is None else float(fraction[0]),
                    finite_fraction=None if fraction is None else float(fraction[1]),
                    **dict(zip(names, [None]*len(names) if v is None else v.tolist()))))
    return output


def calibration(args, actor, mode):
    table = read(args.v167_root/'maps'/f'{actor}_{mode}.json')
    require(table['execution_source'] == saved.MAP_SOURCE, 'wrong map source')
    with np.load(args.v167_root/table['atom_file']) as z:
        requests, ar = z['requests'], z['atom_request']
        n = len(requests)
        errors, support = np.full((n, 3), np.nan), np.full(n, np.nan)
        errors[ar] = z['residual_rsd']
        support[ar] = np.array([e['day_count'] for e in table['candidate']])[z['candidate_key']]
        data = (requests[:, 0].astype(int), requests[:, 1].astype(int), (requests[:, 4]//3600).astype(int),
            z['native_supported'], z['finite_candidate'], errors, requests[:, 3], z['bounds'], support)
        require(set(data[0]) == set(range(2951001, 2951065)), 'calibration day cohort differs')
        return cells(actor, mode, 'calibration64_in_sample', data)


def confirmation(args, actor, mode, reservation):
    root = args.v168_root/'confirmation'
    requests, errors, points, bounds, native, finite, support = [], [], [], [], [], [], []
    episodes = {}
    for day in reservation['confirmation_days']:
        stem = f'{actor}_{day}_{mode}.jsonl.gz'
        for r, p in zip_longest(saved.stream(root/'residuals'/stem), saved.stream(root/'interval_predictions'/stem)):
            require(r is not None and p is not None and r['query_key'] == p['query_key'] and
                r['query_time_s'] == p['query_time_s'], 'residual/bound identity or clock differs')
            require(p['map_source'] == saved.MAP_SOURCE, 'wrong prediction map')
            if p['support_reason'] == 'ineligible':
                continue
            require(r['label']['status'] == 'resolved', 'eligible label missing')
            is_native = not p['support_reason'].startswith('native/')
            require(not p['finite'] or p['reference_finite'], 'finite reference missing')
            err = [r['residual_s'][c] for c in CHANNELS] if is_native else [np.nan]*3
            requests.append((day, episodes.setdefault((day, *r['episode']), len(episodes)), r['query_time_s']//3600))
            errors.append(err)
            points.append(r['label']['full_eta_s']+sum(err))
            bounds.append([p['bounds'] if p['finite'] else [np.nan]*2,
                p['reference_bounds'] if p['reference_finite'] else [np.nan]*2])
            native.append(is_native)
            finite.append(p['finite'])
            support.append(np.nan if p['calibration_days'] is None else p['calibration_days'])
    a = np.asarray(requests, dtype=int)
    return cells(actor, mode, 'confirmation32_held_out', (a[:, 0], a[:, 1], a[:, 2],
        np.array(native), np.array(finite), np.array(errors), np.array(points), np.array(bounds), np.array(support)))


def bucket(args, actor, mode, reservation):
    result = calibration(args, actor, mode)+confirmation(args, actor, mode, reservation)
    write(args.out_root/'cells'/f'{actor}_{mode}.json', result)
    print('V169_ACTOR_MODE_COMPLETE', actor, mode, flush=True)
    return result


def summarize(group, names):
    output = dict(cells=len(group), nonempty_cells=sum(r['selected_episodes'] > 0 for r in group),
        queries=sum(r['selected_queries'] for r in group))
    for name in names:
        v = [r[name] for r in group if r.get(name) is not None and np.isfinite(r[name])]
        output[name] = float(np.mean(v)) if v else None
    return output


def describe(rows, actors, reservation):
    frame = pd.DataFrame(rows)
    summaries, marginal = [], []
    metrics = (*COMPONENTS, 'cancellation_s', 'native_fraction', 'finite_fraction', *BOUND_METRICS)
    uncertainty = ('running_bias_s', 'service_bias_s', 'residence_bias_s', 'joint_bias_s', 'joint_mae_s',
        'coverage', 'above', 'reference_above', 'upper_delta_s', 'candidate_only_upper', 'reference_only_upper')
    for identity, group in frame.groupby(['cohort', 'mode', 'scope', 'mask'], sort=True):
        selected = group.where(pd.notnull(group), None).to_dict('records')
        key = dict(zip(('cohort', 'mode', 'scope', 'mask'), identity))
        expected_days = reservation['calibration_days' if identity[0].startswith('calibration') else 'confirmation_days']
        require(len(selected) == len(actors)*len(expected_days) and
            {(r['actor'], r['day']) for r in selected} == {(a, d) for a in actors for d in expected_days},
            'actor/day grid missing or duplicated')
        result = dict(**key, **summarize(selected, metrics))
        result['crossed_ci95'] = {}
        if identity[3] in ('native', 'matched'):
            for metric in uncertainty:
                if group[metric].isna().any():
                    continue
                f = group.rename(columns={'actor': 'train_seed', 'day': 'eval_seed'})
                draws = saved.hierarchical_bootstrap(f, metric, draws=5000)
                result['crossed_ci95'][metric] = np.percentile(draws, [2.5, 97.5]).tolist()
            for axis in ('actor', 'day'):
                m = group.groupby(axis)[list(uncertainty)].mean()
                marginal.extend(dict(**key, axis=axis, identity=int(i), **r) for i, r in m.to_dict('index').items())
                result[axis+'_range'] = {metric: [float(m[metric].min()), float(m[metric].max())]
                    for metric in uncertainty if m[metric].notna().any()}
        summaries.append(result)
    return summaries, marginal


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v169_*.py', '-v'], check=True)
    model = read(args.v167_root/'model.json')
    registration = read(args.v168_root/'registration.json')
    primary = read(args.v168_root/'summary/status.json')
    require(model == registration['sealed_model'] and model['execution_source'] == saved.MAP_SOURCE,
        'sealed map differs from confirmed snapshot')
    require(primary['status'] == 'v168_independent_confirmation_complete' and primary['paths'] == 768 and
        primary['execution_source'] == '8466f300ab' and primary['map_updates'] == 0, 'V168 not completed/frozen')
    reservation = registration['reservation']
    actors = reservation['actors']
    require(reservation['calibration_days'] == list(range(2951001, 2951065)) and
        reservation['confirmation_days'] == list(range(2852001, 2852033)), 'wrong residual cohorts')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, actor, mode, reservation) for actor in actors for mode in ('old', 'handoff')]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.extend(job.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='actor_mode')), flush=True)
    summaries, marginals = describe(rows, actors, reservation)
    for mode in ('old', 'handoff'):
        for scope in ('all', 'hour2', 'hour13'):
            check = next(r for r in summaries if (r['cohort'], r['mode'], r['scope'], r['mask']) ==
                ('confirmation32_held_out', mode, scope, 'matched'))
            original = next(r for r in primary['readouts'] if (r['mode'], r['scope']) == (mode, scope))
            for metric in ('coverage', 'above', 'reference_above', 'width_delta_s', 'score_delta_s'):
                saved.metadata.close(check[metric], original['estimates'][metric]['mean'])
            saved.metadata.close(check['above']-check['reference_above'],
                check['candidate_only_upper']-check['reference_only_upper'])
    write(args.out_root/'details/marginals.json', marginals)
    write(args.out_root/'summary/status.json', dict(status='v169_readonly_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], map_source=saved.MAP_SOURCE,
        cohort_paths=dict(calibration=1536, confirmation=768), summaries=summaries,
        residual_sign='prediction minus truth, seconds', residence='holding + execution waits + inseparable current-ready remainder',
        weights='query mean within selected episode; equal episodes within actor/day; equal actor/day cells',
        conditional_strata='conditional means use explicitly nonempty cells; missing strata are not zeros',
        calibration_estimates='in-sample descriptive, not independent coverage evidence',
        bootstrap='descriptive crossed actor/day 5000 seed20260803; cohorts not demand-day paired',
        frozen_v168_gate=primary['gate'], frozen_model_equal=read(args.v167_root/'model.json') == model,
        v168_metrics_reproduced=True, new_rollouts=0, new_training=0, interval_fit=False, map_updates=0,
        main_policy_changed=False))
    print('V169_READONLY_COMPLETE_V168_METRICS_REPRODUCED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v167-root', 'v168-root', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    run(parser.parse_args())
