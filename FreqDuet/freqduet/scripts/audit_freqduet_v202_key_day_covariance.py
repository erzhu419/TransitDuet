#!/usr/bin/env python3
"""Partition within/between-day J,Y co-moments for the two fixed V201 keys."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v201_uncertainty_sources as previous


base = previous.previous
saved, read, write, require = base.saved, base.read, base.write, base.require
KEYS = ((0, 7, 13), (0, 12, 13))
SIGNALS = ('signed_remaining_s', 'innovation_total_s', 'innovation_absorbed_s')
FIELDS = (*SIGNALS, 'first_running_under_s')
IX = [saved.VARS.index(n) for n in FIELDS]
COLS = [0, *[1+i for i in IX], *[1+saved.N+next(k for k, pair in enumerate(zip(*saved.TRI))
    if pair == tuple(sorted((i, IX[-1])))) for i in IX[:3]]]
METRICS = ('mix', 'conditional', 'conditional_within_day', 'conditional_between_day',
    'conditional_mean_product', 'supported_delta', 'unsupported_delta',
    'parent_within_delta', 'parent_between_delta', 'parent_total_delta')
ENDPOINTS = ('supported_within_day', 'supported_between_day_centered', 'supported_mean_product_centered',
    'unsupported_centered_cross', 'parent_raw_within_day', 'parent_raw_between_day', 'parent_raw_total')


def within(v):
    return v[..., 5:8]-np.divide(v[..., 1:4]*v[..., 4:5], v[..., :1],
        out=np.zeros_like(v[..., 5:8]), where=v[..., :1] > 0.)


def center(v, parent):
    mu = np.divide(parent[..., 1:5], parent[..., :1], out=np.zeros_like(parent[..., 1:5]), where=parent[..., :1] > 0.)
    out = v.copy()
    out[..., 1:5] -= v[..., :1]*mu[:, :, None, :]
    out[..., 5:8] -= mu[:, :, None, :3]*v[..., 4:5]+mu[:, :, None, 3:4]*v[..., 1:4]
    out[..., 5:8] += v[..., :1]*mu[:, :, None, :3]*mu[:, :, None, 3:4]
    require(np.allclose(within(out), within(v), rtol=1e-8, atol=1e-8), 'day offsets changed within-cell covariance')
    require(np.allclose(out[..., 5:8].sum(axis=2), within(parent), rtol=0., atol=1e-8),
        'all9 centered cross contributions do not reproduce parent within-day covariance')
    return out


def half(raw, centered, keep, weights):
    pooled = base.average_days(centered, weights)
    mass = pooled[..., :1]
    w = base.average_days(within(raw), weights)
    p = np.divide(pooled[..., 1:4]*pooled[..., 4:5], mass,
        out=np.full_like(w, np.nan), where=mass > 0.)
    parts = np.stack((w, pooled[..., 5:8]-p-w, p), axis=-1)
    t = np.where(keep[None, :, :, None, None], np.divide(parts, mass[..., None],
        out=np.full_like(parts, np.nan), where=mass[..., None] > 0.), 0.)
    parent = raw.sum(axis=2)
    parent_pool = base.average_days(parent, weights)
    parent_w = base.average_days(within(parent), weights)
    parent_total = within(parent_pool)
    end = np.stack((*[np.where(keep[None, :, :, None], parts[..., i], 0.).sum(axis=2) for i in range(3)],
        np.where(~keep[None, :, :, None], pooled[..., 5:8], 0.).sum(axis=2),
        parent_w, parent_total-parent_w, parent_total), axis=-2)
    return pooled, t, end


def evaluate(raw, centered, keep, weights):
    a, b = [half(raw[h*32:(h+1)*32], centered[h*32:(h+1)*32], keep, weights[h]) for h in range(2)]
    dm, mm = b[0][..., :1]-a[0][..., :1], (b[0][..., :1]+a[0][..., :1])/2.
    mix = (dm*(a[1].sum(axis=-1)+b[1].sum(axis=-1))/2.).sum(axis=2)
    components = (mm[..., None]*(b[1]-a[1])).sum(axis=2)
    supported = np.where(keep[None, :, :, None], b[0][..., 5:8]-a[0][..., 5:8], 0.).sum(axis=2)
    end = np.stack((a[2], b[2]), axis=1)
    delta = end[:, 1]-end[:, 0]
    metrics = np.stack((mix, components.sum(axis=-1), components[..., 0], components[..., 1], components[..., 2],
        supported, delta[..., 3, :], delta[..., 4, :], delta[..., 5, :], delta[..., 6, :]), axis=-2)
    require(np.allclose(metrics[..., 0, :]+metrics[..., 1, :], metrics[..., 5, :], rtol=1e-8, atol=1e-8, equal_nan=True),
        'mix/conditional decomposition differs')
    require(np.allclose(metrics[..., 5, :]+metrics[..., 6, :], metrics[..., 7, :], rtol=1e-8, atol=1e-8),
        'supported plus unsupported no longer equals parent within-day change')
    require(np.allclose(metrics[..., 7, :]+metrics[..., 8, :], metrics[..., 9, :], rtol=1e-8, atol=1e-8),
        'raw parent within/between partition differs')
    require(np.allclose(metrics[..., 0], metrics[..., 1]-metrics[..., 2], rtol=1e-8, atol=1e-8, equal_nan=True)
        and np.allclose(end[..., 0], end[..., 1]-end[..., 2], rtol=1e-8, atol=1e-8, equal_nan=True), 'J=I-A covariance identity differs')
    return metrics, end


def load_actor(args, actor, mode, scope, support):
    payload = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
    require(payload['source'] == '51a9409ec2', 'V194 source differs')
    parent = base.plan.base.actor_matrix(payload['rows'], scope, support, actor)
    keys = sorted(tuple(k['key']) for k in support if k['actor'] == actor)
    metadata = read(args.v199_root/'metadata'/f'{actor}_{mode}.json')
    require(metadata['source'] == '71af51ae4d' and metadata['paths'] == 64, 'V199 metadata differs')
    with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
        require(z['days'].tolist() == list(base.DAYS) and z['keys'].tolist() == [list(k) for k in keys]
            and z['bins'].tolist() == [list(k) for k in base.previous.BINS], 'frozen day/key/bin order differs')
        m, keep, counts, factors = z['moments'][..., COLS], z['supported_both'], z['distinct_days'], z['parent_balance_factors']
    require(np.array_equal(keep, np.all(counts >= 19, axis=0)), 'sealed V199 mask differs')
    require(np.allclose(m.sum(axis=2), parent[..., COLS], rtol=0., atol=1e-8), 'fixed-key parent moment input differs')
    raw, centers, mask, presence = np.zeros((64, 2, 9, 8)), np.zeros((64, 2, 8)), np.zeros((2, 9), dtype=bool), []
    for ki, key in enumerate(KEYS):
        presence.append(key in keys)
        if key in keys:
            j = keys.index(key)
            raw[:, ki], centers[:, ki], mask[ki] = m[:, j], parent[:, j, COLS], keep[j]
    values = center(raw, centers)
    for h, expected in enumerate(base.common.balance(parent[:32], parent[32:])):
        take = slice(h*32, (h+1)*32)
        require(np.allclose(parent[take, :, :1]*factors[h, None, :, None], expected[..., :1], rtol=0., atol=1e-12),
            'frozen V191 parent factors differ')
        for ki, key in enumerate(KEYS):
            if key in keys:
                factor = factors[h, keys.index(key)]
                raw[take, ki] *= factor
                values[take, ki] *= factor
    return raw, values, mask, presence


def reproduce(point, draws, reference, key):
    with np.load(reference) as z:
        ki = z['keys'].tolist().index(list(key))
        names, points, variances = z['names'].tolist(), z['key_point'][ki], z['key_own_variance'][ki]
    compare = dict(mix=point[0, 0], conditional=point[1, 0], conditional_covariance_change=point[2, 0]+point[3, 0],
        conditional_mean_product_change=point[4, 0], supported_delta=point[5, 0], unsupported_delta=point[6, 0], parent_delta=point[7, 0])
    boot = dict(mix=draws[:, 0, 0], conditional=draws[:, 1, 0], conditional_covariance_change=draws[:, 2, 0]+draws[:, 3, 0],
        conditional_mean_product_change=draws[:, 4, 0], supported_delta=draws[:, 5, 0], unsupported_delta=draws[:, 6, 0], parent_delta=draws[:, 7, 0])
    errors = []
    for name, value in compare.items():
        j = names.index(name)
        require(np.allclose(value, points[j], rtol=0., atol=1e-8)
            and np.allclose(boot[name].var(), variances[j], rtol=1e-8, atol=1e-8), 'V201 fixed-key point/bootstrap variance differs')
        errors.append(abs(float(value-points[j])))
    return max(errors)


def run_case(args, actors, mode, scope):
    support = read(args.v191_root/'details'/f'{mode}_{scope}.json')['common_key_support']
    aw, dw = base.common.weights(len(actors))
    draws, n = base.saved.prior.DRAWS, len(actors)
    point, boot = np.zeros((2, len(METRICS), 3)), np.zeros((draws, 2, len(METRICS), 3))
    pe, be = np.zeros((2, 2, len(ENDPOINTS), 3)), np.zeros((draws, 2, 2, len(ENDPOINTS), 3))
    mass, retained, present, supported_cells = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    records = []
    for ai, actor in enumerate(actors):
        raw, centered, keep, presence = load_actor(args, actor, mode, scope, support)
        m, e = evaluate(raw, centered, keep, np.full((2, 1, 32), 1./32))
        point += m[0]/n
        pe += e[0]/n
        present += presence
        supported_cells += keep.sum(axis=1)
        for h in range(2):
            v = raw[h*32:(h+1)*32, ..., 0].mean(axis=0)
            mass[h] += v.sum(axis=1)/n
            retained[h] += np.where(keep, v, 0.).sum(axis=1)/n
        records.append(dict(actor=actor, common_key_present=presence, supported_cells=keep.sum(axis=1).tolist()))
        for start in range(0, draws, 256):
            stop = min(start+256, draws)
            m, e = evaluate(raw, centered, keep, dw[:, start:stop])
            boot[start:stop] += aw[start:stop, ai, None, None, None]*m
            be[start:stop] += aw[start:stop, ai, None, None, None, None]*e
        target = args.out_root/'days'/f'{actor}_{mode}_{scope}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, days=base.DAYS, keys=KEYS, supported_both=keep, raw=raw, centered=centered,
            fields=['mass', 'sum_J', 'sum_I', 'sum_A', 'sum_Y', 'sum_JY', 'sum_IY', 'sum_AY'])
        print('V202_FIXED_KEY_ACTOR', mode, scope, actor, flush=True)
    require(np.isfinite(boot).all(), 'undefined supported draw;do not drop cells or replace draws')
    results = []
    for ki, key in enumerate(KEYS):
        error = reproduce(point[ki], boot[:, ki], args.v201_root/'details'/f'{mode}_{scope}.npz', key)
        metrics = {name: {label: base.saved.prior.interval(point[ki, mi, si], boot[:, ki, mi, si])
            for si, label in enumerate(('J', 'I', 'A'))} for mi, name in enumerate(METRICS)}
        endpoints = {name: {label: dict(zip(('first32', 'last32'), pe[:, ki, mi, si].tolist()))
            for si, label in enumerate(('J', 'I', 'A'))} for mi, name in enumerate(ENDPOINTS)}
        result = dict(mode=mode, scope=scope, key=list(key), common_actor_keys=int(present[ki]),
            supported_actor_bin_cells=int(supported_cells[ki]), parent_mass=mass[:, ki].tolist(),
            retained_mass=retained[:, ki].tolist(), v201_point_max_abs_error=error, endpoints=endpoints, metrics=metrics)
        results.append(result)
    write(args.out_root/'details'/f'{mode}_{scope}.json', dict(actors=records, results=results))
    target = args.out_root/'details'/f'{mode}_{scope}_draws.npz'
    np.savez_compressed(target, keys=KEYS, metrics=METRICS, endpoints=ENDPOINTS, point=point, boot=boot,
        endpoint_point=pe, endpoint_boot=be)
    print('V202_KEY_DAY_CASE_COMPLETE', mode, scope, flush=True)
    return results


def registration():
    return dict(stage='v202_fixed_key_within_between_day_covariance', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(base.DAYS), frozen=base.registration()['frozen'], v201_source='5d73c20fd9', keys=list(KEYS),
        inputs='Original64 V199/V194 moments,V191 fixed masks/weights,V201 key references;no raw trajectories.',
        signals='J=I-A;I is completed-source total innovation,A is absorbed part,Y is first-road truth-minus-prediction error. '
            'Preserve signed Cov(J,Y)=Cov(I,Y)-Cov(A,Y),including all matched bootstrap draws.',
        parent='Raw full-parent covariance=mean daily within-parent covariance+between-day parent-mean covariance. '
            'The former equals V200 parent/day-centered cross contribution;never add raw between-day covariance back into V200.',
        supported='Within each admitted actor/key/bin,center J/I/A/Y by FULL parent/day means. '
            'Half centered cross mass=W+B+P:W=mean daily within-cell covariance mass; '
            'B=between-day centered cell-mean covariance mass;P=pooled centered cell-mean product mass. '
            'V200 conditional covariance=W+B. Apply the frozen symmetric mix/conditional identity to each part.',
        weights='Fixed V19919-day-in-each-half masks,V191 factors,original episode/query weights,equal12 actors/day. '
            'No selected-mass normalization,new bins or pooled admission. Absent original actor/common-key means zero exposure,not missing archive.',
        unsupported='Keep only aggregate mass and parent/day-centered cross contribution;no conditional decomposition or relaxed admission.',
        resampling='Same10000 shared-actor/independent-half-day V200 draws,seed20260803;reproduce V201 fixed-key points/own variances. '
            'All new95% intervals are exploratory diagnostics on V201-selected keys,no new family gate or replication claim.',
        reporting='Daily raw/centered moments and matched draws stay on server;summary only local. Point/covariance units s^2 at nominal mass.',
        limitations='Development-only finite-bin algebra,not a causal explanation or new forecast evidence. '
            'Centered between-day cell-mean covariance differs from raw parent between-day covariance. '
            'I/A cancellation does not identify a predictor update error or a deployable intervention.',
        raw_capture_files_opened=0, interval_fit=False, predictor_fit=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v202_*.py', '-v'], check=True)
    parent = read(args.v201_root/'summary/status.json')
    require(parent['source'] == '5d73c20fd9' and parent['masks_and_weights_unchanged'] and parent['v200_points_and_intervals_reproduced']
        and parent['protocol']['calibration_days'] == list(base.DAYS), 'requires frozen V201 diagnostic')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite fixed-key protocol')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    actors = sorted({r['actor'] for r in read(args.v191_root/'details/old_clearance.json')['common_key_support']})
    require(len(actors) == 12, 'actor grid differs')
    results = []
    print('V202_KEY_DAY_PROTOCOL_SEALED', flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, actors, mode, scope) for mode, scope in product(('old', 'handoff'), base.PRIMARY)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.extend(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"key_day_covariance"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope'], r['key']))
    write(args.out_root/'summary/status.json', dict(status='v202_key_day_covariance_complete', source=protocol['source'],
        protocol=protocol, results=results, v201_fixed_key_points_and_variances_reproduced=True,
        within_between_identities_checked=True, signed_innovation_cancellation_checked=True,
        masks_and_weights_unchanged=True, unsupported_separate=True, raw_capture_files_opened=0,
        interval_fit=False, predictor_fit=False, confirmation_outcomes_read=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V202_KEY_DAY_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v201-root', 'v199-root', 'v194-root', 'v191-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
