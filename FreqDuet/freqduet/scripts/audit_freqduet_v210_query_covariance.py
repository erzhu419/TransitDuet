#!/usr/bin/env python3
"""Execute V208's frozen observable/covariance test after V209 admission."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_freqduet_v209_query_mixed as prep
import audit_freqduet_v191_common_keys as common


plan, saved = prep.plan, prep.saved
read, write, require = prep.read, prep.write, prep.require
PARTS = ('within_cell', 'relative_cell_parent', 'total')


def project(m, beta):
    operator = np.zeros((4, saved.N))
    operator[np.arange(2), prep.SIGNAL_IX] = 1.
    operator[np.arange(2, 4), prep.TARGET_IX] = 1.
    operator[2:, prep.STATE_IX] -= beta
    second = np.zeros((*m.shape[:-1], saved.N, saved.N))
    second[..., saved.TRI[0], saved.TRI[1]] = m[..., 1+saved.N:]
    second[..., saved.TRI[1], saved.TRI[0]] = m[..., 1+saved.N:]
    return (m[..., 0], np.einsum('ij,...j->...i', operator, m[..., 1:1+saved.N]),
        np.einsum('ij,...jk,lk->...il', operator, second, operator, optimize=True))


def profiles(mass, first, second, mixed, keep):
    cell = np.divide(first, mass[..., None], out=np.zeros_like(first), where=mass[..., None] > 0.)
    parent_mass = mass.sum(axis=2)
    parent = np.divide(first.sum(axis=2), parent_mass[..., None], out=np.zeros_like(first.sum(axis=2)),
        where=parent_mass[..., None] > 0.)
    half_mass = np.stack((mass[:32].mean(axis=0), mass[32:].mean(axis=0)))
    require(np.all(half_mass[:, keep] > 0.), 'frozen supported cell has no half exposure')
    coefficient = np.divide(half_mass.mean(axis=0), half_mass, out=np.zeros_like(half_mass), where=half_mass > 0.)
    coefficient = np.repeat(np.where(keep[None], coefficient, 0.), 32, axis=0)
    s, x, y = cell[..., :2], cell[..., 2:3], cell[..., 3:4]
    sx, sy = second[..., :2, 2], second[..., :2, 3]
    cx = sx-s*first[..., 2:3]
    cy = sy-s*first[..., 3:4]
    uncentered = mixed-s*second[..., 2:3, 3]
    within = uncentered-x*cy-y*cx
    relative = (x-parent[..., None, 2:3])*cy+(y-parent[..., None, 3:4])*cx
    direct = uncentered-parent[..., None, 2:3]*cy-parent[..., None, 3:4]*cx
    require(np.allclose(within+relative, direct, rtol=1e-10, atol=1e-8), 'query covariance decomposition differs')
    parts = np.stack((within, relative, direct), axis=-1)
    daily = (coefficient[..., None, None]*parts).sum(axis=2)
    variances = np.diagonal(second[..., :2, :2], axis1=-2, axis2=-1)-s*first[..., :2]
    variation = (coefficient[..., None]*variances).sum(axis=2)
    exposure = (coefficient*mass).sum(axis=2)
    raw_exposure = np.stack((mass.sum(axis=2), (mass*keep[None]).sum(axis=2)), axis=-1)
    return daily, variation, exposure, raw_exposure


def average(values, aw, dw):
    require(values.shape[:2] == (64, aw.shape[1]), 'crossed actor/day grid differs')
    return np.stack([np.einsum('ba,bd,dak...->bk...', aw, dw[h], values[h*32:(h+1)*32], optimize=True)
        for h in range(2)], axis=1)


def estimates(point, boot, alpha=.05):
    return common.prior.interval(point, boot, alpha=alpha)


def sign(record):
    ci = record['ci']
    return 0 if ci is None else (1 if ci[0] > 0. else (-1 if ci[1] < 0. else 0))


def replicated(rows):
    signs = [sign(r['primary_family']) for r in rows]
    return bool(len(rows) == 2 and signs[0] != 0 and signs[0] == signs[1]
        and all(sign(r['halves'][h]['total']) == signs[0] for r in rows for h in ('first32', 'last32')))


def load_case(args, actors, mode, scope):
    daily, variation, exposure, raw_exposure = [], [], [], []
    supported = 0
    with np.load(args.v205_root/'details'/f'{mode}_{scope}.npz') as z:
        beta = z['beta_point']
        require(z['states'].tolist() == list(plan.plan.STATES), 'saved state order differs')
    for actor in actors:
        with np.load(args.v209_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(prep.DAYS) and z['keys'].tolist() == [list(k) for k in prep.KEYS]
                and z['signals'].tolist() == list(plan.SIGNALS) and z['mixed_fields'].tolist() == list(plan.MIXED_FIELDS)
                and np.array_equal(z['beta'], beta), 'admitted mixed block/beta differs')
            mixed, keep, counts, factors = (z[k] for k in ('raw_mixed_sums', 'supported_both', 'complete_queries', 'parent_balance_factors'))
            distinct = z['distinct_days']
        prep.verify_support(counts, distinct, keep)
        with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
            ix = [z['keys'].tolist().index(list(k)) for k in prep.KEYS]
            require(z['days'].tolist() == list(prep.DAYS) and z['bins'].tolist() == [list(b) for b in prep.BINS]
                and np.array_equal(z['supported_both'][ix], keep) and np.array_equal(z['distinct_days'][:, ix], distinct)
                and np.array_equal(z['parent_balance_factors'][:, ix], factors), 'original support/balance differs')
            m = z['moments'][:, ix]
        require(np.array_equal(m[..., 0] > 1e-12, counts > 0) and np.isfinite(mixed).all(), 'mixed complete-input exposure differs')
        for h in range(2):
            f = factors[h, None, :, None, None]
            m[h*32:(h+1)*32] *= f
            mixed[h*32:(h+1)*32] *= f
        result = profiles(*project(m, beta), mixed, keep)
        for output, value in zip((daily, variation, exposure, raw_exposure), result):
            output.append(value)
        supported += int(keep.sum())
    return [np.stack(v, axis=1) for v in (daily, variation, exposure, raw_exposure)], supported


def run_case(args, actors, mode, scope):
    (daily, variation, exposure, raw_exposure), supported = load_case(args, actors, mode, scope)
    aw, dw = common.weights(len(actors))
    uniform_a = np.full((1, len(actors)), 1./len(actors))
    uniform_d = np.full((2, 1, 32), 1./32)
    point = average(daily, uniform_a, uniform_d)[0]
    boot = average(daily, aw, dw)
    var_point = average(variation, uniform_a, uniform_d)[0]
    var_boot = average(variation, aw, dw)
    exp_point = average(exposure, uniform_a, uniform_d)[0]
    exp_boot = average(exposure, aw, dw)
    valid = (var_boot > 0.) & (exp_boot[..., None] > 0.)
    boot = np.where(valid[..., None], boot, np.nan)
    point = np.where(((var_point > 0.) & (exp_point[..., None] > 0.))[..., None], point, np.nan)
    require(np.allclose(boot[..., :2].sum(axis=-1), boot[..., 2], rtol=1e-10, atol=1e-8, equal_nan=True),
        'matched-draw signed pieces do not reproduce total')
    actor_boot = average(daily, aw, np.broadcast_to(uniform_d, dw.shape))
    day_boot = average(daily, np.broadcast_to(uniform_a, aw.shape), dw)
    actor_var = average(variation, aw, np.broadcast_to(uniform_d, dw.shape))
    day_var = average(variation, np.broadcast_to(uniform_a, aw.shape), dw)
    actor_boot = np.where((actor_var > 0.)[..., None], actor_boot, np.nan)
    day_boot = np.where((day_var > 0.)[..., None], day_boot, np.nan)
    raw_point = average(raw_exposure, uniform_a, uniform_d)[0]
    results = []
    for ki, key in enumerate(prep.KEYS):
        for si, signal in enumerate(plan.SIGNALS):
            p, b = point[:, ki, si], boot[:, :, ki, si]
            p_all, b_all = p.mean(axis=0), b.mean(axis=1)
            results.append(dict(mode=mode, scope=scope, key=list(key), signal=signal,
                units='s^3' if si == 0 else 'min*s^2',
                primary_family=estimates(p_all[2], b_all[:, 2], alpha=.05/16),
                all64={name: estimates(p_all[i], b_all[:, i]) for i, name in enumerate(PARTS)},
                halves={name: {part: estimates(p[h, i], b[:, h, i]) for i, part in enumerate(PARTS)}
                    for h, name in enumerate(('first32', 'last32'))},
                half_difference={name: estimates(p[1, i]-p[0, i], b[:, 1, i]-b[:, 0, i]) for i, name in enumerate(PARTS)},
                actor_only=estimates(p_all[2], actor_boot[:, :, ki, si, 2].mean(axis=1)),
                day_only=estimates(p_all[2], day_boot[:, :, ki, si, 2].mean(axis=1)),
                nominal_within_state_variation=var_point[:, ki, si].tolist(),
                raw_parent_and_supported_mass=raw_point[:, ki].tolist()))
    path = args.out_root/'details'/f'{mode}_{scope}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actors=actors, days=prep.DAYS, keys=prep.KEYS, signals=plan.SIGNALS, parts=PARTS,
        point=point, boot=boot, actor_only=actor_boot, day_only=day_boot, daily_profiles=daily,
        within_state_variation=var_point, exposure=exp_point, valid_draws=valid)
    print('V210_QUERY_ASSOCIATION_CASE_COMPLETE', mode, scope, flush=True)
    return results, supported


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v210_*.py', '-v'], check=True)
    parent = read(args.v209_root/'summary/status.json')
    require(parent['source'] == '8fa2f736b2' and parent['status'] == 'v209_two_query_mixed_sums_admitted'
        and parent['paths'] == 1536 and parent['clock_violations'] == 0 and parent['masks_and_weights_unchanged']
        and max(parent['per_bin_max_abs_error']) <= 1e-8, 'requires complete V209 admission')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite readout')
    protocol = dict(stage='v210_execute_v208_query_covariance', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        v208_protocol=parent['protocol']['v208_protocol'], v209_source='8fa2f736b2',
        execution='Frozen V20816 associations with saved beta/masks/coefficients. No raw archive scan, '
            'new feature/threshold selection,regression,interval fit,confirmation or controller change.',
        actor_day_margins='Actor-only/day-only intervals descriptive;crossed family/both-half criteria govern replication.')
    write(args.out_root/'protocol.json', protocol)
    actors = sorted({r['actor'] for r in parent['results']})
    require(len(actors) == 12, 'original actor grid differs')
    results, supported = [], 0
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        for job in as_completed([pool.submit(run_case, args, actors, mode, scope) for mode, scope in product(prep.MODES, prep.SCOPES)]):
            rows, n = job.result()
            results.extend(rows)
            supported += n
    require(supported == 315 and len(results) == 16, 'frozen support/family differs')
    results.sort(key=lambda r: (r['scope'], r['key'], r['signal'], r['mode']))
    gates = []
    for scope, key, signal in product(prep.SCOPES, prep.KEYS, plan.SIGNALS):
        rows = [r for r in results if (r['scope'], r['key'], r['signal']) == (scope, list(key), signal)]
        gates.append(dict(scope=scope, key=list(key), signal=signal, replicated=replicated(rows)))
    write(args.out_root/'summary/status.json', dict(status='v210_frozen_query_covariance_test_complete',
        source=protocol['source'], protocol=protocol, results=results, replication=gates,
        supported_actor_bin_cells=supported, candidate_actor_bin_cells=864,
        masks_and_weights_unchanged=True, raw_capture_files_opened=0,
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V210_FROZEN_TEST_COMPLETE_REPLICATED', sum(r['replicated'] for r in gates), '/8', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v205-root', 'v209-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
