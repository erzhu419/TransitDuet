#!/usr/bin/env python3
"""Split cell-mean co-movement and parent/day centering using V202 daily moments."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v202_key_day_covariance as previous
import audit_freqduet_v202_cancellation as cancellation


base, read, write, require = previous.base, previous.read, previous.write, previous.require
PARTS = ('within', 'between_cell', 'between_minus_cell_parent_y', 'between_minus_parent_j_cell', 'between_parent',
    'mean_cell', 'mean_minus_cell_parent_y', 'mean_minus_parent_j_cell', 'mean_parent')
GROUPS = dict(conditional=tuple(range(9)), conditional_between=(1, 2, 3, 4), conditional_mean_product=(5, 6, 7, 8),
    between_centering=(2, 3, 4), mean_centering=(6, 7, 8), cell_raw_contribution=(0, 1, 5),
    parent_centering_contribution=(2, 3, 4, 6, 7, 8))
SIGNS = np.array([1., -1., -1., 1.])
FIELDS = ['mass', 'sum_J', 'sum_I', 'sum_A', 'sum_Y', 'sum_JY', 'sum_IY', 'sum_AY']


def day_terms(raw, centered):
    """Both inputs already carry the frozen V191 factors; never apply them again."""
    parent = raw.sum(axis=2)
    mu = np.divide(parent[..., 1:3], parent[..., :1], out=np.zeros_like(parent[..., 1:3]), where=parent[..., :1] > 0.)
    mass, x, y, xy = (raw[..., i] for i in range(4))
    pj, py = mu[..., 0, None], mu[..., 1, None]
    require(np.allclose(centered[..., 0], mass, rtol=0., atol=1e-12)
        and np.allclose(centered[..., 1], x-mass*pj, rtol=0., atol=1e-9)
        and np.allclose(centered[..., 2], y-mass*py, rtol=0., atol=1e-9)
        and np.allclose(centered[..., 3], xy-x*py-y*pj+mass*pj*py, rtol=0., atol=1e-8),
        'V202 full-parent/day centering differs')
    cell_xy = np.divide(x*y, mass, out=np.zeros_like(x), where=mass > 0.)
    return np.stack((mass, x, y, mass*pj, mass*py, cell_xy, x*py, y*pj, mass*pj*py,
        xy-cell_xy, centered[..., 3]), axis=-1)


def half(days, weights):
    pooled = base.average_days(days, weights)
    mass, x, y, pj, py = (pooled[..., i] for i in range(5))
    products = np.stack((x*y, x*py, pj*y, pj*py), axis=-1)
    mean_products = np.divide(products, mass[..., None], out=np.zeros_like(products), where=mass[..., None] > 0.)
    between = (pooled[..., 5:9]-mean_products)*SIGNS
    means = mean_products*SIGNS
    parts = np.concatenate((pooled[..., 9:10], between, means), axis=-1)
    require(np.allclose(parts.sum(axis=-1), pooled[..., 10], rtol=1e-9, atol=1e-8),
        'within plus cell/parent mean partitions do not reproduce centered cross mass')
    return pooled, parts


def evaluate(days, keep, weights):
    a, b = [half(days[h*32:(h+1)*32], weights[h]) for h in range(2)]
    norms = [np.where(keep[None, :, :, None], np.divide(parts, pooled[..., :1],
        out=np.full_like(parts, np.nan), where=pooled[..., :1] > 0.), 0.) for pooled, parts in (a, b)]
    dm, mm = b[0][..., :1]-a[0][..., :1], (b[0][..., :1]+a[0][..., :1])/2.
    conditional = (mm*(norms[1]-norms[0])).sum(axis=2)
    mix = (dm*(norms[1]+norms[0])/2.).sum(axis=(2, 3))
    delta = b[0][..., 10]-a[0][..., 10]
    supported = np.where(keep, delta, 0.).sum(axis=2)
    unsupported = np.where(~keep, delta, 0.).sum(axis=2)
    require(np.allclose(conditional.sum(axis=-1)+mix, supported, rtol=1e-9, atol=1e-8),
        'conditional parts/mix no longer reproduce supported change')
    endpoints = np.stack([np.where(keep[None, :, :, None], parts, 0.).sum(axis=2) for _, parts in (a, b)], axis=1)
    return conditional, endpoints, np.stack((mix, supported, unsupported), axis=-1)


def reproduce(point, boot, extra, extra_boot, reference):
    mapping = dict(mix=(extra[..., 0], extra_boot[..., 0]), conditional=(point.sum(axis=-1), boot.sum(axis=-1)),
        conditional_within_day=(point[..., 0], boot[..., 0]),
        conditional_between_day=(point[..., 1:5].sum(axis=-1), boot[..., 1:5].sum(axis=-1)),
        conditional_mean_product=(point[..., 5:9].sum(axis=-1), boot[..., 5:9].sum(axis=-1)),
        supported_delta=(extra[..., 1], extra_boot[..., 1]), unsupported_delta=(extra[..., 2], extra_boot[..., 2]))
    with np.load(reference) as z:
        require(z['keys'].tolist() == [list(k) for k in previous.KEYS], 'V202 reference keys differ')
        rp, rb, names = z['point'][..., 0], z['boot'][..., 0], z['metrics'].tolist()
    error = 0.
    for name, (p, b) in mapping.items():
        i = names.index(name)
        require(np.allclose(p, rp[:, i], rtol=0., atol=1e-8)
            and np.allclose(b, rb[:, :, i], rtol=0., atol=1e-8), 'V202 point or matched draw differs')
        error = max(error, float(np.abs(p-rp[:, i]).max()), float(np.abs(b-rb[:, :, i]).max()))
    return error


def run_case(args, actors, mode, scope):
    aw, dw = base.common.weights(len(actors))
    n, draws = len(actors), base.saved.prior.DRAWS
    point, boot = np.zeros((2, 9)), np.zeros((draws, 2, 9))
    endpoints, extra = np.zeros((2, 2, 9)), np.zeros((2, 3))
    extra_boot = np.zeros((draws, 2, 3))
    mass, supported, counts = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2, dtype=int)
    for ai, actor in enumerate(actors):
        with np.load(args.v202_root/'days'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(base.DAYS) and z['keys'].tolist() == [list(k) for k in previous.KEYS]
                and z['fields'].tolist() == FIELDS, 'V202 day/key/field grid differs')
            raw, centered, keep = z['raw'][..., [0, 1, 4, 5]], z['centered'][..., [0, 1, 4, 5]], z['supported_both']
        require(raw.shape == centered.shape == (64, 2, 9, 4) and keep.shape == (2, 9), 'V202 daily schema differs')
        days = day_terms(raw, centered)
        p, ep, ex = evaluate(days, keep, np.full((2, 1, 32), 1./32))
        point += p[0]/n
        endpoints += ep[0]/n
        extra += ex[0]/n
        counts += keep.sum(axis=1)
        for h in range(2):
            m = raw[h*32:(h+1)*32, ..., 0].mean(axis=0)
            mass[h] += m.sum(axis=1)/n
            supported[h] += np.where(keep, m, 0.).sum(axis=1)/n
        for start in range(0, draws, 256):
            stop = min(start+256, draws)
            p, _, ex = evaluate(days, keep, dw[:, start:stop])
            boot[start:stop] += aw[start:stop, ai, None, None]*p
            extra_boot[start:stop] += aw[start:stop, ai, None, None]*ex
        print('V203_SAVED_DAY_ACTOR', mode, scope, actor, flush=True)
    require(np.isfinite(boot).all(), 'undefined draw;do not replace cells or masks')
    error = reproduce(point, boot, extra, extra_boot, args.v202_root/'details'/f'{mode}_{scope}_draws.npz')
    reference = read(args.v202_root/'summary/status.json')['results']
    results = []
    for ki, key in enumerate(previous.KEYS):
        old = next(r for r in reference if (r['mode'], r['scope'], r['key']) == (mode, scope, list(key)))
        require(counts[ki] == old['supported_actor_bin_cells'] and np.allclose(mass[:, ki], old['parent_mass'], atol=1e-12, rtol=0.)
            and np.allclose(supported[:, ki], old['retained_mass'], atol=1e-12, rtol=0.), 'frozen V202 support/exposure differs')
        parts = {name: base.saved.prior.interval(point[ki, i], boot[:, ki, i]) for i, name in enumerate(PARTS)}
        groups = {name: base.saved.prior.interval(point[ki, list(ix)].sum(), boot[:, ki, list(ix)].sum(axis=1)) for name, ix in GROUPS.items()}
        whole = np.stack((boot[:, ki, 0], boot[:, ki, [1, 5]].sum(axis=1),
            boot[:, ki, list(GROUPS['parent_centering_contribution'])].sum(axis=1)), axis=1)
        results.append(dict(mode=mode, scope=scope, key=list(key), v202_max_abs_reproduction_error=error,
            supported_actor_bin_cells=int(counts[ki]), parent_mass=mass[:, ki].tolist(), retained_mass=supported[:, ki].tolist(),
            conditional_parts=parts, groups=groups,
            endpoints={name: dict(zip(('first32', 'last32'), endpoints[:, ki, i].tolist())) for i, name in enumerate(PARTS)},
            extra={name: base.saved.prior.interval(extra[ki, i], extra_boot[:, ki, i]) for i, name in enumerate(('mix', 'supported_delta', 'unsupported_delta'))},
            cancellation=dict(between=cancellation.partition(boot[:, ki, 1:5], boot[:, ki, 1:5].sum(axis=1), list(PARTS[1:5])),
                mean_product=cancellation.partition(boot[:, ki, 5:9], boot[:, ki, 5:9].sum(axis=1), list(PARTS[5:9])),
                conditional=cancellation.partition(whole, boot[:, ki].sum(axis=1), ['within', 'cell_raw_means', 'parent_centering']))))
    target = args.out_root/'details'/f'{mode}_{scope}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, keys=previous.KEYS, parts=PARTS, point=point, boot=boot, endpoints=endpoints,
        extra_point=extra, extra_boot=extra_boot)
    print('V203_PARENT_CENTERING_CASE_COMPLETE', mode, scope, flush=True)
    return results


def registration():
    return dict(stage='v203_saved_daily_parent_centering', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        parent_source='8829065fe1', calibration_days=list(base.DAYS), frozen=previous.registration()['frozen'], keys=list(previous.KEYS),
        inputs='V202 saved daily raw/centered moments,mask and V202 matched-draw reference ONLY;no V199/V194 files or trajectories.',
        weights='V202 values ALREADY contain frozen V191 factors and nominal query/episode mass. Do not rebalance or renormalize. '
            'Retain saved V199 masks,original bins,19-day threshold,equal12 actors/day. Parent means use ALL9 cells before masks.',
        algebra='For cell-day means j,y and full parent-day means pJ,pY,weighted centered between-day covariance equals '
            'Cov(j,y)-Cov(j,pY)-Cov(pJ,y)+Cov(pJ,pY). EACH covariance uses this cell daily mass,not parent daily mass. '
            'Pooled mean-product splits into the same4 signed products. Within-day covariance is invariant.',
        readout='Apply unchanged V198 symmetric average-mass times half-difference to every part;sum reproduces V202. '
            'cell_raw_contribution and parent_centering_contribution are algebraic parts,not alternate policy/predictor estimates.',
        uncertainty='Same10000 shared-actor/independent-half-day draws,seed20260803;reproduce V202 points AND individual draws. '
            'Retain full signed covariance cancellation,including negative or>100% allocations. All95% intervals exploratory,not new gates.',
        unsupported='Mass and aggregate centered cross change only;no low-support cell decomposition or relaxed admission.',
        limitations='Reused development keys/days. Centering is a fixed estimand operation,not a causal intervention or measured deployment gain. '
            'These weighted day-mean covariances are not the bootstrap day-variance channels from V201.',
        raw_capture_files_opened=0, earlier_moment_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v203_*.py', '-v'], check=True)
    parent = read(args.v202_root/'summary/status.json')
    require(parent['source'] == '8829065fe1' and parent['masks_and_weights_unchanged'] and parent['within_between_identities_checked']
        and parent['protocol']['calibration_days'] == list(base.DAYS), 'requires frozen V202 daily moments')
    actors = sorted(int(p.stem.split('_')[0]) for p in (args.v202_root/'days').glob('*_old_clearance.npz'))
    require(len(actors) == len(set(actors)) == 12, 'V202 actor grid differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite readout protocol')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    results = []
    print('V203_SAVED_DAY_PROTOCOL_SEALED', flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, actors, mode, scope) for mode, scope in product(('old', 'handoff'), base.PRIMARY)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.extend(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"saved_day_centering"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope'], r['key']))
    write(args.out_root/'summary/status.json', dict(status='v203_parent_centering_complete', source=protocol['source'],
        protocol=protocol, results=results, v202_points_and_matched_draws_reproduced=True, centering_identities_checked=True,
        masks_and_weights_unchanged=True, unsupported_separate=True, raw_capture_files_opened=0, earlier_moment_files_opened=0,
        interval_fit=False, predictor_fit=False, confirmation_outcomes_read=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V203_PARENT_CENTERING_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v202-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    run(p.parse_args())
