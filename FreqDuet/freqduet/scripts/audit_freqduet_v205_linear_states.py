#!/usr/bin/env python3
"""Execute V204's fixed-key linear state decomposition from saved moments only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v204_observable_states as plan


old, base, saved = plan.previous, plan.base, plan.saved
read, write, require = plan.read, plan.write, plan.require
N, Z = len(plan.FIELDS), len(plan.STATES)
J, Y = Z, Z+1
TRI = np.triu_indices(N)
PAIRS = ('state_state', 'residualJ_stateY', 'stateJ_residualY', 'residual_residual')
GROUPS = dict(old.GROUPS, within=(0,))


def outer(a, b):
    return a[..., :, None]*b[..., None, :]


def unpack(values):
    second = np.zeros((*values.shape[:-1], N, N))
    second[..., TRI[0], TRI[1]] = values[..., 1+N:]
    second[..., TRI[1], TRI[0]] = values[..., 1+N:]
    return values[..., 0], values[..., 1:1+N], second


def prepare(values, keep):
    mass, first, second = unpack(values)
    parent_mass, parent_first = mass.sum(axis=2), first.sum(axis=2)
    mu = np.divide(parent_first, parent_mass[..., None], out=np.zeros_like(parent_first), where=parent_mass[..., None] > 0.)
    parent = mass[..., None]*mu[:, :, None, :]
    cc = np.divide(outer(first, first), mass[..., None, None], out=np.zeros_like(second), where=mass[..., None, None] > 0.)
    cp, pp = outer(first, mu[:, :, None, :]), mass[..., None, None]*outer(mu, mu)[:, :, None]
    original = second[..., J, Y]-cp[..., J, Y]-cp[..., Y, J]+pp[..., J, Y]
    # Parent means use ALL9 cells; unsupported cells receive no state partition.
    matrices = np.stack((second, cc, cp, pp), axis=-3)
    matrices = np.where(keep[None, :, :, None, None, None], matrices, 0.)
    firsts = np.where(keep[None, :, :, None, None], np.stack((first, parent), axis=-2), 0.)
    return dict(mass=mass, first=firsts, matrices=matrices, keep=keep,
        unsupported=np.where(~keep[None], original, 0.).sum(axis=2))


def half(cache, start, weights):
    take = slice(start, start+32)
    avg = lambda name: base.average_days(cache[name][take], weights)
    mass, first, q = avg('mass'), avg('first'), avg('matrices')
    x, p = first[..., 0, :], first[..., 1, :]
    denom = mass[..., None, None]
    divide = lambda value: np.divide(value, denom, out=np.zeros_like(value), where=denom > 0.)
    cc, cp, pp = (divide(v) for v in (outer(x, x), outer(x, p), outer(p, p)))
    neg_cp = cp-q[..., 2, :, :]
    parts = np.stack((q[..., 0, :, :]-q[..., 1, :, :], q[..., 1, :, :]-cc,
        neg_cp, np.swapaxes(neg_cp, -1, -2), q[..., 3, :, :]-pp,
        cc, -cp, -np.swapaxes(cp, -1, -2), pp), axis=-3)
    return mass, parts


def actor_weight(values, weights):
    factor = weights.reshape((len(weights),)+(1,)*(values.ndim-1))
    return np.where(factor > 0., factor*values, 0.)


def aggregate(caches, actor_weights, day_weights):
    b = len(actor_weights)
    conditional = np.zeros((b, 2, 9, N, N))
    mix = np.zeros_like(conditional)
    endpoints = np.zeros((b, 2, 2, 9, N, N))
    unsupported = np.zeros((b, 2))
    for ai, cache in enumerate(caches):
        a, c = [half(cache, 32*h, day_weights[h]) for h in range(2)]
        norms = [np.where(cache['keep'][None, :, :, None, None, None],
            np.divide(parts, mass[..., None, None, None], out=np.full_like(parts, np.nan),
                where=mass[..., None, None, None] > 0.), 0.) for mass, parts in (a, c)]
        dm, mm = c[0]-a[0], (c[0]+a[0])/2.
        conditional += actor_weight((mm[..., None, None, None]*(norms[1]-norms[0])).sum(axis=2), actor_weights[:, ai])
        mix += actor_weight((dm[..., None, None, None]*(norms[1]+norms[0])/2.).sum(axis=2), actor_weights[:, ai])
        endpoints += actor_weight(np.stack((a[1].sum(axis=2), c[1].sum(axis=2)), axis=1), actor_weights[:, ai])
        low = base.average_days(cache['unsupported'][32:], day_weights[1])-base.average_days(cache['unsupported'][:32], day_weights[0])
        unsupported += actor_weight(low, actor_weights[:, ai])
    # One reference across keys/actors/halves, strictly within-cell AND within-day.
    reference = endpoints[:, :, :, 0].mean(axis=1).sum(axis=1)
    require(np.allclose(conditional+mix, endpoints[:, 1]-endpoints[:, 0], rtol=1e-8, atol=1e-8),
        'conditional/mix matrix identity differs;do not change support')
    return conditional, mix, endpoints, unsupported, reference


def projection(reference):
    cc = reference[..., :Z, :Z]
    sd = np.sqrt(np.maximum(np.diagonal(cc, axis1=-2, axis2=-1), 0.))
    scale = outer(sd, sd)
    corr = np.divide(cc, scale, out=np.zeros_like(cc), where=scale > 1e-12)
    cross = np.divide(reference[..., J:Y+1, :Z], sd[..., None, :],
        out=np.zeros((*reference.shape[:-2], 2, Z)), where=sd[..., None, :] > 1e-12)
    beta = cross@np.linalg.pinv(corr, rcond=1e-10, hermitian=True)
    beta = np.divide(beta, sd[..., None, :], out=np.zeros_like(beta), where=sd[..., None, :] > 1e-12)
    fitted = np.zeros((*reference.shape[:-2], 2, N))
    fitted[..., :Z] = beta
    residual = np.eye(N)[[J, Y]]-fitted
    left = np.stack((fitted[..., 0, :], residual[..., 0, :], fitted[..., 0, :], residual[..., 0, :]), axis=-2)
    right = np.stack((fitted[..., 1, :], fitted[..., 1, :], residual[..., 1, :], residual[..., 1, :]), axis=-2)
    active = sd > 1e-12
    valid = active.any(axis=-1)
    left[~valid], right[~valid] = np.nan, np.nan
    eigen = np.abs(np.linalg.eigvalsh(corr))
    rank = (eigen > 1e-10*eigen.max(axis=-1, keepdims=True)).sum(axis=-1)
    return left, right, beta, active, rank, valid


def projected(matrices, left, right):
    return np.einsum('bpi,bkqij,bpj->bkqp', left, matrices, right, optimize=True)


def decompose(aggregated):
    conditional, mix, endpoints, _, reference = aggregated
    left, right, beta, active, rank, valid = projection(reference)
    result = [projected(value, left, right) for value in (conditional, mix)]
    b = len(left)
    ep = projected(endpoints.reshape(b, 4, 9, N, N), left, right).reshape(b, 2, 2, 9, 4)
    error = 0.
    for value, original in zip((*result, ep), (conditional, mix, endpoints)):
        delta = value.sum(axis=-1)-original[..., J, Y]
        require(np.allclose(delta[valid], 0., rtol=0., atol=1e-8), 'four state/residual pairs do not reproduce original')
        error = max(error, float(np.abs(delta[valid]).max(initial=0.)))
    return (*result, ep, beta, active, rank, valid, error)


def reproduce(aggregated, reference, index):
    conditional, mix, endpoints, low, _ = aggregated
    extra = np.stack((mix[..., J, Y].sum(axis=-1), (conditional+mix)[..., J, Y].sum(axis=-1), low), axis=-1)
    actual = (conditional[..., J, Y], endpoints[..., J, Y], extra)
    expected = [reference[name][index] for name in ('boot', 'endpoints_boot', 'extra_boot')]
    # V203 stores point endpoints,not endpoint draws; their draw counterpart is
    # recovered directly from its saved V202 daily moments in the caller.
    error = 0.
    for got, want in zip(actual, expected):
        require(np.allclose(got, want, rtol=0., atol=1e-8), 'V203 point/matched-draw reproduction differs')
        error = max(error, float(np.abs(got-want).max(initial=0.)))
    return error


def load_case(args, actors, mode, scope, admission):
    cols, missing = plan.requirements(list(saved.VARS))
    require(not missing, 'required state cross moments missing')
    caches, original_days = [], []
    expected = {(r['actor'], tuple(k['key'])): k for r in admission['results']
        if (r['mode'], r['scope']) == (mode, scope) for k in r['keys']}
    for actor in actors:
        with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(plan.DAYS) and z['bins'].tolist() == [list(k) for k in plan.BINS], 'original day/bin grid differs')
            keys = z['keys'].tolist()
            ix = [keys.index(list(k)) for k in plan.KEYS]
            values = z['moments'][:, ix][..., cols]
            keep, factors = z['supported_both'][ix], z['parent_balance_factors'][:, ix]
            counts = z['distinct_days'][:, ix]
        plan.check_support(values[..., 0], counts, keep)
        for ki, key in enumerate(plan.KEYS):
            require(np.array_equal(keep[ki], expected[actor, key]['supported_both']), 'V204 frozen support differs')
        for h in range(2):
            values[h*32:(h+1)*32] *= factors[h, None, :, None, None]
        mass, first, second = unpack(values)
        raw = np.stack((mass, first[..., J], first[..., Y], second[..., J, Y]), axis=-1)
        with np.load(args.v202_root/'days'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(plan.DAYS) and z['keys'].tolist() == [list(k) for k in plan.KEYS]
                and np.array_equal(z['supported_both'], keep), 'V202 comparison grid differs')
            require(np.allclose(raw, z['raw'][..., [0, 1, 4, 5]], rtol=0., atol=1e-8), 'once-weighted V202 raw inputs differ')
            original_days.append((old.day_terms(raw, z['centered'][..., [0, 1, 4, 5]]), keep))
        caches.append(prepare(values, keep))
    print('V205_SAVED_STATE_MOMENTS_LOADED', mode, scope, flush=True)
    return caches, original_days


def original_endpoints(days, aw, dw):
    result = np.zeros((len(aw), 2, 2, 9))
    for ai, (values, keep) in enumerate(days):
        ep = old.evaluate(values, keep, dw)[1]
        result += actor_weight(ep, aw[:, ai])
    return result


def estimates(point, boot, family=False):
    values = {name: (point[..., i], boot[..., i]) for i, name in enumerate(PAIRS)}
    values.update(mixed=(point[..., 1:3].sum(axis=-1), boot[..., 1:3].sum(axis=-1)),
        original=(point.sum(axis=-1), boot.sum(axis=-1)))
    result = {name: saved.prior.interval(p, b) for name, (p, b) in values.items()}
    if family:
        for name in ('state_state', 'mixed'):
            result[name]['family'] = saved.prior.interval(*values[name], alpha=.05/16)
    return result


def run_case(args, actors, admission, mode, scope):
    caches, days = load_case(args, actors, mode, scope, admission)
    aw, dw = base.common.weights(len(actors))
    weights = np.full((1, len(actors)), 1./len(actors))
    uniform = np.full((2, 1, 32), 1./32)
    with np.load(args.v203_root/'details'/f'{mode}_{scope}.npz') as z:
        require(z['keys'].tolist() == [list(k) for k in plan.KEYS] and z['parts'].tolist() == list(old.PARTS), 'V203 fixed part order differs')
        reference = {name: z[name] for name in ('point', 'boot', 'endpoints', 'extra_point', 'extra_boot')}
    point_values = aggregate(caches, weights, uniform)
    point = decompose(point_values)
    point_ref = dict(boot=reference['point'][None], endpoints_boot=reference['endpoints'][None], extra_boot=reference['extra_point'][None])
    error = reproduce(point_values, point_ref, slice(None))
    draws = len(aw)
    boot, mix = np.empty((draws, 2, 9, 4)), np.empty((draws, 2, 9, 4))
    ep, betas = np.empty((draws, 2, 2, 9, 4)), np.empty((draws, 2, Z))
    active, ranks, valid = np.empty((draws, Z), bool), np.empty(draws, int), np.empty(draws, bool)
    pair_error = point[-1]
    for start in range(0, draws, 128):
        take = slice(start, min(start+128, draws))
        values = aggregate(caches, aw[take], dw[:, take])
        result = decompose(values)
        boot[take], mix[take], ep[take], betas[take], active[take], ranks[take], valid[take] = result[:-1]
        pair_error = max(pair_error, result[-1])
        ref = dict(boot=reference['boot'][take], extra_boot=reference['extra_boot'][take],
            endpoints_boot=original_endpoints(days, aw[take], dw[:, take]))
        error = max(error, reproduce(values, ref, slice(None)))
        if (start//128) % 16 == 0:
            print('V205_MATCHED_STATE_DRAWS', mode, scope, start, '/10000', flush=True)
    require(np.isfinite(boot[valid]).all(), 'undefined state result with an admitted common operator')
    p, pm, pe, beta, pa, pr, pv, _ = point
    results = []
    originals = [r for r in read(args.v203_root/'summary/status.json')['results'] if (r['mode'], r['scope']) == (mode, scope)]
    for ki, key in enumerate(plan.KEYS):
        original = next(r for r in originals if r['key'] == list(key))
        groups = {name: estimates(p[0, ki, list(ix)].sum(axis=0), boot[:, ki, list(ix)].sum(axis=1), name == 'conditional')
            for name, ix in GROUPS.items()}
        total = boot[:, ki].sum(axis=1)
        cancellation = old.cancellation.partition(total, total.sum(axis=1), list(PAIRS)) if np.isfinite(total).all() else None
        results.append(dict(mode=mode, scope=scope, key=list(key), supported_actor_bin_cells=original['supported_actor_bin_cells'],
            parent_mass=original['parent_mass'], retained_mass=original['retained_mass'],
            groups=groups, mix=estimates(pm[0, ki].sum(axis=0), mix[:, ki].sum(axis=1)),
            endpoints={label: estimates(pe[0, h, ki].sum(axis=0), ep[:, h, ki].sum(axis=1)) for h, label in enumerate(('first32', 'last32'))},
            unsupported_original=original['extra']['unsupported_delta'], cancellation=cancellation,
            v203_max_abs_reproduction_error=error, four_pair_max_abs_error=pair_error))
    target = args.out_root/'details'/f'{mode}_{scope}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, keys=plan.KEYS, states=plan.STATES, parts=old.PARTS, pairs=PAIRS,
        point=p[0], boot=boot, mix_point=pm[0], mix_boot=mix, endpoints_point=pe[0], endpoints_boot=ep,
        beta_point=beta[0], beta_boot=betas, active_point=pa[0], active_boot=active, rank_point=pr[0], rank_boot=ranks, valid_boot=valid)
    operator = dict(mode=mode, scope=scope, states=list(plan.STATES), point_rank=int(pr[0]),
        point_active=pa[0].tolist(), zero_variance_states=[n for n, yes in zip(plan.STATES, pa[0]) if not yes],
        coefficient_point=beta[0].tolist() if pv[0] else None, bootstrap_rank_range=[int(ranks.min()), int(ranks.max())],
        bootstrap_active_counts=active.sum(axis=0).tolist(), valid_draws=int(valid.sum()),
        interpretation='One shared column-space projection;collinear state coefficients are not separately identified.')
    print('V205_LINEAR_STATE_CASE_COMPLETE', mode, scope, 'error', error, 'rank', int(pr[0]), flush=True)
    return results, operator


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v205_*.py', '-v'], check=True)
    admission = read(args.v204_root/'summary/status.json')
    sealed = read(args.v204_root/'protocol.json')
    expected = plan.registration()
    expected['source'] = '4b85f9159c'
    require(admission['source'] == '4b85f9159c' and admission['additive_state_test_available']
        and admission['missing_required_fields'] == admission['missing_parent_support'] == []
        and sealed == expected, 'requires complete frozen V204 protocol/inventory')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite sealed diagnostic')
    protocol = dict(stage='v205_v204_linear_state_readout', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        parent_source=sealed['source'], v204=sealed, state_operator_fitted=True, state_associations_estimated=True,
        inputs='V199 saved moments,V202 daily reference,V203 saved point/draws,V204 sealed inventory ONLY.',
        interval_fit=False, predictor_fit=False, raw_capture_files_opened=0, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)
    write(args.out_root/'protocol.json', protocol)
    print('V205_FROZEN_LINEAR_STATE_READOUT_SEALED', flush=True)
    results, operators = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, admission['actors'], admission, mode, scope) for mode, scope in product(plan.MODES, plan.SCOPES)]
        for i, job in enumerate(as_completed(jobs), 1):
            rows, operator = job.result()
            results.extend(rows)
            operators.append(operator)
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"linear_state_case"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope'], r['key']))
    write(args.out_root/'summary/status.json', dict(status='v205_linear_state_decomposition_complete', source=protocol['source'],
        protocol=protocol, results=results, operators=operators,
        primary_contrasts=[dict(mode=r['mode'], scope=r['scope'], key=r['key'], term=t, estimate=r['groups']['conditional'][t]['family'])
            for r in results for t in ('state_state', 'mixed')],
        v203_points_and_individual_draws_reproduced=True, masks_and_weights_unchanged=True, unsupported_separate=True,
        state_operator_fitted=True, state_associations_estimated=True, raw_capture_files_opened=0,
        interval_fit=False, predictor_fit=False, confirmation_outcomes_read=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V205_LINEAR_STATE_COMPLETE_NO_PREDICTOR_OR_POLICY_CHANGE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v202-root', 'v203-root', 'v204-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
