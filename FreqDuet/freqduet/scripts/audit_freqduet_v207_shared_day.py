#!/usr/bin/env python3
"""Read saved moments: shared daily levels, covariance changes and centering."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v206_residual_sources as previous


linear, plan, base = previous.previous, previous.plan, previous.base
read, write, require = previous.read, previous.write, previous.require
matched, partition = previous.matched, previous.partition
LEVELS = ('shared_shared', 'shared_other', 'other_other')
FACTORS = ('cell_cell', 'minus_cell_parent', 'minus_parent_cell', 'parent_parent')


def outer(a):
    return a[..., :, None]*a[..., None, :]


def parent_levels(raw):
    parent = raw.sum(axis=3)
    mass, sums = parent[..., 0], parent[..., 1:3]
    means = np.divide(sums, mass[..., None], out=np.zeros_like(sums), where=mass[..., None] > 0.)
    baseline = np.stack([s.sum(axis=0)/m.sum(axis=0)[..., None] for s, m in
        ((sums[:32], mass[:32]), (sums[32:], mass[32:]))])
    baseline = np.repeat(baseline, 32, axis=0)
    numerator, denominator = (sums-mass[..., None]*baseline).sum(axis=1), mass.sum(axis=1)
    available = denominator > 0.
    # A zero-exposure day contributes no mass,not an observed zero residual mean.
    common = np.divide(numerator, denominator[..., None], out=np.zeros_like(numerator), where=available[..., None])
    spread, specific = means-baseline, means-baseline-common[:, None]
    stats = []
    for h in range(2):
        take = slice(h*32, (h+1)*32)
        total = (mass[take, ..., None, None]*outer(spread[take])).mean(axis=(0, 1))
        shared = (mass[take].mean(axis=1)[..., None, None]*outer(common[take])).mean(axis=0)
        rest = (mass[take, ..., None, None]*outer(specific[take])).mean(axis=(0, 1))
        matched(total, shared+rest, 'common/specific parent level covariance')
        stats.append(dict(total=total, shared=shared, specific=rest, mass=mass[take].mean(axis=(0, 1))))
    return means, common, available, stats


def profiles(raw, keep):
    mass, first, q = raw[..., 0], raw[..., 1:3], raw[..., 3]
    parent, common, available, stats = parent_levels(raw)
    cell = np.divide(first, mass[..., None], out=np.zeros_like(first), where=mass[..., None] > 0.)
    halves = np.stack((mass[:32].mean(axis=0), mass[32:].mean(axis=0)))
    require(np.all(halves[:, keep] > 0.), 'supported point cell has no half exposure')
    coefficient = np.divide(halves.mean(axis=0), halves, out=np.zeros_like(halves), where=halves > 0.)
    coefficient = np.repeat(np.where(keep[None], coefficient, 0.), 32, axis=0)
    weight = coefficient*mass
    g = common[:, None, :, None, :]
    p = parent[:, :, :, None, :]
    within = (coefficient*(q-mass*cell[..., 0]*cell[..., 1])).sum(axis=3)
    pairs = ((cell[..., 0], cell[..., 1]), (cell[..., 0], p[..., 1]),
        (p[..., 0], cell[..., 1]), (p[..., 0], p[..., 1]))
    factors = []
    for (x, y), sign in zip(pairs, (1., -1., -1., 1.)):
        parts = np.stack((weight*g[..., 0]*g[..., 1],
            weight*(g[..., 0]*(y-g[..., 1])+(x-g[..., 0])*g[..., 1]),
            weight*(x-g[..., 0])*(y-g[..., 1])), axis=-1)
        factors.append(sign*parts.sum(axis=3))
    split = np.stack(factors, axis=-2)
    daily = np.concatenate((within[..., None], split.reshape(*within.shape, 12)), axis=-1)
    direct = (coefficient*(q-first[..., 0]*p[..., 1]-p[..., 0]*first[..., 1]+mass*p[..., 0]*p[..., 1])).sum(axis=3)
    error = matched(daily.sum(axis=-1), direct, 'daily centered profile')
    error = max(error, matched(split.sum(axis=-2)[..., :2], np.zeros_like(split[..., 0, :2]), 'shared level cancellation'))
    return daily, common, available, stats, error


def profile_structure(daily):
    values = np.stack((daily[..., 0], daily[..., 1:].sum(axis=-1)), axis=-1)
    stats = []
    for half in (values[:32], values[32:]):
        flat = half.reshape(32, half.shape[1], 4)
        centered = flat-flat.mean(axis=0, keepdims=True)
        shared = centered.mean(axis=1)
        specific = centered-shared[:, None]
        total = np.einsum('daf,dag->fg', centered, centered)/(32*half.shape[1])
        common = shared.T@shared/32.
        rest = np.einsum('daf,dag->fg', specific, specific)/(32*half.shape[1])
        matched(total, common+rest, 'shared/specific daily profile covariance')
        stats.append(dict(total=total, shared=common, specific=rest))
    return stats


def scalar_point(raw, keep):
    point = np.zeros((2, 9))
    uniform = np.full((2, 1, 32), 1./32)
    for ai in range(raw.shape[1]):
        values = raw[:, ai]
        parent = values.sum(axis=2)
        mu = np.divide(parent[..., 1:3], parent[..., :1], out=np.zeros_like(parent[..., 1:3]), where=parent[..., :1] > 0.)
        centered = values.copy()
        centered[..., 1:3] -= values[..., :1]*mu[:, :, None]
        centered[..., 3] -= values[..., 1]*mu[..., 1, None]+values[..., 2]*mu[..., 0, None]
        centered[..., 3] += values[..., 0]*mu[..., 0, None]*mu[..., 1, None]
        point += linear.old.evaluate(linear.old.day_terms(values, centered), keep[ai], uniform)[0][0]/raw.shape[1]
    return point


def frozen_draws(daily, day_weights):
    values = daily.mean(axis=1)
    return base.average_days(values[32:], day_weights[1])-base.average_days(values[:32], day_weights[0])


def response_parts(frozen, target, point):
    result = np.stack((frozen[..., 0]-point[..., 0], frozen[..., 1:].sum(axis=-1)-point[..., 1:].sum(axis=-1),
        target-frozen.sum(axis=-1)), axis=-1)
    matched(result.sum(axis=-1), target-point.sum(axis=-1), 'V206 saved date response')
    return result


def load_case(args, actors, admission, mode, scope, beta):
    cols, missing = plan.requirements(list(linear.saved.VARS))
    require(not missing, 'required moment fields unavailable')
    operator = np.eye(linear.N)[[linear.J, linear.Y]].copy()
    operator[:, :linear.Z] -= beta
    raw, masks = [], []
    expected = {(r['actor'], tuple(k['key'])): k for r in admission['results']
        if (r['mode'], r['scope']) == (mode, scope) for k in r['keys']}
    for actor in actors:
        with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(plan.DAYS) and z['bins'].tolist() == [list(k) for k in plan.BINS], 'original day/bin order differs')
            locations = [z['keys'].tolist().index(list(k)) for k in plan.KEYS]
            values = z['moments'][:, locations][..., cols]
            keep, counts = z['supported_both'][locations], z['distinct_days'][:, locations]
            factors = z['parent_balance_factors'][:, locations]
        plan.check_support(values[..., 0], counts, keep)
        require(all(np.array_equal(keep[ki], expected[actor, key]['supported_both']) for ki, key in enumerate(plan.KEYS)), 'V199/V204 fixed mask differs')
        for h in range(2):
            values[h*32:(h+1)*32] *= factors[h, None, :, None, None]
        mass, first, second = linear.unpack(values)
        projected_first = first@operator.T
        cross = np.einsum('i,...ij,j->...', operator[0], second, operator[1], optimize=True)
        raw.append(np.concatenate((mass[..., None], projected_first, cross[..., None]), axis=-1))
        masks.append(keep)
    return np.stack(raw, axis=1), np.stack(masks)


def number(value):
    return float(value) if np.isfinite(value) else None


def run_case(args, actors, admission, mode, scope):
    with np.load(args.v205_root/'details'/f'{mode}_{scope}.npz') as z:
        require(z['states'].tolist() == list(plan.STATES) and z['keys'].tolist() == [list(k) for k in plan.KEYS], 'frozen point projection differs')
        beta, reference = z['beta_point'], z['point'][..., 3]
    with np.load(args.v206_root/'details'/f'{mode}_{scope}.npz') as z:
        require(z['actors'].tolist() == actors and z['days'].tolist() == list(plan.DAYS)
            and z['keys'].tolist() == [list(k) for k in plan.KEYS] and z['metrics'].tolist() == list(previous.METRICS), 'V206 saved cohort/readout differs')
        point, target = z['point'], z['day_only'][..., 0]
    raw, keep = load_case(args, actors, admission, mode, scope, beta)
    exact = scalar_point(raw, keep)
    error = max(matched(exact, reference, 'V205 point residual parts'), matched(previous.metrics(exact), point, 'V206 point metrics'))
    daily, common, available, level_stats, daily_error = profiles(raw, keep)
    structure = profile_structure(daily)
    point_profiles = daily[32:].mean(axis=(0, 1))-daily[:32].mean(axis=(0, 1))
    error = max(error, matched(point_profiles.sum(axis=-1), point[:, 0], 'frozen conditional point'),
        matched(point_profiles[:, 0], point[:, 1], 'frozen within-day point'))
    _, dw = base.common.weights(len(actors))
    boot = frozen_draws(daily, dw)
    responses = response_parts(boot, target, point_profiles)
    split = boot[..., 1:].reshape(len(boot), 2, 4, 3)
    parent_rows = read(args.v206_root/'summary/status.json')['results']
    results = []
    for ki, key in enumerate(plan.KEYS):
        original = next(r for r in parent_rows if (r['mode'], r['scope'], r['key']) == (mode, scope, list(key)))
        raw_level = split[:, ki, 0, :2].sum(axis=-1)
        center_level = split[:, ki, 1:, :2].sum(axis=(1, 2))
        net_error = matched(raw_level+center_level, np.zeros_like(raw_level), 'shared level matched-draw cancellation')
        temporal = np.stack((boot[:, ki, 0], boot[:, ki, 1:].sum(axis=-1)), axis=-1)
        factors = np.concatenate((boot[:, ki, :1], split[:, ki].sum(axis=-1)), axis=-1)
        levels, shared_profiles = [], []
        for h in range(2):
            stats = level_stats[h]
            total, shared, rest = (stats[n][ki] for n in ('total', 'shared', 'specific'))
            levels.append(dict(half=h, total_covariance=total.tolist(), shared_covariance=shared.tolist(), specific_covariance=rest.tolist(),
                shared_variance_fraction=[number(shared[j, j]/total[j, j]) if total[j, j] > 0. else None for j in range(2)],
                parent_mass=float(stats['mass'][ki])))
            ix = slice(2*ki, 2*ki+2)
            profile = structure[h]
            total_var, shared_var = (float(profile[n][ix, ix].sum()) for n in ('total', 'shared'))
            shared_profiles.append(dict(half=h, total_actor_day_profile_variance=total_var, shared_day_profile_variance=shared_var,
                shared_fraction=shared_var/total_var if total_var > 0. else None))
        results.append(dict(mode=mode, scope=scope, key=list(key), v205_v206_point_max_abs_error=error,
            profile_identity_max_abs_error=max(daily_error, net_error), point=point[ki].tolist(),
            shared_parent_levels=levels, shared_daily_profiles=shared_profiles,
            v206_day_response=partition(responses[:, ki], target[:, ki]-point[ki, 0],
                ['frozen_within_day', 'frozen_relative_cell_parent_means', 'operator_and_mass_response']),
            frozen_temporal=partition(temporal, boot[:, ki].sum(axis=-1), ['within_day', 'relative_cell_parent_means']),
            frozen_centering=partition(factors, boot[:, ki].sum(axis=-1), ['within_day', *FACTORS]),
            shared_location_cancellation=dict(raw_sd=float(raw_level.std()), parent_centering_sd=float(center_level.std()),
                covariance=float(np.mean((raw_level-raw_level.mean())*(center_level-center_level.mean()))), max_abs_net=net_error,
                interpretation='Cancellation is a centering identity,not a newly validated mechanism.'),
            zero_full_parent_exposure_days=[plan.DAYS[i] for i in range(64) if not available[i, ki]],
            supported_actor_bin_cells=original['supported_actor_bin_cells'], parent_mass=original['parent_mass'], retained_mass=original['retained_mass'],
            unsupported_original=original['unsupported_original'], unsupported_residual=None))
    transforms = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    cross_key = []
    for h, stats in enumerate(structure):
        covariance = transforms@stats['shared']@transforms.T
        denom = np.sqrt(max(covariance[0, 0]*covariance[1, 1], 0.))
        cross_key.append(dict(half=h, covariance=covariance.tolist(), correlation=float(covariance[0, 1]/denom) if denom > 0. else None))
    target_path = args.out_root/'details'/f'{mode}_{scope}.npz'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, actors=actors, days=plan.DAYS, keys=plan.KEYS, factors=FACTORS, levels=LEVELS,
        point_operator=beta, daily_profiles=daily, shared_parent_shift=np.where(available[..., None], common, np.nan),
        shared_shift_available=available, frozen_point=point_profiles, frozen_boot=boot, v206_day_only=target,
        responses=responses, profile_total_covariance=np.stack([s['total'] for s in structure]),
        profile_shared_covariance=np.stack([s['shared'] for s in structure]), profile_specific_covariance=np.stack([s['specific'] for s in structure]))
    print('V207_SHARED_DAY_CASE_COMPLETE', mode, scope, 'error', error, 'profile', daily_error, flush=True)
    return results, dict(mode=mode, scope=scope, cross_key_shared_daily_profiles=cross_key)


def registration(parent):
    return dict(stage='v207_shared_day_residual_comovement', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        parent_source='3db715e6c0', calibration_days=list(plan.DAYS), keys=[list(k) for k in plan.KEYS], frozen=parent['protocol']['frozen'],
        fixed='V199 masks,bins,19-day threshold,V191 factors,original denominators,all12 actors and64 days unchanged.',
        projection='Use SAVED V205 point operator only for a fixed-reference diagnostic;never replace V206 refitted readout. '
            'Difference from saved V206 day-only draws remains explicit as operator-and-mass response;no new model fit.',
        levels='Remove each actor/key half mass-weighted parent-mean baseline. Shared day shift is full-parent-mass weighted '
            'across original actors. Parent mean scatter=shared shift+actor-specific deviation. '
            'All9 bins enter parent means;no low-support conditional decomposition. Zero exposure is not an observed zero error.',
        profiles='For each original supported actor/key/bin,freeze c_h=average_half_mass/point_half_mass. '
            'Daily profile=sum c_h*centered residual cross mass. Its half difference reproduces V205 conditional point. '
            'Split into within-cell covariance and relative cell/parent mean product;split4 signed mean products '
            'into common/common,common/other,other/other. Common-location terms cancel by identity.',
        shared='Within EACH half,center each actor daily nominal profile by its32-day mean;common profile=equal-actor mean. '
            'Actor/day covariance matrix=common-day+actor-specific matrices. Keep within/relative cross covariance and '
            'both-key shared covariance,not just diagonal variances. No query-time matching or causal propagation claim.',
        resampling='Reuse V20610000 independent-half SHARED day weights,seed20260803. Frozen within response+frozen relative '
            'response+operator/mass remainder exactly reproduce saved V206 day-only response. Signed covariance allocation,not new gates.',
        limitations='Reused calibration days;offline label diagnostics,not query-time deployable features. '
            'Common parent location scatter and daily covariance-profile uncertainty have different scales/populations;do not add them. '
            'Frozen reference is an attribution anchor,not a new predictor or confidence interval. No independent validation.',
        date_deletions=0, actor_deletions=0, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v207_*.py', '-v'], check=True)
    parent, admission = read(args.v206_root/'summary/status.json'), read(args.v204_root/'summary/status.json')
    require(parent['source'] == '3db715e6c0' and parent['v205_residual_points_and_individual_draws_reproduced']
        and parent['masks_and_weights_unchanged'] and parent['protocol']['calibration_days'] == list(plan.DAYS), 'requires completed frozen V206')
    require(admission['source'] == '4b85f9159c' and len(admission['actors']) == 12, 'original actor cohort differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite diagnostic protocol')
    protocol = registration(parent)
    write(args.out_root/'protocol.json', protocol)
    print('V207_SHARED_DAY_PROTOCOL_SEALED', flush=True)
    results, cases = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, admission['actors'], admission, mode, scope) for mode, scope in product(plan.MODES, plan.SCOPES)]
        for i, job in enumerate(as_completed(jobs), 1):
            rows, case = job.result()
            results.extend(rows)
            cases.append(case)
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"shared_day_case"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope'], r['key']))
    write(args.out_root/'summary/status.json', dict(status='v207_shared_day_residual_analysis_complete', source=protocol['source'],
        protocol=protocol, results=results, cases=cases, v205_v206_points_reproduced=True,
        saved_v206_day_responses_reconstructed=True, shared_covariance_identities_checked=True, masks_and_weights_unchanged=True,
        date_deletions=0, actor_deletions=0, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V207_SHARED_DAY_COMPLETE_NO_MODEL_OR_POLICY_CHANGE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v204-root', 'v205-root', 'v206-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
