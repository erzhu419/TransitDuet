#!/usr/bin/env python3
"""Localize V200 uncertainty using frozen original64 moment statistics only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v200_conditional_decomposition as previous


read, write, require = previous.read, previous.write, previous.require
NAMES = (*previous.TERMS, 'supported_delta', 'unsupported_delta', 'parent_delta')
REPORT = ('mix', 'conditional', 'unsupported_delta', 'parent_delta')
CHANNELS = ('actor', 'day', 'actor_day_interaction')


def key_terms(first, last, keep):
    cells = []
    for half in (first, last):
        mass, x, y, q = (half[..., i] for i in range(4))
        t = np.where(keep, np.divide(q, mass, out=np.full_like(q, np.nan), where=mass > 0.), 0.)
        product_mean = np.where(keep, np.divide(x*y, mass**2, out=np.full_like(q, np.nan), where=mass > 0.), 0.)
        cells.append((np.where(keep, mass, 0.), t, t-product_mean, product_mean))
    a, b = cells
    dm, mm = b[0]-a[0], (b[0]+a[0])/2.
    six = [(dm*(a[1]+b[1])/2.).sum(axis=-1), (mm*(b[1]-a[1])).sum(axis=-1),
        (mm*(b[2]-a[2])).sum(axis=-1), (mm*(b[3]-a[3])).sum(axis=-1),
        (dm*(a[2]+b[2])/2.).sum(axis=-1), (dm*(a[3]+b[3])/2.).sum(axis=-1)]
    delta = last[..., 3]-first[..., 3]
    return np.stack((*six, np.where(keep, delta, 0.).sum(axis=-1),
        np.where(~keep, delta, 0.).sum(axis=-1), delta.sum(axis=-1)), axis=-1)


def variance_channels(actor_day):
    """Integrate multinomial actor draws exactly, conditional on saved day draws."""
    n = len(actor_day)
    actor_mean = actor_day.mean(axis=1)
    day_mean = actor_day.mean(axis=0)
    actor = actor_mean.var(axis=0)/n
    day = day_mean.var(axis=0)
    residual = actor_day-actor_mean[:, None, :]
    interaction = residual.var(axis=0).mean(axis=0)/n
    conditional_actor = actor_day.var(axis=0).mean(axis=0)/n
    require(np.allclose(actor+interaction, conditional_actor, rtol=1e-9, atol=1e-10),
        'actor-main plus interaction differs from conditional actor variance')
    return np.stack((actor, day, interaction))


def covariance_allocation(parts, total):
    require(np.allclose(parts.sum(axis=1), total, rtol=1e-9, atol=1e-8), 'contributions do not sum to total draw')
    allocation = np.mean((parts-parts.mean(axis=0))*(total-total.mean(axis=0))[:, None, :], axis=0)
    require(np.allclose(allocation.sum(axis=0), total.var(axis=0), rtol=1e-9, atol=1e-8),
        'signed covariance allocation does not reproduce total variance')
    return allocation


def leave_day(values, keep):
    means = [v.mean(axis=0, keepdims=True) for v in (values[:32], values[32:])]
    result = []
    for half in range(2):
        v = values[half*32:(half+1)*32]
        omitted = (v.sum(axis=0, keepdims=True)-v)/31.
        args = [np.broadcast_to(m, omitted.shape) for m in means]
        args[half] = omitted
        result.append(key_terms(*args, keep).sum(axis=1))
    return np.concatenate(result)


def reproduce(point, draws, reference):
    errors = []
    for i, name in enumerate(NAMES):
        expected = reference['decomposition'][name]['diagnostic'] if i < 6 else reference['endpoints'][
            name.replace('_delta', '_cross')]['delta']
        actual = previous.saved.prior.interval(point[i], draws[:, i])
        require(actual['valid_draws'] == expected['valid_draws'] == 10000, 'V200 valid draw count differs')
        require(np.allclose([actual['mean'], *actual['ci']], [expected['mean'], *expected['ci']], rtol=0., atol=1e-8),
            'V200 point/diagnostic bootstrap readout differs')
        errors.extend(np.abs(np.array([actual['mean'], *actual['ci']])-np.array([expected['mean'], *expected['ci']])))
        if name in ('mix', 'conditional'):
            actual = previous.saved.prior.interval(point[i], draws[:, i], previous.FAMILY_ALPHA)
            expected = reference['decomposition'][name]['family']
            require(np.allclose(actual['ci'], expected['ci'], rtol=0., atol=1e-8), 'V200 family interval differs')
    return float(max(errors))


def compact(point, actor_point, actor_draws, crossed, key_point, key_draws, day_key, omitted,
        actors, keys, channels):
    actor_parts = actor_draws.transpose(1, 0, 2)
    actor_allocation = covariance_allocation(actor_parts, crossed)
    key_allocation = covariance_allocation(key_draws, crossed)
    key_day_allocation = covariance_allocation(day_key, day_key.sum(axis=1))
    delta = omitted-point
    result = {}
    for name in REPORT:
        i = NAMES.index(name)
        total = channels[:, i].sum()
        ai = np.argsort(-np.abs(actor_allocation[:, i]), kind='stable')[:3]
        ki = np.argsort(-np.abs(key_allocation[:, i]), kind='stable')[:3]
        di = np.argsort(-np.abs(delta[:, i]), kind='stable')[:3]
        result[name] = dict(point=float(point[i]), crossed_bootstrap_variance=float(crossed[:, i].var()),
            crossed_bootstrap_sd=float(crossed[:, i].std()), product_measure_variance=float(total),
            channels={n: dict(variance=float(channels[j, i]), fraction=float(channels[j, i]/total) if total > 0. else None)
                for j, n in enumerate(CHANNELS)},
            finite_mc_variance_gap=float(crossed[:, i].var()-total),
            top_actor_covariance_allocations=[dict(actor=actors[j], point_contribution=float(actor_point[j, i]/len(actors)),
                variance_allocation=float(actor_allocation[j, i])) for j in ai],
            top_key_covariance_allocations=[dict(key=list(keys[j]), point_contribution=float(key_point[j, i]),
                variance_allocation=float(key_allocation[j, i]), own_variance=float(key_draws[:, j, i].var()),
                day_variance_allocation=float(key_day_allocation[j, i])) for j in ki],
            top_day_sensitivities=[dict(day=previous.DAYS[j], half='first32' if j < 32 else 'last32',
                deletion_shift=float(delta[j, i])) for j in di],
            max_abs_day_shift_by_half=[float(np.abs(v[:, i]).max()) for v in (delta[:32], delta[32:])])
    return result, actor_allocation, key_allocation, key_day_allocation


def run_case(args, actors, mode, scope):
    support = read(args.v191_root/'details'/f'{mode}_{scope}.json')['common_key_support']
    keys = sorted({tuple(k['key']) for k in support})
    locations = {k: i for i, k in enumerate(keys)}
    aw, dw = previous.common.weights(len(actors))
    b, n, m = previous.saved.prior.DRAWS, len(actors), len(NAMES)
    actor_point, actor_day = np.zeros((n, m)), np.zeros((n, b, m))
    key_point = np.zeros((len(keys), m))
    key_draws, day_key = np.zeros((b, len(keys), m)), np.zeros((b, len(keys), m))
    omitted = np.zeros((64, m))
    for ai, actor in enumerate(actors):
        values, keep, actor_keys = previous.load_actor(args, actor, mode, scope, support)
        positions = [locations[k] for k in actor_keys]
        first, last = (v.mean(axis=0, keepdims=True) for v in (values[:32], values[32:]))
        p = key_terms(first, last, keep)[0]
        actor_point[ai] = p.sum(axis=0)
        key_point[positions] += p/n
        omitted += leave_day(values, keep)/n
        for start in range(0, b, 256):
            stop = min(start+256, b)
            a = previous.average_days(values[:32], dw[0, start:stop])
            z = previous.average_days(values[32:], dw[1, start:stop])
            k = key_terms(a, z, keep)
            actor_day[ai, start:stop] = k.sum(axis=1)
            key_draws[start:stop, positions] += aw[start:stop, ai, None, None]*k
            day_key[start:stop, positions] += k/n
        print('V201_MOMENT_ACTOR', mode, scope, actor, flush=True)
    point = actor_point.mean(axis=0)
    require(np.isfinite(actor_day).all(), 'undefined decomposition; do not replace draws or masks')
    actor_crossed = actor_day*aw.T[:, :, None]
    crossed = actor_crossed.sum(axis=0)
    reference = next(r for r in read(args.v200_root/'summary/status.json')['results'] if (r['mode'], r['scope']) == (mode, scope))
    error = reproduce(point, crossed, reference)
    channels = variance_channels(actor_day)
    require(np.allclose(day_key.sum(axis=1), actor_day.mean(axis=0), rtol=1e-9, atol=1e-8), 'day/key totals differ')
    summary, actor_alloc, key_alloc, day_alloc = compact(point, actor_point, actor_crossed, crossed,
        key_point, key_draws, day_key, omitted, actors, keys, channels)
    target = args.out_root/'details'/f'{mode}_{scope}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, actors=actors, keys=keys, names=NAMES, actor_point=actor_point,
        actor_bootstrap_mean=actor_day.mean(axis=1), actor_covariance_allocation=actor_alloc,
        key_point=key_point, key_covariance_allocation=key_alloc, key_day_covariance_allocation=day_alloc,
        key_own_variance=key_draws.var(axis=0), days=previous.DAYS, day_deletion_shift=omitted-point,
        variance_channels=channels, crossed_variance=crossed.var(axis=0), point=point)
    result = dict(mode=mode, scope=scope, v200_readout_max_abs_error=error, metrics=summary)
    print('V201_UNCERTAINTY_CASE_COMPLETE', mode, scope, flush=True)
    return result


def registration():
    return dict(stage='v201_readonly_uncertainty_sources', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(previous.DAYS), frozen=previous.registration()['frozen'], v200_source='52e8a70164',
        inputs='V199/V194 moment cells,V191 support/factors,V200 readout only;no raw archives or confirmation days.',
        scope='Four primary mode/scope cases from V200;hour13 is overlapping diagnostic,not another replication.',
        fixed='Original64 days,bins,19-day masks,parent/day centering,V191 factors and original denominators remain fixed.',
        resampling='Reuse10000 V200 shared-actor/independent-half-day draws,seed20260803;reproduce V200 point and intervals. '
            'No new significance gates. All9 scalar readouts are diagnostics;compact summary reports mix,conditional,unsupported,parent.',
        variance='Under independent actor/day resampling,theta_i(D) is the actor conditional readout. '
            'actor=Var_i(E_D theta_i)/n;day=Var_D(mean_i theta_i);interaction=E_D Var_i(theta_i-E_D theta_i)/n. '
            'These sum EXACTLY to the law-of-total-variance product measure integrating multinomial actor draws. '
            'The paired10000 crossed sample variance differs by finite Monte Carlo error;report it,do not force equality.',
        allocation='Actor and exact(direction,road,reference-hour) key contributions use signed Cov(part,total) in the SAME crossed draws. '
            'All allocations sum to empirical total variance;negative cancellation remains. Key grouping is attribution only, '
            'AFTER actor-level decomposition,never pooled admission or replacement of any mask. Actor/key views are NOT additive to channels.',
        day_sensitivity='Delete one shared day from one half using1/31 day weights,other half unchanged;retain original masks/factors. '
            'Report shift from full point,not a new estimand/gate or grounds to exclude a day. These shifts are not additive variance parts.',
        reporting='Keep all actor/key/day arrays server-side;compact summary top3 by absolute signed covariance allocation '
            'or absolute day sensitivity for descriptive localization only,no selected-case inference.',
        limitations='Conditional bootstrap variance,not physical uncertainty or causal variance decomposition. '
            'Only original supported subset is decomposed;unsupported readouts remain aggregate and unnormalized. '
            'Reused calibration days cannot establish held-future forecast/control gain.',
        raw_capture_files_opened=0, interval_fit=False, predictor_fit=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v201_*.py', '-v'], check=True)
    parent = read(args.v200_root/'summary/status.json')
    require(parent['source'] == '52e8a70164' and parent['source_paths'] == 1536 and parent['v191_weights_fixed']
        and parent['v199_masks_and_exposures_reproduced'] and parent['no_mass_renormalization']
        and parent['protocol']['calibration_days'] == list(previous.DAYS), 'requires frozen V200 readout')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite uncertainty registration')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    actors = parent['actors']
    require(len(actors) == 12 and actors == sorted(actors), 'actor grid differs')
    print('V201_UNCERTAINTY_PROTOCOL_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, actors, mode, scope) for mode, scope in product(('old', 'handoff'), previous.PRIMARY)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"moment_uncertainty"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope']))
    write(args.out_root/'summary/status.json', dict(status='v201_uncertainty_localization_complete', source=protocol['source'],
        protocol=protocol, results=results, v200_points_and_intervals_reproduced=True,
        variance_identities_checked=True, signed_allocations_preserved=True, masks_and_weights_unchanged=True,
        day_deletion_diagnostic_only=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V201_UNCERTAINTY_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v200-root', 'v199-root', 'v194-root', 'v191-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
