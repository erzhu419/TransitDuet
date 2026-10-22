#!/usr/bin/env python3
"""Localize V205 residual uncertainty without new data or controller changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v205_linear_states as previous


plan, base = previous.plan, previous.base
read, write, require = previous.read, previous.write, previous.require
partition = previous.old.cancellation.partition
METRICS = ('total', 'within_cell_day', 'between_day', 'centered_mean_product')
CHANNELS = ('actor_at_point_days', 'day_at_point_actors', 'remaining_joint_response')


def metrics(parts):
    return np.stack((parts.sum(axis=-1), parts[..., 0], parts[..., 1:5].sum(axis=-1), parts[..., 5:9].sum(axis=-1)), axis=-1)


def residual(matrices, beta):
    operator = np.broadcast_to(np.eye(previous.N)[[previous.J, previous.Y]], (len(beta), 2, previous.N)).copy()
    operator[..., :previous.Z] -= beta
    return np.einsum('bi,bkqij,bj->bkq', operator[:, 0], matrices, operator[:, 1], optimize=True)


def fixed_actor_sum(actor_points, weights):
    return tuple(np.tensordot(weights, values, axes=(1, 0)) for values in actor_points)


def refitted(values):
    _, _, beta, _, rank, valid = previous.projection(values[-1])
    require(valid.all(), 'undefined sensitivity operator;do not replace it with zero or change support')
    return metrics(residual(values[0], beta)), rank


def anchors(joint, actor_only, day_only, point):
    result = np.stack((actor_only-point, day_only-point, joint-actor_only-day_only+point), axis=-1)
    require(np.allclose(result.sum(axis=-1), joint-point, rtol=0., atol=1e-8), 'anchored responses do not reproduce joint')
    return result


def deletion_weights(n):
    actor = np.full((n, n), 1./(n-1))
    np.fill_diagonal(actor, 0.)
    day = np.full((2, 64, 32), 1./32)
    for h in range(2):
        day[h, h*32:(h+1)*32] = (1.-np.eye(32))/31.
    return actor, day


def matched(actual, expected, label):
    require(np.allclose(actual, expected, rtol=0., atol=1e-8), label+' differs from V205')
    return float(np.abs(actual-expected).max(initial=0.))


def reproduce(values, beta, expected):
    point, mix, endpoints, _, reference = values
    fitted_beta = previous.projection(reference)[2]
    errors = [matched(fitted_beta, beta, 'shared operator'), matched(residual(point, beta), expected[0], 'residual parts'),
        matched(residual(mix, beta), expected[1], 'residual mix')]
    ep = residual(endpoints.reshape(len(beta), 4, 9, previous.N, previous.N), beta).reshape(len(beta), 2, 2, 9)
    errors.append(matched(ep, expected[2], 'residual half endpoints'))
    return max(errors)


def point_actors(caches, uniform):
    values = [previous.aggregate([cache], np.ones((1, 1)), uniform) for cache in caches]
    return tuple(np.concatenate([v[i] for v in values], axis=0) for i in range(5))


def grouped_days(caches, aw, dw, saved_beta):
    b, n = len(aw), len(caches)
    joint, day = None, None
    contributions = np.empty((b, n, 2, len(METRICS)))
    for ai, cache in enumerate(caches):
        values = previous.aggregate([cache], np.ones((b, 1)), dw)
        weighted = tuple(previous.actor_weight(v, aw[:, ai]) for v in values)
        if joint is None:
            joint = tuple(v.copy() for v in weighted)
            day = tuple(v/n for v in values)
        else:
            joint = tuple(a+b for a, b in zip(joint, weighted))
            day = tuple(a+b/n for a, b in zip(day, values))
        contributions[:, ai] = previous.actor_weight(metrics(residual(values[0], saved_beta)), aw[:, ai])
    return joint, day, contributions


def top_sensitivity(shifts, labels, column):
    order = np.argsort(-np.abs(shifts[:, column]), kind='stable')[:3]
    return [dict(label=int(labels[i]), shift=float(shifts[i, column])) for i in order]


def registration(parent):
    return dict(stage='v206_saved_moment_residual_uncertainty', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        parent_source='09530a00dc', calibration_days=list(plan.DAYS), keys=[list(k) for k in plan.KEYS],
        frozen=parent['protocol']['v204']['frozen'], states=list(plan.STATES),
        inputs='Original64 V199 moments,V202 reference,V204 frozen admission,V205 saved parts/operators/draws ONLY.',
        fixed='V199 masks/9 bins/19-day threshold,V191 parent/half factors,original query/episode denominators,full parent/day centering.',
        reproduction='Reuse V20510000 shared-actor/independent-half-day draws,seed20260803;'
            'reproduce residual parts,mix,endpoints and shared operators before attribution.',
        temporal='Residual total=within-cell/day+between-day covariance+pooled centered-mean product. '
            'Retain signed Cov(part,total);also split the4 between terms and raw-cell/parent-centering cancellation. '
            'These algebraic partitions are NOT the actor/day resampling channels.',
        resampling='For F(A,D) refit the SAME V205 shared operator;F00 uses uniform actors/days. '
            'Actor anchor=F(A,D0)-F00;day anchor=F(A0,D)-F00;remaining=F(A,D)-F(A,D0)-F(A0,D)+F00. '
            'Their sum is exactly the saved joint-minus-point draw. Report signed covariance allocations and own variances;'
            'do NOT call them orthogonal ANOVA or force remaining variance nonnegative. '
            'V201 analytic linear actor integration is invalid for this refitted operator.',
        actors='Per-actor contributions use the SAME saved joint operator,not an actor-specific fit. '
            'Sum weighted actor contributions reproduces each saved residual draw. Actor attribution and anchor/temporal views overlap,not additive.',
        sensitivity='Delete ONE shared day from one fixed half(1/31,other half1/32),or ONE actor(1/11);'
            'refit the unchanged pooled operator,keep original cell masks/factors. Report all64/12 sensitivities server-side;'
            'top3 absolute shifts are localization only,not exclusion or inference on selected days/actors.',
        unsupported='Original aggregate JY/mass only;no unsupported residual decomposition or selected-mass renormalization.',
        limitations='Reused development days. Point-anchored bootstrap sensitivity,not causal uncertainty or an orthogonal product-measure '
            'variance attribution. Day-mean covariance and day-resampling variance differ. No new significance gate,ETA fit or policy claim.',
        raw_capture_files_opened=0, interval_fit=False, predictor_fit=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def run_case(args, actors, admission, mode, scope):
    caches, _ = previous.load_case(args, actors, mode, scope, admission)
    with np.load(args.v205_root/'details'/f'{mode}_{scope}.npz') as z:
        require(z['keys'].tolist() == [list(k) for k in plan.KEYS] and z['states'].tolist() == list(plan.STATES)
            and z['parts'].tolist() == list(previous.old.PARTS) and z['pairs'].tolist() == list(previous.PAIRS)
            and z['valid_boot'].all(), 'V205 saved residual schema or validity differs')
        reference = {name: z[name][..., 3] for name in ('point', 'boot', 'mix_point', 'mix_boot', 'endpoints_point', 'endpoints_boot')}
        beta_point, beta_boot = z['beta_point'], z['beta_boot']
    aw, dw = base.common.weights(len(actors))
    n, b = len(actors), len(aw)
    uniform = np.full((2, 1, 32), 1./32)
    fixed = point_actors(caches, uniform)
    values = fixed_actor_sum(fixed, np.full((1, n), 1./n))
    error = reproduce(values, beta_point[None], [reference[k][None] for k in ('point', 'mix_point', 'endpoints_point')])
    point = metrics(reference['point'])
    target = metrics(reference['boot'])
    actor_point = metrics(residual(fixed[0], np.broadcast_to(beta_point, (n, 2, previous.Z))))/n
    matched(actor_point.sum(axis=0), point, 'point actor allocation')
    actor_contribution = np.empty((b, n, 2, len(METRICS)))
    actor_only, day_only = np.empty_like(target), np.empty_like(target)
    rank_actor, rank_day = np.empty(b, int), np.empty(b, int)
    for start in range(0, b, 128):
        take = slice(start, min(start+128, b))
        joint, day, contribution = grouped_days(caches, aw[take], dw[:, take], beta_boot[take])
        error = max(error, reproduce(joint, beta_boot[take], [reference[k][take] for k in ('boot', 'mix_boot', 'endpoints_boot')]))
        actor_contribution[take] = contribution
        matched(contribution.sum(axis=1), target[take], 'draw actor allocation')
        actor_only[take], rank_actor[take] = refitted(fixed_actor_sum(fixed, aw[take]))
        day_only[take], rank_day[take] = refitted(day)
        if (start//128) % 16 == 0:
            print('V206_SAVED_RESIDUAL_DRAWS', mode, scope, start, '/10000', flush=True)
    responses = anchors(target, actor_only, day_only, point)
    actor_deleted, day_weights = deletion_weights(n)
    actor_leave, actor_leave_rank = refitted(fixed_actor_sum(fixed, actor_deleted))
    day_leave_values = previous.aggregate(caches, np.full((64, n), 1./n), day_weights)
    day_leave, day_leave_rank = refitted(day_leave_values)
    day_shift, actor_shift = day_leave-point, actor_leave-point
    rows, actor_allocations = [], np.empty((n, 2, len(METRICS)))
    parent_rows = read(args.v205_root/'summary/status.json')['results']
    for ki, key in enumerate(plan.KEYS):
        original = next(r for r in parent_rows if (r['mode'], r['scope'], r['key']) == (mode, scope, list(key)))
        sections = {}
        for mi, name in enumerate(METRICS):
            allocation = partition(actor_contribution[:, :, ki, mi], target[:, ki, mi], [str(a) for a in actors])
            actor_allocations[:, ki, mi] = allocation['signed_allocation']
            order = np.argsort(-np.abs(actor_allocations[:, ki, mi]), kind='stable')[:3]
            sections[name] = dict(point=float(point[ki, mi]), bootstrap_sd=float(target[:, ki, mi].std()),
                anchors=partition(responses[:, ki, mi], target[:, ki, mi]-point[ki, mi], list(CHANNELS)),
                top_actor_contributions=[dict(actor=actors[j], point=float(actor_point[j, ki, mi]),
                    signed_variance_allocation=float(actor_allocations[j, ki, mi]), own_variance=float(actor_contribution[:, j, ki, mi].var()),
                    fraction=allocation['shares'][j] if allocation['shares'] is not None else None) for j in order],
                top_actor_deletion_shifts=top_sensitivity(actor_shift[:, ki], actors, mi),
                top_day_deletion_shifts=top_sensitivity(day_shift[:, ki], plan.DAYS, mi),
                max_abs_day_shift_by_half=[float(np.abs(x[:, ki, mi]).max()) for x in (day_shift[:32], day_shift[32:])])
        temporal = reference['boot'][:, ki]
        centered = temporal[:, [2, 3, 4, 6, 7, 8]].sum(axis=1)
        raw = temporal[:, [0, 1, 5]].sum(axis=1)
        rows.append(dict(mode=mode, scope=scope, key=list(key), v205_max_abs_error=error, metrics=sections,
            temporal_cancellation=partition(target[:, ki, 1:], target[:, ki, 0], list(METRICS[1:])),
            between_cancellation=partition(temporal[:, 1:5], temporal[:, 1:5].sum(axis=1), list(previous.old.PARTS[1:5])),
            raw_centering_cancellation=partition(np.stack((raw, centered), axis=1), target[:, ki, 0], ['raw_cell', 'parent_centering']),
            supported_actor_bin_cells=original['supported_actor_bin_cells'], parent_mass=original['parent_mass'], retained_mass=original['retained_mass'],
            unsupported_original=original['unsupported_original'], unsupported_residual=None))
    target_path = args.out_root/'details'/f'{mode}_{scope}.npz'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, actors=actors, days=plan.DAYS, keys=plan.KEYS, metrics=METRICS, channels=CHANNELS,
        point=point, saved_joint=target, actor_only=actor_only, day_only=day_only, responses=responses,
        actor_point=actor_point, actor_contribution=actor_contribution, actor_covariance_allocation=actor_allocations,
        actor_deletion_shift=actor_shift, day_deletion_shift=day_shift, rank_actor=rank_actor, rank_day=rank_day,
        rank_actor_delete=actor_leave_rank, rank_day_delete=day_leave_rank)
    print('V206_RESIDUAL_SOURCES_CASE_COMPLETE', mode, scope, 'error', error, flush=True)
    return rows


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v206_*.py', '-v'], check=True)
    parent, admission = read(args.v205_root/'summary/status.json'), read(args.v204_root/'summary/status.json')
    require(parent['source'] == '09530a00dc' and parent['v203_points_and_individual_draws_reproduced']
        and parent['masks_and_weights_unchanged'] and not parent['main_policy_changed']
        and parent['protocol']['v204']['calibration_days'] == list(plan.DAYS), 'requires frozen completed V205')
    require(admission['source'] == '4b85f9159c' and len(admission['actors']) == 12, 'original actor grid differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite readout protocol')
    protocol = registration(parent)
    write(args.out_root/'protocol.json', protocol)
    print('V206_RESIDUAL_SOURCE_PROTOCOL_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, admission['actors'], admission, mode, scope) for mode, scope in product(plan.MODES, plan.SCOPES)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.extend(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":4,"unit":"residual_source_case"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope'], r['key']))
    write(args.out_root/'summary/status.json', dict(status='v206_residual_uncertainty_sources_complete', source=protocol['source'],
        protocol=protocol, results=results, v205_residual_points_and_individual_draws_reproduced=True,
        signed_covariance_identities_checked=True, refitted_shared_operator_preserved=True, masks_and_weights_unchanged=True,
        unsupported_separate=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V206_RESIDUAL_SOURCES_COMPLETE_NO_POLICY_OR_PREDICTOR_CHANGE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v202-root', 'v204-root', 'v205-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
