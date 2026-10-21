#!/usr/bin/env python3
"""Read V194 saved moments only under its sealed component-association plan."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_freqduet_v194_mechanism as saved


common, prior = saved.common, saved.prior
read, write, require = saved.read, saved.write, saved.require
N, SIZE, VARS = saved.N, saved.SIZE, saved.VARS
SIGNALS = ('signed_remaining_s', 'innovation_total_s', 'innovation_absorbed_s')
SELECTED = (*SIGNALS, *saved.OUTCOMES)
CONTROLS = (*prior.CONTROLS, 'innovation_rms_s')
CI = [VARS.index(c) for c in CONTROLS]
SI = [VARS.index(c) for c in SELECTED]
PRIMARY = ('clearance', 'hour13_moving_launch')
TESTS = {'H1': ('first_running_under_s', 1), 'H3': ('non_running_under_s', -1)}
FAMILY_ALPHA = .05 / 8
TARGETS = (*saved.OUTCOMES, 'all_running_under_s', 'non_running_under_s',
    'later_minus_first_under_s', 'full_minus_first_under_s')
CONTRAST = np.zeros((len(TARGETS), len(SELECTED)))
CONTRAST[:7, 3:] = np.eye(7)
CONTRAST[7, 3:5] = 1.
CONTRAST[8, 5:9] = 1.
CONTRAST[9, 3:5] = (-1., 1.)
CONTRAST[10, [3, 9]] = (-1., 1.)


def registration():
    return dict(stage='v195_frozen_mechanism_readout', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        parent_source='51a9409ec2', calibration_days=list(saved.DAYS),
        frozen=saved.registration()['frozen'], readout=saved.registration()['next_analysis'],
        controls=list(CONTROLS), signals=list(SIGNALS), targets=list(TARGETS),
        inputs='V194 per-key/day joint moments and V191 saved support/point reference only; no trajectory access',
        covariance_scale='Original nominal episode/query mass, fixed V191 balance, equal actor/day averages. '
            'Do not separately renormalize selected mass in halves or bootstrap draws. Report retained mass.',
        projection='One control design from average balanced-half WITHIN-day covariance, '
            'same operator for every additive signal/outcome and for within/between/total partitions; recomputed each draw.',
        all64='Equal mean of fixed halves; shared actor draws, independent32-day draws in each half.',
        interpretation='Calibration-only adjusted linear association, not held-future prediction or causal control benefit.',
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0)


def second(m):
    result = np.zeros((*m.shape[:-1], N, N))
    result[..., saved.TRI[0], saved.TRI[1]] = m[..., 1+N:]
    result[..., saved.TRI[1], saved.TRI[0]] = m[..., 1+N:]
    return result


def actor_matrix(rows, scope, expected, actor):
    require(len(rows) == 64 and {r['day'] for r in rows} == set(saved.DAYS), 'original64 grid differs')
    keys = sorted(tuple(r['key']) for r in expected if r['actor'] == actor)
    require(len(keys) > 0 and len(set(keys)) == len(keys), 'empty/duplicate common keys')
    locations = {k: i for i, k in enumerate(keys)}
    m = np.zeros((64, len(keys), SIZE))
    lookup = {r['day']: r for r in rows}
    for di, day in enumerate(saved.DAYS):
        cells = [c for c in lookup[day]['scopes'] if c['scope'] == scope]
        require(len(cells) == 1, 'scope missing/duplicated')
        seen = set()
        for g in cells[0]['groups']:
            k = tuple(g['key'])
            require(k in locations and k not in seen, 'common key differs/duplicated')
            seen.add(k)
            require(len(g['moments']) == SIZE, 'extended moment dimensions differ')
            m[di, locations[k]] = g['moments']
    require(np.isfinite(m).all() and np.all(m[..., 0] >= 0.), 'invalid moment values')
    support = {tuple(r['key']): r for r in expected if r['actor'] == actor}
    for i, k in enumerate(keys):
        counts = [int(np.count_nonzero(v[:, i, 0])) for v in (m[:32], m[32:])]
        require(min(counts) >= 19 and counts == [support[k]['first_days'], support[k]['last_days']],
            'V191 distinct-day support differs')
    return m


def day_parts(m):
    first, seconds = m[..., :1+N], second(m.sum(axis=1))
    scaled = np.divide(first[..., 1:], np.sqrt(np.maximum(first[..., :1], 0.)),
        out=np.zeros_like(first[..., 1:]), where=first[..., :1] > 0.)
    within = seconds - np.einsum('dki,dkj->dij', scaled, scaled)
    return first, seconds, within


def weighted_covariance(first, seconds, weights):
    # Recenter each key under the resampled days, rather than reweighting old centers.
    f = (weights @ first.reshape(len(first), -1)).reshape(-1, first.shape[1], 1+N)
    scaled = np.divide(f[..., 1:], np.sqrt(np.maximum(f[..., :1], 0.)),
        out=np.zeros_like(f[..., 1:]), where=f[..., :1] > 0.)
    covariance = (weights @ seconds.reshape(len(seconds), -1)).reshape(-1, N, N)
    covariance -= np.einsum('bki,bkj->bij', scaled, scaled)
    return covariance, f[..., 0].sum(axis=1)


def projection(reference):
    cc = reference[..., CI, :][..., :, CI]
    sd = np.sqrt(np.maximum(np.diagonal(cc, axis1=-2, axis2=-1), 0.))
    scale = sd[..., :, None] * sd[..., None, :]
    corr = np.divide(cc, scale, out=np.zeros_like(cc), where=scale > 1e-12)
    cross = reference[..., SI, :][..., :, CI]
    cross = np.divide(cross, sd[..., None, :], out=np.zeros_like(cross), where=sd[..., None, :] > 1e-12)
    beta = cross @ np.linalg.pinv(corr, rcond=1e-10, hermitian=True)
    beta = np.divide(beta, sd[..., None, :], out=np.zeros_like(beta), where=sd[..., None, :] > 1e-12)
    result = np.broadcast_to(np.eye(N)[SI], (*reference.shape[:-2], len(SI), N)).copy()
    result[..., :, CI] = -beta
    return result


def projected(covariance, operator):
    return operator @ covariance @ np.swapaxes(operator, -1, -2)


def identities(covariance):
    # These linear identities must survive the common projection and resampling.
    require(np.allclose(covariance[..., 1, :], covariance[..., 0, :] + covariance[..., 2, :],
        rtol=1e-7, atol=2e-6), 'I=A+J projected identity differs')
    require(np.allclose(covariance[..., 9, :], covariance[..., 3:9, :].sum(axis=-2),
        rtol=1e-7, atol=2e-6), 'additive projected component identity differs')


def measures(covariance):
    cov = np.einsum('...j,tj->...t', covariance[..., 0, :], CONTRAST)
    variance = np.einsum('ti,...ij,tj->...t', CONTRAST, covariance, CONTRAST)
    signal = covariance[..., 0, 0]
    cov = np.where(signal[..., None] > 1e-12, cov, np.nan)
    denominator = np.sqrt(np.maximum(signal[..., None], 0.) * np.maximum(variance, 0.))
    rho = np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 1e-12)
    return cov, rho


def intervals(point, boot, family=False):
    result = dict(covariance=prior.interval(point[0], boot[0]), correlation=prior.interval(point[1], boot[1]))
    if family:
        result['family_covariance'] = prior.interval(point[0], boot[0], FAMILY_ALPHA)
    return result


def reproduce(covariances, masses, expected):
    for half, label in enumerate(('first', 'last')):
        saved.close(masses[half], expected['support']['balanced'][label])
        for si, signal in enumerate(prior.SIGNALS):
            for block, unique in (('baseline', False), ('unique', True)):
                value = common.measure(covariances[half, :prior.N, :prior.N], si, unique)[0]
                want = expected['stats'][signal+'|'+block]['arms']['balanced'][label]['mean']
                require((want is None) == (not np.isfinite(value)), 'V191 missing estimate differs')
                if want is not None:
                    require(abs(float(value)-want) < 1e-8, 'V191 correlation/centering differs')


def summarize(pc, pw, bc, bw, scope):
    r, br = projection(pw.mean(axis=0)), projection(bw.mean(axis=0))
    result, signal_details = {}, {}
    for part, p, b in (('within_day', pw, bw), ('between_day', pc-pw, bc-bw), ('total', pc, bc)):
        p, b = projected(p, r), projected(b, br)
        identities(p)
        identities(b)
        p = np.stack((p[0], p[1], p.mean(axis=0), p[1]-p[0]))
        b = np.stack((b[0], b[1], b.mean(axis=0), b[1]-b[0]))
        cov, rho = measures(p[:3])
        dcov, drho = measures(b[:3])
        # Differences are differences of estimates, not correlations of a difference matrix.
        cov, rho = np.concatenate((cov, [cov[1]-cov[0]])), np.concatenate((rho, [rho[1]-rho[0]]))
        dcov, drho = np.concatenate((dcov, [dcov[1]-dcov[0]])), np.concatenate((drho, [drho[1]-drho[0]]))
        result[part] = {split: {target: intervals((cov[h, ti], rho[h, ti]), (dcov[h, :, ti], drho[h, :, ti]),
            family=scope in PRIMARY and part == 'within_day' and split == 'all64' and target in {v[0] for v in TESTS.values()})
            for ti, target in enumerate(TARGETS)} for h, split in enumerate(('first32', 'last32', 'all64', 'delta'))}
        signal_details[part] = {split: {signal: {target: prior.interval(p[h, si, 3+ti], b[h, :, si, 3+ti])
            for ti, target in enumerate(saved.OUTCOMES)} for si, signal in enumerate(SIGNALS)}
            for h, split in enumerate(('first32', 'last32', 'all64', 'delta'))}
    # A between-day contrast is not an independent residualization/model.
    for ti in range(len(TARGETS)):
        within = np.einsum('...j,j->...', projected(pw, r)[..., 0, :], CONTRAST[ti])
        between = np.einsum('...j,j->...', projected(pc-pw, r)[..., 0, :], CONTRAST[ti])
        total = np.einsum('...j,j->...', projected(pc, r)[..., 0, :], CONTRAST[ti])
        require(np.allclose(total, within+between, rtol=1e-8, atol=1e-7), 'day partition differs')
    return result, signal_details


def run_case(args, actors, mode, scope):
    expected = read(args.v191_root/'details'/f'{mode}_{scope}.json')
    aw, dw = common.weights(len(actors))
    pc, pw = np.zeros((2, N, N)), np.zeros((2, N, N))
    bc, bw = np.zeros((2, prior.DRAWS, N, N)), np.zeros((2, prior.DRAWS, N, N))
    mass, bmass = np.zeros(2), np.zeros((2, prior.DRAWS))
    for ai, actor in enumerate(actors):
        cell = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
        require(cell['source'] == '51a9409ec2' and all((r['actor'], r['mode']) == (actor, mode)
            and r['v190_moments_reproduced'] for r in cell['rows']), 'V194 source/identity/admission differs')
        m = actor_matrix(cell['rows'], scope, expected['common_key_support'], actor)
        for half, values in enumerate(common.balance(m[:32], m[32:])):
            first, seconds, within = day_parts(values)
            c, mm = weighted_covariance(first, seconds, np.full((1, 32), 1./32))
            pc[half] += c[0]/len(actors)
            pw[half] += within.mean(axis=0)/len(actors)
            mass[half] += mm[0]/len(actors)
            for start in range(0, prior.DRAWS, 64):
                stop = min(start+64, prior.DRAWS)
                c, mm = weighted_covariance(first, seconds, dw[half, start:stop])
                factor = aw[start:stop, ai]
                bc[half, start:stop] += factor[:, None, None]*c
                bw[half, start:stop] += factor[:, None, None]*(dw[half, start:stop] @ within.reshape(32, -1)).reshape(-1, N, N)
                bmass[half, start:stop] += factor*mm
        print('V195_MOMENT_ACTOR', mode, scope, actor, flush=True)
    reproduce(pc, mass, expected['result'])
    print('V195_V191_POINTS_REPRODUCED', mode, scope, flush=True)
    stats, details = summarize(pc, pw, bc, bw, scope)
    result = dict(mode=mode, scope=scope, common_keys=len(expected['common_key_support']),
        analysis_mass=dict(zip(('first32', 'last32'), mass.tolist())),
        stats=stats, v191_points_reproduced=True)
    write(args.out_root/'details'/f'{mode}_{scope}.json', dict(result=result, component_signal_covariances=details,
        bootstrap_analysis_mass={label: prior.interval(mass[h], bmass[h]) for h, label in enumerate(('first32', 'last32'))}))
    write(args.out_root/'cells'/f'{mode}_{scope}.json', result)
    print('V195_MECHANISM_CASE_COMPLETE', mode, scope, flush=True)
    return result


def sign_check(value, sign):
    if value['mean'] is None or value['ci'] is None:
        return None
    return value['ci'][0] > 0. if sign > 0 else value['ci'][1] < 0.


def gates(results):
    lookup = {(r['mode'], r['scope']): r['stats']['within_day'] for r in results}
    output = {}
    for scope, hypothesis in product(PRIMARY, TESTS):
        target, sign = TESTS[hypothesis]
        checks = {}
        for mode in ('old', 'handoff'):
            for split in ('first32', 'last32', 'all64'):
                key = 'family_covariance' if split == 'all64' else 'covariance'
                checks[mode+'|'+split] = sign_check(lookup[mode, scope][split][target][key], sign)
        status = 'replicated' if all(v is True for v in checks.values()) else (
            'inconclusive' if any(v is None for v in checks.values()) else 'not_replicated')
        output[scope+'|'+hypothesis] = dict(status=status, target=target, expected_sign=sign, checks=checks)
    return output


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v195_*.py', '-v'], check=True)
    admission = read(args.v194_root/'summary/status.json')
    plan = read(args.v194_root/'protocol.json')
    require(admission['source'] == plan['source'] == '51a9409ec2' and admission['paths'] == 1536
        and admission['v190_moments_reproduced'] and admission['common_support_19_each_half_preserved']
        and not admission['mechanism_associations_estimated'] and not admission['main_policy_changed'], 'V194 not admitted')
    require(plan['calibration_days'] == list(saved.DAYS) and plan['next_analysis'] == saved.registration()['next_analysis'],
        'V194 sealed analysis differs')
    # Actor IDs come from already admitted moment metadata, never from raw archives.
    actors = sorted({int(p.stem.split('_')[0]) for p in (args.v194_root/'cells').glob('*_old.json')})
    require(len(actors) == 12, '12 admitted actors required')
    protocol = registration()
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite sealed readout')
    write(args.out_root/'protocol.json', protocol)
    print('V195_FROZEN_READOUT_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_case, args, actors, m, s) for m, s in product(('old', 'handoff'), saved.SCOPES)]
        for i, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"saved_moment_readout"}' % i, flush=True)
    results.sort(key=lambda r: (r['mode'], r['scope']))
    decision = gates(results)
    write(args.out_root/'summary/status.json', dict(status='v195_mechanism_readout_complete', source=protocol['source'],
        protocol=protocol, results=results, gates=decision, source_paths=1536, actors=actors,
        v191_points_reproduced=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V195_MECHANISM_COMPLETE_NO_FIT', decision, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v194-root', 'v191-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
