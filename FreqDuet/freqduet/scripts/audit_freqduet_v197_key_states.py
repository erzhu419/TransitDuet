#!/usr/bin/env python3
"""Read original64 joint moments: common-key state mix and J/first-road covariance."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v196_first_leg as previous


base, saved, prior, common = previous.parent, previous.saved, previous.prior, previous.common
read, write, require = previous.read, previous.write, previous.require
N, DAYS = previous.N, previous.DAYS
FIELDS = (*base.CONTROLS, 'signed_remaining_s', 'first_running_under_s')
IX = [base.VARS.index(f) for f in FIELDS]
FOCUS_KEYS = ((0, 12, 13), (0, 11, 13), (1, 11, 13))
CORE = ('raw', 'state_covariance', 'residualJ_stateY', 'stateJ_residualY', 'mixed', 'residual')


def registration():
    return dict(stage='v197_common_key_state_composition', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), moment_source='51a9409ec2', reference_source='f6c09a6dd1',
        frozen=saved.registration()['frozen'], inputs='V194 saved joint moments,V191 support,V196 result reference only',
        support='Exact V191 actor/mode/road/entry-hour common keys with19 complete-input days in EACH half.',
        weights='Original nominal query/episode mass,equal actor/day,V191 fixed support-only balance;no outcome-selected rows or keys.',
        states=list(FIELDS), groups=previous.GROUPS,
        questions=['Do observable state means and within-day variances change inside the same common keys?',
            'Do state-J and state-first-road-error covariances/correlations change?',
            'Under the SAME pooled linear projection,does raw covariance change follow state covariance,mixed terms or residual covariance?'],
        means='Balanced point key masses are equal across halves;mean changes therefore retain the same key mixture. '
            'Bootstrap uses original resampled masses,not a second key renormalization. '
            'Mean shifts are NOT additive explanations of within-key/day-centered covariance.',
        state_statistics='Mean,within-day variance,between-day variance,covariance/correlation with raw J and Y; '
            'mean delta scaled by average-half within-key SD. Variances/covariances per retained mass; '
            'primary covariance decomposition remains unnormalized nominal mass.',
        projection='Unchanged V19524-control pooled WITHIN-day operator;one reference for both halves,actors,keys and partitions; '
            'recompute once per bootstrap draw,no per-key or per-half regressions.',
        decomposition='J=fJ+rJ,Y=fY+rY;raw=cov(fJ,fY)+cov(rJ,fY)+cov(fJ,rY)+cov(rJ,rY). '
            'mixed=sum of the two cross terms. Allocate state-state products symmetrically across fixed control blocks.',
        aggregate_mixed='Average-half mixed terms are constrained by the pooled projection;near-zero all64 is not validation.',
        reporting='Save ALL pooled road-key state points and actor/key decomposition points server-side. '
            'Three V196-selected keys are development diagnostics only,never independent replication or new admission masks.',
        focus_keys=[list(k) for k in FOCUS_KEYS], parts='Within-day primary;between-day and total separately,using the same operator.',
        inference=dict(draws=prior.DRAWS, seed=prior.SEED, actor='shared actors', day='independent32-day draws per fixed half',
            fixed='masks/support balance', family='8 within-day delta contrasts:2 modes x2 primary scopes x(state_covariance,mixed); '
                'two-sided percentile alpha=.05/8', diagnostic='All state/field/block/other-part intervals95%;direct delta,not differing significance.',
            missing='Zero-variance correlation/standardization is unknown,not zero;no low-support pooling or19-day relaxation.'),
        limitations='Reused development days. First/second moments cannot identify nonlinear conditional distributions, '
            'causal composition effects or a counterfactual state-matched population. No predictive or policy superiority claim.',
        stop='V195 gates unchanged;no predictor/interval fit,threshold search,confirmation,replay,training or policy promotion.',
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0)


def ratio(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 1e-12)


def state_values(c, w, sums):
    mass = sums[..., :1]
    means = ratio(sums[..., np.array(IX)+1], mass)
    vd = np.maximum(np.diagonal(w, axis1=-2, axis2=-1), 0.)
    bd = np.maximum(np.diagonal(c-w, axis1=-2, axis2=-1), 0.)
    values = dict(mean=means, within_variance=ratio(vd[..., IX], mass), between_variance=ratio(bd[..., IX], mass))
    for name, target in (('J', previous.J), ('Y', previous.Y)):
        cov = w[..., IX, target]
        values['cov_'+name] = ratio(cov, mass)
        values['corr_'+name] = ratio(cov, np.sqrt(vd[..., IX]*vd[..., target, None]))
    scale = np.sqrt(np.maximum(np.diagonal(c.mean(axis=0), axis1=-2, axis2=-1)[..., IX], 0.) / mass.mean(axis=0))
    return values, ratio(means[1]-means[0], scale)


def state_summary(c, w, sums, bc, bw, bs):
    points, shifts = state_values(c, w, sums)
    draws, bshifts = state_values(bc, bw, bs)
    return {f: dict(metrics={name: dict(first32=common.number(p[0, i]), last32=common.number(p[1, i]),
        delta=prior.interval(p[1, i]-p[0, i], draws[name][1, :, i]-draws[name][0, :, i]))
        for name, p in points.items()}, within_key_sd_scaled_mean_delta=prior.interval(shifts[i], bshifts[:, i]))
        for i, f in enumerate(FIELDS)}


def components(c, operator):
    x, y = operator[..., 0, :], operator[..., 3, :]
    fx, fy = previous.EJ-x, previous.EY-y
    dot = lambda a, b: previous.bilinear(a, c, b)
    result = dict(raw=c[..., previous.J, previous.Y], state_covariance=dot(fx, fy),
        residualJ_stateY=dot(x, fy), stateJ_residualY=dot(fx, y), residual=dot(x, y))
    result['mixed'] = result['residualJ_stateY']+result['stateJ_residualY']
    require(np.allclose(result['raw'], result['state_covariance']+result['mixed']+result['residual'],
        rtol=1e-8, atol=2e-7), 'raw state/mixed/residual identity differs')
    for name, fields in previous.GROUPS.items():
        ix = [base.VARS.index(f) for f in fields]
        a, b = np.zeros_like(fx), np.zeros_like(fy)
        a[..., ix], b[..., ix] = fx[..., ix], fy[..., ix]
        result['state|'+name] = .5*(dot(a, fy)+dot(fx, b))
        result['mixed|'+name] = dot(a, y)+dot(x, b)
    for name, term in (('state', 'state_covariance'), ('mixed', 'mixed')):
        require(np.allclose(sum(v for k, v in result.items() if k.startswith(name+'|')), result[term],
            rtol=1e-8, atol=2e-7), 'fixed block state allocation differs')
    return result


def summarize(c, w, bc, bw, scope):
    r, br = base.projection(w.mean(axis=0)), base.projection(bw.mean(axis=0))
    stats = {}
    for part, p, b in (('within_day', w, bw), ('between_day', c-w, bc-bw), ('total', c, bc)):
        point, draws = components(p, r), components(b, br)
        stats[part] = previous.summarize(point, draws)
        if part == 'within_day' and scope in base.PRIMARY:
            for term in ('state_covariance', 'mixed'):
                stats[part][term]['delta_family'] = prior.interval(point[term][1]-point[term][0],
                    draws[term][1]-draws[term][0], .05/8)
    return stats, r


def key_point(c, w, sums, operator):
    values, shifts = state_values(c, w, sums)
    terms = components(w, operator)
    return dict(mass=sums[:, 0].tolist(),
        terms={k: dict(first32=float(v[0]), last32=float(v[1]), delta=float(v[1]-v[0])) for k, v in terms.items()},
        states={f: dict(metrics={k: [common.number(v[0, i]), common.number(v[1, i]), common.number(v[1, i]-v[0, i])]
            for k, v in values.items()}, within_key_sd_scaled_mean_delta=common.number(shifts[i])) for i, f in enumerate(FIELDS)})


def key_moments(m):
    sums = m[..., :1+N].mean(axis=0)
    scaled = sums[..., 1:]/np.sqrt(sums[..., :1])
    total = base.second(m.mean(axis=0))-np.einsum('ki,kj->kij', scaled, scaled)
    within = previous.key_within(m).mean(axis=0)
    return total, within, sums


def reproduce(stats, reference, mode, scope):
    expected = next(r for r in reference['results'] if (r['mode'], r['scope']) == (mode, scope))
    for part in stats:
        for now, old in (('raw', 'raw'), ('residual', 'adjusted'), ('state_covariance', 'joint_projection')):
            for split in ('first32', 'last32', 'all64', 'delta'):
                got, want = stats[part][now][split], expected['stats'][part][old][split]
                require(got['valid_draws'] == want['valid_draws'] == prior.DRAWS and
                    np.allclose([got['mean'], *got['ci']], [want['mean'], *want['ci']], rtol=1e-7, atol=2e-7),
                    'V196 point/bootstrap interval differs')


def key_signs(roads):
    output = {}
    for field in FIELDS:
        output[field] = {}
        for metric in ('mean', 'within_variance', 'cov_J', 'cov_Y', 'corr_J', 'corr_Y'):
            deltas = [r['states'][field]['metrics'][metric][2] for r in roads]
            output[field][metric] = dict(positive=sum(x is not None and x > 0. for x in deltas),
                negative=sum(x is not None and x < 0. for x in deltas), zero=sum(x == 0. for x in deltas),
                unknown=sum(x is None for x in deltas))
    return output


def run_case(args, actors, reference, mode, scope):
    support = read(args.v191_root/'details'/f'{mode}_{scope}.json')
    aw, dw = common.weights(len(actors))
    c, w, sums = np.zeros((2, N, N)), np.zeros((2, N, N)), np.zeros((2, 1+N))
    bc, bw, bs = np.zeros((2, prior.DRAWS, N, N)), np.zeros((2, prior.DRAWS, N, N)), np.zeros((2, prior.DRAWS, 1+N))
    keys, pooled = [], {}
    for ai, actor in enumerate(actors):
        cell = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
        require(cell['source'] == '51a9409ec2' and all((r['actor'], r['mode']) == (actor, mode)
            and r['v190_moments_reproduced'] for r in cell['rows']), 'moment source/identity differs')
        m = base.actor_matrix(cell['rows'], scope, support['common_key_support'], actor)
        names = sorted(tuple(k['key']) for k in support['common_key_support'] if k['actor'] == actor)
        halves = []
        for h, v in enumerate(common.balance(m[:32], m[32:])):
            first, seconds, wd = base.day_parts(v)
            cc, _ = base.weighted_covariance(first, seconds, np.full((1, 32), 1./32))
            ss = first.sum(axis=1)
            c[h] += cc[0]/len(actors)
            w[h] += wd.mean(axis=0)/len(actors)
            sums[h] += ss.mean(axis=0)/len(actors)
            halves.append(key_moments(v))
            require(np.allclose(halves[-1][0].sum(axis=0), cc[0], rtol=1e-8, atol=2e-7), 'per-key covariance sum differs')
            for start in range(0, prior.DRAWS, 64):
                stop = min(start+64, prior.DRAWS)
                cc, _ = base.weighted_covariance(first, seconds, dw[h, start:stop])
                weight = aw[start:stop, ai]
                bc[h, start:stop] += weight[:, None, None]*cc
                bw[h, start:stop] += weight[:, None, None]*(dw[h, start:stop] @ wd.reshape(32, -1)).reshape(-1, N, N)
                bs[h, start:stop] += weight[:, None]*(dw[h, start:stop] @ ss)
        for ki, key in enumerate(names):
            record = tuple(np.array([half[j][ki] for half in halves])/len(actors) for j in range(3))
            keys.append((actor, key, record))
            if key not in pooled:
                pooled[key] = tuple(np.zeros_like(v) for v in record)
            for target, value in zip(pooled[key], record):
                target[:] += value
        print('V197_MOMENT_ACTOR', mode, scope, actor, flush=True)
    base.reproduce(c, sums[:, 0], support['result'])
    stats, operator = summarize(c, w, bc, bw, scope)
    reproduce(stats, reference, mode, scope)
    print('V197_V196_POINTS_AND_INTERVALS_REPRODUCED', mode, scope, flush=True)
    for i, expected in enumerate((c, w, sums)):
        require(np.allclose(sum(v[i] for v in pooled.values()), expected, rtol=1e-8, atol=2e-7), 'pooled key moments do not sum')
    roads = [dict(key=list(k), **key_point(*v, operator)) for k, v in sorted(pooled.items())]
    actor_keys = [dict(actor=a, key=list(k), terms={name: [float(v[0]), float(v[1]), float(v[1]-v[0])]
        for name, v in components(record[1], operator).items() if name in CORE}) for a, k, record in keys]
    for name in CORE:
        require(np.isclose(sum(r['terms'][name][2] for r in actor_keys), stats['within_day'][name]['delta']['mean'],
            rtol=1e-8, atol=2e-7), 'actor/key state decomposition does not sum')
    result = dict(mode=mode, scope=scope, common_keys=len(keys), pooled_keys=len(roads), analysis_mass=sums[:, 0].tolist(),
        stats=stats, states=state_summary(c, w, sums, bc, bw, bs), key_delta_signs=key_signs(roads),
        focus_keys=[{k: v for k, v in r.items() if k != 'states'} for r in roads if tuple(r['key']) in FOCUS_KEYS],
        v196_points_and_intervals_reproduced=True)
    write(args.out_root/'details'/f'{mode}_{scope}.json', dict(result=result, pooled_road_states=roads,
        actor_key_terms=actor_keys, reference_vectors=operator.tolist()))
    print('V197_KEY_STATES_CASE_COMPLETE', mode, scope, flush=True)
    return result


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v197_*.py', '-v'], check=True)
    reference = read(args.v196_root/'summary/status.json')
    require(reference['source'] == 'f6c09a6dd1' and reference['protocol']['calibration_days'] == list(DAYS)
        and reference['v195_points_and_intervals_reproduced'] and not reference['interval_fit'] and not reference['main_policy_changed'],
        'requires frozen completed V196')
    actors = sorted({k['actor'] for k in read(args.v191_root/'details/old_clearance.json')['common_key_support']})
    require(len(actors) == 12, 'actor grid differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    print('V197_STATE_COMPOSITION_PROTOCOL_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        for i, job in enumerate(as_completed([pool.submit(run_case, args, actors, reference, mode, scope)
                for mode, scope in product(('old', 'handoff'), saved.SCOPES)]), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"saved_moment_state_audit"}' % i, flush=True)
    write(args.out_root/'summary/status.json', dict(status='v197_common_key_state_audit_complete', source=protocol['source'],
        protocol=protocol, results=sorted(results, key=lambda r: (r['mode'], r['scope'])),
        v196_points_and_intervals_reproduced=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V197_KEY_STATES_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v194-root', 'v191-root', 'v196-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
