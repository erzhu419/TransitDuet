#!/usr/bin/env python3
"""Decompose V195 first-road half changes using admitted V194 moments only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v195_mechanism as parent


read, write, require = parent.read, parent.write, parent.require
prior, common, saved = parent.prior, parent.common, parent.saved
N, DAYS = parent.N, saved.DAYS
J, Y = parent.VARS.index('signed_remaining_s'), parent.VARS.index('first_running_under_s')
EJ, EY = np.eye(N)[J], np.eye(N)[Y]
GROUPS = dict(clock_geometry=('query_offset_min', 'remaining_stops', 'remaining_km', 'first_fraction'),
    forecast_bounds=('horizon_min', 'running_min', 'service_min', 'lower_radius_min', 'upper_radius_min'),
    leg_support=('native_leg_mean_min', 'native_n', 'native_latest_age_min', 'matched_n',
        'matched_latest_age_min', 'matched_mean_age_min', 'matched_entry_age_min', 'predicted_entry'),
    completed_service=('service_prefix_under_min', 'service_prefix_latest_age_min', 'service_prefix_n'),
    current_stage=('stage_service', 'stage_command', 'stage_ready'), other_rms=('innovation_rms_s',))
TERMS = ('raw', 'signal_projection', 'target_projection', 'joint_projection', 'projection', 'adjusted')


def registration():
    return dict(stage='v196_first_road_half_change_decomposition', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), moment_source='51a9409ec2', reference_source='759c662f20',
        frozen=saved.registration()['frozen'], inputs='V194 saved moments,V191 common support,V195 result reference only',
        support='Exact V191 common actor/mode/road/entry-hour keys,19 complete-input days in EACH half; fixed support-mass balance.',
        weights='Original episode/query mass,equal nominal actor/day; no selected-mass or leave-out renormalization.',
        target='J versus first remaining-road truth-minus-prediction running error',
        hypotheses=dict(raw='The half change is already present before fixed query-control adjustment.',
            projection='The common-control projection contributes to or cancels that half change.',
            concentration='Actor/common-road-key contributions may cancel or concentrate the adjusted point change.'),
        algebra='adjusted=raw-cov(fittedJ,Y)-cov(J,fittedY)+cov(fittedJ,fittedY); '
            'projection=sum of last three terms; delta=last32-first32 for EVERY term.',
        projection='Same V19524 controls and mean-half WITHIN-day operator; no per-half/actor/key refit. '
            'Recompute the single pooled operator in every bootstrap draw.',
        blocks=GROUPS, block_allocation='Symmetric allocation of all cross-control products; coordinate-dependent accounting,not causal shares.',
        parts='Within-day is primary; between-day and total use the SAME within-day operator and remain descriptive.',
        contributions='Within-day actor and actor/common-key covariances under the SAME operator and original weights. '
            'Pool identical direction/road/entry-hour labels only for reporting,never for support admission. '
            'Report all actor and all pooled-key point contributions server-side,actor diagnostic95% intervals; '
            'largest3 absolute key deltas are display only,not new selected tests.',
        leave_out='Subtract each fixed group contribution without refitting or renormalizing; '
            'point accounting sensitivity only,not a rerun or a deployment counterfactual.',
        inference=dict(draws=prior.DRAWS, seed=prior.SEED, actor='shared actor draws',
            day='independent32-day multinomial draws in each fixed half',
            fixed='common-key masks and support balancing',
            family='8 primary within-day delta contrasts:2 modes x2 primary scopes x(raw,projection); alpha=.05/8,two-sided',
            diagnostic='95% for adjusted delta,halves,all64,control blocks,actor contributions,between-day,total and overlapping hour13',
            interpretation='Test half DIFFERENCES directly; point sign reversals do not establish opposite population signs. '
                'No selected-key significance tests or ratios of signed cancelling contributions.'),
        primary_scopes=list(parent.PRIMARY), diagnostic='hour13 overlaps clearance',
        stop='V195 failed gates remain failed; no propagation correction,interval/predictor fit,threshold search or policy promotion.',
        limitations='Reused calibration days; second-moment decomposition is not a causal or nonlinear composition analysis. '
            'Fixed-weight key subtraction does not show what retraining or removing a road would do.',
        interval_fit=False, predictor_fit=False, confirmation_started=False, new_replays=0, new_training=0)


def bilinear(a, c, b):
    return np.einsum('...i,...ij,...j->...', a, c, b)


def decompose(c, operator, blocks=True):
    x, y = operator[..., 0, :], operator[..., 3, :]
    fx, fy = EJ-x, EY-y
    terms = dict(raw=c[..., J, Y], signal_projection=-bilinear(fx, c, EY),
        target_projection=-bilinear(EJ, c, fy), joint_projection=bilinear(fx, c, fy))
    terms['projection'] = sum(terms[k] for k in TERMS[1:4])
    terms['adjusted'] = bilinear(x, c, y)
    require(np.allclose(terms['raw']+terms['projection'], terms['adjusted'], rtol=1e-8, atol=2e-7),
        'four-term covariance identity differs')
    if blocks:
        groups = {}
        for name, fields in GROUPS.items():
            ix = [parent.VARS.index(f) for f in fields]
            a, b = np.zeros_like(fx), np.zeros_like(fy)
            a[..., ix], b[..., ix] = fx[..., ix], fy[..., ix]
            groups[name] = -bilinear(a, c, EY)-bilinear(EJ, c, b)+.5*(bilinear(a, c, fy)+bilinear(fx, c, b))
        require(np.allclose(sum(groups.values()), terms['projection'], rtol=1e-8, atol=2e-7),
            'cross-control block allocation differs')
        terms.update({'block|'+k: v for k, v in groups.items()})
    return terms


def summarize(points, draws, primary=False):
    result = {}
    for name, p in points.items():
        b = draws[name]
        values = (p[0], p[1], p.mean(axis=0), p[1]-p[0])
        samples = (b[0], b[1], b.mean(axis=0), b[1]-b[0])
        result[name] = {split: prior.interval(value, sample) for split, value, sample in
            zip(('first32', 'last32', 'all64', 'delta'), values, samples)}
        if primary and name in ('raw', 'projection'):
            result[name]['delta_family'] = prior.interval(values[-1], samples[-1], .05/8)
    return result


def key_within(m):
    mass, sums = m[..., :1], m[..., 1:1+N]
    scaled = np.divide(sums, np.sqrt(np.maximum(mass, 0.)), out=np.zeros_like(sums), where=mass > 0.)
    return parent.second(m)-np.einsum('...i,...j->...ij', scaled, scaled)


def point_record(values):
    return {name: dict(first32=float(p[0]), last32=float(p[1]), all64=float(p.mean()), delta=float(p[1]-p[0]))
        for name, p in values.items() if name in TERMS}


def contribution_summary(records, baseline):
    require(len(records) > 0, 'missing contribution groups')
    for term in TERMS:
        for split in ('first32', 'last32', 'all64', 'delta'):
            require(np.isclose(sum(r['terms'][term][split] for r in records), baseline[term][split]['mean'],
                rtol=1e-8, atol=2e-7), 'group contribution sum differs')
    delta = np.array([r['terms']['adjusted']['delta'] for r in records])
    raw = np.array([r['terms']['raw']['delta'] for r in records])
    remain = {term: np.array([[baseline[term][split]['mean']-r['terms'][term][split]
        for split in ('first32', 'last32', 'delta')] for r in records]) for term in ('raw', 'adjusted')}
    return dict(groups=len(records), positive_delta_groups=int((delta > 0.).sum()), negative_delta_groups=int((delta < 0.).sum()),
        positive_delta_sum=float(delta[delta > 0.].sum()), negative_delta_sum=float(delta[delta < 0.].sum()),
        raw_positive_delta_sum=float(raw[raw > 0.].sum()), raw_negative_delta_sum=float(raw[raw < 0.].sum()),
        largest3_absolute_delta=[records[i] for i in np.argsort(-np.abs(delta), kind='stable')[:3]],
        leave_one_out={term: dict(delta_range=[float(v[:, 2].min()), float(v[:, 2].max())],
            negative_to_positive_point_halves=int(((v[:, 0] < 0.) & (v[:, 1] > 0.)).sum()),
            nonpositive_delta=int((v[:, 2] <= 0.).sum())) for term, v in remain.items()})


def reproduce(stats, reference, mode, scope):
    expected = next(r for r in reference['results'] if (r['mode'], r['scope']) == (mode, scope))
    for part, values in stats.items():
        for split in ('first32', 'last32', 'all64', 'delta'):
            want = expected['stats'][part][split]['first_running_under_s']['covariance']
            got = values['adjusted'][split]
            require(got['valid_draws'] == want['valid_draws'] == prior.DRAWS, 'V195 valid draw count differs')
            require(np.allclose([got['mean'], *got['ci']], [want['mean'], *want['ci']], rtol=1e-7, atol=2e-7),
                'V195 adjusted first-road point/bootstrap interval differs')


def run_case(args, actors, reference, mode, scope):
    expected = read(args.v191_root/'details'/f'{mode}_{scope}.json')
    aw, dw = common.weights(len(actors))
    pc, pw = np.zeros((2, N, N)), np.zeros((2, N, N))
    bc, bw = np.zeros((2, prior.DRAWS, N, N)), np.zeros((2, prior.DRAWS, N, N))
    mass = np.zeros(2)
    actor_days, key_points = [], []
    for ai, actor in enumerate(actors):
        cell = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
        require(cell['source'] == '51a9409ec2' and all((r['actor'], r['mode']) == (actor, mode)
            and r['v190_moments_reproduced'] for r in cell['rows']), 'moment source/identity differs')
        m = parent.actor_matrix(cell['rows'], scope, expected['common_key_support'], actor)
        keys = sorted(tuple(k['key']) for k in expected['common_key_support'] if k['actor'] == actor)
        days, key_cov = [], []
        for h, values in enumerate(common.balance(m[:32], m[32:])):
            first, seconds, within = parent.day_parts(values)
            kcov = key_within(values).mean(axis=0)
            require(np.allclose(kcov.sum(axis=0), within.mean(axis=0), rtol=1e-8, atol=2e-7), 'key/day covariance sum differs')
            days.append(within)
            key_cov.append(kcov/len(actors))
            c, mm = parent.weighted_covariance(first, seconds, np.full((1, 32), 1./32))
            pc[h] += c[0]/len(actors)
            pw[h] += within.mean(axis=0)/len(actors)
            mass[h] += mm[0]/len(actors)
            for start in range(0, prior.DRAWS, 64):
                stop = min(start+64, prior.DRAWS)
                c, _ = parent.weighted_covariance(first, seconds, dw[h, start:stop])
                factor = aw[start:stop, ai, None, None]
                bc[h, start:stop] += factor*c
                bw[h, start:stop] += factor*(dw[h, start:stop] @ within.reshape(32, -1)).reshape(-1, N, N)
        actor_days.append(np.array(days))
        key_points.extend(dict(actor=actor, key=list(key), covariance=np.array(key_cov)[:, ki]) for ki, key in enumerate(keys))
        print('V196_MOMENT_ACTOR', mode, scope, actor, flush=True)
    parent.reproduce(pc, mass, expected['result'])
    r, br = parent.projection(pw.mean(axis=0)), parent.projection(bw.mean(axis=0))
    stats = {part: summarize(decompose(p, r), decompose(b, br), part == 'within_day' and scope in parent.PRIMARY)
        for part, p, b in (('within_day', pw, bw), ('between_day', pc-pw, bc-bw), ('total', pc, bc))}
    reproduce(stats, reference, mode, scope)
    print('V196_V195_POINTS_AND_INTERVALS_REPRODUCED', mode, scope, flush=True)
    actor_records, key_records, pooled_keys = [], [], {}
    bootstrap_sums = {name: np.zeros((2, prior.DRAWS)) for name in TERMS}
    for ai, (actor, days) in enumerate(zip(actors, actor_days)):
        points = decompose(days.mean(axis=1)/len(actors), r, blocks=False)
        draws = {k: np.empty((2, prior.DRAWS)) for k in TERMS}
        for h, daily in enumerate(days):
            for start in range(0, prior.DRAWS, 64):
                stop = min(start+64, prior.DRAWS)
                c = (dw[h, start:stop] @ daily.reshape(32, -1)).reshape(-1, N, N)*aw[start:stop, ai, None, None]
                for name, values in decompose(c, br[start:stop], blocks=False).items():
                    draws[name][h, start:stop] = values
        for name in TERMS:
            bootstrap_sums[name] += draws[name]
        actor_records.append(dict(actor=actor, terms=point_record(points), diagnostic=summarize(points, draws)))
    for name, values in decompose(bw, br, blocks=False).items():
        require(np.allclose(bootstrap_sums[name], values, rtol=1e-8, atol=2e-7), 'bootstrap actor contributions do not sum')
    for entry in key_points:
        key_records.append(dict(actor=entry['actor'], key=entry['key'], terms=point_record(decompose(entry['covariance'], r, blocks=False))))
        key = tuple(entry['key'])
        pooled_keys.setdefault(key, np.zeros((2, N, N)))[:] += entry['covariance']
    roads = [dict(key=list(k), terms=point_record(decompose(c, r, blocks=False))) for k, c in sorted(pooled_keys.items())]
    compact_actors = [{k: v for k, v in row.items() if k != 'diagnostic'} for row in actor_records]
    result = dict(mode=mode, scope=scope, common_keys=len(key_records), analysis_mass=mass.tolist(), stats=stats,
        actor_summary=contribution_summary(compact_actors, stats['within_day']),
        road_summary=contribution_summary(roads, stats['within_day']), v195_points_and_intervals_reproduced=True)
    contribution_summary(key_records, stats['within_day'])
    write(args.out_root/'details'/f'{mode}_{scope}.json', dict(result=result, actors=actor_records,
        actor_road_keys=key_records, pooled_road_keys=roads, reference_vectors=r.tolist()))
    write(args.out_root/'cells'/f'{mode}_{scope}.json', result)
    print('V196_FIRST_ROAD_CASE_COMPLETE', mode, scope, flush=True)
    return result


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v196_*.py', '-v'], check=True)
    reference = read(args.v195_root/'summary/status.json')
    require(reference['source'] == '759c662f20' and reference['protocol']['calibration_days'] == list(DAYS)
        and reference['v191_points_reproduced'] and not reference['interval_fit'] and not reference['main_policy_changed'],
        'requires completed frozen V195 readout')
    actors = reference['actors']
    require(len(actors) == 12 and actors == sorted(set(actors)), 'actor grid differs')
    fields = [f for block in GROUPS.values() for f in block]
    require(len(fields) == len(set(fields)) == 24 and set(fields) == set(parent.CONTROLS), 'fixed control partition differs')
    protocol = registration()
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    write(args.out_root/'protocol.json', protocol)
    print('V196_FIRST_ROAD_PROTOCOL_SEALED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        for i, job in enumerate(as_completed([pool.submit(run_case, args, actors, reference, m, s)
                for m, s in product(('old', 'handoff'), saved.SCOPES)]), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":6,"unit":"saved_moment_decomposition"}' % i, flush=True)
    write(args.out_root/'summary/status.json', dict(status='v196_first_road_decomposition_complete', source=protocol['source'],
        protocol=protocol, results=sorted(results, key=lambda r: (r['mode'], r['scope'])),
        v195_points_and_intervals_reproduced=True, raw_capture_files_opened=0, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V196_FIRST_ROAD_COMPLETE_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v194-root', 'v191-root', 'v195-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
