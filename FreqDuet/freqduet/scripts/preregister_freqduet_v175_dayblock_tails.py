#!/usr/bin/env python3
"""Preregister a day-block upper-tail envelope; audit calibration only, no fit."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v171_temporal_tails as prior


fit, ledger = prior.original, prior.ledger
read, write, require, stream = fit.read, fit.write, fit.require, fit.stream
CALIBRATION = tuple(range(2951001, 2951065))
CONFIRMATION = tuple(range(3352001, 3352065))
SAMPLERS = {day: 1751000000+i for i, day in enumerate(CONFIRMATION)}
TARGETS = ('clearance', 'hour13_moving_launch')
METRICS = ('above', 'below', 'coverage', 'width_s', 'score_s', 'coarse_above',
    'coarse_score_delta_s', 'v167_above', 'v167_score_delta_s', 'point_bias_s', 'headroom_s')


def key_scopes(key, fields):
    values = dict(zip(fields, key))
    return tuple(scope for scope, selected in zip(TARGETS, (values['query_phase'] == 'clearance',
        values['query_time_stratum'] == 'hour13' and values['native_stage_kind'] == 'moving-launch')) if selected)


def freshness(args):
    old = read(args.inventory_file)
    known = {int(k) for k in old['day_references']} | {d for g in old['groups'] for d in g['days']}
    collisions = [dict(path='V159 inventory', day=d) for d in sorted(known & set(CONFIRMATION))]
    roots = set(ledger.archives.result_roots(args.parent)) | {args.v172_root.parent.parent}
    files, names = set(), 0
    headers = {'protocol.json', 'preregistration.json', 'registration.json', 'model.json', 'manifest.json'}
    for root in sorted(roots):
        for directory, children, filenames in os.walk(root):
            path = Path(directory)
            children[:] = sorted(d for d in children if d not in ledger.SKIP and
                path/d != args.out_root.parent and len((path/d).relative_to(root).parts) <= 5)
            if path.name in ledger.RAW | ledger.RECEIPTS:
                children[:] = []
                for name in filenames:
                    names += 1
                    day = ledger.filename_day(name)
                    if day in CONFIRMATION:
                        collisions.append(dict(path=str(path/name), day=day))
            for name in filenames:
                if path.name not in ledger.RAW and name.endswith('.json') and (
                        name in headers or path.name in ledger.RECEIPTS):
                    files.add(path/name)
    for index, path in enumerate(sorted(files), 1):
        value = read(path)
        days = ledger.extract.day_ids(value) & set(CONFIRMATION)
        samplers = ledger.metadata.sampler_ids(value) & set(SAMPLERS.values())
        if days or samplers:
            collisions.append(dict(path=str(path), days=sorted(days), samplers=sorted(samplers)))
        if index % 5000 == 0:
            print('V175_FRESHNESS_METADATA', index, len(files), flush=True)
    generators = 0
    for root in args.generator_root:
        for path in root.glob('*.py'):
            if path.name != Path(__file__).name:
                generators += 1
                hits = ledger.metadata.literal_conflicts(path, set(CONFIRMATION) | set(SAMPLERS.values()))
                if hits:
                    collisions.append(dict(path=str(path), literals=sorted(hits)))
    result = dict(result_roots=len(roots), metadata_files=len(files), artifact_names=names,
        generators=generators, collisions=collisions,
        scope='known archive inventory, raw artifact names, registration/model headers, '
            'path/pair receipts and generator literals only; no confirmation outcomes or raw labels')
    write(args.out_root/'details/freshness.json', result)
    require(not collisions, 'reserved day/sampler collision; stop without substituting days')
    return result


def registration(model, fresh):
    require(model['status'] == 'sealed' and model['execution_source'] == 'a5b45df520' and
        model['calibration_days'] == list(CALIBRATION) and len(model['actors']) == 12,
        'requires frozen V172 64-day calibration')
    base = model['frozen_protocol']
    require(base['execution_source'] == 'f5d152f0bc' and not fresh['collisions'] and
        base['frozen']['point_source'] == '1b783ea8f3' and base['frozen']['native_window_s'] == 3600. and
        base['frozen']['tails'] == [.1, .1] and base['support']['minimum_calibration_days'] == 19,
        'frozen forecast/support or freshness differs')
    result = deepcopy(base)
    result.update(status='preregistered_not_fitted_not_collected', stage='v175_dayblock_upper_tail_envelope',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    samplers = {str(d): base['reservation']['sampler_seeds'][str(d)] for d in CALIBRATION}
    require(not set(samplers.values()) & set(SAMPLERS.values()), 'sampler overlap')
    result['reservation'].update(confirmation_days=list(CONFIRMATION),
        sampler_seeds={**samplers, **{str(d): SAMPLERS[d] for d in CONFIRMATION}},
        excluded_confirmation_days=list(range(2852001, 2852033))+list(range(3152001, 3152033)),
        confirmation_paths=64*12*2, split_unit='whole demand day and sampler shared across actors and modes',
        size_reason='exactly 64 fresh confirmation days, emphasizing independent days rather than actor repeats; '
            'fixed budget, not a power guarantee or target-until-pass rule')
    result['upper_tail_envelope'] = dict(candidate_count=1, targets=list(TARGETS),
        membership='query_phase == clearance OR (query_time_stratum == hour13 AND native_stage_kind == moving-launch); '
            'query-time state only; the union is applied once, not two stacked corrections',
        base='unchanged V172 key, nominal tails, day-mass ranks and physical truncation',
        support='use exactly the V172 finite mask; >=19 original distinct contributing calibration days; '
            'native-low-support, low-key and unseen keys always abstain, no reference fallback',
        resampling='within each key, sort original D contributing days; draw D day slots with replacement; '
            'each selected day carries all its joint R/S/D atoms and its equal-episode/equal-query weights',
        repeated_days='a day drawn m times receives m/D mass; never collapse duplicates, count them as independent '
            'support, or recompute ranks from the number of unique resampled days',
        ranks='hold original D fixed: k=floor((D+1)/10); in every draw invert the weighted joint-total '
            'CDF at k/D, using the existing smallest-value-with-mass-at-least rule without interpolation',
        random='RandomState(20260803), reset per key; 5000 draws in sorted contributing-day order; '
            'not query bootstrap, not independent channel bootstrap, not actor repeats',
        envelope='sort 5000 Qlow draws, take one-based order rank 250 (the fixed 5th percentile without '
            'interpolation); Qlow_new=min(Qlow_V172, Qlow_draw[250]); Qhigh_new=Qhigh_V172',
        interval='L_new=L_V172; U_new=max(0,m-Qlow_new), m=frozen V151; '
            'outside target union copy V172 exactly; do not update any frozen map in place',
        monotonicity='same finite mask and lower bound; U_new >= U_V172 for every finite query; '
            'no upper contraction and no extra loss of coverage relative to V172 on a matched query',
        no_search='no choice of key, confidence level, bootstrap seed/draws, radius, stop, day subset, '
            'mixing coefficient or endpoint shrinkage using calibration or confirmation scores',
        interpretation='bootstrap uncertainty envelope for the empirical upper-tail calibration; '
            'not an exact conditional/future-day coverage guarantee and not a new point prediction')
    result['comparator'].update(primary='unchanged V167 coarse reference',
        secondary=['unchanged V167 conditional map', 'unchanged V172 temporal map'],
        matching='all comparisons on exactly the proposed candidate-finite mask; missing reference is inconclusive')
    result['inference'].update(grid='12 actors x 64 fresh confirmation days in both modes and all six scopes',
        additional='report upper expansion, new/repaired misses, upper exceedance, width/lower/upper score '
            'decomposition, per-day tails and actor/day marginals; calibration statistics are in-sample only',
        uncertainty='retain the 5000-draw shared-demand-day crossed bootstrap and existing one-sided gates')
    result['isolation'].update(first='this stage: protocol, freshness, frozen-calibration support and baseline '
            'day variability only; no new quantiles, interval construction, collection or training',
        calibration_seal='separate authorization required; construct exactly one new envelope from only the '
            '64 V166 calibration days, reproduce support/mask/lower bounds and seal before collecting fresh days',
        confirmation='only after separate authorization and mapping seal: all 1536 assigned paths '
            '(12 actors x 64 fresh demand days x 2 modes); apply the sealed envelope once, no updates',
        development='historical diagnostics motivate the upper-tail question only; numerical design is fixed '
            'before this calibration readout, and no confirmation outcome is used to select parameters',
        fixed_budget='existing 64 calibration days; exactly 64 reserved confirmation days if separately '
            'authorized; no replacements, optional additions, subgroup admission or post-failure shrinkage',
        no_reuse='no V168/V173 confirmation records in fitting, support, bootstrap or threshold selection; '
            'new confirmation outcomes evaluated once after mapping seal, never update the envelope',
        failure='unchanged full six-scope/two-mode conjunction including strict all/clearance coarse-score '
            'improvement; a wider band alone is not success; any failed gate ends this registered candidate')
    result['parents'] = dict(frozen_v172='a5b45df520', frozen_v167='e6fc851ec1',
        calibration_capture='20d914af03', calibration_admission='c990e3c99b')
    result['execution'] = dict(collector_submitted=False, new_days_collected=0, new_rollouts=0,
        new_training=0, interval_fit=False, map_updates=0, confirmation_outcomes_read=False, main_policy_changed=False)
    result['freshness'] = fresh
    return result


def metrics(point, truth, bounds, coarse, secondary):
    def score(b):
        lo, hi = b
        return (hi-lo)+10.*max(lo-truth, 0.)+10.*max(truth-hi, 0.)
    lo, hi = bounds
    return np.array([float(truth > hi), float(truth < lo), float(lo <= truth <= hi), hi-lo,
        score(bounds), float(truth > coarse[1]), score(bounds)-score(coarse),
        float(truth > secondary[1]), score(bounds)-score(secondary), point-truth, hi-point])


def bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require((table['execution_source'], table['actor'], table['mode']) == ('a5b45df520', actor, mode), 'map differs')
    fields = model['frozen_protocol']['conditioning']['candidate_key']
    keys = table['candidate']
    membership = [key_scopes(e['key'], fields) for e in keys]
    key_support = [dict(scope=scope, keys=len(entries), finite_keys=sum(e['status'] == 'finite' for e in entries),
        day_histogram=dict(sorted(Counter(e['day_count'] for e in entries).items())))
        for scope, entries in ((scope, [e for e, scopes in zip(keys, membership) if scope in scopes]) for scope in TARGETS)]
    with np.load(args.v172_root/chunk['atom_file']) as z:
        q, ri, ki = z['requests'], z['atom_request'], z['candidate_key']
        finite, bounds, secondary = z['finite_candidate'], z['bounds'], z['frozen_v167_bounds']
        require(set(q[:, 0].astype(int)) == set(CALIBRATION) and len(q) == table['eligible_queries'],
            'calibration atom cohort differs')
        target = np.zeros(len(q), dtype=bool)
        moving = np.zeros(len(q), dtype=bool)
        target[ri] = np.array([bool(scopes) for scopes in membership])[ki]
        stage = fields.index('native_stage_kind')
        moving[ri] = np.array([e['key'][stage] == 'moving-launch' for e in keys])[ki]
        masks = {'clearance': (q[:, 2].astype(int) & 4) != 0,
            'hour13_moving_launch': (q[:, 4]//3600 == 13) & moving}
        require(np.array_equal(target[finite], (masks['clearance'] | masks['hour13_moving_launch'])[finite]),
            'query-time target union differs from key membership')
        require(z['finite_reference'][finite].all() and z['finite_frozen_v167'][finite].all(), 'matched reference missing')
        support = {(c['day'], c['scope']): c for c in table['cells']}
        rows = []
        for day in CALIBRATION:
            selected = np.flatnonzero((q[:, 0] == day) & finite & target)
            wanted = {tuple(q[i, 5:].astype(int)): i for i in selected}
            values = {}
            for residual in stream(args.capture_root/'calibration/residuals'/f'{actor}_{day}_{mode}.jsonl.gz'):
                key = tuple(residual['query_key'])
                if key not in wanted:
                    continue
                i = wanted.pop(key)
                require(residual['query_time_s'] == q[i, 4] and residual['label']['status'] == 'resolved',
                    'exact calibration label/clock missing')
                values[i] = metrics(q[i, 3], residual['label']['full_eta_s'], bounds[i, 0], bounds[i, 1], secondary[i])
                if not wanted:
                    break
            require(not wanted, 'calibration labels incomplete; no dropping selected queries')
            for scope in TARGETS:
                indices = selected[masks[scope][selected]]
                episodes, inverse, counts = np.unique(q[indices, 1], return_inverse=True, return_counts=True)
                original = support[day, scope]
                require(len(indices) == original['finite_queries'] and len(episodes) == original['finite_episodes'] and
                    len(episodes) > 0, 'finite target denominator differs')
                vectors = np.array([values[i] for i in indices])
                weighted = (vectors/counts[inverse, None]).sum(axis=0)/len(episodes)
                rows.append(dict(actor=actor, mode=mode, day=day, scope=scope,
                    finite_queries=len(indices), finite_fraction=original['finite_fraction'],
                    **dict(zip(METRICS, weighted.tolist()))))
    write(args.out_root/'cells'/f'{actor}_{mode}.json', dict(cells=rows, key_support=key_support))
    print('V175_CALIBRATION_BUCKET_COMPLETE', actor, mode, flush=True)
    return dict(cells=rows, key_support=[dict(actor=actor, mode=mode, **r) for r in key_support], support=table['cells'])


def summarize(rows, actors):
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) == len(actors)*64 and set(index) ==
        set(product(actors, CALIBRATION)), 'calibration crossed grid incomplete')
    matrices = {m: np.array([[index[a, d][m]
        for d in CALIBRATION] for a in actors]) for m in METRICS}
    return dict(cells=len(rows), means={m: float(x.mean()) for m, x in matrices.items()},
        day_ranges={m: [float(x.mean(axis=0).min()), float(x.mean(axis=0).max())] for m, x in matrices.items()},
        day_mean_std={m: float(x.mean(axis=0).std()) for m, x in matrices.items()},
        finite_fraction=float(np.mean([r['finite_fraction'] for r in rows])),
        interpretation='frozen V172 in-sample calibration baselines only; no candidate bands or validation claim')


def run(args):
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite a preregistration')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v175_*.py', '-v'], check=True)
    model, old = read(args.v172_root/'model.json'), read(args.v167_root/'model.json')
    audit = read(args.audit_root/'summary/status.json')
    require(model['frozen_v167_source'] == old['execution_source'] == 'e6fc851ec1' and
        model['capture_root'] == str(args.capture_root) and audit['status'] == 'v166_readback_passed' and
        (audit['audit_source'], audit['capture_source']) == ('c990e3c99b', '20d914af03') and
        audit['calibration_days'] == list(CALIBRATION) and audit['passed_paths'] == 1536 and
        audit['failed_paths'] == 0, 'requires admitted V166 calibration')
    protocol = registration(model, freshness(args))
    write(args.out_root/'protocol.json', protocol)
    print('V175_PROTOCOL_SEALED_NO_FIT_NO_COLLECTION', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, model, chunk) for chunk in model['chunks']]
        for job in as_completed(jobs):
            results.append(job.result())
    cells = [c for r in results for c in r['cells']]
    support = prior.support_release([c for r in results for c in r['support']], model['actors'], CALIBRATION)
    require(support == model['support_admission'] and read(args.v172_root/'model.json') == model and
        read(args.v167_root/'model.json') == old, 'support or frozen headers changed')
    write(args.out_root/'details/calibration_day_cells.json', cells)
    write(args.out_root/'details/key_support.json', [k for r in results for k in r['key_support']])
    readouts = [dict(mode=m, scope=s, **summarize([r for r in cells if (r['mode'], r['scope']) == (m, s)],
        model['actors'])) for m, s in product(('old', 'handoff'), TARGETS)]
    write(args.out_root/'summary/status.json', dict(status='v175_protocol_and_calibration_audit_complete',
        execution_source=protocol['execution_source'], protocol=protocol, support=support,
        calibration_readouts=readouts, key_support=[k for r in results for k in r['key_support']],
        interval_fit=False, map_updates=0, collector_submitted=False, new_days_collected=0, new_rollouts=0,
        new_training=0, confirmation_outcomes_read=False, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], main_policy_changed=False,
        next='Preregistered only. Separate authorization is required for interval construction; no collection now.'))
    print('V175_PREREGISTERED_CALIBRATION_ONLY_COMPLETE', support['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v172-root', 'v167-root', 'capture-root', 'audit-root', 'inventory-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--generator-root', type=Path, action='append', required=True)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
