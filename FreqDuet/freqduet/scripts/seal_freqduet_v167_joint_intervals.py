#!/usr/bin/env python3
"""Seal V161/V165 joint maps on the admitted V166 64-day cohort only."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v166_fresh_capture as entry
import seal_freqduet_v163_joint_intervals as fit


read, write, require, close = fit.read, fit.write, fit.require, fit.close
CAPTURE_SOURCE, AUDIT_SOURCE = '20d914af03', 'c990e3c99b'
PROTOCOL_SOURCE = '4a6d783667'


def validate_admission(sealed, audit):
    r = sealed['reservation']
    require((audit['status'], audit['audit_source'], audit['capture_source'], audit['protocol_source'],
        audit['expected_paths'], audit['passed_paths'], audit['failed_paths']) ==
        ('v166_readback_passed', AUDIT_SOURCE, CAPTURE_SOURCE, PROTOCOL_SOURCE, 1536, 1536, 0),
        'requires complete V166 metadata admission')
    require(audit['calibration_days'] == r['calibration_days'] == list(entry.fresh.CALIBRATION) and
        audit['actors'] == r['actors'] and audit['modes'] == r['modes'] and
        audit['counts'].get('eligible_unresolved', 0) == 0 and audit['calibration_input_eligible'] and
        audit['query_metadata_complete'] and audit['query_support_and_label_clocks_checked'],
        'calibration cohort, labels or metadata admission changed')
    require(not audit['interval_fit'] and not audit['confirmation_started'] and
        not audit['confirmation_outcomes_read'] and audit['confirmation_paths'] == 0,
        'requires unfitted calibration and unopened confirmation')


def build_bucket(args, sealed, actor, mode):
    root, cal_days = args.capture_root/'calibration', sealed['reservation']['calibration_days']
    candidate_keys, reference_keys, episode_keys = {}, {}, {}
    groups, refs = defaultdict(list), defaultdict(list)
    requests, vectors, atom_requests, atom_keys, atom_refs = [], [], [], [], []
    observed = Counter()
    for day in cal_days:
        name = f'{actor}_{day}_{mode}'
        receipt = read(root/'capture_cells'/(name+'.json'))
        require((receipt['execution_source'], receipt['protocol_source'], receipt['seed'], receipt['day'],
            receipt['mode'], receipt['sampler_seed']) == (CAPTURE_SOURCE, PROTOCOL_SOURCE, actor, day, mode,
            sealed['reservation']['sampler_seeds'][str(day)]), 'capture assignment/source changed')
        observed.update(receipt['counts'])
        count = 0
        for q, residual in zip_longest(fit.stream(root/'queries'/(name+'.jsonl.gz')),
                fit.stream(root/'residuals'/(name+'.jsonl.gz'))):
            require(q is not None and residual is not None and q['query_key'] == residual['query_key'] and
                q['query_time_s'] == residual['query_time_s'], 'calibration query/residual join changed')
            require((q['actor'], q['day'], q['mode']) == (actor, day, mode), 'wrong calibration cohort')
            count += 1
            if not q['denominator_eligible']:
                continue
            require(residual['label']['status'] == 'resolved', 'unresolved eligible label blocks seal')
            episode = (day, *q['episode'])
            ep = episode_keys.setdefault(episode, len(episode_keys))
            ri = len(requests)
            requests.append((day, ep, fit.scope_bits(q), q['point_s'], q['query_time_s'], *q['query_key']))
            if not q['native_supported']:
                continue
            require(q['candidate_key'] is not None and residual['residual_s'] is not None,
                'supported calibration query lacks a joint vector')
            key = tuple(q['candidate_key'])
            ref = tuple(q['condition_values'][f] for f in sealed['comparator']['key'])
            ki = candidate_keys.setdefault(key, len(candidate_keys))
            ci = reference_keys.setdefault(ref, len(reference_keys))
            ai = len(vectors)
            errors = [residual['residual_s'][c] for c in fit.CHANNELS]
            close(sum(errors), q['point_s']-residual['label']['full_eta_s'])
            vectors.append(errors)
            atom_requests.append(ri)
            atom_keys.append(ki)
            atom_refs.append(ci)
            groups[ki].append(ai)
            refs[ci].append(ai)
        require(count == receipt['query_record_count'], 'calibration query count changed')
    request = np.asarray(requests, dtype=float)
    day, ep, scope = (request[:, i].astype(np.int64) for i in range(3))
    atom_requests = np.asarray(atom_requests, dtype=np.int64)
    vectors = np.asarray(vectors, dtype=float)
    require(np.isfinite(vectors).all(), 'nonfinite joint residual')
    require(len(request) == observed['denominator_eligible'] and len(vectors) == observed['native_supported'],
        'fitting changed the admitted cohort')
    candidate, reference = [None]*len(candidate_keys), [None]*len(reference_keys)
    weights = np.zeros((len(vectors), 2), dtype=np.int64)
    for output, keys, indices, column in ((candidate, candidate_keys, groups, 0), (reference, reference_keys, refs, 1)):
        for key, index in keys.items():
            selected = np.asarray(indices[index], dtype=np.int64)
            ri = atom_requests[selected]
            record, denominator = fit.fit_key(vectors[selected], day[ri], ep[ri])
            output[index] = dict(key=list(key), **record)
            weights[selected, column] = denominator
    native = np.zeros(len(request), dtype=bool)
    native[atom_requests] = True
    bounds = np.full((len(request), 2, 2), np.nan)
    for ai, ri in enumerate(atom_requests):
        for column, record in enumerate((candidate[atom_keys[ai]], reference[atom_refs[ai]])):
            value = fit.interval(request[ri, 3], record)
            if value is not None:
                bounds[ri, column] = value
    finite = np.isfinite(bounds[:, 0]).all(axis=1)
    ref_finite = np.isfinite(bounds[:, 1]).all(axis=1)
    require(not np.any(finite & ~native) and not np.any(finite & ~ref_finite), 'invalid candidate/reference mask')
    require(np.all(bounds[finite, 0, 0] <= bounds[finite, 0, 1]) and np.all(bounds[finite, 0] >= 0.),
        'invalid physical interval bounds')
    stem = f'{actor}_{mode}'
    (args.out_root/'atoms').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_root/'atoms'/(stem+'.npz'), residual_rsd=vectors,
        atom_request=atom_requests, candidate_key=np.asarray(atom_keys), reference_key=np.asarray(atom_refs),
        mass_denominator=weights, requests=request, episode_keys=np.asarray(list(episode_keys), dtype=np.int64),
        native_supported=native, bounds=bounds, finite_candidate=finite, finite_reference=ref_finite)
    cells = fit.support_cells(actor, mode, cal_days, day, ep, scope, finite, native, ref_finite)
    result = dict(actor=actor, mode=mode, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        candidate=candidate, reference=reference, cells=cells, queries=observed['queries'],
        eligible_queries=len(request), joint_atoms=len(vectors), finite_queries=int(finite.sum()),
        native_abstentions=int((~native).sum()), key_abstentions=int((native & ~finite).sum()),
        atom_file='atoms/'+stem+'.npz')
    write(args.out_root/'maps'/(stem+'.json'), result)
    print('V167_BUCKET_COMPLETE', actor, mode, flush=True)
    return result


def seal(args):
    require(not (args.out_root/'model.json').exists(), 'mapping already sealed; do not refit or overwrite')
    for pattern in ('test_v163_*.py', 'test_v167_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    sealed, _ = entry.parents(args)
    audit = read(args.audit_root/'summary/status.json')
    validate_admission(sealed, audit)
    reservation = sealed['reservation']
    require(not (args.capture_root/'confirmation').exists(), 'unexpected confirmation data')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(build_bucket, args, sealed, a, m) for a in reservation['actors'] for m in fit.protocol.MODES]
        for done, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='actor_mode_map')), flush=True)
    results.sort(key=lambda r: (r['actor'], r['mode']))
    cells = [c for r in results for c in r['cells']]
    admission = fit.release(cells, reservation['actors'], reservation['calibration_days'])
    totals = {f: sum(r[f] for r in results) for f in ('queries', 'eligible_queries', 'joint_atoms',
        'finite_queries', 'native_abstentions', 'key_abstentions')}
    require(totals['queries'] == audit['counts']['queries'] and
        totals['eligible_queries'] == audit['counts']['denominator_eligible'] and
        totals['joint_atoms'] == audit['counts']['native_supported'] and
        totals['finite_queries'] == sum(r['queries_in_keys_at_minimum'] for r in audit['key_day_support'].values()),
        'fit support does not reproduce independent V166 readback')
    support = []
    for mode in fit.protocol.MODES:
        selected = [r for r in results if r['mode'] == mode]
        entries = [e for r in selected for e in r['candidate']]
        support.append(dict(mode=mode, keys=len(entries), finite_keys=sum(e['status'] == 'finite' for e in entries),
            day_count_histogram=dict(sorted(Counter(e['day_count'] for e in entries).items())),
            native_abstentions=sum(r['native_abstentions'] for r in selected),
            key_abstentions=sum(r['key_abstentions'] for r in selected)))
    model = dict(status='sealed', stage='v167_v161_v165_joint_calibration_maps',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], calibration_days=reservation['calibration_days'],
        actors=reservation['actors'], modes=list(fit.protocol.MODES), frozen_protocol=sealed,
        capture_root=str(args.capture_root), capture_source=CAPTURE_SOURCE, audit_source=AUDIT_SOURCE,
        estimator_implementation_source='35ae15b185',
        mixture_mass='atom mass = 1 / stored integer mass_denominator, separately for candidate and reference',
        request_columns=['day', 'episode_id', 'scope_bits', 'fixed_v151_point_s', 'query_time_s', 'trip', 'stop', 'direction'],
        episode_columns=['day', 'follower_trip', 'follower_bus', 'target_stop', 'direction'],
        vector_columns=list(fit.CHANNELS), bound_axes=['request', 'candidate/reference', 'lower/upper'],
        absent_bounds='NaN with explicit finite masks; never zero-filled',
        unseen_keys='permanent abstention; no coarse pooling into candidate',
        chunks=[dict(actor=r['actor'], mode=r['mode'], map_file=f"maps/{r['actor']}_{r['mode']}.json",
            atom_file=r['atom_file']) for r in results], support_admission=admission,
        confirmation_release_eligible=admission['decision'] == 'pass', confirmation_started=False,
        confirmation_outcomes_read=False, main_policy_changed=False)
    summary = dict(status='v167_maps_sealed_calibration_only', execution_source=model['execution_source'],
        capture_source=CAPTURE_SOURCE, audit_source=AUDIT_SOURCE, protocol_source=PROTOCOL_SOURCE,
        estimator_implementation_source='35ae15b185', calibration_days=reservation['calibration_days'],
        calibration_paths=1536, actors=12, modes=list(fit.protocol.MODES), totals=totals,
        candidate_keys=sum(len(r['candidate']) for r in results),
        finite_candidate_keys=sum(e['status'] == 'finite' for r in results for e in r['candidate']),
        reference_keys=sum(len(r['reference']) for r in results), key_support=support, support_admission=admission,
        confirmation_release_eligible=model['confirmation_release_eligible'], confirmation_started=False,
        confirmation_outcomes_read=False, new_rollouts=0, new_training=0, old_v163_maps_changed=False,
        v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1], main_policy_changed=False,
        evidence='calibration support and sealed maps only; no independent coverage or controller admission')
    (args.out_root/'summary').mkdir(parents=True, exist_ok=True)
    fit.csv_rows(args.out_root/'summary/mode_scope.csv', admission['mode_scopes'])
    for field in ('actor', 'day'):
        fit.csv_rows(args.out_root/'summary'/('by_'+field+'.csv'), fit.marginal(cells, field))
    write(args.out_root/'support_cells.json', cells)
    write(args.out_root/'model.json', model)
    require(read(args.out_root/'model.json') == model, 'sealed model JSON roundtrip differs')
    write(args.out_root/'summary/status.json', summary)
    print('V167_MAPS_SEALED_CALIBRATION_ONLY', admission['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'audit-root', 'out-root', 'v165-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    seal(parser.parse_args())
