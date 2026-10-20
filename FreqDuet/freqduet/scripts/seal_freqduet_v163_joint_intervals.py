#!/usr/bin/env python3
"""Construct and seal V161 maps using the audited V162 calibration days only."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
from itertools import zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v162_calibration_capture as capture


protocol = capture.protocol
read, write, require, close = capture.read, capture.write, capture.require, capture.metadata.close
stream = capture.metadata.stream
CHANNELS = ('running', 'service', 'residence')
SCOPES = protocol.SCOPES


def hierarchy(days, episodes):
    pairs, inverse, queries = np.unique(np.column_stack((days, episodes)), axis=0,
        return_inverse=True, return_counts=True)
    contributing_days, day_index, episode_counts = np.unique(pairs[:, 0],
        return_inverse=True, return_counts=True)
    return contributing_days, (episode_counts[day_index]*queries)[inverse]


def inverse_day_mass(total, denominator, rank):
    order = np.argsort(total, kind='stable')
    values, den = total[order], denominator[order]
    cumulative = np.cumsum(1./den.astype(np.longdouble), dtype=np.longdouble)
    i = min(int(np.searchsorted(cumulative, rank, side='left')), len(values)-1)
    # Resolve exact day-mass boundaries without floating-point tie shifts.
    denominators, counts = np.unique(den[:i+1], return_counts=True)
    mass = sum((Fraction(int(n), int(d)) for n, d in zip(counts, denominators)), Fraction())
    while mass < rank:
        i += 1
        mass += Fraction(1, int(den[i]))
    while i and mass-Fraction(1, int(den[i])) >= rank:
        mass -= Fraction(1, int(den[i]))
        i -= 1
    return float(values[i])


def fit_key(vectors, days, episodes):
    contributing, denominator = hierarchy(days, episodes)
    count = len(contributing)
    entry = dict(status='abstain_less_than_19_days', days=contributing.tolist(), day_count=count,
        queries=len(days), episodes=len(np.unique(episodes)), qlow_s=None, qhigh_s=None)
    ranks = protocol.rank_levels(count)
    if ranks is not None:
        total = vectors.sum(axis=1)
        entry.update(status='finite', ranks=ranks,
            qlow_s=inverse_day_mass(total, denominator, ranks['lower_rank']),
            qhigh_s=inverse_day_mass(total, denominator, ranks['upper_rank']))
        require(entry['qlow_s'] <= entry['qhigh_s'], 'joint residual quantiles reversed')
    return entry, count*denominator


def interval(point, entry, native_supported=True):
    if not native_supported or entry is None or entry['status'] != 'finite':
        return None
    require(np.isfinite(point) and point >= 0., 'invalid fixed point')
    return max(0., point-entry['qhigh_s']), max(0., point-entry['qlow_s'])


def scope_bits(q):
    phase = q['condition_values']['query_phase']
    hour = q['observable']['hour']
    return 1 | (2 if phase == 'active' else 4) | (8 if hour == 2 else 0) | (16 if hour == 13 else 0)


def support_cells(actor, mode, cal_days, days, episodes, scopes, finite, native, reference_finite):
    rows = []
    for day in cal_days:
        for i, scope in enumerate(SCOPES):
            mask = (days == day) & ((scopes & (1 << i)) != 0)
            selected = np.flatnonzero(mask)
            if len(selected):
                unique, inv, n = np.unique(episodes[selected], return_inverse=True, return_counts=True)
                finite_counts = np.bincount(inv, weights=finite[selected], minlength=len(unique))
                fraction = float(np.mean(finite_counts/n))
                finite_episodes = int(np.count_nonzero(finite_counts))
            else:
                unique, fraction, finite_episodes = [], None, 0
            rows.append(dict(actor=actor, day=day, mode=mode, scope=scope,
                eligible_queries=int(mask.sum()), eligible_episodes=len(unique),
                finite_queries=int(finite[mask].sum()), finite_episodes=finite_episodes,
                native_abstentions=int((~native[mask]).sum()),
                key_abstentions=int((native[mask] & ~finite[mask]).sum()),
                missing_reference=int((finite[mask] & ~reference_finite[mask]).sum()),
                finite_fraction=fraction))
    return rows


def release(rows, actors, days):
    output, expected = [], {(a, d) for a in actors for d in days}
    for mode in protocol.MODES:
        for scope in SCOPES:
            cells = [r for r in rows if (r['mode'], r['scope']) == (mode, scope)]
            complete = len(cells) == len(expected) and {(r['actor'], r['day']) for r in cells} == expected
            nonempty = complete and all(r['eligible_episodes'] > 0 and r['finite_episodes'] > 0 for r in cells)
            fractions = [r['finite_fraction'] for r in cells if r['finite_fraction'] is not None]
            fraction = float(np.mean(fractions)) if complete and len(fractions) == len(cells) else None
            missing = sum(r['missing_reference'] for r in cells)
            admitted = nonempty and fraction >= .8 and not missing
            output.append(dict(mode=mode, scope=scope, cells=len(cells), complete_grid=complete,
                every_cell_has_finite_estimate=nonempty, finite_fraction=fraction,
                minimum_cell_fraction=min(fractions) if fractions else None,
                eligible_queries=sum(r['eligible_queries'] for r in cells),
                finite_queries=sum(r['finite_queries'] for r in cells),
                native_abstentions=sum(r['native_abstentions'] for r in cells),
                key_abstentions=sum(r['key_abstentions'] for r in cells), missing_reference=missing,
                support_admitted=admitted))
    return dict(decision='pass' if all(r['support_admitted'] for r in output) else 'insufficient_support',
        mode_scopes=output, failed=[r['mode']+'/'+r['scope'] for r in output if not r['support_admitted']])


def build_bucket(args, sealed, actor, mode):
    root, cal_days = args.capture_root/'calibration', sealed['reservation']['calibration_days']
    candidate_keys, reference_keys, episode_keys = {}, {}, {}
    groups, refs = defaultdict(list), defaultdict(list)
    requests, vectors, atom_requests, atom_keys, atom_refs = [], [], [], [], []
    observed = Counter()
    for day in cal_days:
        name = f'{actor}_{day}_{mode}'
        receipt = read(root/'capture_cells'/(name+'.json'))
        require(receipt['execution_source'] == '25a2eddeaf', 'capture source changed')
        observed.update(receipt['counts'])
        count = 0
        for q, residual in zip_longest(stream(root/'queries'/(name+'.jsonl.gz')),
                stream(root/'residuals'/(name+'.jsonl.gz'))):
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
            requests.append((day, ep, scope_bits(q), q['point_s'], q['query_time_s'], *q['query_key']))
            if not q['native_supported']:
                continue
            require(q['candidate_key'] is not None and residual['residual_s'] is not None,
                'supported calibration query lacks a joint vector')
            key = tuple(q['candidate_key'])
            ref = tuple(q['condition_values'][f] for f in sealed['comparator']['key'])
            ki, ci = candidate_keys.setdefault(key, len(candidate_keys)), reference_keys.setdefault(ref, len(reference_keys))
            ai = len(vectors)
            errors = [residual['residual_s'][c] for c in CHANNELS]
            close(sum(errors), q['point_s']-residual['label']['full_eta_s'])
            vectors.append(errors)
            atom_requests.append(ri)
            atom_keys.append(ki)
            atom_refs.append(ci)
            groups[ki].append(ai)
            refs[ci].append(ai)
        require(count == receipt['query_record_count'], 'calibration query count changed')
    request = np.asarray(requests, dtype=float)
    day, ep, scope = request[:, 0].astype(np.int64), request[:, 1].astype(np.int64), request[:, 2].astype(np.int64)
    atom_requests = np.asarray(atom_requests, dtype=np.int64)
    vectors = np.asarray(vectors, dtype=float)
    require(np.isfinite(vectors).all(), 'nonfinite joint residual')
    require(len(request) == observed['denominator_eligible'] and len(vectors) == observed['native_supported'],
        'calibration fitting silently changed the audited cohort')
    candidate, reference = [None]*len(candidate_keys), [None]*len(reference_keys)
    weights = np.zeros((len(vectors), 2), dtype=np.int64)
    for output, keys, indices, column in ((candidate, candidate_keys, groups, 0), (reference, reference_keys, refs, 1)):
        for key, index in keys.items():
            selected = np.asarray(indices[index], dtype=np.int64)
            ri = atom_requests[selected]
            entry, denominator = fit_key(vectors[selected], day[ri], ep[ri])
            output[index] = dict(key=list(key), **entry)
            weights[selected, column] = denominator
    native = np.zeros(len(request), dtype=bool)
    native[atom_requests] = True
    bounds = np.full((len(request), 2, 2), np.nan)
    for ai, ri in enumerate(atom_requests):
        for column, entry in enumerate((candidate[atom_keys[ai]], reference[atom_refs[ai]])):
            value = interval(request[ri, 3], entry)
            if value is not None:
                bounds[ri, column] = value
    finite = np.isfinite(bounds[:, 0]).all(axis=1)
    ref_finite = np.isfinite(bounds[:, 1]).all(axis=1)
    require(not np.any(finite & ~native) and not np.any(finite & ~ref_finite), 'invalid candidate/reference mask')
    require(np.all(bounds[finite, 0, 0] <= bounds[finite, 0, 1]) and np.all(bounds[finite, 0] >= 0.),
        'invalid physical interval bounds')
    # Keep the joint vectors, exact mixture masses and both matched bound arrays remote.
    stem = f'{actor}_{mode}'
    (args.out_root/'atoms').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_root/'atoms'/(stem+'.npz'), residual_rsd=vectors,
        atom_request=atom_requests, candidate_key=np.asarray(atom_keys), reference_key=np.asarray(atom_refs),
        mass_denominator=weights, requests=request, episode_keys=np.asarray(list(episode_keys), dtype=np.int64),
        native_supported=native, bounds=bounds, finite_candidate=finite, finite_reference=ref_finite)
    cells = support_cells(actor, mode, cal_days, day, ep, scope, finite, native, ref_finite)
    result = dict(actor=actor, mode=mode, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        candidate=candidate, reference=reference, cells=cells, queries=observed['queries'],
        eligible_queries=len(request), joint_atoms=len(vectors), finite_queries=int(finite.sum()),
        native_abstentions=int((~native).sum()), key_abstentions=int((native & ~finite).sum()),
        atom_file='atoms/'+stem+'.npz')
    write(args.out_root/'maps'/(stem+'.json'), result)
    print('V163_BUCKET_COMPLETE', actor, mode, flush=True)
    return result


def csv_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def marginal(rows, field):
    result = []
    for mode in protocol.MODES:
        for scope in SCOPES:
            selected = [r for r in rows if (r['mode'], r['scope']) == (mode, scope)]
            for key in sorted({r[field] for r in selected}):
                cells = [r for r in selected if r[field] == key]
                fractions = [r['finite_fraction'] for r in cells]
                result.append({field: key, 'mode': mode, 'scope': scope, 'cells': len(cells),
                    'finite_fraction': float(np.mean(fractions)) if all(v is not None for v in fractions) else None,
                    'finite_queries': sum(r['finite_queries'] for r in cells),
                    'eligible_queries': sum(r['eligible_queries'] for r in cells)})
    return result


def seal(args):
    require(not (args.out_root/'model.json').exists(), 'mapping already sealed; do not refit or overwrite')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v163_*.py', '-v'], check=True)
    sealed, _ = capture.parent_protocol(args)
    audit = read(args.audit_root/'summary/status.json')
    require((audit['status'], audit['audit_source'], audit['capture_source'], audit['passed_paths'], audit['failed_paths']) ==
        ('v162_readback_passed', 'c5bef892c6', '25a2eddeaf', 768, 0), 'requires complete V162 metadata admission')
    reservation = sealed['reservation']
    require(audit['calibration_days'] == reservation['calibration_days'] and audit['actors'] == reservation['actors'] and
        audit['counts'].get('eligible_unresolved', 0) == 0, 'calibration cohort or labels changed')
    require(not (args.capture_root/'confirmation').exists(), 'unexpected confirmation data')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(build_bucket, args, sealed, a, m) for a in reservation['actors'] for m in protocol.MODES]
        for done, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print(f'V163_PROGRESS {done}/{len(jobs)}', flush=True)
    results.sort(key=lambda r: (r['actor'], r['mode']))
    cells = [c for r in results for c in r['cells']]
    admission = release(cells, reservation['actors'], reservation['calibration_days'])
    totals = {f: sum(r[f] for r in results) for f in ('queries', 'eligible_queries', 'joint_atoms',
        'finite_queries', 'native_abstentions', 'key_abstentions')}
    require(totals['queries'] == audit['counts']['queries'] and
        totals['eligible_queries'] == audit['counts']['denominator_eligible'] and
        totals['joint_atoms'] == audit['counts']['native_supported'] and
        totals['finite_queries'] == sum(r['queries_in_keys_at_minimum'] for r in audit['key_day_support'].values()),
        'fit support does not reproduce independent V162 readback')
    model = dict(status='sealed', stage='v163_v161_joint_calibration_maps',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], calibration_days=reservation['calibration_days'],
        actors=reservation['actors'], modes=list(protocol.MODES), frozen_protocol=sealed,
        capture_root=str(args.capture_root), capture_source='25a2eddeaf', audit_source='c5bef892c6',
        mixture_mass='atom mass = 1 / stored integer mass_denominator, separately for candidate and reference',
        request_columns=['day', 'episode_id', 'scope_bits', 'fixed_v151_point_s', 'query_time_s', 'trip', 'stop', 'direction'],
        episode_columns=['day', 'follower_trip', 'follower_bus', 'target_stop', 'direction'],
        vector_columns=list(CHANNELS), bound_axes=['request', 'candidate/reference', 'lower/upper'],
        absent_bounds='NaN with explicit finite masks; never zero-filled',
        chunks=[dict(actor=r['actor'], mode=r['mode'], map_file=f"maps/{r['actor']}_{r['mode']}.json",
            atom_file=r['atom_file']) for r in results], support_admission=admission,
        confirmation_release_eligible=admission['decision'] == 'pass', confirmation_started=False,
        confirmation_outcomes_read=False, main_policy_changed=False)
    summary = dict(status='v163_maps_sealed_calibration_only', execution_source=model['execution_source'],
        calibration_days=reservation['calibration_days'], calibration_paths=768, actors=12, modes=list(protocol.MODES),
        totals=totals, candidate_keys=sum(len(r['candidate']) for r in results),
        finite_candidate_keys=sum(e['status'] == 'finite' for r in results for e in r['candidate']),
        reference_keys=sum(len(r['reference']) for r in results), support_admission=admission,
        confirmation_started=False, confirmation_outcomes_read=False, new_rollouts=0, new_training=0,
        v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1], main_policy_changed=False,
        evidence='calibration support and deterministic maps only; no independent coverage or controller admission')
    (args.out_root/'summary').mkdir(parents=True, exist_ok=True)
    csv_rows(args.out_root/'summary/mode_scope.csv', admission['mode_scopes'])
    for field in ('actor', 'day'):
        csv_rows(args.out_root/'summary'/('by_'+field+'.csv'), marginal(cells, field))
    write(args.out_root/'support_cells.json', cells)
    write(args.out_root/'model.json', model)
    require(read(args.out_root/'model.json') == model, 'sealed model JSON roundtrip differs')
    write(args.out_root/'summary/status.json', summary)
    print('V163_MAPS_SEALED_CALIBRATION_ONLY', admission['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'audit-root', 'out-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    seal(parser.parse_args())
