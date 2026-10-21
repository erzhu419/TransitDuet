#!/usr/bin/env python3
"""Seal one V171 temporal joint map on the existing 64 calibration days only."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v171_temporal_tails as plan


fit = plan.original
read, write, require = fit.read, fit.write, fit.require
V167_SOURCE, V171_SOURCE = 'e6fc851ec1', 'f5d152f0bc'


def validate_parents(model, protocol, admission, audit):
    require(model['status'] == 'sealed' and model['execution_source'] == V167_SOURCE and
        model['calibration_days'] == list(range(2951001, 2951065)), 'requires frozen V167 64-day maps')
    require(admission['status'] == 'v171_protocol_and_calibration_support_complete' and
        admission['execution_source'] == V171_SOURCE and admission['protocol'] == protocol and
        admission['support']['decision'] == 'pass', 'requires sealed V171 protocol and support admission')
    require(protocol['execution_source'] == V171_SOURCE and
        protocol['reservation']['calibration_days'] == model['calibration_days'] and
        protocol['reservation']['actors'] == model['actors'] and
        protocol['conditioning']['candidate_key'] ==
        model['frozen_protocol']['conditioning']['candidate_key']+['query_time_stratum'] and
        protocol['frozen'] == model['frozen_protocol']['frozen'] and
        protocol['support'] == model['frozen_protocol']['support'] and
        protocol['inference']['scopes'] == list(plan.SCOPES), 'frozen V171 design differs')
    reservation = protocol['reservation']
    require(reservation['confirmation_days'] == list(plan.CONFIRMATION) and
        [reservation['sampler_seeds'][str(d)] for d in plan.CONFIRMATION] ==
        [plan.SAMPLERS[d] for d in plan.CONFIRMATION] and not any(protocol['execution'].values()),
        'confirmation reservation or original execution receipt differs')
    require((audit['status'], audit['audit_source'], audit['capture_source'], audit['passed_paths'],
        audit['failed_paths']) == ('v166_readback_passed', 'c990e3c99b', '20d914af03', 1536, 0) and
        audit['calibration_days'] == model['calibration_days'] and
        audit['counts'].get('eligible_unresolved', 0) == 0 and audit['query_metadata_complete'] and
        audit['query_support_and_label_clocks_checked'], 'requires complete V166 label/state/clock admission')


def temporal_fit(entries, old_keys, requests, atom_request, vectors):
    times = requests[atom_request, 4]
    codes, inverse = np.unique(old_keys*3+plan.strata(times), return_inverse=True)
    candidate, records = [], []
    mass = np.zeros(len(vectors), dtype=np.int64)
    for index, code in enumerate(codes):
        selected = np.flatnonzero(inverse == index)
        ri = atom_request[selected]
        entry, denominator = fit.fit_key(vectors[selected], requests[ri, 0].astype(np.int64),
            requests[ri, 1].astype(np.int64))
        old_key, period = divmod(int(code), 3)
        candidate.append(dict(key=[*entries[old_key]['key'], plan.STRATA[period]], **entry))
        records.append(dict(old_key=old_key, stratum=plan.STRATA[period], days=entry['day_count']))
        mass[selected] = denominator
    total_mass = np.bincount(inverse, weights=1./mass, minlength=len(candidate))
    require(np.allclose(total_mass, 1., rtol=0., atol=1e-9), 'new-key mixture mass is not one')
    return candidate, inverse, mass, records, float(np.max(np.abs(total_mass-1.)))


def build_bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v167_root/chunk['map_file'])
    support = read(args.v171_root/'cells'/f'{actor}_{mode}.json')
    require((table['execution_source'], table['actor'], table['mode']) == (V167_SOURCE, actor, mode),
        'frozen bucket source differs')
    # V171 read all unsupported hour13 contexts; this cohort contains no moving ones.
    require(support['metadata']['unsupported_hour13_moving'] == 0,
        'V171 unsupported moving membership requires a different cohort readback')
    with np.load(args.v167_root/chunk['atom_file']) as z:
        requests, ri = z['requests'], z['atom_request']
        vectors, old_keys = z['residual_rsd'], z['candidate_key']
        native, baseline = z['native_supported'], z['finite_candidate']
        require(set(requests[:, 0].astype(int)) == set(model['calibration_days']) and
            len(requests) == table['eligible_queries'] and len(vectors) == table['joint_atoms'] and
            np.isfinite(vectors).all(), 'fitting changed the admitted calibration cohort')
        candidate, keys, mass, records, mass_error = temporal_fit(table['candidate'], old_keys, requests, ri, vectors)
        require(records == read(args.v171_root/'details/key_support'/f'{actor}_{mode}.json'),
            'new-key distinct-day support differs from V171')
        bounds = z['bounds'].copy()
        bounds[:, 0] = np.nan
        for index, entry in enumerate(candidate):
            selected = ri[keys == index]
            if entry['status'] == 'finite':
                point = requests[selected, 3]
                bounds[selected, 0, 0] = np.maximum(0., point-entry['qhigh_s'])
                bounds[selected, 0, 1] = np.maximum(0., point-entry['qlow_s'])
        finite, reference = np.isfinite(bounds[:, 0]).all(axis=1), z['finite_reference']
        require(not np.any(finite & ~native) and not np.any(finite & ~baseline) and
            not np.any(finite & ~reference), 'invalid native or matched frozen-comparator mask')
        require(np.all(bounds[finite, 0, 0] <= bounds[finite, 0, 1]) and
            np.all(bounds[finite, 0] >= 0.), 'invalid physical interval')
        field = model['frozen_protocol']['conditioning']['candidate_key'].index('native_stage_kind')
        moving = np.zeros(len(requests), dtype=bool)
        moving[ri] = np.array([e['key'][field] == 'moving-launch' for e in table['candidate']])[old_keys]
        masks = [(requests[:, 2].astype(int) & (1 << i)) != 0 for i in range(len(fit.SCOPES))]
        masks.append(masks[-1] & moving)
        days, episodes = requests[:, 0].astype(int), requests[:, 1].astype(int)
        cells = [dict(actor=actor, mode=mode,
            **plan.support_cell(d, s, mask, days, episodes, native, finite, baseline))
            for d in model['calibration_days'] for s, mask in zip(plan.SCOPES, masks)]
        require(cells == support['cells'], 'actual fitted masks do not reproduce V171 support cells')
        weights = z['mass_denominator'].copy()
        changed = int(np.count_nonzero(weights[:, 0] != mass))
        weights[:, 0] = mass
        atom_file = f'atoms/{actor}_{mode}.npz'
        (args.out_root/'atoms').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_root/atom_file, residual_rsd=vectors, atom_request=ri,
            candidate_key=keys, reference_key=z['reference_key'], mass_denominator=weights,
            requests=requests, episode_keys=z['episode_keys'], native_supported=native, bounds=bounds,
            finite_candidate=finite, finite_reference=reference,
            frozen_v167_bounds=z['bounds'][:, 0], finite_frozen_v167=baseline)
        with np.load(args.out_root/atom_file) as saved:
            require(np.array_equal(saved['mass_denominator'], weights) and
                np.array_equal(saved['finite_candidate'], finite) and
                np.array_equal(saved['bounds'], bounds, equal_nan=True), 'saved weights/bounds readback differs')
    strata = [dict(stratum=s, keys=sum(r['stratum'] == s for r in records),
        finite_keys=sum(r['stratum'] == s and r['days'] >= 19 for r in records),
        day_histogram=dict(sorted(Counter(r['days'] for r in records if r['stratum'] == s).items())))
        for s in plan.STRATA]
    result = dict(actor=actor, mode=mode, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        candidate=candidate, reference=table['reference'], cells=cells, atom_file=atom_file,
        queries=table['queries'], eligible_queries=len(requests), joint_atoms=len(vectors),
        finite_queries=int(finite.sum()), native_abstentions=int((~native).sum()),
        key_abstentions=int((native & ~finite).sum()), recomputed_weight_atoms=len(mass),
        changed_weight_atoms=changed, maximum_key_mass_error=mass_error, strata=strata,
        v171_support_reproduced=True, frozen_reference_reused=True, saved_arrays_readback_passed=True)
    write(args.out_root/'maps'/f'{actor}_{mode}.json', result)
    print('V172_BUCKET_SEALED', actor, mode, flush=True)
    return result


def seal(args):
    require(args.out_root not in (args.v167_root, args.v171_root) and
        not (args.out_root/'model.json').exists(), 'use an unsealed independent output root')
    for pattern in ('test_v163_*.py', 'test_v171_*.py', 'test_v172_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    parent = read(args.v167_root/'model.json')
    protocol = read(args.v171_root/'protocol.json')
    admission = read(args.v171_root/'summary/status.json')
    audit = read(args.audit_root/'summary/status.json')
    validate_parents(parent, protocol, admission, audit)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(build_bucket, args, parent, c) for c in parent['chunks']]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='actor_mode_map')), flush=True)
    rows.sort(key=lambda r: (r['actor'], r['mode']))
    cells = [c for row in rows for c in row['cells']]
    release = plan.support_release(cells, parent['actors'], parent['calibration_days'])
    require(release == admission['support'], 'fitted support release differs from V171')
    require(read(args.v167_root/'model.json') == parent and
        read(args.v171_root/'protocol.json') == protocol, 'frozen parent changed')
    totals = {f: sum(r[f] for r in rows) for f in ('queries', 'eligible_queries', 'joint_atoms',
        'finite_queries', 'native_abstentions', 'key_abstentions', 'recomputed_weight_atoms', 'changed_weight_atoms')}
    require(totals['queries'] == audit['counts']['queries'] and
        totals['eligible_queries'] == audit['counts']['denominator_eligible'] and
        totals['joint_atoms'] == audit['counts']['native_supported'], 'calibration totals differ from V166')
    model = dict(status='sealed', stage='v172_v171_temporal_joint_maps',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_protocol=protocol,
        calibration_days=parent['calibration_days'], actors=parent['actors'], modes=parent['modes'],
        capture_root=parent['capture_root'], capture_source='20d914af03', audit_source='c990e3c99b',
        protocol_root=str(args.v171_root), protocol_source=V171_SOURCE,
        frozen_v167_root=str(args.v167_root), frozen_v167_source=V167_SOURCE,
        mixture_mass='atom mass = 1 / stored integer mass_denominator; candidate recomputed within each new key; '
            'reference copied unchanged from V167',
        request_columns=parent['request_columns'], episode_columns=parent['episode_columns'],
        vector_columns=parent['vector_columns'], bound_axes=parent['bound_axes'],
        absent_bounds=parent['absent_bounds'], unseen_keys=parent['unseen_keys'],
        chunks=[dict(actor=r['actor'], mode=r['mode'], map_file=f"maps/{r['actor']}_{r['mode']}.json",
            atom_file=r['atom_file']) for r in rows], support_admission=release,
        confirmation_release_eligible=release['decision'] == 'pass', confirmation_started=False,
        confirmation_outcomes_read=False, main_policy_changed=False)
    summary = dict(status='v172_temporal_maps_sealed_calibration_only', execution_source=model['execution_source'],
        protocol_source=V171_SOURCE, frozen_v167_source=V167_SOURCE, calibration_days=parent['calibration_days'],
        calibration_paths=1536, actors=12, modes=parent['modes'], totals=totals,
        candidate_keys=sum(len(r['candidate']) for r in rows),
        finite_candidate_keys=sum(e['status'] == 'finite' for r in rows for e in r['candidate']),
        strata=[dict(actor=r['actor'], mode=r['mode'], groups=r['strata']) for r in rows],
        support_admission=release, v171_support_reproduced=True, saved_arrays_readback_passed=True,
        maximum_key_mass_error=max(r['maximum_key_mass_error'] for r in rows), frozen_reference_reused=True,
        interval_fit=True, confirmation_release_eligible=model['confirmation_release_eligible'],
        confirmation_started=False, confirmation_outcomes_read=False, new_days_collected=0,
        new_rollouts=0, new_training=0, v167_changed=False, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], main_policy_changed=False,
        next='Mapping sealed only; wait for separate authorization before confirmation collection.',
        evidence='calibration-only deterministic construction and support, not independent coverage or controller admission')
    write(args.out_root/'support_cells.json', cells)
    write(args.out_root/'model.json', model)
    require(read(args.out_root/'model.json') == model, 'sealed model JSON roundtrip differs')
    write(args.out_root/'summary/status.json', summary)
    print('V172_MAPS_SEALED_CALIBRATION_ONLY', release['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v167-root', 'v171-root', 'audit-root', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    seal(parser.parse_args())
