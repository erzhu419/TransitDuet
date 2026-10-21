#!/usr/bin/env python3
"""Fit and seal V182's shared lower-bound errors on original V166 calibration only."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v182_shared_lower as plan
import reserve_freqduet_v183_confirmation as reservation
import audit_freqduet_v180_service_increment as states


read, write, require, close = plan.read, plan.write, plan.require, plan.close
previous, temporal = plan.prior.previous, plan.prior.temporal
fit = temporal.original
METRICS = ('below', 'above', 'coverage', 'width_s', 'score_s', 'v172_score_delta_s',
    'lower_delta_s', 'zero_cap', 'upper_cap', 'zero_width')


def shared_fit(error, days, episodes):
    contributing, within_day = fit.hierarchy(days, episodes)
    count = len(contributing)
    entry = dict(status='abstain_less_than_19_days', days=contributing.tolist(), day_count=count,
        queries=len(days), episodes=len(np.unique(episodes)), delta_s=None)
    ranks = fit.protocol.rank_levels(count)
    denominator = count*within_day
    if ranks is not None:
        require(np.isfinite(error).all(), 'nonfinite anchored lower-bound error')
        entry.update(status='finite', ranks=ranks,
            delta_s=fit.inverse_day_mass(error, within_day, ranks['upper_rank']))
        close(float(np.sum(1./denominator)), 1.)
    return entry, denominator


def corrected(bounds, baseline, target, signs, entries):
    result = bounds.copy()
    result[target] = np.nan
    for sign, entry in enumerate(entries):
        selected = baseline & target & (signs == sign)
        if entry['status'] == 'finite':
            lower, upper = bounds[selected].T
            result[selected, 0] = np.minimum(upper, np.maximum(0., lower-entry['delta_s']))
            result[selected, 1] = upper
    finite = np.isfinite(result).all(axis=1)
    require(not np.any(finite & ~baseline), 'original abstention was rescued')
    require(np.array_equal(result[~target], bounds[~target], equal_nan=True), 'non-target bounds changed')
    require(np.array_equal(result[finite, 1], bounds[finite, 1]), 'finite V172 upper changed')
    require(np.all(result[finite, 0] >= 0.) and np.all(result[finite, 0] <= result[finite, 1]), 'physical cap violated')
    return result, finite


def parents(args):
    model = read(args.v172_root/'model.json')
    protocol = read(args.v182_root/'protocol.json')
    support = read(args.v182_root/'summary/status.json')
    reserved = read(args.v183_root/'registration.json')
    reservation.validate(protocol, support, model)
    require(reserved['source'] == '1ce0cfc596' and reserved['status'] == 'reserved_not_fitted_not_collected' and
        reserved['v182_protocol'] == protocol and reserved['confirmation_days'] == list(reservation.DAYS) and
        reserved['sampler_seeds'] == {str(d): reservation.SAMPLERS[d] for d in reservation.DAYS} and
        reserved['freshness']['decision'] == 'fresh' and not any(reserved['execution'].values()),
        'requires unchanged V183 reservation, not confirmation records')
    audit = read(args.v166_audit_root/'summary/status.json')
    require(audit['status'] == 'v166_readback_passed' and audit['passed_paths'] == 1536 and
        audit['failed_paths'] == 0 and audit['calibration_days'] == list(plan.CALIBRATION) and
        audit['query_metadata_complete'] and audit['query_support_and_label_clocks_checked'] and
        model['capture_source'] == '20d914af03' and Path(model['capture_root']) == args.v166_root,
        'requires original64 admitted captures and unchanged clocks')
    require(len(model['chunks']) == 24 and {(c['actor'], c['mode']) for c in model['chunks']} ==
        set(product(protocol['actors'], protocol['modes'])), 'actor/mode grid incomplete')
    return model, protocol, support, reserved


def load_queries(args, model, chunk, table, archive):
    actor, mode = chunk['actor'], chunk['mode']
    requests, native, baseline = (archive[k] for k in ('requests', 'native_supported', 'finite_candidate'))
    days, episodes = requests[:, 0].astype(int), requests[:, 1].astype(int)
    require(set(days) == set(plan.CALIBRATION), 'not the original64 calibration cohort')
    indices = {tuple(e['key']): i for i, e in enumerate(table['candidate'])}
    signs, moving = np.full(len(requests), -1, dtype=np.int8), np.zeros(len(requests), dtype=bool)
    state_ids = np.full((len(requests), len(states.PARTITIONS)), -1, dtype=np.int32)
    state_keys = {level: {} for level in states.PARTITIONS}
    for count, day in enumerate(plan.CALIBRATION, 1):
        root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
        receipt = read(root/'capture_cells'/(name+'.json'))
        require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
            ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'],
            'capture source/deployment differs')
        wanted = {tuple(requests[i, 5:].astype(int)): i for i in np.flatnonzero(days == day)}
        atoms, selected, records = {}, [], 0
        for q, r in zip_longest(*(previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
            require(q is not None and r is not None and q['query_key'] == r['query_key'] and
                q['query_time_s'] == r['query_time_s'] and q['episode'] == r['episode'] and
                (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/residual join differs')
            records += 1
            previous.add_first(atoms, q, r)
            if not q['denominator_eligible']:
                continue
            i = wanted.pop(tuple(q['query_key']))
            require(q['point_s'] == requests[i, 3] and q['query_time_s'] == requests[i, 4] and
                archive['episode_keys'][episodes[i]].tolist() == [day, *q['episode']] and
                q['native_supported'] == bool(native[i]), 'calibration coordinates differ')
            bits = previous.saved.fit.scope_bits(q)
            require(bits == int(requests[i, 2]) and q['observable']['hour'] == int(q['query_time_s']//3600),
                'scope clock differs')
            moving[i] = bool(bits & 16) and q['native_stage']['kind'] == 'moving-launch'
            period = temporal.STRATA[int(temporal.strata(np.array([q['query_time_s']]))[0])]
            key = (*q['candidate_key'], period) if q['native_supported'] else None
            ki = indices.get(key)
            entry = None if ki is None else table['candidate'][ki]
            require(bool(native[i] and entry is not None and entry['status'] == 'finite') == bool(baseline[i]),
                'original V172 key support differs')
            require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
            if bits & 4 or moving[i]:
                features = states.prior.features(q, entry)
                for col, (level, fields) in enumerate(states.PARTITIONS.items()):
                    identity = tuple(features[f] for f in fields)
                    state_ids[i, col] = state_keys[level].setdefault(identity, len(state_keys[level]))
            position = q['observable']['position']
            selected.append((q['query_time_s'], q['clocks']['query_ready_order'], i,
                None if position is None else position['remaining_stops']))
        require(not wanted and records == receipt['query_record_count'], 'calibration path incomplete')
        prefix = previous.Prefix(atoms)
        for now, order, i, remaining in sorted(selected):
            history = prefix.query(now, remaining)['global']
            if int(requests[i, 2]) & 4 or moving[i]:
                signs[i] = plan.prior.service_sign(history)
        print('V184_CALIBRATION_PREFIX', actor, mode, count, '/64', flush=True)
    return signs, moving, state_ids, state_keys


def observations(bounds, updated, truth, selected, target, shift):
    lo, hi = updated[selected].T
    old = bounds[selected, 0]
    y = truth[selected]
    score = hi-lo+10.*np.maximum(lo-y, 0.)+10.*np.maximum(y-hi, 0.)
    old_score = hi-old+10.*np.maximum(old-y, 0.)+10.*np.maximum(y-hi, 0.)
    delta = old-lo+10.*(np.maximum(lo-y, 0.)-np.maximum(old-y, 0.))
    require(np.allclose(score-old_score, delta, rtol=0., atol=1e-8), 'registered score identity differs')
    raw = old-shift[selected]
    return np.column_stack((y < lo, y > hi, (lo <= y) & (y <= hi), hi-lo, score, delta, lo-old,
        target[selected] & (raw < 0.), target[selected] & (raw > hi), lo == hi))


def cell_diagnostics(requests, masks, finite, values, target, state_ids, state_keys, actor, mode):
    days, episodes = requests[:, 0].astype(int), requests[:, 1].astype(int)
    rows, profiles = [], []
    for day, (scope, mask) in product(plan.CALIBRATION, masks.items()):
        selected = np.flatnonzero((days == day) & mask & finite)
        require(len(selected) > 0, 'missing finite calibration diagnostic cell')
        eps, inv, count = np.unique(episodes[selected], return_inverse=True, return_counts=True)
        weights = 1./(len(eps)*count[inv])
        rows.append(dict(actor=actor, day=day, mode=mode, scope=scope,
            values=(weights@values[selected]).tolist()))
        if scope not in plan.TARGET_SCOPES:
            continue
        require(np.all(target[selected]), 'diagnostic scope escaped target union')
        for col, (level, keys) in enumerate(state_keys.items()):
            for key, code in keys.items():
                use = state_ids[selected, col] == code
                if np.any(use):
                    profiles.append(dict(actor=actor, day=day, mode=mode, scope=scope, level=level, key=list(key),
                        mass=float(weights[use].sum()), values=(weights[use]@values[selected[use]]).tolist()))
    return rows, profiles


def bucket(args, model, support, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require((table['execution_source'], table['actor'], table['mode']) == ('a5b45df520', actor, mode), 'V172 map differs')
    require(all(e['day_count'] >= 19 for e in table['candidate'] if e['status'] == 'finite'),
        'original V172 key-day support weakened')
    with np.load(args.v172_root/chunk['atom_file']) as stored:
        archive = {k: stored[k] for k in ('requests', 'episode_keys', 'atom_request', 'residual_rsd',
            'native_supported', 'finite_candidate', 'bounds')}
    requests, baseline = archive['requests'], archive['finite_candidate']
    bounds = archive['bounds'][:, 0]
    require(np.array_equal(np.isfinite(bounds).all(axis=1), baseline), 'V172 finite mask differs')
    signs, moving, state_ids, state_keys = load_queries(args, model, chunk, table, archive)
    target = plan.prior.target_mask(requests[:, 2].astype(int), moving)
    ri = archive['atom_request']
    truth = np.full(len(requests), np.nan)
    truth[ri] = requests[ri, 3]-archive['residual_rsd'].sum(axis=1)
    require(np.isfinite(truth[baseline]).all(), 'finite baseline lacks same-query joint labels')
    entries, denominator = [], np.zeros(len(requests), dtype=np.int64)
    shift = np.zeros(len(requests))
    for code, sign in enumerate(plan.SIGNS):
        use = baseline & target & (signs == code)
        entry, den = shared_fit(bounds[use, 0]-truth[use], requests[use, 0].astype(int), requests[use, 1].astype(int))
        entry.update(sign=sign)
        expected = next(p for p in support['parameters'] if (p['actor'], p['mode'], p['sign']) == (actor, mode, sign))
        require(entry['days'] == expected['contributing_days'] and entry['day_count'] == expected['days'],
            'actual shared support differs from V182 metadata admission')
        denominator[use] = den
        if entry['status'] == 'finite':
            shift[use] = entry['delta_s']
        entries.append(entry)
    updated, finite = corrected(bounds, baseline, target, signs, entries)
    require(np.array_equal(finite, baseline), 'V182 support-mask equivalence not reproduced')
    bits = requests[:, 2].astype(int)
    masks = {s: (bits & (1 << i)) != 0 for i, s in enumerate(temporal.SCOPES[:-1])}
    masks['hour13_moving_launch'] = moving
    days, episodes = requests[:, 0].astype(int), requests[:, 1].astype(int)
    cells = [dict(actor=actor, mode=mode, **temporal.support_cell(d, s, mask, days, episodes,
        archive['native_supported'], finite, baseline)) for d, (s, mask) in product(plan.CALIBRATION, masks.items())]
    require(cells == [dict(c, baseline_fraction=c['finite_fraction']) for c in table['cells']],
        'original six-scope support cells differ')
    values = np.full((len(requests), len(METRICS)), np.nan)
    values[finite] = observations(bounds, updated, truth, finite, target, shift)
    diagnostics, profiles = cell_diagnostics(requests, masks, finite, values, target, state_ids, state_keys, actor, mode)
    reasons = Counter(native_abstention=int((~archive['native_supported']).sum()),
        v172_abstention=int((archive['native_supported'] & ~baseline).sum()),
        prefix_abstention=int((baseline & target & (signs < 0)).sum()),
        parameter_abstention=int((baseline & target & (signs >= 0) & ~finite).sum()))
    audit_file = f'audit/{actor}_{mode}.npz'
    (args.out_root/'audit').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_root/audit_file, lower=updated[:, 0], finite=finite,
        target=target, signs=signs, shared_mass_denominator=denominator)
    with np.load(args.out_root/audit_file) as saved:
        require(all(np.array_equal(saved[k], v, equal_nan=True) for k, v in dict(lower=updated[:, 0], finite=finite,
            target=target, signs=signs, shared_mass_denominator=denominator).items()), 'saved arrays differ')
        rebuilt, rebuilt_finite = corrected(bounds, baseline, saved['target'], saved['signs'], entries)
        require(np.array_equal(rebuilt[:, 0], saved['lower'], equal_nan=True) and
            np.array_equal(rebuilt_finite, saved['finite']), 'saved map application differs')
    require(read(args.v172_root/chunk['map_file']) == table, 'original V172 map changed')
    write(args.out_root/'details'/f'{actor}_{mode}_states.json', dict(metrics=list(METRICS),
        partitions={k: list(v) for k, v in states.PARTITIONS.items()}, rows=profiles,
        interpretation='in-sample contributions, equal physical-episode/query within each actor/day/scope; no state selection'))
    result = dict(actor=actor, mode=mode, source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], parameters=entries,
        cells=cells, calibration_diagnostics=diagnostics, audit_file=audit_file,
        parent_atom_file=chunk['atom_file'], eligible_queries=len(requests), target_finite_queries=int((target & finite).sum()),
        recomputed_weight_queries=int(np.count_nonzero(denominator)), reasons=dict(reasons),
        upper_unchanged=np.array_equal(updated[:, 1], bounds[:, 1], equal_nan=True), non_target_unchanged=True,
        finite_mask_unchanged=True, saved_arrays_readback_passed=True)
    write(args.out_root/'maps'/f'{actor}_{mode}.json', result)
    require(read(args.out_root/'maps'/f'{actor}_{mode}.json') == result, 'sealed parameter JSON readback differs')
    print('V184_SHARED_MAP_SEALED', actor, mode, [(e['sign'], e['day_count'], e['delta_s']) for e in entries], flush=True)
    return result


def seal(args):
    model, protocol, support, reserved = parents(args)
    require(not (args.out_root/'registration.json').exists(), 'use an unused output root; never overwrite a seal')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v184_*.py', '-v'], check=True)
    write(args.out_root/'registration.json', dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        frozen_protocol=protocol, confirmation_reservation=reserved,
        authorized='original64-day fitting/sealing only; no confirmation collection', calibration_days=list(plan.CALIBRATION)))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for job in as_completed([pool.submit(bucket, args, model, support, c) for c in model['chunks']]):
            rows.append(job.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=24, unit='actor_mode_map')), flush=True)
    rows.sort(key=lambda r: (r['actor'], r['mode']))
    parameters = [dict(actor=r['actor'], mode=r['mode'], **p) for r in rows for p in r['parameters']]
    require(len(parameters) == 48 and all(p['status'] == 'finite' and p['day_count'] >= 19 for p in parameters),
        'shared parameter admission failed; no regrouping or weaker support')
    cells = [c for r in rows for c in r['cells']]
    release = temporal.support_release(cells, model['actors'], list(plan.CALIBRATION))
    require(release == support['v172_baseline_support'] and release['decision'] == 'pass', 'support release differs')
    require(all(r['upper_unchanged'] and r['non_target_unchanged'] and r['saved_arrays_readback_passed'] for r in rows),
        'sealed geometry/readback admission failed')
    require(sum(r['eligible_queries'] for r in rows) == support['input_original_calibration_eligible_queries'],
        'original query denominator changed')
    require(parents(args) == (model, protocol, support, reserved), 'frozen inputs changed')
    reasons = {k: sum(r['reasons'][k] for r in rows) for k in rows[0]['reasons']}
    require(reasons['native_abstention'] == support['native_abstentions'] and
        reasons['v172_abstention'] == support['v172_abstentions'], 'original abstention counts changed')
    diagnostics = [c for r in rows for c in r['calibration_diagnostics']]
    averages = [dict(mode=m, scope=s, values=dict(zip(METRICS, np.mean([
        c['values'] for c in diagnostics if (c['mode'], c['scope']) == (m, s)], axis=0).tolist())))
        for m, s in product(protocol['modes'], temporal.SCOPES)]
    sealed = dict(status='sealed', stage='v184_v182_shared_lower_bound_error',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_protocol=protocol,
        calibration_days=list(plan.CALIBRATION), actors=model['actors'], modes=model['modes'],
        base_model_root=str(args.v172_root), base_model_source='a5b45df520',
        capture_root=str(args.v166_root), capture_source='20d914af03',
        protocol_root=str(args.v182_root), protocol_source='a57afaa337',
        confirmation_reservation_root=str(args.v183_root), confirmation_reservation=reserved,
        chunks=[dict(actor=r['actor'], mode=r['mode'], map_file=f"maps/{r['actor']}_{r['mode']}.json",
            audit_file=r['audit_file'], parent_atom_file=r['parent_atom_file']) for r in rows],
        support_admission=release, confirmation_release_eligible=True, confirmation_collection_authorized=False,
        confirmation_started=False, confirmation_outcomes_read=False, main_policy_changed=False)
    write(args.out_root/'model.json', sealed)
    require(read(args.out_root/'model.json') == sealed, 'sealed model JSON readback differs')
    write(args.out_root/'summary/status.json', dict(status='v184_shared_lower_sealed_calibration_only',
        source=sealed['source'], protocol_source='a57afaa337', reservation_source='1ce0cfc596',
        calibration_days=list(plan.CALIBRATION), calibration_paths=1536, actor_mode_maps=24,
        parameters=parameters, support_admission=release, reasons=reasons,
        eligible_queries=sum(r['eligible_queries'] for r in rows),
        target_finite_queries=sum(r['target_finite_queries'] for r in rows),
        recomputed_weight_queries=sum(r['recomputed_weight_queries'] for r in rows),
        invariants=dict(upper_unchanged=True, non_target_unchanged=True, finite_mask_unchanged=True,
            original_key_19_days_preserved=True, shared_19_days_preserved=True, saved_arrays_readback_passed=True,
            v182_protocol_unchanged=True, v183_reservation_unchanged=True, frozen_parents_unchanged=True),
        calibration_diagnostics=averages, interval_fit=True, fitted_parameter_values=48,
        confirmation_collection_authorized=False, confirmation_started=False, confirmation_outcomes_read=False,
        collector_submitted=False, new_days_collected=0, new_rollouts=0, new_training=0, main_policy_changed=False,
        evidence='original64-day in-sample fit/support/geometry only, not independent coverage or controller superiority',
        next='Mapping sealed. Keep V18332 confirmation days unstarted pending separate authorization.'))
    print('V184_SEALED_CALIBRATION_ONLY_CONFIRMATION_NOT_STARTED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v172-root', 'v182-root', 'v183-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    seal(parser.parse_args())
