#!/usr/bin/env python3
"""Construct the frozen V175 upper envelope from V172 calibration atoms only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from math import lcm
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import preregister_freqduet_v175_dayblock_tails as plan


fit = plan.fit
read, write, require = plan.read, plan.write, plan.require
DRAWS, SEED, ORDER = 5000, 20260803, 250


@lru_cache(maxsize=46)
def day_counts(day_count):
    draws = np.random.RandomState(SEED).randint(0, day_count, size=(DRAWS, day_count))
    return np.bincount((draws+np.arange(DRAWS)[:, None]*day_count).ravel(),
        minlength=DRAWS*day_count).reshape(DRAWS, day_count)


def mass_at_least(counts, column, scale, rank):
    if column.dtype != object:
        return counts @ column >= rank*scale
    # Integer limbs retain exact CDF ties even when the common denominator exceeds int64.
    base = 1 << 32
    limbs = (int(scale*counts.shape[1]).bit_length()+31)//32
    digits = np.array([[(int(v) >> (32*j)) & (base-1) for j in range(limbs)] for v in column], dtype=np.int64)
    mass = counts @ digits
    carry = np.zeros(len(counts), dtype=np.int64)
    for j in range(limbs):
        value = mass[:, j]+carry
        mass[:, j], carry = value % base, value//base
    require(not carry.any(), 'integer CDF limb capacity differs')
    above, equal = np.zeros(len(counts), dtype=bool), np.ones(len(counts), dtype=bool)
    target = rank*scale
    for j in reversed(range(limbs)):
        digit = (target >> (32*j)) & (base-1)
        above |= equal & (mass[:, j] > digit)
        equal &= mass[:, j] == digit
    return above | equal


def block_order_statistic(total, day_index, denominator, counts, rank, order):
    """Invert the order of bootstrap inverses without materializing 5000 query tapes."""
    values, inverse = np.unique(total, return_inverse=True)
    d = counts.shape[1]
    require(np.all(counts.sum(axis=1) == d) and 1 <= order <= len(counts), 'day-slot bootstrap differs')
    scale = lcm(*(int(v) for v in np.unique(denominator)))
    dtype = np.int64 if scale*d <= np.iinfo(np.int64).max else object
    cdf = np.zeros((d, len(values)), dtype=dtype)
    weights = np.array([scale//int(v) for v in denominator], dtype=dtype)
    np.add.at(cdf, (day_index, inverse), weights)
    cdf = np.cumsum(cdf, axis=1)
    require(all(int(v) == scale for v in cdf[:, -1]), 'within-day episode/query mass differs from one')
    original_mass = cdf.sum(axis=0)
    original_index = next(i for i, v in enumerate(original_mass) if int(v) >= rank*scale)
    lo, hi = 0, len(values)-1
    while lo < hi:
        mid = (lo+hi)//2
        # Qlow_b <= x iff bootstrap CDF_b(x) >= k/D (equivalently day mass >= k).
        n = np.count_nonzero(mass_at_least(counts, cdf[:, mid], scale, rank))
        if n >= order:
            hi = mid
        else:
            lo = mid+1
    below = 0 if lo == 0 else int(np.count_nonzero(mass_at_least(counts, cdf[:, lo-1], scale, rank)))
    at = int(np.count_nonzero(mass_at_least(counts, cdf[:, lo], scale, rank)))
    require(below < order <= at, 'bootstrap order statistic not bracketed')
    return float(values[lo]), dict(original_qlow_s=float(values[original_index]),
        draws_strictly_below=below, draws_at_or_below=at, unique_total_errors=len(values),
        exact_integer_cdf=True, common_denominator_bits=scale.bit_length(), limb_arithmetic=dtype is object)


def envelope(entry, vectors, days, episodes, saved_denominator):
    contributing, denominator = fit.hierarchy(days, episodes)
    d = len(contributing)
    require(d == entry['day_count'] and contributing.tolist() == entry['days'] and d >= 19 and
        entry['status'] == 'finite' and np.array_equal(d*denominator, saved_denominator),
        'original independent-day support or equal-episode/query weights differ')
    rank = fit.protocol.rank_levels(d)['lower_rank']
    qlow, audit = block_order_statistic(vectors.sum(axis=1), np.searchsorted(contributing, days),
        denominator, day_counts(d), rank, ORDER)
    require(audit['original_qlow_s'] == entry['qlow_s'], 'exact CDF does not reproduce V172 Qlow')
    result = deepcopy(entry)
    result['qlow_s'] = min(entry['qlow_s'], qlow)
    return result, dict(day_count=d, atoms=len(vectors), bootstrap_qlow_order250_s=qlow,
        sealed_qlow_s=result['qlow_s'], **audit)


def invariants(original, updated, finite, target):
    require(np.array_equal(np.isfinite(updated).all(axis=1), finite), 'finite/abstention mask changed')
    require(np.array_equal(original[:, 0], updated[:, 0], equal_nan=True), 'lower bound changed')
    require(np.array_equal(original[~target], updated[~target], equal_nan=True), 'non-target interval changed')
    delta = updated[finite, 1]-original[finite, 1]
    require((delta >= 0.).all(), 'upper bound contracted')
    require((updated[finite, 0] >= 0.).all() and
        (updated[finite, 0] <= updated[finite, 1]).all(), 'invalid physical interval')
    return dict(finite_mask_unchanged=True, lower_bound_unchanged=True, non_target_unchanged=True,
        upper_never_contracts=True, finite_queries=int(finite.sum()),
        target_finite_queries=int((target & finite).sum()), expanded_queries=int(np.count_nonzero(delta)),
        upper_expansion_query_sum_s=float(delta.sum()), maximum_upper_expansion_s=float(delta.max(initial=0.)))


def parents(args):
    model = read(args.v172_root/'model.json')
    protocol = read(args.v175_root/'protocol.json')
    receipt = read(args.v175_root/'summary/status.json')
    require(receipt['status'] == 'v175_protocol_and_calibration_audit_complete' and
        receipt['execution_source'] == protocol['execution_source'] == '5f8cdf4296' and
        receipt['protocol'] == protocol and not receipt['interval_fit'] and
        not receipt['collector_submitted'] and not receipt['confirmation_outcomes_read'] and
        receipt['support'] == model['support_admission'], 'requires completed frozen V175 support/protocol')
    expected = plan.registration(model, protocol['freshness'])
    expected['execution_source'] = protocol['execution_source']
    require(expected == protocol and model['calibration_days'] == list(plan.CALIBRATION) and
        protocol['reservation']['confirmation_days'] == list(plan.CONFIRMATION), 'frozen V175 design differs')
    return model, protocol


def build_bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    parent = read(args.v172_root/chunk['map_file'])
    require((parent['execution_source'], parent['actor'], parent['mode']) == ('a5b45df520', actor, mode), 'map source differs')
    candidate = deepcopy(parent['candidate'])
    fields = model['frozen_protocol']['conditioning']['candidate_key']
    target_keys = np.array([bool(plan.key_scopes(e['key'], fields)) for e in candidate])
    diagnostics = []
    with np.load(args.v172_root/chunk['atom_file']) as z:
        q, ri, keys, vectors = z['requests'], z['atom_request'], z['candidate_key'], z['residual_rsd']
        mass, original, finite = z['mass_denominator'][:, 0], z['bounds'][:, 0], z['finite_candidate']
        require(set(q[:, 0].astype(int)) == set(plan.CALIBRATION) and np.isfinite(vectors).all(),
            'requires exactly the existing 64 calibration days')
        updated, target = original.copy(), np.zeros(len(q), dtype=bool)
        target[ri] = target_keys[keys]
        stage = fields.index('native_stage_kind')
        moving = np.zeros(len(q), dtype=bool)
        moving[ri] = np.array([e['key'][stage] == 'moving-launch' for e in candidate])[keys]
        query_target = ((q[:, 2].astype(int) & 4) != 0) | ((q[:, 4]//3600 == 13) & moving)
        require(np.array_equal(target[finite], query_target[finite]), 'key/query target membership differs')
        wanted = [i for i, e in enumerate(candidate) if target_keys[i] and e['status'] == 'finite']
        for done, i in enumerate(wanted, 1):
            selected = np.flatnonzero(keys == i)
            requests = ri[selected]
            candidate[i], audit = envelope(candidate[i], vectors[selected], q[requests, 0].astype(int),
                q[requests, 1].astype(int), mass[selected])
            audit.update(key_index=i, key=candidate[i]['key'])
            diagnostics.append(audit)
            if candidate[i]['qlow_s'] != parent['candidate'][i]['qlow_s']:
                require(finite[requests].all(), 'low-support interval gained a finite bound')
                updated[requests, 1] = np.maximum(0., q[requests, 3]-candidate[i]['qlow_s'])
            if done % 20 == 0:
                print('V176_KEY_PROGRESS', actor, mode, done, len(wanted), flush=True)
        for i, (old, new) in enumerate(zip(parent['candidate'], candidate)):
            expected = dict(old, qlow_s=new['qlow_s']) if i in wanted else old
            require(new == expected and new['qhigh_s'] == old['qhigh_s'] and
                (i not in wanted or new['qlow_s'] <= old['qlow_s']), 'unexpected map/abstention change')
        checks = invariants(original, updated, finite, target)
        stem = f'{actor}_{mode}'
        sidecar = 'audit/'+stem+'.npz'
        (args.out_root/'audit').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_root/sidecar, bounds=updated, finite_candidate=finite, target=target)
        with np.load(args.out_root/sidecar) as stored:
            require(np.array_equal(stored['bounds'], updated, equal_nan=True) and
                np.array_equal(stored['target'], target), 'saved bound readback differs')
            invariants(original, stored['bounds'], stored['finite_candidate'], stored['target'])
    result = dict(actor=actor, mode=mode, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        candidate=candidate, reference=parent['reference'], cells=parent['cells'], checks=checks,
        parent_atom_file=chunk['atom_file'], audit_file=sidecar, target_finite_keys=len(wanted),
        widened_keys=sum(candidate[i]['qlow_s'] < parent['candidate'][i]['qlow_s'] for i in wanted),
        candidate_keys=len(candidate), abstaining_keys=sum(e['status'] != 'finite' for e in candidate),
        limb_keys=sum(e['limb_arithmetic'] for e in diagnostics), original_qlow_reproduced=True)
    write(args.out_root/'details'/('quantiles_'+stem+'.json'), diagnostics)
    write(args.out_root/'maps'/(stem+'.json'), result)
    print('V176_BUCKET_SEALED', actor, mode, flush=True)
    return result


def aggregate(args):
    model, protocol = parents(args)
    registration = read(args.out_root/'registration.json')
    require(registration['base_model'] == model and registration['frozen_protocol'] == protocol and
        registration['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'sealing source/parents differ')
    require(not (args.out_root/'model.json').exists(), 'do not overwrite a sealed envelope')
    names = {f"{c['actor']}_{c['mode']}.json" for c in model['chunks']}
    require({p.name for p in (args.out_root/'maps').glob('*.json')} == names, 'map matrix incomplete')
    rows = [read(args.out_root/'maps'/name) for name in names]
    rows.sort(key=lambda r: (r['actor'], r['mode']))
    for r in rows:
        require(r['execution_source'] == registration['execution_source'] and r['original_qlow_reproduced'] and
            all(r['checks'][k] for k in ('finite_mask_unchanged', 'lower_bound_unchanged',
                'non_target_unchanged', 'upper_never_contracts')), 'bucket invariants/source differ')
    release = plan.prior.support_release([c for r in rows for c in r['cells']], model['actors'], plan.CALIBRATION)
    require(release == model['support_admission'], 'frozen support not reproduced')
    totals = {m: sum(r[m] for r in rows) for m in
        ('candidate_keys', 'target_finite_keys', 'widened_keys', 'abstaining_keys', 'limb_keys')}
    totals.update({m: sum(r['checks'][m] for r in rows) for m in
        ('finite_queries', 'target_finite_queries', 'expanded_queries', 'upper_expansion_query_sum_s')})
    totals['maximum_upper_expansion_s'] = max(r['checks']['maximum_upper_expansion_s'] for r in rows)
    sealed = dict(status='sealed', stage='v176_v175_dayblock_upper_envelope',
        execution_source=registration['execution_source'], frozen_protocol=protocol,
        protocol_root=str(args.v175_root), protocol_source='5f8cdf4296',
        base_model_root=str(args.v172_root), base_model_source='a5b45df520',
        calibration_days=list(plan.CALIBRATION), actors=model['actors'], modes=model['modes'],
        capture_root=model['capture_root'], capture_source=model['capture_source'],
        frozen_v167_root=model['frozen_v167_root'], frozen_v167_source='e6fc851ec1',
        chunks=[dict(actor=r['actor'], mode=r['mode'], map_file=f"maps/{r['actor']}_{r['mode']}.json",
            audit_file=r['audit_file'], parent_atom_file=r['parent_atom_file']) for r in rows],
        support_admission=release, totals=totals, unseen_keys=model['unseen_keys'],
        confirmation_release_eligible=release['decision'] == 'pass', confirmation_started=False,
        confirmation_outcomes_read=False, collector_submitted=False, main_policy_changed=False)
    require(read(args.v172_root/'model.json') == registration['base_model'] and
        read(args.v175_root/'protocol.json') == protocol, 'frozen headers changed during construction')
    write(args.out_root/'model.json', sealed)
    require(read(args.out_root/'model.json') == sealed, 'sealed model readback differs')
    write(args.out_root/'summary/status.json', dict(status='v176_upper_envelope_sealed_calibration_only',
        execution_source=sealed['execution_source'], protocol_source='5f8cdf4296', maps=len(rows),
        totals=totals, support_admission=release,
        invariants=dict(upper_never_contracts=True, lower_bound_unchanged=True, non_target_unchanged=True,
            finite_mask_unchanged=True, low_support_abstention_unchanged=True, original_qlow_reproduced=True),
        bootstrap_draws=DRAWS, bootstrap_seed=SEED, envelope_order=ORDER, exact_integer_cdf=True,
        calibration_days=list(plan.CALIBRATION), reserved_confirmation_days=list(plan.CONFIRMATION),
        interval_fit=True, frozen_map_updates=0, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], new_days_collected=0, new_rollouts=0, new_training=0,
        confirmation_started=False, confirmation_outcomes_read=False, collector_submitted=False,
        main_policy_changed=False, next='Sealed only; keep confirmation unopened until separate authorization.'))
    print('V176_ENVELOPE_SEALED_NO_CONFIRMATION', flush=True)


def build(args):
    require(not (args.out_root/'registration.json').exists(), 'use a new unprepared output root')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v176_*.py', '-v'], check=True)
    model, protocol = parents(args)
    write(args.out_root/'registration.json', dict(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        base_model=model, frozen_protocol=protocol, bootstrap_draws=DRAWS, bootstrap_seed=SEED,
        envelope_order=ORDER, confirmation_started=False, main_policy_changed=False))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(build_bucket, args, model, chunk) for chunk in model['chunks']]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS', i, 24, 'actor_mode_envelope', flush=True)
    aggregate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'aggregate'))
    for name in ('out-root', 'v175-root', 'v172-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'build': build, 'aggregate': aggregate}[args.command](args)
