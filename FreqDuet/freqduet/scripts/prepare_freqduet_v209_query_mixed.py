#!/usr/bin/env python3
"""Supplement V208's two fixed-beta mixed sums from original64 archives only."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_freqduet_v199_conditional_moments as previous
import preregister_freqduet_v208_query_covariance as plan


saved, prior = previous.saved, previous.prior
read, write, require = previous.read, previous.write, previous.require
DAYS, BINS, KEYS = tuple(previous.DAYS), previous.BINS, plan.plan.KEYS
SCOPES, MODES = plan.plan.SCOPES, plan.plan.MODES
STATE_IX = [saved.VARS.index(s) for s in plan.plan.STATES]
SIGNAL_IX = [saved.VARS.index(s) for s in plan.SIGNALS]
TARGET_IX = [saved.VARS.index(s) for s in ('signed_remaining_s', 'first_running_under_s')]
CLOCKS = ('native_completion_lead_s', 'matched_completion_lead_s', 'service_availability_lead_s',
    'first_label_lead_s', 'full_label_lead_s')


def mixed_sums(vector, beta):
    v = np.asarray(vector)
    residual = v[TARGET_IX]-beta@v[STATE_IX]
    result = v[SIGNAL_IX]*residual[0]*residual[1]
    require(np.isfinite(result).all(), 'nonfinite aligned mixed sum')
    return result


def query_clocks(q, r, item, service, timing):
    now, native, matched = q['query_time_s'], item[3], item[2]
    require(matched['n'] >= 1 and service['n'] >= 9, 'completed-prefix support changed')
    leads = np.array([now-native['latest'], now-matched['latest'], now-service['latest'],
        timing[2], r['label']['available_time_s']-now])
    require(np.isfinite(leads).all() and np.all(leads[:3] > 0.) and np.all(leads[3:] >= 0.),
        'query feature/label clock violation')
    return leads


def same_denominators(base, reference):
    expected = {tuple(r['episode']): (r['eligible_queries'], r['finite_interval_queries']) for r in reference}
    require({ep: tuple(v) for ep, v in base.items()} == expected, 'original episode/query denominators differ')


def verify_cells(got, facts, expected, expected_facts):
    require(got.shape == expected.shape and facts.shape == expected_facts.shape, 'V199 per-bin schema differs')
    errors = [float(np.max(np.abs(got[..., sl]-expected[..., sl]), initial=0.))
        for sl in (slice(0, 1), slice(1, 1+saved.N), slice(1+saved.N, None))]
    errors.append(float(np.max(np.abs(facts-expected_facts), initial=0.)))
    require(max(errors) <= 1e-8, 'V199 per-bin mass/first/second/facts differ')
    return np.array(errors)


def verify_support(counts, expected_counts, keep):
    actual = np.stack([(c > 0).sum(axis=0) for c in (counts[:32], counts[32:])])
    require(np.array_equal(actual, expected_counts) and np.array_equal(keep, np.all(actual >= 19, axis=0)),
        'V199 support changed;do not replace mask')


def scan(args, actor, day, mode, table, geometry, parent, admitted, beta):
    require(day in DAYS and (parent['actor'], parent['day'], parent['mode']) == (actor, day, mode)
        and parent['v190_moments_reproduced'], 'only original64 admitted paths')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
        ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'], 'capture differs')
    path, rows, entries, history = saved.load_timeline(root, name, geometry)
    prefix = prior.RiskPrefix(rows, history)
    mapping = {tuple(e['key']): e for e in table['candidate']}
    atoms, selected, query_count = {}, [], 0
    streams = (prior.running.previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))
    for q, r in zip_longest(*streams):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and q['episode'] == r['episode']
            and q['query_time_s'] == r['query_time_s'] and (q['actor'], q['day'], q['mode']) == (actor, day, mode),
            'same physical query/label join differs')
        query_count += 1
        prior.running.previous.add_first(atoms, q, r)
        if not q['denominator_eligible']:
            continue
        scopes = [s for s in saved.SCOPES if (s == 'clearance' and q['condition_values']['query_phase'] == 'clearance')
            or (s == 'hour13' and q['observable']['hour'] == 13) or (s == 'hour13_moving_launch'
            and q['observable']['hour'] == 13 and q['native_stage']['kind'] == 'moving-launch')]
        if not scopes:
            continue
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
        now = q['query_time_s']
        period = prior.running.temporal.STRATA[int(prior.running.temporal.strata(np.array([now]))[0])]
        entry = mapping.get((*q['candidate_key'], period)) if q['native_supported'] else None
        bounds = prior.running.previous.saved.fit.interval(q['point_s'], entry, q['native_supported'])
        require(bounds is None or entry['day_count'] >= 19, 'frozen interval support weakened')
        selected.append((q, r, scopes, bounds))
    require(query_count == receipt['query_record_count'], 'incomplete archive query stream')
    service = prior.running.previous.Prefix(atoms)
    base = {s: {} for s in saved.SCOPES}
    moments, facts, mixed = ({s: defaultdict(dict) for s in saved.SCOPES} for _ in range(3))
    counts = {s: np.zeros((2, 9), dtype=np.int64) for s in SCOPES}
    checked, minima = 0, np.full(len(CLOCKS), np.inf)
    for q, r, scopes, bounds in sorted(selected, key=lambda v: (v[0]['query_time_s'], v[0]['clocks']['query_ready_order'])):
        ep, now = tuple(q['episode']), q['query_time_s']
        h = service.query(now, None)['global']
        item = None if bounds is None else prior.leg(q, geometry, entries, history, prefix)
        vector = None if item is None or item[2] is None or h['mean'] is None else prior.features(q, r, bounds, item, h)
        included = [] if vector is None else [s for s in scopes if item[0] in admitted[s]]
        if included:
            i, a, j = saved.innovation_parts(prefix, item[0], now, item[3]['mean'], item[4])
            saved.close(j, vector[0])
            components, timing = saved.component_labels(q, r, item, path, geometry, entries)
            values = [*vector, i, a, *components]
            extended = saved.moment(values)
            require(np.allclose(saved.old_moment(extended), prior.moment(vector), rtol=0., atol=1e-8), 'query moments changed')
            bi = BINS.index(previous.plan.stratum(vector[previous.FRACTION], vector[previous.HORIZON]))
            minima = np.minimum(minima, query_clocks(q, r, item, h, timing))
            checked += 1
        for scope in scopes:
            base[scope].setdefault(ep, np.zeros(2))[:] += (1., float(bounds is not None))
            if scope not in included:
                continue
            key = item[0]
            moments[scope][ep].setdefault(key, np.zeros((9, saved.SIZE)))[bi] += extended
            facts[scope][ep].setdefault(key, np.zeros((9, 1+len(saved.FACTS))))[bi] += np.r_[1., timing]
            if scope in SCOPES and key in KEYS:
                mixed[scope][ep].setdefault(key, np.zeros((9, 2)))[bi] += mixed_sums(values, beta[scope])
                counts[scope][KEYS.index(key), bi] += 1
    require(checked == parent['clock_queries'], 'V194 per-path complete-input count differs')
    result = []
    for scope in saved.SCOPES:
        expected = next(s for s in parent['scopes'] if s['scope'] == scope)
        require(len(base[scope]) == expected['eligible_episodes'], 'original eligible-episode count differs')
        values, meta = saved.collapse(moments[scope], base[scope]), saved.collapse(facts[scope], base[scope])
        original = {tuple(g['key']): g for g in expected['groups']}
        require(set(values) == set(meta) == set(original) <= admitted[scope], 'V194 parent query cohort differs')
        for key in original:
            previous.verify_partition(values[key][None], meta[key][None],
                np.array([original[key]['moments']]), np.array([original[key]['facts']]))
        result.append(dict(scope=scope, groups=values, facts=meta, base=base[scope],
            mixed=saved.collapse(mixed[scope], base[scope])))
    return dict(day=day, archive_queries=query_count, clock_queries=checked, scopes=result, counts=counts,
        min_clock_leads=[float(x) if np.isfinite(x) else None for x in minima])


def bucket(args, chunk, days=DAYS, persist=True):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source'] == 'a5b45df520', 'V172 source differs')
    geometry = prior.clocks.native.load_geometry(args.runtime)
    parent = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
    require(parent['source'] == '51a9409ec2', 'V194 source differs')
    lookup = {r['day']: r for r in parent['rows']}
    metadata = read(args.v199_root/'metadata'/f'{actor}_{mode}.json')
    require(metadata['source'] == '71af51ae4d' and metadata['paths'] == 64, 'V199 metadata differs')
    refs, admitted, beta = {}, {}, {}
    for scope in saved.SCOPES:
        with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
            require(z['days'].tolist() == list(DAYS) and z['bins'].tolist() == [list(b) for b in BINS], 'V199 cohort/grid differs')
            refs[scope] = {name: z[name] for name in ('keys', 'moments', 'facts', 'parent_balance_factors', 'distinct_days', 'supported_both')}
        admitted[scope] = {tuple(k) for k in refs[scope]['keys'].tolist()}
        if scope in SCOPES:
            with np.load(args.v205_root/'details'/f'{mode}_{scope}.npz') as z:
                require(z['states'].tolist() == list(plan.plan.STATES), 'V205 state order differs')
                beta[scope] = z['beta_point']
            with np.load(args.v207_root/'details'/f'{mode}_{scope}.npz') as z:
                require(beta[scope].shape == (2, 9) and np.array_equal(beta[scope], z['point_operator']), 'fixed beta differs')
    mixed = {s: np.zeros((len(days), 2, 9, 2)) for s in SCOPES}
    counts = {s: np.zeros((len(days), 2, 9), dtype=np.int64) for s in SCOPES}
    rows, error = [], np.zeros(4)
    for di, day in enumerate(days):
        row = scan(args, actor, day, mode, table, geometry, lookup[day], admitted, beta)
        original_di = DAYS.index(day)
        for cell in row['scopes']:
            scope, ref = cell['scope'], refs[cell['scope']]
            expected = next(r['episodes'] for r in metadata['episode_denominators'] if r['day'] == day and r['scope'] == scope)
            same_denominators(cell['base'], expected)
            for ki, key_array in enumerate(ref['keys']):
                key = tuple(key_array)
                got = cell['groups'].get(key, np.zeros((9, saved.SIZE)))
                facts = cell['facts'].get(key, np.zeros((9, 1+len(saved.FACTS))))
                error = np.maximum(error, verify_cells(got, facts, ref['moments'][original_di, ki], ref['facts'][original_di, ki]))
            if scope in SCOPES:
                for ki, key in enumerate(KEYS):
                    mixed[scope][di, ki] = cell['mixed'].get(key, np.zeros((9, 2)))
                counts[scope][di] = row['counts'][scope]
                require(np.isfinite(mixed[scope][di]).all(), 'mixed sums nonfinite')
        rows.append({k: row[k] for k in ('day', 'archive_queries', 'clock_queries', 'min_clock_leads')})
        if (di+1) % 8 == 0:
            print('V209_READONLY_DAYS', actor, mode, di+1, '/', len(days), flush=True)
    support = []
    if persist:
        require(tuple(days) == DAYS, 'partial paths cannot pass full support admission')
        for scope in SCOPES:
            ref = refs[scope]
            ix = [ref['keys'].tolist().index(list(k)) for k in KEYS]
            fixed_counts, keep = ref['distinct_days'][:, ix], ref['supported_both'][ix]
            verify_support(counts[scope], fixed_counts, keep)
            target = args.out_root/'cells'/f'{actor}_{mode}_{scope}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, days=DAYS, keys=KEYS, bins=BINS, signals=plan.SIGNALS,
                mixed_fields=plan.MIXED_FIELDS, raw_mixed_sums=mixed[scope], complete_queries=counts[scope],
                beta=beta[scope], distinct_days=fixed_counts, supported_both=keep,
                parent_balance_factors=ref['parent_balance_factors'][:, ix])
            support.append(dict(scope=scope, supported_cells=int(keep.sum()), queries=int(counts[scope].sum())))
    result = dict(actor=actor, mode=mode, paths=len(days), archive_queries=sum(r['archive_queries'] for r in rows),
        clock_queries=sum(r['clock_queries'] for r in rows), mixed_scope_queries=sum(int(c.sum()) for c in counts.values()),
        per_bin_max_abs_error=error.tolist(), support=support, rows=rows,
        pre_entry_baselines_checked=True, clock_violations=0, original_denominators_reproduced=True)
    if persist:
        write(args.out_root/'metadata'/f'{actor}_{mode}.json', result)
    return result


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v209_*.py', '-v'], check=True)
    parent = read(args.v208_root/'summary/status.json')
    require(parent['source'] == '5e9b7c7d63' and parent['missing_aligned_fields'] == list(plan.MIXED_FIELDS)
        and parent['supported_actor_bin_cells'] == 315 and parent['masks_and_weights_unchanged'], 'requires frozen V208')
    model = read(args.v172_root/'model.json')
    require(model['execution_source'] == 'a5b45df520' and model['calibration_days'] == list(DAYS)
        and len(model['actors']) == 12 and len(model['chunks']) == 24 and
        {(c['actor'], c['mode']) for c in model['chunks']} == set(product(model['actors'], MODES)), 'actor/day/mode cohort differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preparation')
    protocol = dict(stage='v209_original64_two_mixed_sum_supplement', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), v208_protocol=parent['protocol'],
        execution='Read-only existing1536 paths;fixed four-path preflight before24 actor/mode workers. '
            'Reproduce V194 counts/clocks/all9 parent moments and V199 all3-scope per-bin moments/facts/denominators. '
            'Persist ONLY two fixed-beta mixed sums for two frozen keys/two primary scopes,unweighted counts and original support/factors.',
        storage='Original episode/query normalized RAW sums;V191 factors stored separately,not applied. '
            'Retain all9 bins including unsupported;no new masks or selected-mass normalization.',
        associations_estimated=False, interval_fit=False, predictor_fit=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)
    write(args.out_root/'protocol.json', protocol)
    print('V209_PROTOCOL_SEALED_NO_ASSOCIATION_OR_FIT', flush=True)
    preflight = [bucket(args, c, days=(DAYS[0], DAYS[-1]), persist=False)
        for c in model['chunks'] if c['actor'] == min(model['actors'])]
    require(sum(r['paths'] for r in preflight) == 4 and sum(r['mixed_scope_queries'] for r in preflight) > 0,
        'fixed preflight lacks aligned query evidence')
    write(args.out_root/'preflight/status.json', dict(status='v209_four_path_preflight_passed', paths=4, results=preflight))
    print('V209_FOUR_PATH_PREFLIGHT_PASSED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, c) for c in model['chunks']]
        for i, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}' % i, flush=True)
    original = read(args.v199_root/'summary/status.json')
    require(sum(r['paths'] for r in results) == 1536 and sum(r['clock_queries'] for r in results) == original['clock_queries']
        and sum(r['archive_queries'] for r in results) == original['archive_queries'], 'full V199 archive/query cohort differs')
    retained = sum(s['supported_cells'] for r in results for s in r['support'])
    require(retained == parent['supported_actor_bin_cells'], 'fixed support total differs')
    write(args.out_root/'summary/status.json', dict(status='v209_two_query_mixed_sums_admitted', source=protocol['source'],
        protocol=protocol, paths=1536, preflight_paths=4, actors=12, days=64, mixed_files=48,
        clock_queries=sum(r['clock_queries'] for r in results), archive_queries=original['archive_queries'],
        mixed_scope_queries=sum(r['mixed_scope_queries'] for r in results),
        per_bin_max_abs_error=np.max([r['per_bin_max_abs_error'] for r in results], axis=0).tolist(),
        supported_actor_bin_cells=retained, candidate_actor_bin_cells=864, masks_and_weights_unchanged=True,
        source_and_label_clocks_checked=True, clock_violations=0, original_denominators_reproduced=True,
        results=[{k: v for k, v in r.items() if k != 'rows'} for r in sorted(results, key=lambda r: (r['actor'], r['mode']))],
        raw_archives_modified=False, associations_estimated=False, interval_fit=False, predictor_fit=False,
        confirmation_outcomes_read=False, confirmation_started=False, new_replays=0, new_training=0,
        map_updates=0, main_policy_changed=False))
    print('V209_MIXED_SUMS_ADMITTED_NO_ASSOCIATION_OR_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'runtime', 'v166-root', 'v172-root', 'v194-root', 'v199-root', 'v205-root', 'v207-root', 'v208-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    run(p.parse_args())
