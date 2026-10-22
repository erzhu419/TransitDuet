#!/usr/bin/env python3
"""Partition original64 archives under V198, admit parent identity and support only."""

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

import preregister_freqduet_v198_conditional_moments as plan


saved, common, prior = plan.saved, plan.common, plan.saved.prior
read, write, require = plan.read, plan.write, plan.require
DAYS, SCOPES = plan.DAYS, saved.SCOPES
BINS = list(product(range(3), repeat=2))
FRACTION, HORIZON = (saved.VARS.index(n) for n in ('first_fraction', 'horizon_min'))


def verify_partition(moments, facts, parent, parent_facts):
    require(moments.shape == (len(parent), 9, saved.SIZE)
        and facts.shape == (len(parent), 9, 1+len(saved.FACTS)), 'conditional schema differs')
    require(np.isfinite(moments).all() and np.isfinite(facts).all()
        and np.all(moments[..., 0] >= 0.), 'invalid conditional moments')
    actual, clocks = moments.sum(axis=1), facts.sum(axis=1)
    require(np.allclose(actual, parent, rtol=0., atol=1e-8), 'all9 cells do not reproduce V194 parent moments')
    require(np.allclose(clocks, parent_facts, rtol=0., atol=1e-8), 'all9 cells do not reproduce V194 clock facts')
    require(np.allclose(moments[..., 0], facts[..., 0], rtol=0., atol=1e-12), 'moment/clock admission mass differs')
    error = np.abs(actual-parent)
    return [float(error[:, sl].max(initial=0.)) for sl in (slice(0, 1), slice(1, 1+saved.N), slice(1+saved.N, None))] + [
        float(np.abs(clocks-parent_facts).max(initial=0.))]


def support_summary(mass, parent):
    """Use parent-key balance for every bin, retaining unsupported exposure."""
    require(mass.shape == (*parent.shape[:2], 9), 'conditional mass dimensions differ')
    require(np.allclose(mass.sum(axis=2), parent[..., 0], rtol=0., atol=1e-12), 'conditional mass differs from parent')
    counts, keep = plan.supported(mass)
    a, b = parent[:32, :, 0].mean(axis=0), parent[32:, :, 0].mean(axis=0)
    require(np.all(a > 0.) and np.all(b > 0.), 'parent common key missing in a half')
    reference = (a+b)/2.
    factors = np.stack((reference/a, reference/b))
    balanced = mass*np.repeat(factors, 32, axis=0)[..., None]
    inherited = common.balance(parent[:32], parent[32:])
    for half, expected in zip((balanced[:32], balanced[32:]), inherited):
        require(np.allclose(half.sum(axis=2), expected[..., 0], rtol=0., atol=1e-12), 'V191 parent balance changed')
    bins = []
    for bi, (f, h) in enumerate(BINS):
        retained = balanced[:, :, bi]*keep[None, :, bi]
        total = balanced[:, :, bi]
        bins.append(dict(fraction_bin=f, horizon_bin=h, candidate_cells=len(keep),
            observed_cells=int(np.any(counts[:, :, bi] > 0, axis=0).sum()),
            supported_cells=int(keep[:, bi].sum()), abstained_cells=int((~keep[:, bi]).sum()),
            first_days_range=[int(counts[0, :, bi].min()), int(counts[0, :, bi].max())],
            last_days_range=[int(counts[1, :, bi].min()), int(counts[1, :, bi].max())],
            balanced_mass=[float(v.mean(axis=0).sum()) for v in (total[:32], total[32:])],
            retained_balanced_mass=[float(v.mean(axis=0).sum()) for v in (retained[:32], retained[32:])]))
    return counts, keep, factors, bins


def scan(args, actor, day, mode, table, geometry, parent, admitted):
    require(day in DAYS and (parent['actor'], parent['day'], parent['mode']) == (actor, day, mode)
        and parent['v190_moments_reproduced'], 'V194 path identity differs')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
        ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'], 'capture differs')
    path, rows, entries, history = saved.load_timeline(root, name, geometry)
    prefix = prior.RiskPrefix(rows, history)
    mapping = {tuple(e['key']): e for e in table['candidate']}
    atoms, selected, query_count = {}, [], 0
    for q, r in zip_longest(*(prior.running.previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and q['episode'] == r['episode']
            and q['query_time_s'] == r['query_time_s'] and (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/label join differs')
        query_count += 1
        prior.running.previous.add_first(atoms, q, r)
        if not q['denominator_eligible']:
            continue
        scopes = [s for s in SCOPES if (s == 'clearance' and q['condition_values']['query_phase'] == 'clearance') or
            (s == 'hour13' and q['observable']['hour'] == 13) or
            (s == 'hour13_moving_launch' and q['observable']['hour'] == 13 and q['native_stage']['kind'] == 'moving-launch')]
        if not scopes:
            continue
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
        now = q['query_time_s']
        period = prior.running.temporal.STRATA[int(prior.running.temporal.strata(np.array([now]))[0])]
        entry = mapping.get((*q['candidate_key'], period)) if q['native_supported'] else None
        bounds = prior.running.previous.saved.fit.interval(q['point_s'], entry, q['native_supported'])
        require(bounds is None or entry['day_count'] >= 19, 'frozen interval support weakened')
        selected.append((q, r, scopes, bounds))
    require(query_count == receipt['query_record_count'], 'incomplete archived query stream')
    service = prior.running.previous.Prefix(atoms)
    base = {s: {} for s in SCOPES}
    moments, facts = ({s: defaultdict(dict) for s in SCOPES} for _ in range(2))
    clock_queries = 0
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
            extended = saved.moment([*vector, i, a, *components])
            require(np.allclose(saved.old_moment(extended), prior.moment(vector), rtol=0., atol=1e-8), 'query moments changed')
            f, hbin = plan.stratum(vector[FRACTION], vector[HORIZON])
            bi = BINS.index((f, hbin))
            clock_queries += 1
        for scope in scopes:
            base[scope].setdefault(ep, np.zeros(2))[:] += (1., float(bounds is not None))
            if scope in included:
                key = item[0]
                moments[scope][ep].setdefault(key, np.zeros((9, saved.SIZE)))[bi] += extended
                facts[scope][ep].setdefault(key, np.zeros((9, 1+len(saved.FACTS))))[bi] += np.r_[1., timing]
    require(clock_queries == parent['clock_queries'], 'V194 complete-input query cohort changed')
    result, max_error = [], np.zeros(4)
    for scope in SCOPES:
        expected = next(s for s in parent['scopes'] if s['scope'] == scope)
        require(len(base[scope]) == expected['eligible_episodes'], 'original episode denominator differs')
        values, meta = saved.collapse(moments[scope], base[scope]), saved.collapse(facts[scope], base[scope])
        original = {tuple(g['key']): g for g in expected['groups']}
        require(set(values) == set(meta) == set(original) <= admitted[scope], 'V194 admitted parent keys changed')
        for key in original:
            max_error = np.maximum(max_error, verify_partition(values[key][None], meta[key][None],
                np.array([original[key]['moments']]), np.array([original[key]['facts']])))
        result.append(dict(scope=scope, groups=values, facts=meta,
            episode_denominators=[dict(episode=list(ep), eligible_queries=int(v[0]), finite_interval_queries=int(v[1]))
                for ep, v in base[scope].items()]))
    return dict(actor=actor, day=day, mode=mode, archive_queries=query_count, clock_queries=clock_queries,
        parent_max_abs_error=max_error.tolist(), scopes=result)


def bucket(args, chunk, days=DAYS, persist=True):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require(table['execution_source'] == 'a5b45df520', 'V172 source differs')
    geometry = prior.clocks.native.load_geometry(args.runtime)
    payload = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
    require(payload['source'] == '51a9409ec2', 'V194 source differs')
    references = {r['day']: r for r in payload['rows']}
    support = {s: read(args.v191_root/'details'/f'{mode}_{s}.json')['common_key_support'] for s in SCOPES}
    keys = {s: sorted(tuple(g['key']) for g in support[s] if g['actor'] == actor) for s in SCOPES}
    parents = {s: plan.base.actor_matrix(payload['rows'], s, support[s], actor) for s in SCOPES}
    matrices = {s: np.zeros((len(days), len(keys[s]), 9, saved.SIZE)) for s in SCOPES}
    facts = {s: np.zeros((len(days), len(keys[s]), 9, 1+len(saved.FACTS))) for s in SCOPES}
    denominators, clock_queries, archive_queries, max_error = [], 0, 0, np.zeros(4)
    for di, day in enumerate(days):
        row = scan(args, actor, day, mode, table, geometry, references[day], {s: set(v) for s, v in keys.items()})
        clock_queries += row['clock_queries']
        archive_queries += row['archive_queries']
        max_error = np.maximum(max_error, row['parent_max_abs_error'])
        for cell in row['scopes']:
            scope = cell['scope']
            for ki, key in enumerate(keys[scope]):
                if key in cell['groups']:
                    matrices[scope][di, ki] = cell['groups'][key]
                    facts[scope][di, ki] = cell['facts'][key]
            denominators.append(dict(day=day, scope=scope, episodes=cell['episode_denominators']))
        if (di+1) % 8 == 0:
            print('V199_ARCHIVE_DAYS', actor, mode, di+1, '/64', flush=True)
    result = dict(actor=actor, mode=mode, paths=len(days), clock_queries=clock_queries, archive_queries=archive_queries,
        parent_max_abs_error=max_error.tolist(), scopes=[])
    if persist:
        require(tuple(days) == tuple(DAYS), 'partial paths cannot be admitted for support')
        for scope in SCOPES:
            counts, keep, factors, bins = support_summary(matrices[scope][..., 0], parents[scope])
            target = args.out_root/'cells'/f'{actor}_{mode}_{scope}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, days=DAYS, keys=keys[scope], bins=BINS, moments=matrices[scope], facts=facts[scope],
                parent_balance_factors=factors, distinct_days=counts, supported_both=keep)
            result['scopes'].append(dict(scope=scope, parent_keys=len(keys[scope]),
                supported_cells=int(keep.sum()), candidate_cells=int(keep.size), bins=bins))
        write(args.out_root/'metadata'/f'{actor}_{mode}.json', dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            actor=actor, mode=mode, episode_denominators=denominators, paths=len(days),
            clock_queries=clock_queries, archive_queries=archive_queries, parent_max_abs_error=max_error.tolist()))
    return result


def aggregate(results):
    rows = []
    for mode, scope in product(('old', 'handoff'), SCOPES):
        cells = [s for r in results if r['mode'] == mode for s in r['scopes'] if s['scope'] == scope]
        require(len(cells) == 12, 'actor scope grid incomplete')
        bins = []
        for bi, (f, h) in enumerate(BINS):
            inputs = [c['bins'][bi] for c in cells]
            b = dict(fraction_bin=f, horizon_bin=h)
            for name in ('candidate_cells', 'observed_cells', 'supported_cells', 'abstained_cells'):
                b[name] = sum(v[name] for v in inputs)
            for name in ('first_days_range', 'last_days_range'):
                b[name] = [min(v[name][0] for v in inputs), max(v[name][1] for v in inputs)]
            for name in ('balanced_mass', 'retained_balanced_mass'):
                b[name] = np.mean([v[name] for v in inputs], axis=0).tolist()
            bins.append(b)
        total, retained = (np.sum([b[n] for b in bins], axis=0) for n in ('balanced_mass', 'retained_balanced_mass'))
        rows.append(dict(mode=mode, scope=scope, parent_keys=sum(c['parent_keys'] for c in cells),
            candidate_cells=sum(c['candidate_cells'] for c in cells), supported_cells=sum(c['supported_cells'] for c in cells),
            balanced_mass=total.tolist(), retained_balanced_mass=retained.tolist(),
            abstained_balanced_mass=(total-retained).tolist(), retained_mass_fraction=(retained/total).tolist(), bins=bins))
    return rows


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v199_*.py', '-v'], check=True)
    parent = read(args.v198_root/'summary/status.json')
    frozen = plan.registration()
    require(parent['source'] == 'd347cfadd3' and parent['protocol']['calibration_days'] == list(DAYS)
        and parent['protocol']['required'] == frozen['required'] and not parent['schema_review_required'], 'V198 frozen protocol differs')
    model = read(args.v172_root/'model.json')
    require(model['execution_source'] == 'a5b45df520' and model['calibration_days'] == list(DAYS)
        and len(model['actors']) == 12 and {(c['actor'], c['mode']) for c in model['chunks']} ==
        set(product(model['actors'], ('old', 'handoff'))) and len(model['chunks']) == 24, 'actor/day/mode grid differs')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite sealed preparation')
    protocol = dict(stage='v199_readonly_conditional_moment_preparation', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), frozen=frozen['frozen'], v198_source='d347cfadd3', v194_source='51a9409ec2',
        required=frozen['required'], bin_order=BINS, packed_second_order='numpy.triu_indices(39)', facts=list(saved.FACTS),
        storage='Raw original V194 weights in NPZ;V191 parent-key factors saved separately,shared by ALL9 bins. '
            'Zero exposure only after complete archive/parent verification. Keep unsupported moments,never renormalize.',
        admission='All9 cells reproduce every parent mass,first,second and clock moment at absolute1e-8. '
            'Distinct positive-mass days use inherited1e-12 rounding tolerance;each cell requires19 EACH half;otherwise abstain.',
        execution='Original64 archive read only. Four fixed V194 preflight paths before full1536-path scan. '
            'Support masks only,no conditional association estimates or tests.',
        interval_fit=False, predictor_fit=False, conditional_associations_estimated=False,
        confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)
    write(args.out_root/'protocol.json', protocol)
    print('V199_PROTOCOL_SEALED_NO_ASSOCIATION_READOUT', flush=True)
    preflight = [bucket(args, c, days=(DAYS[0], DAYS[-1]), persist=False)
        for c in model['chunks'] if c['actor'] == min(model['actors'])]
    require(sum(r['paths'] for r in preflight) == 4 and sum(r['clock_queries'] for r in preflight) > 0, 'fixed preflight incomplete')
    write(args.out_root/'preflight/status.json', dict(status='v199_parent_preflight_passed', paths=4, results=preflight))
    print('V199_PARENT_PREFLIGHT_PASSED', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for i, job in enumerate(as_completed([pool.submit(bucket, args, c) for c in model['chunks']]), 1):
            results.append(job.result())
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}' % i, flush=True)
    require(sum(r['paths'] for r in results) == 1536, 'incomplete original64 scan')
    expected = read(args.v194_root/'summary/status.json')
    require(sum(r['clock_queries'] for r in results) == expected['clock_queries'], 'V194 full-cohort query count differs')
    summary = dict(status='v199_conditional_moments_and_support_complete', source=protocol['source'], protocol=protocol,
        paths=1536, preflight_paths=4, actors=12, days=64, modes=['old', 'handoff'],
        clock_queries=sum(r['clock_queries'] for r in results), archive_queries=sum(r['archive_queries'] for r in results),
        parent_max_abs_error=np.max([r['parent_max_abs_error'] for r in results], axis=0).tolist(),
        parent_error_order=['mass', 'first', 'second', 'facts'], parent_all9_moments_reproduced=True,
        source_and_label_clocks_checked=True, original_episode_denominators_preserved=True,
        parent_balance_shared_across_bins=True, conditional_support_frozen=True, unsupported_moments_retained=True,
        results=aggregate(results), raw_archives_modified=False, interval_fit=False, predictor_fit=False,
        conditional_associations_estimated=False, confirmation_outcomes_read=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)
    write(args.out_root/'summary/status.json', summary)
    print('V199_PARENT_AND_SUPPORT_ADMISSION_COMPLETE_NO_ASSOCIATIONS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'runtime', 'v166-root', 'v172-root', 'v191-root', 'v194-root', 'v198-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    run(p.parse_args())
