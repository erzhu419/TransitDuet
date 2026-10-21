#!/usr/bin/env python3
"""Seal temporal-tail validation and count calibration support; never fit bands."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from math import isfinite
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import seal_freqduet_v163_joint_intervals as original
import preregister_freqduet_v165_fresh_calibration as ledger


read, write, require, close = original.read, original.write, original.require, original.close
CONFIRMATION = tuple(range(3152001, 3152033))
SAMPLERS = {day: 1711000000+i for i, day in enumerate(CONFIRMATION)}
SCOPES = (*original.SCOPES, 'hour13_moving_launch')
STRATA = ('other', 'hour2', 'hour13')


def strata(times):
    hour = (times//3600).astype(int)
    return np.where(hour == 2, 1, np.where(hour == 13, 2, 0))


def supported(keys, days, periods, request_indices, request_count):
    code = keys*len(STRATA)+periods
    pairs = np.unique(np.column_stack((code, days)), axis=0)
    day_count = np.bincount(pairs[:, 0], minlength=int(code.max())+1)
    finite = np.zeros(request_count, dtype=bool)
    finite[request_indices] = day_count[code] >= original.protocol.MIN_DAYS
    records = [dict(old_key=int(k//3), stratum=STRATA[int(k % 3)], days=int(day_count[k]))
        for k in np.unique(code)]
    return finite, records


def support_cell(day, scope, mask, days, episodes, native, finite, baseline):
    selected = np.flatnonzero(mask & (days == day))
    result = dict(day=int(day), scope=scope, eligible_queries=len(selected), eligible_episodes=0,
        finite_queries=int(finite[selected].sum()), finite_episodes=0, finite_fraction=None,
        baseline_fraction=None, native_abstentions=int((~native[selected]).sum()),
        key_abstentions=int((native[selected] & ~finite[selected]).sum()))
    if len(selected):
        unique, inv, count = np.unique(episodes[selected], return_inverse=True, return_counts=True)
        n = np.bincount(inv, weights=finite[selected], minlength=len(unique))
        b = np.bincount(inv, weights=baseline[selected], minlength=len(unique))
        result.update(eligible_episodes=len(unique), finite_episodes=int(np.count_nonzero(n)),
            finite_fraction=float(np.mean(n/count)), baseline_fraction=float(np.mean(b/count)))
    return result


def moving_mask(args, actor, mode, requests, native, atom_requests, atom_keys, entries, field):
    moving = np.zeros(len(requests), dtype=bool)
    moving[atom_requests] = np.array([entries[k]['key'][field] == 'moving-launch' for k in atom_keys])
    missing = (~native) & (requests[:, 4]//3600 == 13)
    opened = matched = 0
    for day in np.unique(requests[missing, 0]).astype(int):
        selected = np.flatnonzero(missing & (requests[:, 0] == day))
        wanted = {tuple(requests[i, 5:].astype(int)): i for i in selected}
        path = args.capture_root/'calibration/queries'/f'{actor}_{day}_{mode}.jsonl.gz'
        opened += 1
        for q in original.stream(path):
            key = tuple(q['query_key'])
            if key not in wanted:
                continue
            i = wanted.pop(key)
            require((q['actor'], q['day'], q['mode']) == (actor, day, mode) and
                q['denominator_eligible'] and not q['native_supported'] and
                q['query_time_s'] == requests[i, 4], 'unsupported query metadata join differs')
            moving[i] = q['native_stage']['kind'] == 'moving-launch'
            matched += 1
            if not wanted:
                break
        require(not wanted, 'eligible unsupported hour13 query metadata missing')
    return moving, dict(calibration_query_files_opened=opened, unsupported_hour13_contexts_checked=matched,
        unsupported_hour13_moving=int((moving & missing).sum()))


def bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v167_root/chunk['map_file'])
    require((table['execution_source'], table['actor'], table['mode']) == ('e6fc851ec1', actor, mode), 'map source differs')
    with np.load(args.v167_root/chunk['atom_file']) as z:
        q, ri, keys = z['requests'], z['atom_request'], z['candidate_key']
        native, baseline = z['native_supported'], z['finite_candidate']
        days, episodes = q[:, 0].astype(int), q[:, 1].astype(int)
        require(set(days) == set(model['calibration_days']), 'calibration cohort differs')
        finite, records = supported(keys, days[ri], strata(q[ri, 4]), ri, len(q))
        require(not np.any(finite & ~baseline) and not np.any(finite & ~z['finite_reference']),
            'time stratification expanded support or lost frozen comparator')
        moving, metadata = moving_mask(args, actor, mode, q, native, ri, keys, table['candidate'],
            model['frozen_protocol']['conditioning']['candidate_key'].index('native_stage_kind'))
        bits = q[:, 2].astype(int)
        masks = {s: (bits & (1 << i)) != 0 for i, s in enumerate(original.SCOPES)}
        masks['hour13_moving_launch'] = masks['hour13'] & moving
        cells = [dict(actor=actor, mode=mode, **support_cell(d, s, mask, days, episodes, native, finite, baseline))
            for d in model['calibration_days'] for s, mask in masks.items()]
    result = dict(actor=actor, mode=mode, cells=cells, metadata=metadata,
        strata=[dict(stratum=s, keys=len([k for k in records if k['stratum'] == s]),
            finite_keys=sum(k['days'] >= 19 for k in records if k['stratum'] == s),
            day_histogram=dict(sorted(Counter(k['days'] for k in records if k['stratum'] == s).items()))) for s in STRATA])
    write(args.out_root/'details/key_support'/f'{actor}_{mode}.json', records)
    write(args.out_root/'cells'/f'{actor}_{mode}.json', result)
    print('V171_SUPPORT_BUCKET_COMPLETE', actor, mode, flush=True)
    return result


def support_release(cells, actors, days):
    expected = {(a, d) for a in actors for d in days}
    rows = []
    for mode in ('old', 'handoff'):
        for scope in SCOPES:
            selected = [c for c in cells if (c['mode'], c['scope']) == (mode, scope)]
            complete = len(selected) == len(expected) and {(c['actor'], c['day']) for c in selected} == expected
            nonempty = complete and all(c['eligible_episodes'] and c['finite_episodes'] for c in selected)
            fraction = float(np.mean([c['finite_fraction'] for c in selected])) if nonempty else None
            rows.append(dict(mode=mode, scope=scope, cells=len(selected), complete_grid=complete,
                every_cell_finite=bool(nonempty), finite_fraction=fraction,
                baseline_fraction=float(np.mean([c['baseline_fraction'] for c in selected])) if nonempty else None,
                finite_queries=sum(c['finite_queries'] for c in selected),
                native_abstentions=sum(c['native_abstentions'] for c in selected),
                key_abstentions=sum(c['key_abstentions'] for c in selected),
                support_admitted=bool(nonempty and fraction >= .8)))
    return dict(decision='pass' if all(r['support_admitted'] for r in rows) else 'insufficient_support',
        mode_scopes=rows, failed=[r['mode']+'/'+r['scope'] for r in rows if not r['support_admitted']])


def freshness(args, model):
    old = read(args.inventory_file)
    roots = set(ledger.archives.result_roots(args.parent)) | {args.v167_root.parent.parent}
    known = {int(k) for k in old['day_references']} | {d for g in old['groups'] for d in g['days']}
    collisions = [dict(path='V159 inventory', day=d) for d in sorted(known & set(CONFIRMATION))]
    files, names = set(), 0
    metadata_names = {'protocol.json', 'preregistration.json', 'registration.json', 'model.json', 'manifest.json'}
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
                if name.endswith('.json') and (name in metadata_names or path.name in ledger.RECEIPTS):
                    files.add(path/name)
    for i, path in enumerate(sorted(files), 1):
        value = read(path)
        days = ledger.extract.day_ids(value) & set(CONFIRMATION)
        samplers = ledger.metadata.sampler_ids(value) & set(SAMPLERS.values())
        if days or samplers:
            collisions.append(dict(path=str(path), days=sorted(days), samplers=sorted(samplers)))
        if i % 5000 == 0:
            print('V171_FRESHNESS_METADATA', i, len(files), flush=True)
    for root in args.generator_root:
        for path in root.glob('*.py'):
            if path.name != Path(__file__).name:
                hits = ledger.metadata.literal_conflicts(path, set(CONFIRMATION) | set(SAMPLERS.values()))
                if hits:
                    collisions.append(dict(path=str(path), literals=sorted(hits)))
    require(not set(CONFIRMATION) & set(model['calibration_days']), 'cohort overlap')
    result = dict(result_roots=len(roots), metadata_files=len(files), artifact_names=names, collisions=collisions,
        scope='known FreqDuet archive inventory, recorder filenames, protocol/registration/model headers, '
            'path/pair sampler receipts and declared generator literals; no outcome summaries or raw labels')
    write(args.out_root/'details/freshness.json', result)
    require(not collisions, 'new confirmation day/sampler collision; do not substitute days')
    return result


def registration(model, fresh):
    require(model['execution_source'] == 'e6fc851ec1' and model['status'] == 'sealed' and
        model['calibration_days'] == list(range(2951001, 2951065)), 'requires frozen V167 64-day calibration')
    require(not fresh['collisions'], 'new confirmation reservation not fresh')
    result = deepcopy(model['frozen_protocol'])
    old_key = result['conditioning']['candidate_key']
    result.update(status='preregistered_not_collected', stage='v171_temporal_tail_validation',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    result['conditioning'].update(candidate_key=[*old_key, 'query_time_stratum'],
        query_time_stratum='hour2 iff floor(query_time_s/3600)==2; hour13 iff ==13; other otherwise',
        candidate_count=1, selection='fixed three strata; no hour/window/horizon/stop-ID search')
    result['reservation'].update(confirmation_days=list(CONFIRMATION),
        sampler_seeds={**{str(d): model['frozen_protocol']['reservation']['sampler_seeds'][str(d)]
            for d in model['calibration_days']}, **{str(d): SAMPLERS[d] for d in CONFIRMATION}},
        excluded_confirmation_days=list(range(2852001, 2852033)), confirmation_paths=32*len(model['actors'])*2,
        size_reason='retain the existing 64 calibration days and reserve exactly 32 fresh confirmation days; '
            'fixed resource design, not a power guarantee or target-until-pass rule',
        calibration_role='existing V166 64 calibration days only; support now, interval construction only after authorization')
    result['inference'].update(scopes=list(SCOPES),
        moving_scope='hour13 and recorded query-time native_stage.kind == moving-launch; all eligible requests '
            'including native unsupported; not defined by finite intervals or realized labels',
        migration='separately report calibration and fresh-confirmation hour13_moving_launch coverage, both tails, '
            'finite fraction, score and actor/day marginals for candidate/V167/coarse on the candidate-finite mask; '
            'calibration is in-sample, not an independent success criterion',
        migration_uncertainty='descriptive confirmation-minus-calibration tail deltas: 5000 draws, seed20260803, '
            'shared resampled actor axis, independent resampled calibration-day and confirmation-day axes; no migration gate')
    result['gates']['applies_to'] = 'both modes and all six registered scopes, including hour13_moving_launch'
    result['comparator'].update(map_source='e6fc851ec1', primary='unchanged V167 coarse reference',
        secondary='unchanged V167 conditional map; report paired differences on the same candidate-finite mask')
    result['estimator']['implementation'] = (
        'same V161 empirical ranks and joint atoms; future mixtures must recompute '
        'equal-day/episode/query weights within each new key, never reuse old-key atom weights')
    result['isolation'].update(first='protocol and calibration-support counts only; no collector or estimator now',
        release='all six calibration support scopes: complete actor/day grid, nonempty finite estimate in every cell, '
            'mean finite fraction >=0.8, no missing frozen comparator; otherwise stop without regrouping or added days',
        calibration_seal='if separately authorized and support passed, construct one temporal map on the existing '
            '64 calibration days, seal maps/support before opening the new 32 confirmation days',
        fixed_budget='existing 64 calibration days only; exactly 32 reserved confirmation days if later authorized; '
            'no outcome-driven additions, replacements, regrouping or optional stopping',
        development='prior V168/V170 diagnostics motivate the questions only; this stage reads no confirmation '
            'outcomes and uses calibration-only state/support counts for numerical design',
        no_reuse='no consumed V168 records in fitting, support checks or threshold selection; no stop exceptions, '
            'quantile/window changes, online adaptation or selecting successful scopes')
    result['execution'] = dict(collector_submitted=False, new_days_collected=0, new_rollouts=0, new_training=0,
        interval_fit=False, confirmation_outcomes_read=False, main_policy_changed=False)
    result['parents'] = dict(frozen_map='e6fc851ec1', calibration_capture='20d914af03', calibration_admission='c990e3c99b')
    result['freshness'] = fresh
    return result


def gate(readouts):
    expected = {(m, s) for m in ('old', 'handoff') for s in SCOPES}
    identities = [(r['mode'], r['scope']) for r in readouts]
    if len(identities) != len(expected) or set(identities) != expected:
        return dict(decision='inconclusive', reason='incomplete_or_duplicate_grid', controller_admitted=False)
    failures = []
    for r in readouts:
        fields = ('coverage_lower95', 'below_upper95', 'above_upper95', 'finite_fraction_lower95', 'score_delta_upper95')
        if (not r['complete_actor_day_grid'] or r['finite_unresolved_n'] or r['missing_reference_n'] or
                any(r.get(f) is None or not isfinite(r[f]) for f in fields)):
            return dict(decision='inconclusive', reason='missing_evidence', controller_admitted=False)
        tests = dict(coverage=r['coverage_lower95'] >= .8, below=r['below_upper95'] <= .1,
            above=r['above_upper95'] <= .1, support=r['finite_fraction_lower95'] >= .8,
            score=r['score_delta_upper95'] < 0. if r['scope'] in ('all', 'clearance') else r['score_delta_upper95'] <= 0.)
        failures.extend(r['mode']+'/'+r['scope']+'/'+name for name, passed in tests.items() if not passed)
    return dict(decision='pass' if not failures else 'not_admitted', failures=failures, controller_admitted=False)


def run(args):
    require(not (args.out_root/'protocol.json').exists(), 'protocol already sealed; do not overwrite')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v171_*.py', '-v'], check=True)
    model = read(args.v167_root/'model.json')
    admission = read(args.audit_root/'summary/status.json')
    require(admission['status'] == 'v166_readback_passed' and admission['passed_paths'] == 1536 and
        admission['failed_paths'] == 0 and admission['audit_source'] == 'c990e3c99b', 'calibration admission missing')
    plan = registration(model, freshness(args, model))
    write(args.out_root/'protocol.json', plan)
    print('V171_PROTOCOL_SEALED_BEFORE_SUPPORT_CHECK', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, model, chunk) for chunk in model['chunks']]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=24, unit='actor_mode_support')), flush=True)
    cells = [c for row in rows for c in row['cells']]
    release = support_release(cells, model['actors'], model['calibration_days'])
    require(read(args.v167_root/'model.json') == model, 'frozen V167 model changed')
    write(args.out_root/'details/actor_day_support.json', cells)
    write(args.out_root/'summary/status.json', dict(status='v171_protocol_and_calibration_support_complete',
        execution_source=plan['execution_source'], protocol=plan, support=release,
        context_readback=dict(sum((Counter(r['metadata']) for r in rows), Counter())),
        strata=[dict(actor=r['actor'], mode=r['mode'], groups=r['strata']) for r in rows],
        frozen_v167_unchanged=True, new_days_collected=0, new_rollouts=0, new_training=0, interval_fit=False,
        confirmation_outcomes_read=False, collector_submitted=False, main_policy_changed=False,
        next='Support only, not coverage evidence. Wait for separate authorization; no collection or fitting started.'))
    print('V171_PREREGISTERED_SUPPORT_COMPLETE_NO_COLLECTION_NO_FIT', release['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v167-root', 'capture-root', 'audit-root', 'inventory-file', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--generator-root', type=Path, action='append', required=True)
    parser.add_argument('--workers', type=int, default=2)
    run(parser.parse_args())
