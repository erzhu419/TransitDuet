#!/usr/bin/env python3
"""Read back the fixed V165 calibration cohort; no replay, fit or confirmation."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
import json
from math import isfinite
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v166_fresh_capture as entry


audit, capture = entry.audit, entry.capture
read, write, require, stream = entry.read, entry.write, entry.require, entry.stream
CAPTURE_SOURCE, PROTOCOL_SOURCE = '20d914af03', '4a6d783667'


def receipts(receipt, cell, actor, day, mode, sampler):
    for record in (receipt, cell):
        require((record['seed'], record['day'], record['mode'], record['sampler_seed']) ==
            (actor, day, mode, sampler), 'path assignment differs')
        require(record['execution_source'] == CAPTURE_SOURCE and not record['main_policy_changed'] and
            all(record[f+'_updates'] == 0 for f in ('actor', 'critic', 'dual')), 'capture source/controller changed')
    require(cell['protocol_source'] == PROTOCOL_SOURCE and cell['all_queries_recorded_before_target_labels']
        and not cell['interval_fit'], 'capture protocol/workflow changed')


def shard_coverage(shards, pair_count):
    covered = []
    for s in shards:
        require(s['status'] == 'v166_calibration_shard_complete' and s['execution_source'] == CAPTURE_SOURCE and
            s['protocol_source'] == PROTOCOL_SOURCE and s['confirmation_paths'] == 0 and
            not s['interval_fit'] and not s['main_policy_changed'], 'shard source/freeze differs')
        require(s['end']-s['start'] == 8 and s['start'] % 8 == 0 and
            s['calibration_paths'] == 16, 'shard assignment differs')
        covered.extend(range(s['start'], s['end']))
    require(sorted(covered) == list(range(pair_count)), 'shard gaps or overlap')


def live_record(raw, ready, rank):
    now = raw['query_time_s']
    require(raw['query_key'] == list(capture.key(ready)) and now == ready['ready_time_s'] and
        raw['ready_callback_order'] == rank == raw['events_seen']['ready']-1, 'live ready identity/clock differs')
    require(raw['evidence'] == {k: v for k, v in ready.items() if k not in ('trip_id', 'station_id', 'direction')},
        'live evidence differs from ready capture')
    require(raw['query_load'] is None and not raw['query_load_available'], 'late load entered live snapshot')
    require(isfinite(raw['own_progress_m']) and isfinite(raw['own_speed_mps']), 'own query AVL missing')
    for field in ('last_arrival_s', 'last_departure_s'):
        t = raw['events_seen'][field]
        require(t is None or t <= now, 'future event in live snapshot')
    follower = raw['follower']
    if follower is not None:
        require((follower['bus_id'], follower['trip_id']) ==
            (ready['predicted_follower_bus_id'], ready['predicted_follower_trip_id']) and
            isfinite(follower['progress_m']) and isfinite(follower['speed_mps']) and bool(follower['state']),
            'live physical follower/AVL differs')


def audit_path(args, sealed, actor, day, mode):
    root, name = args.capture_root/'calibration', f'{actor}_{day}_{mode}'
    receipt, cell = (read(root/d/(name+'.json')) for d in ('paths', 'capture_cells'))
    receipts(receipt, cell, actor, day, mode, sealed['reservation']['sampler_seeds'][str(day)])
    ready_rows = list(stream(root/'ready_labels'/(name+'.jsonl.gz')))
    ready = {capture.key(r): r for r in ready_rows}
    ranks = {k: i for i, k in enumerate(ready)}
    raw_rows = list(stream(root/'query_snapshots'/(name+'.jsonl.gz')))
    snapshots = {tuple(r['query_key']): r for r in raw_rows}
    require(len(ready_rows) == len(ready) == receipt['ready_labels'] == len(raw_rows) == len(snapshots) ==
        receipt['live_query_snapshots'] == cell['live_ready_snapshots'], 'ready snapshot count differs')
    require(set(ready) == set(snapshots), 'ready snapshot identities incomplete')
    for k, row in ready.items():
        live_record(snapshots[k], row, ranks[k])
    counts, keys, seen = Counter(), Counter(), set()
    cutoff = receipt['contract']['clock']['demand_end_time_s']
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals', 'traces')]
    for q, residual, row in zip_longest(*sources):
        require(q is not None and residual is not None and row is not None, 'query/residual/trace count differs')
        k = tuple(q['query_key'])
        require(k not in seen, 'duplicate query key')
        seen.add(k)
        require(set(q['observable']['evidence']) == set(capture.extraction.EVIDENCE), 'observable evidence fields missing')
        position = q['observable']['position']
        if position is not None:
            require(set(position) == {'origin_index', 'target_index', 'progress_m', 'remaining_stops',
                'remaining_distance_m', 'initial_residence_prediction_s'} and
                all(isfinite(v) for v in position.values()), 'query position fields incomplete')
        audit.query_record(q, residual, row, snapshots[k], ready, ranks, sealed, actor, day, mode, cutoff, counts)
        if q['native_supported'] and residual['residual_s'] is not None:
            keys[json.dumps(q['candidate_key'], separators=(',', ':'))] += 1
    require(counts['queries'] == receipt['decisions'] == cell['query_record_count'], 'lower query cohort incomplete')
    for field, count in cell['counts'].items():
        require(counts[field] == count, 'capture counter differs: '+field)
    counts['live_ready_snapshots'] = len(snapshots)
    return dict(actor=actor, day=day, mode=mode, counts=dict(counts), key_counts=dict(keys), status='passed')


def audit_case(args, sealed, case):
    actor, day, mode = case
    try:
        result = audit_path(args, sealed, *case)
    except Exception as exc:
        result = dict(actor=actor, day=day, mode=mode, status='failed', error=type(exc).__name__+': '+str(exc))
    write(args.out_root/'paths'/f'{actor}_{day}_{mode}.json', result)
    return result


def matrix(args, sealed):
    expected = [(a, d, m) for a, d in capture.calibration_cases(sealed) for m in capture.MODES]
    names = {f'{a}_{d}_{m}' for a, d, m in expected}
    root = args.capture_root/'calibration'
    for directory, suffix in (('paths', '.json'), ('capture_cells', '.json'), ('records', '.npz'),
            ('terminal', '.json'), ('ready_labels', '.jsonl.gz'), ('query_snapshots', '.jsonl.gz'),
            ('queries', '.jsonl.gz'), ('residuals', '.jsonl.gz'), ('traces', '.jsonl.gz')):
        require({p.name for p in (root/directory).glob('*'+suffix)} == {n+suffix for n in names},
            'calibration matrix incomplete or contaminated: '+directory)
    shards = [read(p) for p in (args.capture_root/'shards').glob('*.json')]
    shard_coverage(shards, len(expected)//2)
    require(not (args.capture_root/'confirmation').exists(), 'unexpected confirmation collection')
    return expected, len(shards)


def main(args):
    for pattern in ('test_v162_capture_audit.py', 'test_v166_capture_audit.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    sealed, _ = entry.parents(args)
    proof = read(args.capture_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and
        proof['execution_source'] == CAPTURE_SOURCE and proof['protocol_source'] == PROTOCOL_SOURCE and
        all(proof[k] for k in entry.EQUIVALENCE), 'V166 passive admission missing')
    expected, shards = matrix(args, sealed)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(audit_case, args, sealed, case) for case in expected]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            if done % 32 == 0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='saved_path')), flush=True)
    failures = [r for r in rows if r['status'] != 'passed']
    passed = [r for r in rows if r['status'] == 'passed']
    counts = dict(sum((Counter(r['counts']) for r in passed), Counter()))
    complete = not failures and len(passed) == len(expected)
    eligible = complete and counts.get('eligible_unresolved', 0) == 0
    marginals = {field: [dict(group=r['group'], paths=r['paths'],
        queries=r['counts'].get('queries', 0), eligible=r['counts'].get('denominator_eligible', 0),
        native_supported=r['counts'].get('native_supported', 0),
        eligible_unresolved=r['counts'].get('eligible_unresolved', 0)) for r in audit.marginals(passed, field)]
        for field in ('actor', 'day')}
    summary = dict(status='v166_readback_passed' if complete else 'v166_readback_failed',
        capture_source=CAPTURE_SOURCE, protocol_source=PROTOCOL_SOURCE,
        audit_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=sealed['reservation']['calibration_days'], actors=sealed['reservation']['actors'],
        modes=list(capture.MODES), expected_paths=len(expected), passed_paths=len(passed),
        failed_paths=len(failures), failures=failures[:12], shard_count=shards, counts=counts,
        by_mode=audit.marginals(passed, 'mode'), marginals=marginals,
        key_day_support=audit.key_support(passed, sealed['support']['minimum_calibration_days']),
        query_metadata_complete=complete, query_support_and_label_clocks_checked=complete,
        calibration_input_eligible=eligible, conditional_key_counts_only=True, interval_fit=False,
        confirmation_paths=0, confirmation_started=False, confirmation_outcomes_read=False,
        replay=False, training=False, main_policy_changed=False, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1],
        next='Calibration metadata admitted; separately authorize frozen V161 joint-map construction using '
            'only these 64 days. Confirmation stays closed until support admission.' if eligible else
            'Stop before fitting; inspect failed path receipts or unresolved eligible labels on server.')
    write(args.out_root/'summary/status.json', summary)
    print('V166_READBACK_COMPLETE', summary['status'], 'passed', len(passed), 'failed', len(failures), flush=True)
    require(complete, 'saved calibration metadata failed readback; inspect server path receipts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'out-root', 'v165-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    main(parser.parse_args())
