#!/usr/bin/env python3
"""Passive V165 calibration capture, admitted on consumed V162 preflight paths."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v162_capture as audit
import preregister_freqduet_v165_fresh_calibration as fresh


capture = fresh.capture
read, write, require = capture.read, capture.write, capture.require
stream = capture.metadata.stream
CONSUMED = (12011, 2451001, 1450000000)
EQUIVALENCE = ('original_endpoints_actions_arrays_terminal_clocks_exact',
    'v162_query_packets_snapshots_labels_and_traces_exact',
    'query_state_support_identity_and_clocks_checked', 'frozen_v151_forecasts_and_components_exact')


def validate_protocol(sealed, reference):
    require(sealed['execution_source'] == '4a6d783667' and
        sealed['status'] == 'preregistered_not_collected' and
        sealed['stage'] == 'v165_fixed_fresh_64day_calibration', 'requires sealed V165 protocol')
    require(reference['execution_source'] == '09553bc920' and
        all(sealed[k] == reference[k] for k in fresh.METHOD_FIELDS), 'frozen V161 method changed')
    r, original = sealed['reservation'], reference['reservation']
    require(r['calibration_days'] == list(fresh.CALIBRATION) and
        r['confirmation_days'] == original['confirmation_days'] and
        r['excluded_development_days'] == original['calibration_days'], 'reserved cohort changed')
    samplers = {str(d): fresh.SAMPLERS[d] for d in fresh.CALIBRATION}
    samplers.update({str(d): original['sampler_seeds'][str(d)] for d in original['confirmation_days']})
    require(r['sampler_seeds'] == samplers and r['actors'] == original['actors'] and
        len(r['actors']) == 12 and r['modes'] == original['modes'] == list(capture.MODES) and
        r['checkpoint_ep'] == 199 and r['calibration_paths'] == 1536, 'path assignment changed')


def parents(args):
    sealed, reference = read(args.v165_root/'protocol.json'), read(args.v161_root/'protocol.json')
    validate_protocol(sealed, reference)
    service, old = read(args.v151_root/'model.json'), read(args.v146_root/'model.json')
    require((service['status'], service['execution_source']) == ('sealed', '1b783ea8f3'), 'V151 changed')
    require((old['status'], old['lower_extension_s'], old['upper_extension_s']) == ('sealed', 3., 0.), 'V146 changed')
    require(not set(service['calibration_days']) & set(sealed['reservation']['calibration_days']),
        'fresh calibration overlaps V151 prior')
    return sealed, service


def require_preflight(args):
    proof = read(args.out_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and
        proof['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and
        proof['protocol_source'] == '4a6d783667' and all(proof[k] for k in EQUIVALENCE),
        'requires successful same-source passive equivalence preflight')


def capture_case(args, actor, day, mode, sampler, root, sealed, service):
    name = f'{actor}_{day}_{mode}'
    final = root/'capture_cells'/(name+'.json')
    if final.exists():
        old = read(final)
        require((old['seed'], old['day'], old['mode'], old['sampler_seed'], old['execution_source']) ==
            (actor, day, mode, sampler, os.environ['FREQDUET_DIAGNOSTIC_COMMIT']), 'stored capture assignment changed')
        return old
    receipt = capture.collect(root, args.runtime, actor, day, mode, sampler)
    require(receipt['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'stored rollout source changed')
    path, geometry, counts = capture.save_queries(root, args.runtime, actor, day, mode, service, sealed)
    counts.update(capture.attach_labels(root, name, path, geometry))
    result = dict(seed=actor, day=day, mode=mode, sampler_seed=sampler, counts=dict(counts),
        query_record_count=counts['queries'], live_ready_snapshots=receipt['live_query_snapshots'],
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], protocol_source='4a6d783667',
        all_queries_recorded_before_target_labels=True, interval_fit=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V166_CAPTURE_PATH_COMPLETE', actor, day, mode, flush=True)
    return result


def exact_stream(left, right, label):
    count = 0
    for a, b in zip_longest(left, right):
        require(a is not None and b is not None and a == b, f'{label} differs at row {count}')
        count += 1
    require(count > 0, label+' empty')
    return count


def preflight(args):
    sealed, service = parents(args)
    for pattern in ('test_v162_calibration_capture.py', 'test_v166_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    prior = read(args.capture_root/'preflight/summary/status.json')
    require(prior['status'] == 'v162_capture_preflight_passed' and
        prior['execution_source'] == '25a2eddeaf', 'original V162 passive proof missing')
    receipts = []
    for mode in capture.MODES:
        actor, day, sampler = CONSUMED
        root, baseline = args.out_root/'preflight_paths', args.capture_root/'preflight_paths'
        cell = capture_case(args, actor, day, mode, sampler, root, sealed, service)
        name = f'{actor}_{day}_{mode}'
        old, new = read(args.archive_root/'paths'/(name+'.json')), read(root/'paths'/(name+'.json'))
        for field in ('values', 'contract'):
            capture.collector.capture.pilot.joint.assert_same_state(old[field], new[field], 'V166_original_'+field)
        capture.collector.capture.pilot.joint.assert_same_state(read(args.archive_root/'terminal'/(name+'.json')),
            read(root/'terminal'/(name+'.json')), 'V166_original_terminal_clocks')
        rows = list(stream(root/'traces'/(name+'.jsonl.gz')))
        capture.collector.capture.passive.assert_arrays_and_actions(args.archive_root/'records'/(name+'.npz'),
            root/'records'/(name+'.npz'), rows)
        capture.collector.capture.identity.assert_decisions(list(stream(args.archive_root/'traces'/(name+'.jsonl.gz'))), rows)
        identical = {}
        for directory in ('query_snapshots', 'queries', 'ready_labels', 'residuals', 'traces'):
            relative = Path(directory)/(name+'.jsonl.gz')
            identical[directory] = exact_stream(stream(baseline/relative), stream(root/relative), directory)
        ready = {capture.key(r): r for r in stream(root/'ready_labels'/(name+'.jsonl.gz'))}
        ranks = {k: i for i, k in enumerate(ready)}
        snapshots = {tuple(r['query_key']): r for r in stream(root/'query_snapshots'/(name+'.jsonl.gz'))}
        require(set(ready) == set(snapshots), 'ready snapshot identities missing')
        queries = {tuple(q['query_key']): q for q in stream(root/'queries'/(name+'.jsonl.gz'))}
        residuals = list(stream(root/'residuals'/(name+'.jsonl.gz')))
        counts = Counter()
        require(len(rows) == len(residuals) == len(queries) == cell['query_record_count'], 'query cohort incomplete')
        for row, residual in zip(rows, residuals):
            k = capture.key(row)
            audit.query_record(queries[k], residual, row, snapshots[k], ready, ranks, sealed,
                actor, day, mode, new['contract']['clock']['demand_end_time_s'], counts)
        reproduced = 0
        for saved in stream(args.v159_root/'residuals'/(name+'.jsonl.gz')):
            if saved['forecasts'] is not None:
                q = queries[tuple(saved['query_key'])]
                capture.close(q['point_s'], saved['forecasts']['v151']['point_s'], 'V151 point changed')
                if saved['components'] is not None:
                    for c, value in saved['components']['v151']['prediction_s'].items():
                        capture.close(q['prediction_coarse'][c], value, 'V151 component changed')
                reproduced += 1
        receipts.append(dict(actor=actor, day=day, mode=mode, sampler_seed=sampler,
            identical_records=identical, query_audit_counts=dict(counts), v151_forecasts_reproduced=reproduced))
    write(args.out_root/'preflight/summary/status.json', dict(status='v166_passive_equivalence_passed',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], protocol_source='4a6d783667',
        consumed_paths=receipts, **{k: True for k in EQUIVALENCE}, new_calibration_days_collected=0,
        confirmation_collected=0, interval_fit=False, training=False, main_policy_changed=False))
    print('V166_PREFLIGHT_PASSED_FRESH_CALIBRATION_NOT_STARTED', flush=True)


def pair_case(args, index):
    require_preflight(args)
    sealed, service = parents(args)
    actor, day = capture.calibration_cases(sealed)[index]
    sampler = sealed['reservation']['sampler_seeds'][str(day)]
    for mode in capture.MODES:
        capture_case(args, actor, day, mode, sampler, args.out_root/'calibration', sealed, service)
    return index


def run_batch(args):
    require_preflight(args)
    sealed, _ = parents(args)
    cases = capture.calibration_cases(sealed)
    require(0 <= args.start < args.end <= len(cases) and 1 <= args.workers <= 8, 'invalid calibration shard')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(pair_case, args, i) for i in range(args.start, args.end)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='calibration_actor_day')), flush=True)
    write(args.out_root/'shards'/f'{args.start}-{args.end}.json', dict(status='v166_calibration_shard_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], protocol_source='4a6d783667',
        start=args.start, end=args.end, calibration_paths=2*(args.end-args.start),
        confirmation_paths=0, interval_fit=False, main_policy_changed=False))
    print('V166_CALIBRATION_SHARD_COMPLETE', args.start, args.end, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run'))
    for name in ('out-root', 'runtime', 'v165-root', 'v161-root', 'v151-root', 'v146-root',
            'archive-root', 'v159-root', 'capture-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'preflight': preflight, 'run': run_batch}[args.command](args)
