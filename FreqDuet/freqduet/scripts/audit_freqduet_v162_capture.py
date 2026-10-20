#!/usr/bin/env python3
"""Read saved calibration queries, support and labels; never fit or replay."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import zip_longest
import json
from math import isfinite
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v162_calibration_capture as capture


read, write, require, close = capture.read, capture.write, capture.require, capture.close
stream = capture.metadata.stream
CAPTURE_SOURCE = '25a2eddeaf'
CHANNELS = {'running', 'service', 'residence'}


def support_record(q, counts):
    now, stage, support = q['query_time_s'], q['native_stage'], q['support']
    require(set(support) == {'service', 'completed_components', 'current_residence', 'v151_detail'},
        'support channels missing')
    kind, reason = stage['kind'], stage['reason']
    counts['stage/'+kind] += 1
    counts['native_reason/'+reason] += 1
    if kind in ('service', 'ready-residence'):
        require(0 <= stage['survivor_n'] <= stage['pool_n'] <= stage['past_n'], 'native support counts differ')
        lag = stage['latest_survivor_lag_s']
        require(lag is None or lag > 0., 'native stage uses same-clock or future support')
        expected = None if reason != 'finite' else (
            'finite_historical_fallback' if stage['historical_fallback'] else 'finite_window')
        require(stage['support_class'] == expected, 'native support class differs')
        require((reason == 'finite') == (stage['survivor_n'] >= 9), 'native rank support threshold differs')
        require(stage['start_time_s'] <= now, 'current stage starts in the future')
    elif kind == 'known-command':
        require(stage['command_time_s'] < now and stage['ready_time_s'] <= now, 'unobserved holding command')
    for call in support['service']:
        close(call['query_time_s'], now)
        require(isfinite(call['predicted_arrival_time_s']), 'service arrival clock missing')
        value = call['support']
        counts['service_calls'] += 1
        if value is None:
            counts['explicit_service_fallback'] += 1
            continue
        latest = value['latest_current_s']
        require(latest is None or latest < now, 'V151 service support not strictly completed')
        require((latest is None) == (value['local_n'] == 0), 'service support clock/count differ')
        require(0 <= value['local_survivor_n'] <= value['local_n'] and
            0 <= value['prior_survivor_n'] <= value['prior_n'] and
            0 <= value['prior_survivor_days'] <= value['prior_days'] <= 8 and value['conditional_mass'] > 0.,
            'historical service support missing or invalid')
    position = q['observable']['position']
    if position is not None:
        require(len(support['completed_components']) == position['remaining_stops']+1,
            'native route support incomplete')
    for stop in support['completed_components']:
        require(set(stop['components']) == {'running', 'service', 'holding', 'ticks'}, 'native channel missing')
        for value in stop['components'].values():
            if value is not None:
                require(value['latest'] < now and value['count'] > 0, 'component support not strictly completed')
                counts['completed_component_supports'] += 1
    residence = support['current_residence']
    if residence is not None:
        require(residence['query_time_s'] == now and residence['latest_selected_s'] < now and
            residence['latest_survivor_s'] < now, 'residence support not strictly completed')
        require(0 < residence['survivor_n'] <= residence['selected_n'], 'residence support missing')
        close(residence['mean_remaining_s'], position['initial_residence_prediction_s'])
        counts['current_residence_supports'] += 1


def query_record(q, residual, row, raw, ready, ranks, sealed, actor, day, mode, cutoff, counts):
    k, now = tuple(q['query_key']), q['query_time_s']
    require((q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query cohort differs')
    require(k == capture.key(row) == tuple(residual['query_key']), 'query/residual/trace order differs')
    capture.validate_snapshot(raw, row, ranks[k])
    require(now == row['prediction_time_s'] == residual['query_time_s'], 'query time differs')
    follower = list(capture.extraction.audit.stats.follower(row))
    require(q['predicted_follower'] == follower and
        q['episode'] == residual['episode'] == [*follower, *k[1:]], 'physical follower/episode differs')
    require(isfinite(raw['own_progress_m']) and isfinite(raw['own_speed_mps']), 'live own AVL missing')
    live = raw['follower']
    if live is not None:
        require(live['bus_id'] == follower[1] and isfinite(live['progress_m']) and
            isfinite(live['speed_mps']) and bool(live['state']), 'live follower AVL incomplete')
        counts['live_follower_snapshots'] += 1
    clocks = q['clocks']
    require(clocks['ready_snapshot_time_s'] == now and clocks['query_ready_order'] == ranks[k] and
        clocks['policy_snapshot_time_s'] == clocks['load_sample_time_s'] == row['policy_time_s'] >= now,
        'query/policy/load sampling clocks differ')
    require(clocks['query_load'] is None and not clocks['query_load_available'] and
        'query_load' not in q['observable'], 'late load entered query features')
    counts['late_load_explicitly_unavailable'] += 1
    counts['policy_after_query'] += int(row['policy_time_s'] > now)
    state = q['observable']
    expected = capture.extraction.observable(row, None, state['position_status'], cutoff)
    for field in ('query_time_s', 'hour', 'demand_end_time_s', 'clearance', 'evidence'):
        require(state[field] == expected[field], 'observable state differs: '+field)
    require(not ({'label', 'truth_coarse', 'residual_s', 'available_time_s'} & set(q)), 'target label in query packet')
    band = capture.horizon_band(q['point_s'])
    condition = dict(actor=actor, mode=mode, query_phase='clearance' if now >= cutoff else 'active',
        predicted_target_phase=None if band is None else 'clearance' if now+q['point_s'] >= cutoff else 'active',
        direction=k[2], target_stop=k[1], position_status=state['position_status'],
        native_stage_kind=q['native_stage']['kind'], native_support_class=q['native_stage']['support_class'],
        predicted_horizon_band=band)
    require(q['condition_values'] == condition, 'conditioning uses different state/clocks')
    eligible = raw['follower_evidence_valid'] and band is not None
    supported = eligible and q['prediction_parts'] is not None and q['native_stage']['support_class'] is not None
    require(q['denominator_eligible'] == eligible and q['native_supported'] == supported, 'abstention denominator differs')
    require(q['candidate_key'] == ([condition[f] for f in sealed['conditioning']['candidate_key']] if supported else None),
        'candidate key missing or changed')
    support_record(q, counts)
    current = q['native_stage'].get('current_key')
    if current is not None:
        current = tuple(current)
        require(ready[current]['bus_id'] == follower[1], 'current stage belongs to another vehicle')
        if state['position_status'] == 'observed_same_clock_ready':
            require(ready[current]['ready_time_s'] == now and ranks[current] < ranks[k],
                'same-clock follower ready callback not yet observed')
            counts['earlier_same_clock_follower_ready'] += 1
    label = residual['label']
    require(label == capture.extraction.label_record(dict(ready=ready), row), 'target identity/availability label differs')
    if label['status'] == 'resolved':
        require(label['available_time_s'] == label['target_ready_time_s'] >= now, 'label availability precedes target')
        close(label['full_eta_s'], label['available_time_s']-now)
        counts['same_clock_target_labels'] += int(label['available_time_s'] == now)
    else:
        require(residual['truth_coarse'] is None and residual['residual_s'] is None, 'unresolved label encoded as residual')
    parts, coarse = q['prediction_parts'], q['prediction_coarse']
    if parts is not None:
        require(set(coarse) == CHANNELS, 'prediction joint channel lost')
        close(sum(parts.values()), q['point_s'])
        close(sum(coarse.values()), q['point_s'])
    joint = label['status'] == 'resolved' and coarse is not None
    require(joint == (residual['residual_s'] is not None), 'joint residual availability differs')
    if joint:
        truth, errors = residual['truth_coarse'], residual['residual_s']
        require(set(truth) == set(errors) == CHANNELS, 'joint residual channel lost')
        for channel in CHANNELS:
            close(coarse[channel]-truth[channel], errors[channel])
        close(sum(truth.values()), label['full_eta_s'])
        close(sum(errors.values()), q['point_s']-label['full_eta_s'])
    counts['queries'] += 1
    counts['denominator_eligible'] += int(eligible)
    counts['native_supported'] += int(supported)
    counts['supported_resolved'] += int(supported and joint)
    counts['joint_resolved'] += int(joint)
    counts['eligible_unresolved'] += int(eligible and label['status'] != 'resolved')
    counts['supported_unresolved'] += int(supported and label['status'] != 'resolved')
    counts['eligible_native_abstentions'] += int(eligible and not supported)
    counts['label/'+label['status']] += 1
    counts['queries/'+condition['query_phase']] += 1
    counts['queries/hour'+str(state['hour'])] += 1


def audit_path(root, sealed, actor, day, mode):
    name = f'{actor}_{day}_{mode}'
    receipt = read(root/'paths'/(name+'.json'))
    cell = read(root/'capture_cells'/(name+'.json'))
    for record in (receipt, cell):
        require((record['seed'], record['day'], record['mode'], record['sampler_seed']) ==
            (actor, day, mode, sealed['reservation']['sampler_seeds'][str(day)]), 'path assignment differs')
        require(record['execution_source'] == CAPTURE_SOURCE and not record['main_policy_changed'] and
            all(record[f+'_updates'] == 0 for f in ('actor', 'critic', 'dual')), 'capture source/controller changed')
    require(cell['all_queries_recorded_before_target_labels'] and not cell['interval_fit'], 'capture workflow changed')
    for directory, suffix in (('records', '.npz'), ('terminal', '.json')):
        require((root/directory/(name+suffix)).is_file(), 'capture artifact missing: '+directory)
    ready_rows = list(stream(root/'ready_labels'/(name+'.jsonl.gz')))
    ready = {capture.key(r): r for r in ready_rows}
    ranks = {k: i for i, k in enumerate(ready)}
    raw_rows = list(stream(root/'query_snapshots'/(name+'.jsonl.gz')))
    snapshots = {tuple(r['query_key']): r for r in raw_rows}
    require(len(ready_rows) == len(ready) == receipt['ready_labels'] == len(raw_rows) ==
        len(snapshots) == receipt['live_query_snapshots'] == cell['live_ready_snapshots'], 'ready snapshot count differs')
    require(set(ready) == set(snapshots), 'ready snapshot identities incomplete')
    counts, seen, keys = Counter(), set(), Counter()
    cutoff = receipt['contract']['clock']['demand_end_time_s']
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals', 'traces')]
    for q, residual, row in zip_longest(*sources):
        require(q is not None and residual is not None and row is not None, 'query/residual/trace count differs')
        k = tuple(q['query_key'])
        require(k not in seen, 'duplicate query key')
        seen.add(k)
        query_record(q, residual, row, snapshots[k], ready, ranks, sealed, actor, day, mode, cutoff, counts)
        if q['native_supported'] and residual['residual_s'] is not None:
            keys[json.dumps(q['candidate_key'], separators=(',', ':'))] += 1
    require(counts['queries'] == receipt['decisions'] == cell['query_record_count'], 'lower query cohort incomplete')
    for field, count in cell['counts'].items():
        require(counts[field] == count, 'capture counter differs: '+field)
    return dict(actor=actor, day=day, mode=mode, counts=dict(counts), key_counts=dict(keys), status='passed')


def audit_case(args, sealed, actor, day, mode):
    try:
        result = audit_path(args.capture_root/'calibration', sealed, actor, day, mode)
    except Exception as exc:
        result = dict(actor=actor, day=day, mode=mode, status='failed', error=type(exc).__name__+': '+str(exc))
    write(args.out_root/'paths'/f'{actor}_{day}_{mode}.json', result)
    return result


def marginals(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return [dict(group=k, paths=len(v), counts=dict(sum((Counter(r['counts']) for r in v), Counter())))
        for k, v in sorted(groups.items())]


def key_support(rows, minimum):
    days, queries = defaultdict(set), Counter()
    for row in rows:
        for key, n in row['key_counts'].items():
            days[key].add(row['day'])
            queries[key] += n
    result = {}
    for mode in capture.MODES:
        keys = [k for k in days if json.loads(k)[1] == mode]
        result[mode] = dict(keys=len(keys), keys_at_minimum=sum(len(days[k]) >= minimum for k in keys),
            distinct_day_distribution=dict(sorted(Counter(len(days[k]) for k in keys).items())),
            supported_resolved_queries=sum(queries[k] for k in keys),
            queries_in_keys_at_minimum=sum(queries[k] for k in keys if len(days[k]) >= minimum))
    return result


def main(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v162_capture_audit.py', '-v'], check=True)
    sealed, _ = capture.parent_protocol(args)
    proof = read(args.capture_root/'preflight/summary/status.json')
    require(proof['status'] == 'v162_capture_preflight_passed' and proof['execution_source'] == CAPTURE_SOURCE,
        'original capture preflight missing')
    expected = {(a, d, m) for a, d in capture.calibration_cases(sealed) for m in capture.MODES}
    root = args.capture_root/'calibration'
    names = {f'{a}_{d}_{m}.json' for a, d, m in expected}
    require({p.name for p in (root/'capture_cells').glob('*.json')} == names, 'calibration matrix incomplete or contaminated')
    require({p.name for p in (root/'paths').glob('*.json')} == names, 'path matrix incomplete or contaminated')
    shards = [read(p) for p in (args.capture_root/'shards').glob('*.json')]
    covered = [i for s in shards for i in range(s['start'], s['end'])]
    require(sorted(covered) == list(range(len(capture.calibration_cases(sealed)))), 'shard gaps or overlap')
    require(all(s['status'] == 'v162_calibration_shard_complete' and s['confirmation_paths'] == 0 and
        not s['interval_fit'] for s in shards), 'shard completion/freeze differs')
    require(not (args.capture_root/'confirmation').exists(), 'unexpected confirmation collection directory')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(audit_case, args, sealed, *case) for case in sorted(expected)]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            if done % 32 == 0:
                print(f'V162_AUDIT_PROGRESS {done}/{len(jobs)}', flush=True)
    failures = [r for r in rows if r['status'] != 'passed']
    passed = [r for r in rows if r['status'] == 'passed']
    counts = dict(sum((Counter(r['counts']) for r in passed), Counter()))
    summary = dict(status='v162_readback_passed' if not failures else 'v162_readback_failed',
        capture_source=CAPTURE_SOURCE, audit_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=sealed['reservation']['calibration_days'], actors=sealed['reservation']['actors'],
        modes=list(capture.MODES), expected_paths=len(expected), passed_paths=len(passed), failed_paths=len(failures),
        failures=failures[:12], shard_count=len(shards), counts=counts,
        by_mode=marginals(passed, 'mode'), key_day_support=key_support(passed, sealed['support']['minimum_calibration_days']),
        confirmation_paths=0, interval_fit=False, replay=False, training=False,
        main_policy_changed=False, v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1])
    write(args.out_root/'summary/status.json', summary)
    for field in ('actor', 'day'):
        data = marginals(passed, field)
        fields = sorted({k for r in data for k in r['counts']})
        with (args.out_root/'summary'/('by_'+field+'.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[field, 'paths', *fields])
            writer.writeheader()
            writer.writerows({field: r['group'], 'paths': r['paths'], **r['counts']} for r in data)
    print('V162_READBACK_COMPLETE', summary['status'], 'paths', len(passed), 'failed', len(failures), flush=True)
    require(not failures, 'saved calibration metadata failed readback; inspect server path receipts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'out-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    main(parser.parse_args())
