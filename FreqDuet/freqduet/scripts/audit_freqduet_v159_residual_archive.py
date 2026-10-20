#!/usr/bin/env python3
"""Read back V159 without forecasting, replaying or fitting any interval."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from itertools import combinations_with_replacement, zip_longest
import json
from math import isclose, isfinite
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

import extract_freqduet_v159_joint_residuals as extract


read, write = extract.read, extract.write
frozen, audit = extract.frozen, extract.audit


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b):
    require(isfinite(a) and isfinite(b) and isclose(a, b, rel_tol=0., abs_tol=1e-7),
        f'residual readback differs: {a} != {b}')


def stream(path):
    with gzip.open(path, 'rt', encoding='ascii') as handle:
        for line in handle:
            yield json.loads(line)


def metrics(episode, errors, full):
    close(sum(errors.values()), full)
    result = dict(episode=episode, full_bias_s=full, full_mae_s=abs(full),
        full_mse_s2=full*full, cancellation_s=(sum(abs(v) for v in errors.values())-abs(full))/2.)
    result.update({'e_'+k: v for k, v in errors.items()})
    result.update({'m_'+a+'__'+b: errors[a]*errors[b]
        for a, b in combinations_with_replacement(errors, 2)})
    return result


def describe(rows, selected):
    frame = pd.DataFrame([r for r, keep in zip(rows, selected) if keep])
    require(not frame.empty, 'expected V158 cohort became empty')
    grouped = frame.groupby('episode', sort=False).mean(numeric_only=True)
    return dict(queries=len(frame), episodes=len(grouped), metrics=grouped.mean().to_dict())


def query_record(record, row, ready, ranks, cutoff, geometry, counts):
    key, now = tuple(record['query_key']), record['query_time_s']
    require(key == audit.stats.key(row), 'saved query identity/order changed')
    close(now, row['prediction_time_s'])
    close(now, row['evidence']['ready_time_s'])
    require(record['query_ready_order'] == ranks[key], 'query callback order changed')
    require(record['predicted_follower'] == list(audit.stats.follower(row)), 'follower identity changed')
    require(record['episode'] == [*audit.stats.follower(row), *key[1:]], 'episode membership changed')
    expected_label = extract.label_record(dict(ready=ready), row)
    require(record['label'] == expected_label, 'same-vehicle label/availability clock differs')
    state = record['observable']
    expected = extract.observable(row, None, state['position_status'], cutoff)
    for field in ('query_time_s', 'hour', 'demand_end_time_s', 'clearance', 'evidence'):
        require(state[field] == expected[field], 'query-observable state changed: '+field)
    require(set(state) <= set(expected) | {'position'}, 'unregistered query-state field')
    require(('query_load' in state) == ('query_load' in expected), 'later policy load entered query state')
    if 'query_load' in state:
        require(state['query_load'] == expected['query_load'], 'query load differs')
    counts['policy_callback_after_query'] += int(row['policy_time_s'] > now)
    counts['query_load_present'] += int('query_load' in state and state['query_load'] is not None)
    counts['query_validity_not_exported'] += int('follower_evidence_valid' not in state)
    if record['label']['status'] == 'resolved':
        available = record['label']['available_time_s']
        require(available >= now, 'label claims availability before future target-ready')
        counts['label_available_same_clock'] += int(available == now)
    else:
        require(record['forecasts'] is None and record['components'] is None and record['support'] is None,
            'unresolved target was encoded as a scored residual')
    if state['position'] is not None:
        p, route = state['position'], geometry[bool(key[2])]
        target = route['index'][key[1]]
        require(p['target_index'] == target and 0 <= p['origin_index'] <= target, 'invalid position indices')
        require(p['remaining_stops'] == target-p['origin_index'], 'remaining-stop state differs')
        close(p['remaining_distance_m'], route['positions'][target]-p['progress_m'])
        require(p['remaining_distance_m'] >= -1e-6 and p['initial_residence_prediction_s'] >= 0.,
            'invalid observable geometry/residence')
        if state['position_status'] == 'observed_same_clock_ready':
            current = tuple(record['components']['v151']['inseparable_current_ready_key'])
            require(ready[current]['ready_time_s'] == now and ranks[current] < ranks[key],
                'same-clock state used a later callback')
            counts['same_clock_ready_state_verified'] += 1


def support_record(record, counts):
    now, support = record['query_time_s'], record['support']
    if record['forecasts'] is None:
        return
    require(support is not None, 'forecast lost its support record')
    for call in support['service']:
        close(call['query_time_s'], now)
        value = call['support']
        counts['service_calls'] += 1
        if value is None:
            counts['service_call_native_fallback'] += 1
            continue
        for field in ('local_n', 'local_survivor_n', 'prior_n', 'prior_days',
                'prior_survivor_n', 'prior_survivor_days', 'latest_current_s',
                'historical_fallback', 'conditional_mass'):
            require(field in value, 'service support missing '+field)
        latest = value['latest_current_s']
        require((latest is None) == (value['local_n'] == 0), 'service local support clock/count differ')
        require(latest is None or latest < now, 'service support includes an unfinished label')
        require(0 <= value['local_survivor_n'] <= value['local_n'] and
            0 <= value['prior_survivor_days'] <= value['prior_days'] <= 8 and
            0 <= value['prior_survivor_n'] <= value['prior_n'] and value['conditional_mass'] > 0.,
            'service support counts differ')
        counts['service_low_prior_survivor_days'] += int(value['prior_survivor_days'] < 8)
    for call in support['running']:
        prefix, prior = call['current_prefix'], call['fixed_prior']
        require(prefix['count'] > 0 and prefix['latest_age_s'] > 0., 'running prefix lacks completed support')
        require(prior['latest_current_s'] < now, 'running support includes unfinished label')
        close(now-prior['latest_current_s'], prefix['latest_age_s'])
        n = prior['prior_days']
        require(0 <= n <= 8, 'running prior-day support differs')
        close(prior['historical_weight'], n/(n+1))
        counts['running_calls'] += 1
        counts['running_zero_prior_days'] += int(n == 0)
        counts['running_single_prior_day'] += int(n == 1)
    if record['components'] is None:
        counts['opaque_position_not_reconstructed'] += int(record['observable']['position'] is None)
        return
    position = record['observable']['position']
    require(position is not None, 'structured residual lacks observable position')
    require(len(support['completed_components']) == position['remaining_stops']+1,
        'completed component support misses route positions')
    for stop in support['completed_components']:
        require(set(stop['components']) == {'running', 'service', 'holding', 'ticks'},
            'completed support channel lost')
        for channel, value in stop['components'].items():
            if value is None:
                counts['explicit_absent_native_support/'+channel] += 1
            else:
                require(value['latest'] < now and value['count'] > 0,
                    'native component support includes unfinished label')
                counts['completed_component_supports'] += 1
    counts['shadow_service_calls_not_separately_exported'] += int('eight_day_service' not in support)
    counts['shadow_service_mean_changed'] += int(abs(record['components']['v151']['prediction_s']['service']-
        record['components']['eight_day']['prediction_s']['service']) > 1e-7)
    if not record['components']['v151']['holding_separable']:
        counts['joint_residence_conditional_support_not_exported'] += int('current_residence' not in support)


def component_record(record):
    label, forecasts, packets = record['label'], record['forecasts'], record['components']
    if forecasts is None:
        require(packets is None, 'components without full forecast')
        return {}
    require(set(forecasts) == {'v151', 'eight_day'}, 'matched prediction arm lost')
    for forecast in forecasts.values():
        close(forecast['point_s']-label['full_eta_s'], forecast['residual_s'])
    if packets is None:
        return {}
    require(set(packets) == set(forecasts), 'component arms differ')
    result = {}
    for name, packet in packets.items():
        prediction, truth, errors = [packet[k] for k in ('prediction_s', 'truth_s', 'residual_s')]
        require(set(prediction) == set(truth) == set(errors) == set(frozen.COARSE), 'joint channel lost')
        close(sum(prediction.values()), forecasts[name]['point_s'])
        close(sum(truth.values()), label['full_eta_s'])
        for channel in frozen.COARSE:
            close(prediction[channel]-truth[channel], errors[channel])
        result[name+'/coarse'] = metrics(tuple(record['episode']), errors, forecasts[name]['residual_s'])
        if packet['holding_separable']:
            require(packet['inseparable_current_ready_key'] is None, 'holding identifiability differs')
            h, ticks = packet['holding_residual_s'], packet['execution_wait_residual_s']
            close(h+ticks, errors['residence'])
            result[name+'/holding_identified'] = metrics(tuple(record['episode']),
                dict(running=errors['running'], service=errors['service'], holding=h, execution_wait=ticks),
                forecasts[name]['residual_s'])
        else:
            require(packet['holding_residual_s'] is None and packet['execution_wait_residual_s'] is None,
                'inseparable residence mislabeled as pure holding')
    require(packets['v151']['truth_s'] == packets['eight_day']['truth_s'], 'matched target truth differs')
    return result


def case(args, seed, day):
    geometry = audit.frozen.native.model.load_geometry(args.runtime)
    expected = read(args.v158_root/'cells'/f'{seed}_{day}.json')
    receipt = read(args.v159_root/'cells'/f'{seed}_{day}.json')
    counts, extracted_counts = Counter(), Counter()
    for mode in ('old', 'handoff'):
        name = f'{seed}_{day}_{mode}'
        ready = {audit.stats.key(r): r for r in stream(args.archive_root/'ready_labels'/(name+'.jsonl.gz'))}
        ranks = {k: i for i, k in enumerate(ready)}
        cutoff = read(args.archive_root/'paths'/(name+'.json'))['contract']['clock']['demand_end_time_s']
        groups, times = defaultdict(list), defaultdict(list)
        for order, (row, record) in enumerate(zip_longest(
                stream(args.archive_root/'traces'/(name+'.jsonl.gz')),
                stream(args.v159_root/'residuals'/(name+'.jsonl.gz')))):
            require(row is not None and record is not None, 'raw/saved query count differs')
            require((record['actor'], record['day'], record['mode'], record['query_order']) ==
                (seed, day, mode, order), 'residual path identity/order differs')
            local = Counter()
            query_record(record, row, ready, ranks, cutoff, geometry, local)
            support_record(record, local)
            for field, entry in component_record(record).items():
                groups[field].append(entry)
                times[field].append(record['query_time_s'])
            for field, count in local.items():
                counts[mode+'/'+field] += count
            extracted_counts[mode+'/queries'] += 1
            extracted_counts[mode+'/label/'+record['label']['status']] += 1
            if record['components'] is not None:
                extracted_counts[mode+'/structured'] += 1
                extracted_counts[mode+'/joint_residence_not_pure_holding'] += int(
                    not record['components']['v151']['holding_separable'])
            elif record['forecasts'] is not None:
                extracted_counts[mode+'/opaque_full_endpoint'] += 1
        reproduced = {}
        for field, rows in groups.items():
            ts = times[field]
            for scope in frozen.SCOPES:
                selected = [scope == 'all' or (scope == 'active' and t < cutoff) or
                    (scope == 'clearance' and t >= cutoff) or scope == 'hour'+str(int(t//3600)) for t in ts]
                if any(selected):
                    reproduced[field+'/'+scope] = describe(rows, selected)
        extract.reconcile(reproduced, {g: v for g, v in expected['modes'][mode]['groups'].items()
            if '/coarse/' in g or '/holding_identified/' in g})
    require(dict(extracted_counts) == receipt['counts'], 'streamed residual counts differ from V159 receipt')
    write(args.out_root/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day,
        counts=dict(counts), residual_counts=dict(extracted_counts), v158_readback_reproduced=True))


def run(args):
    parent = read(args.v159_root/'summary/status.json')
    plan = parent['registration']
    require(parent['status'] == 'v159_joint_residual_extraction_complete' and
        plan['execution_source'] == '6b5dfe58a8', 'requires original completed V159 archive')
    require((plan['window_s'], plan['tails'], plan['v146_extensions_s']) ==
        (3600., [.1, .1], [3., 0.]), 'frozen interval contract differs')
    cases = [(seed, day) for seed in plan['actors'] for day in plan['days']]
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v159_readback.py', '-v'], check=True)
    case(args, *cases[0])
    print('V159_READBACK_FIRST_PAIR_VERIFIED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args, s, d) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V159_READBACK_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    service_registration = read(args.v159_root.parents[1]/'protocol_v6_v151_historical_service/protocol.json')
    write(args.out_root/'summary/status.json', dict(status='v159_residual_readback_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cases), paths=2*len(cases),
        residual_counts=dict(sum((Counter(c['residual_counts']) for c in cells), Counter())),
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        core_residual_clock_state_checks_passed=True, v158_readback_reproduced=True,
        independent_inventory=parent['inventory'], point_forecasts_recomputed=False,
        known_used_cohorts=dict(v145_development_days=plan['days'],
            v151_prior_fit_days=service_registration['calibration_days'],
            v151_previous_confirmation_days=service_registration['confirmation_days']),
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        interval_fit=False, main_policy_changed=False, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1],
        limitations='Previously exposed development data. V159 has V151 service-call support but no '
            'separate eight-day service-call log or current joint-residence conditional support. '
            'Query-time follower-validity is recoverable from raw traces, not exported as a state field. '
            'Opaque full residuals remain outside the joint-component cohort. '
            'Strict label eligibility is available_time_s < subsequent query time, including clock ties.'))
    print('V159_RESIDUAL_READBACK_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v159-root', 'v158-root', 'out-root', 'runtime'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())
