#!/usr/bin/env python3
"""Shadow stop-phase ETA on saved V137 paths; no replay or action selection."""

import argparse
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v139_fallback_phase as phase
from freqduet_stop_phase_eta import ServiceResiduals, from_stop


prior, model = phase.prior, phase.model
read, write, close, key = phase.read, phase.write, phase.close, phase.key


def metrics(rows):
    if not rows:
        return dict(n=0, baseline_mae_s=None, shadow_mae_s=None, paired_absolute_error_change_s=None)
    b, c, truth = [np.asarray([r[k] for r in rows]) for k in ('baseline_s', 'shadow_s', 'truth_s')]
    return dict(n=len(rows), baseline_mae_s=float(np.mean(abs(b-truth))),
        shadow_mae_s=float(np.mean(abs(c-truth))),
        baseline_bias_s=float(np.mean(b-truth)), shadow_bias_s=float(np.mean(c-truth)),
        paired_absolute_error_change_s=float(np.mean(abs(c-truth)-abs(b-truth))))


def prepare(path, geometry):
    launches = {trip: dict(time_s=r['actual_s']) for trip, r in path['dispatch'].items()}
    history = model.build_history(path['arrivals'], path['departures'], path['ready'], launches, geometry)
    services = ServiceResiduals([(k[2], k[1], path['ready'][k]['ready_time_s'],
        path['ready'][k]['ready_time_s']-r['time_s']) for k, r in path['arrivals'].items()])
    for k, r in path['trace'].items():
        close(r['policy_time_s']+max(1., r['chosen_s']), path['departures'][k]['time_s'],
            'saved command reproduces known execution finish clock')
    return history, services, phase.timelines(path, geometry)


def predict(path, row, indexed, geometry, history, services):
    e = row['evidence']
    retained = dict(applied=False, status='unchanged_nonstop_or_other_source')
    if e['follower_source'] != model.clock.JOURNEY or row['original_prediction']['status'] != 'not_causally_moving':
        return retained
    trip, now, direction = e['predicted_follower_trip_id'], e['ready_time_s'], bool(row['direction'])
    motion = indexed[trip].motion
    i, j = bisect_left(motion.dt, now), bisect_left(motion.at, now)
    if not j or (i and motion.at[j-1] <= motion.dt[i-1]):
        return dict(applied=False, status='unchanged_no_completed_pending_stop')
    arrival = motion.arrivals[j-1]
    k = key(arrival)
    route = geometry[direction]
    result = from_stop(history, services, route, direction, route['index'][k[1]],
        route['index'][int(row['station_id'])], now, arrival['time_s'],
        path['ready'][k]['ready_time_s'], path['trace'].get(k))
    # The future labels below are attached only after this function returns.
    return dict(result, current_stop_key=list(k))


def read_path(args, seed, day, mode, geometry):
    path = prior.load_path(args.v137_root, args.v135_root, seed, day, mode)
    history, services, indexed = prepare(path, geometry)
    predictions, rows, unavailable = {}, [], Counter()
    for k, r in path['trace'].items():
        result = predict(path, r, indexed, geometry, history, services)
        predictions[k] = result
        e, baseline = r['evidence'], r['online']['gap_s']
        target = e['predicted_follower_trip_id'], k[1], k[2]
        label = path['ready'].get(target)
        if baseline is None or label is None or label['bus_id'] != e['predicted_follower_bus_id']:
            unavailable['missing_forecast_or_same_vehicle_label'] += 1
            continue
        truth = label['ready_time_s']-e['ready_time_s']
        if truth < 0:
            unavailable['same_follower_already_ready'] += 1
            continue
        record = dict(trip_id=k[0], station_id=k[1], direction=int(k[2]),
            baseline_s=baseline, shadow_s=result['gap_s'] if result['applied'] else baseline,
            truth_s=truth, applied=result['applied'], forecast=result,
            original_fallback=r['original_prediction']['status'] == 'not_causally_moving')
        if result['applied'] and result['service_residual'] is not None:
            current = tuple(result['current_stop_key'])
            record['service_remaining_truth_s'] = path['ready'][current]['ready_time_s']-e['ready_time_s']
        rows.append(record)
    applied = [r for r in rows if r['applied']]
    residuals = [r for r in applied if 'service_remaining_truth_s' in r]
    status = dict(seed=seed, day=day, mode=mode, decisions=len(path['trace']),
        outcomes=metrics(rows), original_fallback_support=metrics([r for r in rows if r['original_fallback']]),
        applied_support=metrics(applied), by_phase={p: metrics([r for r in applied if r['forecast']['phase'] == p])
            for p in sorted({r['forecast']['phase'] for r in applied})},
        statuses=dict(Counter(r['status'] for r in predictions.values())), unavailable_labels=dict(unavailable),
        service_residual_interval=dict(n=len(residuals), coverage_p10_p90=None if not residuals else
            float(np.mean([r['forecast']['service_residual']['p10'] <= r['service_remaining_truth_s'] <=
                r['forecast']['service_residual']['p90'] for r in residuals]))))
    write(args.out_root/'details'/f'{seed}_{day}_{mode}.json', dict(summary=status, rows=rows))
    print('V140_PASSIVE_PATH_DONE', seed, day, mode, flush=True)
    return path, rows, status


def pair(args, seed, day, levels):
    source = read(args.v139_root/'details'/f'{seed}_{day}.json')['changed_source_actions']
    lookup = {m: {key(r): r for r in levels[m][1]} for m in levels}
    rows = []
    for r in source:
        b, c = [lookup[m][key(r)] for m in ('old', 'handoff')]
        rows.append(dict(trip_id=r['trip_id'], station_id=r['station_id'], direction=r['direction'],
            switch=r['switch'], baseline_s=c['baseline_s']-b['baseline_s'],
            shadow_s=c['shadow_s']-b['shadow_s'], truth_s=c['truth_s']-b['truth_s'],
            stop_extension_applied=b['applied'] or c['applied']))
    summary = dict(seed=seed, day=day, source_switch_gap_change=metrics(rows),
        supported_extension=metrics([r for r in rows if r['stop_extension_applied']]),
        by_switch={s: metrics([r for r in rows if r['switch'] == s]) for s in sorted({r['switch'] for r in rows})},
        largest_errors=sorted(rows, key=lambda r: -abs(r['shadow_s']-r['truth_s']))[:3])
    write(args.out_root/'details'/f'{seed}_{day}_paired.json', dict(summary=summary, rows=rows))
    return rows, summary


def run(args):
    parent = read(args.v139_root/'summary/status.json')
    if parent['status'] != 'v139_readonly_fallback_phase_complete' or parent['main_policy_changed']:
        raise AssertionError('requires completed frozen V139 phase diagnosis')
    geometry = model.load_geometry(args.runtime)
    paths, pairs, pooled_rows, pooled_pairs = [], [], [], []
    for case in parent['cases']:
        seed, day = case['seed'], case['day']
        levels = {m: read_path(args, seed, day, m, geometry) for m in ('old', 'handoff')}
        paths.extend(levels[m][2] for m in levels)
        pooled_rows.extend(r for m in levels for r in levels[m][1])
        pair_rows, summary = pair(args, seed, day, levels)
        pairs.append(summary)
        pooled_pairs.extend(pair_rows)
    write(args.out_root/'summary/status.json', dict(status='v140_passive_stop_eta_complete', paths=paths, pairs=pairs,
        applied_support=metrics([r for r in pooled_rows if r['applied']]),
        original_fallback_support=metrics([r for r in pooled_rows if r['original_fallback']]),
        source_switch_gap_change=metrics(pooled_pairs),
        consumed_paths=12, new_rollouts=0, new_demand_days=0, actor_updates=0, critic_updates=0,
        dual_updates=0, action_selection_calls=0, main_policy_changed=False,
        limitations='Consumed outcome-selected development paths, not independent confirmation or realized '
            'control gains. One fixed shadow extension; no parameter search. Original one-hour completed '
            'history window/all-past fallback is unchanged. Unsupported conditional residuals retain native '
            'ETA and are counted. The service interval covers current residual service only, not full ETA '
            'uncertainty. Future current-vehicle ready/departure and command values are excluded until '
            'strictly observed. Future labels are read only for scoring.'))
    print('V140_PASSIVE_STOP_ETA_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'v139-root', 'runtime', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v140_*.py', '-v'], check=True)
    run(args)
