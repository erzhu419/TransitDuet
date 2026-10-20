#!/usr/bin/env python3
"""Account for frozen V141 clearance and source-switch errors on saved paths."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v141_ready_eta as frozen
import audit_freqduet_v147_full_eta as prior


PARTS = ('current_service', 'running', 'downstream_service', 'holding',
    'execution_wait', 'current_ready_joint', 'unallocated')
read, write, key, close = prior.read, prior.write, prior.key, prior.close


def zeros():
    return dict.fromkeys(PARTS, 0.)


def overlap(start, stop, now, end):
    return max(0., min(stop, end)-max(start, now))


def actual_parts(path, row, route, joint_stop=None):
    trip, now = prior.follower(row)[0], row['prediction_time_s']
    target = (trip, int(row['station_id']), bool(row['direction']))
    end = path['ready'][target]['ready_time_s']
    parts, previous = zeros(), path['dispatch'][trip]['actual_s']
    for station in route['ids'][1:route['index'][target[1]]+1]:
        k = trip, station, target[2]
        arrival, departure = path['arrivals'][k], path['departures'][k]
        ready = path['ready'][k]['ready_time_s']
        parts['running'] += overlap(previous, arrival['time_s'], now, end)
        channel = 'current_service' if arrival['time_s'] < now <= ready else 'downstream_service'
        parts[channel] += overlap(arrival['time_s'], ready, now, end)
        if k == joint_stop:
            parts['current_ready_joint'] += overlap(ready, departure['time_s'], now, end)
        else:
            held = departure['realized_holding']-arrival['realized_holding']
            hold_start = departure['time_s']-held
            if hold_start < ready-1e-7:
                raise AssertionError('executed holding does not fit ready/departure clocks')
            parts['holding'] += overlap(hold_start, departure['time_s'], now, end)
            parts['execution_wait'] += overlap(ready, hold_start, now, end)
        previous = departure['time_s']
    close(sum(parts.values()), end-now)
    return parts


def add_tail(parts, tail):
    parts['running'] += tail['running_s']
    parts['downstream_service'] += tail['service_s']+tail['target_service_s']
    parts['holding'] += tail['holding_s']
    parts['execution_wait'] += tail['ticks_s']


def component_packet(path, row, saved, geometry, history, services, durations, indexed):
    stop = frozen.frozen.predict(path, row, indexed, geometry, history, services)
    now, direction = row['prediction_time_s'], bool(row['direction'])
    route, packet, joint = geometry[direction], zeros(), None
    phase = row['online']['status']
    if stop['applied'] or stop['status'] == 'ready_without_prior_command':
        current = tuple(stop['current_stop_key'])
        origin, target = route['index'][current[1]], route['index'][row['station_id']]
        if stop['applied']:
            close(stop['gap_s'], saved['v140_s'])
            phase = stop['phase']
            if phase == 'service_in_progress':
                packet['current_service'] = stop['service_residual']['mean']
                if origin < target:
                    packet['holding'] = history.estimate(('holding', direction, current[1]), now)['mean']
                    packet['execution_wait'] = history.estimate(('ticks', direction, current[1]), now)['mean']
            else:
                command = path['trace'][current]
                channel = 'holding' if command['chosen_s'] > 0 else 'execution_wait'
                packet[channel] = stop['current_remaining_s']
            close(sum(packet.values()), stop['current_remaining_s'])
        else:
            ready = frozen.predict_ready(history, durations, route, direction, origin, target,
                now, path['ready'][current]['ready_time_s'])
            if not ready['applied']:
                return None, None, 'unallocated/'+ready['status']
            close(ready['gap_s'], saved['v141_s'])
            packet['current_ready_joint'], joint, phase = ready['ready_residual']['mean'], current, 'ready_joint'
        if origin < target:
            tail = frozen.model.component_prediction(history, route, direction, origin, target,
                float(route['positions'][origin]), now)
            add_tail(packet, tail)
    elif row['online']['component_used']:
        add_tail(packet, row['online'])
    else:
        return None, None, 'unallocated/'+phase
    close(sum(packet.values()), saved['v141_s'])
    return packet, joint, phase


def error_record(episode, error, parts, structured):
    close(sum(parts.values()), error)
    sign = float(np.sign(error))
    return dict(episode=episode, full_bias_s=error, full_mae_s=abs(error),
        structured_fraction=float(structured),
        **{p+'_bias_s': parts[p] for p in PARTS},
        **{p+'_mae_allocation_s': sign*parts[p] for p in PARTS})


def decompose(path, row, saved, truth, geometry, history, services, durations, indexed):
    prediction, joint, phase = component_packet(path, row, saved, geometry,
        history, services, durations, indexed)
    error = saved['v141_s']-truth
    if prediction is None:
        parts = zeros()
        parts['unallocated'] = error
    else:
        actual = actual_parts(path, row, geometry[bool(row['direction'])], joint)
        parts = {p: prediction[p]-actual[p] for p in PARTS}
    return parts, prediction is not None, phase


def prepare(root, seed, day, mode, geometry):
    name = f'{seed}_{day}_{mode}'
    path = frozen.prior.load_path(root, root, seed, day, mode)
    path['shadows'] = {tuple(r['query']): r for r in prior.trace(root/'shadows'/(name+'.jsonl.gz'))}
    history, services, indexed = frozen.frozen.prepare(path, geometry)
    durations = frozen.ServiceResiduals([(k[2], k[1], r['time_s'],
        r['time_s']-path['ready'][k]['ready_time_s']) for k, r in path['departures'].items()])
    path['cutoff'] = read(root/'paths'/(name+'.json'))['contract']['clock']['demand_end_time_s']
    return path, (history, services, durations, indexed)


def labelled(path, k):
    row, saved = path['trace'][k], path['shadows'][k]
    status, truth = prior.target_label(row, path['ready'])
    if status != 'resolved' or any(saved[v+'_s'] is None for v in prior.VERSIONS):
        return None
    return truth


def burden(rows, denominator):
    fields = set(error_record([0], 0., zeros(), False))-{'episode'}
    return dict(queries=denominator, episodes=denominator,
        metrics={f: sum(r[f] for r in rows)/denominator for f in fields})


def pair_case(root, v147, out, seed, day, geometry):
    target = out/'cells'/f'{seed}_{day}.json'
    if target.exists():
        if read(target)['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
            raise AssertionError('cached component audit has a different source')
        return
    expected = read(v147/'cells'/f'{seed}_{day}.json')
    paths, objects = {}, {}
    for mode in ('old', 'handoff'):
        paths[mode], objects[mode] = prepare(root, seed, day, mode, geometry)
    switch = set()
    for k in paths['old']['trace'].keys() & paths['handoff']['trace'].keys():
        b, c = [paths[m]['trace'][k] for m in ('old', 'handoff')]
        sb, sc = [paths[m]['shadows'][k] for m in ('old', 'handoff')]
        if prior.source(b, sb) != prior.source(c, sc) and all(labelled(paths[m], k) is not None for m in paths):
            if prior.follower(b) != prior.follower(c):
                raise AssertionError('V147 matched-identity switch cohort changed')
            switch.add(k)
    if len(switch) != expected['counts'].get('identity_same/source_changed', 0):
        raise AssertionError('source-switch support differs from V147')
    modes, decomposed, counts = {}, {}, Counter()
    for mode, path in paths.items():
        clearance, selected = [], {}
        for k, row in path['trace'].items():
            in_clearance = row['prediction_time_s'] >= path['cutoff']
            if not in_clearance and k not in switch:
                continue
            truth = labelled(path, k)
            if truth is None:
                continue
            saved = path['shadows'][k]
            parts, structured, phase = decompose(path, row, saved, truth, geometry, *objects[mode])
            error = saved['v141_s']-truth
            selected[k] = (parts, structured, phase, error)
            if in_clearance:
                episode = (*prior.follower(row), *k[1:])
                clearance.append(error_record(episode, error, parts, structured))
                counts[mode+'/clearance_phase/'+phase] += 1
        stats = prior.describe(clearance)
        previous = expected['modes'][mode]['stats']['clearance']
        if stats['queries'] != previous['queries']:
            raise AssertionError('clearance query support differs from V147')
        close(stats['metrics']['full_bias_s'], previous['metrics']['v141_bias_s'])
        close(stats['metrics']['full_mae_s'], previous['metrics']['v141_mae_s'])
        modes[mode], decomposed[mode] = dict(clearance=stats), selected
    rows = []
    for k in switch:
        b, c = [decomposed[m][k] for m in ('old', 'handoff')]
        parts = {p: c[0][p]-b[0][p] for p in PARTS}
        rows.append(error_record(k, c[3]-b[3], parts, b[1] and c[1]))
        counts['switch_phase/'+b[2]+' -> '+c[2]] += 1
    paired = dict(switch_burden=burden(rows, expected['paired']['paired/all']['queries']))
    if rows:
        paired['switch_conditional'] = prior.describe(rows)
        close(paired['switch_conditional']['metrics']['full_mae_s'], expected['paired'][
            'paired/identity_same/source_changed']['metrics']['v141_delta_mae_s'])
    write(target, dict(seed=seed, day=day, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        modes=modes, paired=paired, counts=dict(counts), v147_endpoints_and_support_reproduced=True))


def run(args):
    parent = read(args.v147_root/'summary/status.json')
    model = read(args.v146_root/'model.json')
    if parent['status'] != 'v147_readonly_full_eta_complete' or (
            model['lower_extension_s'], model['upper_extension_s']) != (3., 0.):
        raise AssertionError('requires completed V147 and frozen V146')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v148_eta_components.py', '-v'], check=True)
    geometry = frozen.model.load_geometry(args.runtime)
    cases = list(itertools.product(parent['actors'], parent['days']))
    root = args.v146_root/'confirmation'
    pair_case(root, args.v147_root, args.out_root, *cases[0], geometry)
    print('V148_SAVED_PAIR_PREFLIGHT_COMPLETE', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(pair_case, root, args.v147_root, args.out_root, s, d, geometry) for s, d in cases]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 32 == 0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=len(cases), unit='saved_pair')), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    write(args.out_root/'summary/status.json', dict(status='v148_readonly_eta_components_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=2*len(cases),
        results={m: prior.aggregate(cells, lambda c: c['modes'][m]) for m in ('old', 'handoff')},
        paired=prior.aggregate(cells, lambda c: c['paired']),
        phase_counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        v147_endpoints_and_support_reproduced=True, frozen_component_reconstruction=True,
        saved_point_files_changed=False, point_parameters_changed=False, calibration_refit=False,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Consumed-path clock accounting, not control efficacy or causal mediation. '
            'Frozen historical component calculations must reproduce saved V141 totals. Opaque fallback '
            'error stays unallocated; its other entries are zero allocation, not measured zero errors. '
            'Current ready residence remains joint holding/execution when its predictor is joint. '
            'MAE allocations are sign(total error)*component error and may be negative. Conditional '
            'switch groups retain incomplete crossed support; switch burden uses all valid matched '
            'decisions with observed no-event contribution zero, never missing outcomes zero. '
            'V146 offsets, points, window, tails and main stay unchanged.'))
    print('V148_READONLY_ETA_COMPONENTS_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v146-root', 'v147-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
