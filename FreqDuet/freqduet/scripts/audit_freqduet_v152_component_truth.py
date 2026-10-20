#!/usr/bin/env python3
"""Read-only true-clock component audit of sealed V151 confirmation paths."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v151_confirmation as prior


frozen, parts, stats = prior.frozen, prior.parts, prior.stats
read, write, close = prior.read, prior.write, prior.close
candidate = frozen.candidate


def initial_packet(path, row, position, status, support, objects):
    history, services, durations, indexed = objects
    packet, joint = parts.zeros(), None
    p = position
    direction, now = p['direction'], p['now']
    if status == 'historical_current_service':
        packet['current_service'] = support['mean']
        if p['origin'] < p['target']:
            station = p['route']['ids'][p['origin']]
            for source, dest in (('holding', 'holding'), ('ticks', 'execution_wait')):
                packet[dest] = history.estimate((source, direction, station), now)['mean']
    elif status == 'observed_same_clock_ready':
        packet['current_ready_joint'] = p['initial_s']
        joint = (stats.follower(row)[0], p['route']['ids'][p['origin']], direction)
    elif p['initial_s'] != 0.:
        motion = indexed[stats.follower(row)[0]].motion
        arrival = motion.arrivals[bisect_left(motion.at, now)-1]
        current = stats.key(arrival)
        stop = candidate.frozen.frozen.frozen.from_stop(history, services, p['route'], direction,
            p['origin'], p['target'], now, arrival['time_s'],
            path['ready'][current]['ready_time_s'], path['trace'].get(current))
        if stop['status'] == 'ready_without_prior_command':
            packet['current_ready_joint'], joint = p['initial_s'], current
        elif stop['applied'] and stop['phase'] == 'service_in_progress':
            packet['current_service'] = stop['service_residual']['mean']
            if p['origin'] < p['target']:
                for source, dest in (('holding', 'holding'), ('ticks', 'execution_wait')):
                    packet[dest] = history.estimate((source, direction, current[1]), now)['mean']
        elif stop['applied'] and stop['phase'] == 'known_command':
            channel = 'holding' if path['trace'][current]['chosen_s'] > 0 else 'execution_wait'
            packet[channel] = p['initial_s']
        else:
            raise AssertionError('structured current residence cannot be allocated')
    close(sum(packet.values()), p['initial_s'])
    return packet, joint


def structured_packet(path, row, geometry, objects, phases, law=None, events=False):
    p, status, support = candidate.position(path, row, objects[-1], geometry, objects, law, events)
    if p is None:
        return None, None, status
    packet, joint = initial_packet(path, row, p, status, support, objects)
    elapsed, history = p['initial_s'], objects[0]
    for i in range(p['origin']+1, p['target']+1):
        station, direction, now = p['route']['ids'][i], p['direction'], p['now']
        running = history.estimate(('running', direction, station), now)
        native = history.estimate(('service', direction, station), now)
        if running is None or native is None:
            return None, None, 'missing_downstream_component'
        fraction = ((p['route']['positions'][i]-p['progress'])/
            (p['route']['positions'][i]-p['route']['positions'][i-1])) if i == p['origin']+1 else 1.
        value = fraction*running['mean']
        packet['running'] += value
        elapsed += value
        proposal = (phases.estimate(direction, station, now, now+elapsed) if law is None else
            law.estimate(direction, station, now, now+elapsed))
        service = native if proposal is None else proposal
        packet['downstream_service'] += service['mean']
        elapsed += service['mean']
        if i < p['target']:
            for source, dest in (('holding', 'holding'), ('ticks', 'execution_wait')):
                value = history.estimate((source, direction, station), now)
                if value is None:
                    return None, None, 'missing_downstream_'+source
                packet[dest] += value['mean']
                elapsed += value['mean']
    close(sum(packet.values()), elapsed)
    return packet, joint, status


def coarse(packet):
    return dict(service=packet['current_service']+packet['downstream_service'],
        running=packet['running'], residence=packet['holding']+packet['execution_wait']+
        packet['current_ready_joint'])


def component_errors(predictions, actual):
    truth = coarse(actual)
    result = {}
    for name, packet in predictions.items():
        values = coarse(packet)
        errors = {k: values[k]-truth[k] for k in truth}
        total = sum(errors.values())
        close(total, sum(packet.values())-sum(actual.values()))
        result[name+'_full_mae_s'], result[name+'_full_bias_s'] = abs(total), total
        for k, e in errors.items():
            result[name+'_'+k+'_bias_s'] = e
            result[name+'_'+k+'_mae_s'] = abs(e)
        result[name+'_cancellation_s'] = (sum(abs(v) for v in errors.values())-abs(total))/2.
    for field in ('full_mae', 'service_mae', 'running_mae', 'residence_mae', 'cancellation'):
        result['delta_'+field+'_s'] = result['joint_'+field+'_s']-result['v149_'+field+'_s']
    close(result['delta_full_mae_s'], sum(result['delta_'+k+'_mae_s'] for k in truth)-
        2*result['delta_cancellation_s'])
    return result


def case(root, out, runtime, seed, day, model):
    geometry = candidate.frozen.model.load_geometry(runtime)
    modes, counts, launches = {}, Counter(), []
    expected = read(root/'cells'/f'{seed}_{day}.json')
    for mode in frozen.MODES:
        path, objects = parts.prepare(root/'confirmation', seed, day, mode, geometry)
        path['_observed_ready'] = candidate.ObservedReady(path['ready'])
        phases = candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        law = candidate.ServiceLaw(model['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
        saved = {tuple(r['query']): r for r in stats.trace(root/'confirmation/candidate_shadows'/
            f'{seed}_{day}_{mode}.jsonl.gz')}
        if saved.keys() != path['trace'].keys():
            raise AssertionError('saved V151 query support changed')
        groups, endpoints = defaultdict(list), defaultdict(list)
        for k, row in path['trace'].items():
            label, truth = stats.target_label(row, path['ready'])
            s = saved[k]
            if label != 'resolved' or s['points']['v141'] is None:
                continue
            episode = (*stats.follower(row), *k[1:])
            labels = ['all']
            if row['prediction_time_s'] >= path['cutoff']:
                labels.append('clearance')
            if int(row['prediction_time_s']//3600) in (2, 13):
                labels.append('hour'+str(int(row['prediction_time_s']//3600)))
            record = frozen.record(episode, s['points'], truth)
            for scope in labels:
                endpoints[scope].append({f: record[f] for f in ('episode', 'v149_mae_s',
                    'joint_mae_s', 'v149_bias_s', 'joint_bias_s')})
            reference, rjoint, rstatus = structured_packet(path, row, geometry, objects, phases)
            if reference is None:
                reference, rjoint, _ = parts.component_packet(path, row, path['shadows'][k], geometry, *objects)
            joint, jjoint, jstatus = structured_packet(path, row, geometry, objects, phases, law, True)
            if jstatus != s['details']['joint']['status']:
                raise AssertionError('reconstructed V151 state differs from frozen forecast')
            if joint is None:
                joint, jjoint = reference, rjoint
            for name, packet in (('v149', reference), ('joint', joint)):
                if packet is not None:
                    close(sum(packet.values()), s['points'][name])
            structured = reference is not None and joint is not None
            counts[mode+('/paired_components' if structured else '/opaque_reference')] += 1
            launch = 'clearance' in labels and jstatus == 'observed_same_clock_launch'
            if structured or launch:
                actual = parts.actual_parts(path, row, geometry[bool(row['direction'])])
                close(sum(actual.values()), truth)
            if structured:
                values = component_errors(dict(v149=reference, joint=joint), actual)
                values.update(episode=episode)
                for scope in labels:
                    groups[scope].append(values)
                if rjoint is None and jjoint is None:
                    holding = dict(episode=episode)
                    for name, packet in (('v149', reference), ('joint', joint)):
                        e = packet['holding']-actual['holding']
                        holding[name+'_holding_bias_s'], holding[name+'_holding_mae_s'] = e, abs(e)
                    holding['delta_holding_mae_s'] = holding['joint_holding_mae_s']-holding['v149_holding_mae_s']
                    for scope in labels:
                        groups[scope+'/separable_holding'].append(holding)
                if 'hour2' in labels and jstatus == 'structured':
                    groups['hour2/downstream_only'].append(values)
            if launch:
                launches.append(dict(seed=seed, day=day, mode=mode, query=list(k),
                    time_s=row['prediction_time_s'], follower=list(stats.follower(row)), truth_s=truth,
                    points={v: s['points'][v] for v in ('v141', 'v149', 'events', 'joint')},
                    reference_status=rstatus, native_status=row['online']['status'],
                    native_component_used=row['online']['component_used'],
                    original_status=row['original_prediction']['status'], actual=actual,
                    reference_components=reference, joint_components=joint,
                    dispatch_s=path['dispatch'][stats.follower(row)[0]]['actual_s'],
                    spatial_gap_m=row['evidence']['spatial_gap_m']))
        end = {g: stats.describe(rows) for g, rows in endpoints.items()}
        for scope in ('all', 'clearance', 'hour13'):
            before = expected['modes'][mode][scope]
            if end[scope]['queries'] != before['queries']:
                raise AssertionError('original V151 endpoint support changed')
            for field, value in end[scope]['metrics'].items():
                close(value, before['metrics'][field])
        modes[mode] = dict(components={g: stats.describe(rows) for g, rows in groups.items()}, endpoints=end)
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes,
        counts=dict(counts), clearance_launches=launches))


def run(args):
    parent, model = read(args.v151_root/'summary/status.json'), read(args.v151_root/'model.json')
    if parent['execution_source'] != '1b783ea8f3' or not parent['labels_complete'] or model['status'] != 'sealed':
        raise AssertionError('requires complete frozen V151 and sealed prior')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v152_component_truth.py', '-v'], check=True)
    case(args.v151_root, args.out_root, args.runtime, *frozen.CASES[0], model)
    print('V152_FIRST_SAVED_PAIR_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v151_root, args.out_root, args.runtime, s, d, model)
            for s, d in frozen.CASES[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V152_COMPONENT_PROGRESS', n, len(frozen.CASES), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    results = {m: {section: stats.aggregate(cells, lambda c: c['modes'][m][section])
        for section in ('components', 'endpoints')} for m in frozen.MODES}
    write(args.out_root/'summary/status.json', dict(status='v152_readonly_component_truth_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_source='1b783ea8f3',
        cases=len(cells), saved_paths=2*len(cells), results=results,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        clearance_launches=[r for c in cells for r in c['clearance_launches']],
        original_endpoints_reproduced=True, exact_component_clock_partition=True,
        original_joint_prediction_gate=parent['joint_prediction_gate'], new_rollouts=0,
        calibration_refit=False, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, v146_extensions_s=[3., 0.], window_s=3600., native_tails=[.1, .1],
        limitations='Consumed-path diagnostic, not a new gate or causal control effect. Component MAE '
            'is absolute component error, not signed full-MAE allocation. Residence includes holding, '
            'execution ticks and unsplit ready residence; holding-only comparisons exclude unsplit '
            'ready predictors. Opaque reference forecasts remain outside paired component estimates. '
            'Sparse launch cases are individual records without crossed confidence intervals. '
            'Cancellation is half the difference between component absolute-error sum and full MAE. '
            'All means retain within-follower-stop episode and equal actor/day weighting.'))
    print('V152_READONLY_COMPONENT_TRUTH_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v151-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
