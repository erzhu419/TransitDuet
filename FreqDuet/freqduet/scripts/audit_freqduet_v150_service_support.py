#!/usr/bin/env python3
"""Diagnose saved V149 service support and fallback inputs without changing points."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v149_confirmation as previous
from audit_freqduet_v143_interval_support import support_context


parts, frozen = previous.components, previous.frozen
read, write, close = previous.read, previous.write, previous.close
SUPPORT = ('none', 'one_to_eight', 'nine_plus')
QUERY_GROUPS = (*SUPPORT, 'no_downstream_stop', 'retained_structured', 'opaque')


def support_group(n):
    return 'none' if n == 0 else 'one_to_eight' if n < 9 else 'nine_plus'


def stop_predictions(position, history, phases):
    """Recover only the frozen V149 calculation; attach no future labels here."""
    p, rows = position, []
    elapsed = p['initial_s']
    for i in range(p['origin']+1, p['target']+1):
        station, direction, now = p['route']['ids'][i], p['direction'], p['now']
        running = history.estimate(('running', direction, station), now)
        original = history.estimate(('service', direction, station), now)
        if running is None or original is None:
            raise AssertionError('saved structured tail lost its frozen component history')
        fraction = ((p['route']['positions'][i]-p['progress'])/
            (p['route']['positions'][i]-p['route']['positions'][i-1])) if i == p['origin']+1 else 1.
        elapsed += fraction*running['mean']
        proposal = phases.estimate(direction, station, now, now+elapsed)
        selected = original if proposal is None else proposal
        if max(running['latest'], original['latest'], selected['latest']) >= now:
            raise AssertionError('frozen reconstruction used unfinished labels')
        rows.append(dict(station=station, predicted_arrival_s=float(now+elapsed),
            phase_n=0 if proposal is None else proposal['count'],
            original_s=original['mean'], selected_s=selected['mean'],
            historical_fallback=False if proposal is None else proposal['historical_fallback']))
        elapsed += selected['mean']
        if i < p['target']:
            for channel in ('holding', 'ticks'):
                value = history.estimate((channel, direction, station), now)
                if value is None or value['latest'] >= now:
                    raise AssertionError('frozen downstream residence lost its completed history')
                elapsed += value['mean']
    return rows, float(elapsed)


def boundary(path, row, point, indexed, history, services, geometry):
    now, e = row['prediction_time_s'], row['evidence']
    trip, direction = e['predicted_follower_trip_id'], bool(row['direction'])
    route, motion = geometry[direction], indexed[trip].motion
    progress = float(route['positions'][route['index'][row['station_id']]])-e['spatial_gap_m']
    observed = indexed[trip].inspect(now, progress)
    launch = path['dispatch'][trip]['actual_s']
    native_launch = row['online'].get('launch_observed_s')
    data = dict(launch_at_query_timestamp=float(launch == now),
        native_launch_observation_logged=float(native_launch is not None and native_launch <= now),
        simultaneous_event_timestamp=float(bool(observed['same_clock_events'])),
        own_query_load_present=float('query_load' in row))
    if point['status'] == 'no_completed_service_survivor':
        j = bisect_left(motion.at, now)
        arrival = motion.arrivals[j-1]
        current = parts.key(arrival)
        elapsed = now-arrival['time_s']
        context, survivors = support_context(services, direction, current[1], now, elapsed)
        if survivors or observed['reason'] != 'pending_stop':
            raise AssertionError('saved service-support fallback not explained by its strict prefix')
        ready = path['ready'][current]['ready_time_s']
        if ready < now:
            raise AssertionError('service fallback already had a strictly earlier ready event')
        command = path['trace'].get(current)
        tag = ('no_completed_history' if context['past_n'] == 0 else
            'survivor_depleted_pool_ge9' if context['pool_n'] >= 9 else 'survivor_depleted_pool_lt9')
        data.update(pool_n=float(context['pool_n']), past_n=float(context['past_n']),
            elapsed_s=elapsed, retrospective_current_service_remaining_s=ready-now,
            ready_at_query_timestamp=float(ready == now),
            current_follower_query_load_strictly_observed=float(command is not None and
                command['policy_time_s'] < now and 'query_load' in command))
    else:
        if observed['origin'] is not None:
            raise AssertionError('saved strict-segment fallback has a strict moving origin')
        if launch == now and abs(progress) < 1e-8:
            tag = 'launch_at_query'
            available = True
            for i in range(1, route['index'][row['station_id']]+1):
                channels = ('running', 'service', 'holding', 'ticks') if i < route['index'][row['station_id']] else ('running', 'service')
                for channel in channels:
                    value = history.estimate((channel, direction, route['ids'][i]), now)
                    available = available and value is not None
                    if value is not None and value['latest'] >= now:
                        raise AssertionError('launch history probe used simultaneous labels')
            data['all_required_component_means_available'] = float(available)
        elif 'arrival' in observed['same_clock_events']:
            tag = 'arrival_at_query'
        elif 'departure' in observed['same_clock_events']:
            tag = 'departure_at_query'
        else:
            tag = observed['reason']
    return tag, data


def budgets(rows):
    result = {}
    for group in QUERY_GROUPS:
        result['query/'+group] = parts.prior.describe([dict(episode=r['episode'],
            fraction=float(r['group'] == group),
            base_allocated_bias_s=r['base'] if r['group'] == group else 0.,
            phase_allocated_bias_s=r['phase'] if r['group'] == group else 0.) for r in rows])
    for group in (*SUPPORT, 'retained_structured'):
        result['stop/'+group] = parts.prior.describe([dict(episode=r['episode'],
            base_allocated_bias_s=r['stop_parts'][group][0],
            phase_allocated_bias_s=r['stop_parts'][group][1]) for r in rows])
    return result


def case(root, out, geometry, seed, day):
    expected = read(root/'readout/cells'/f'{seed}_{day}.json')
    modes, counts, schemas = {}, Counter(), defaultdict(set)
    for mode in frozen.MODES:
        name = f'{seed}_{day}_{mode}'
        path, objects = parts.prepare(root/'confirmation', seed, day, mode, geometry)
        history, services, durations, indexed = objects
        phases = frozen.candidate.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        saved = {tuple(r['query']): r for r in parts.prior.trace(root/'confirmation/candidate_shadows'/(name+'.jsonl.gz'))}
        if saved.keys() != path['trace'].keys():
            raise AssertionError('saved support and frozen decision keys differ')
        query_rows, query_groups, stops, fallback = [], defaultdict(list), defaultdict(list), defaultdict(list)
        for k, row in path['trace'].items():
            p = saved[k]
            schemas['evidence'].update(row['evidence'])
            schemas['online'].update(row['online'])
            schemas['decision'].update(row)
            status, truth = parts.prior.target_label(row, path['ready'])
            if status != 'resolved' or p['points']['base'] is None:
                continue
            base_structured = (row['online']['component_used'] or
                path['shadows'][k]['v140_status'] == 'passive_stop_component' or
                path['shadows'][k]['v141_status'] == 'passive_unissued_ready')
            episode = [*parts.prior.follower(row), *k[1:]]
            if p['status'] in ('no_strict_physical_segment', 'no_completed_service_survivor'):
                tag, data = boundary(path, row, p, indexed, history, services, geometry)
                prefix = 'retained_structured/' if base_structured else 'opaque/'
                fallback[prefix+tag].append(dict(episode=episode, **data,
                    full_bias_s=p['points']['base']-truth, full_mae_s=abs(p['points']['base']-truth)))
                counts[mode+'/fallback/'+prefix+tag] += 1
                counts[mode+'/fallback_source/'+prefix+tag+'/'+row['evidence']['follower_source']] += 1
            if row['prediction_time_s'] < path['cutoff']:
                continue
            packet, joint, _ = parts.component_packet(path, row, path['shadows'][k], geometry, *objects)
            stop_parts = {s: [0., 0.] for s in (*SUPPORT, 'retained_structured')}
            if packet is None:
                group, base, phase = 'opaque', 0., 0.
                counts[mode+'/clearance/opaque_unknown_service'] += 1
            else:
                actual = parts.actual_parts(path, row, geometry[bool(row['direction'])], joint)
                base = packet['downstream_service']-actual['downstream_service']
                phase = base+p['points']['service']-p['points']['base']
                if p['status'] != 'structured':
                    group = 'retained_structured'
                    stop_parts[group] = [base, phase]
                else:
                    position, reason = frozen.candidate.state(path, row, indexed, geometry, history, services, durations)
                    if reason != 'structured':
                        raise AssertionError('frozen V149 structured state was not reproduced')
                    predictions, total = stop_predictions(position, history, phases)
                    close(total, p['points']['joint'])
                    if sum(s['phase_n'] == 0 for s in predictions) != p['phase_unsupported_stops'] or (
                            sum(s['phase_n'] > 0 for s in predictions) != p['phase_supported_stops']):
                        raise AssertionError('per-stop support differs from saved V149 metadata')
                    group = support_group(min(s['phase_n'] for s in predictions)) if predictions else 'no_downstream_stop'
                    base_total, phase_total = 0., 0.
                    # Truths below cannot change frozen phase selection or the saved point.
                    for s in predictions:
                        target = row['evidence']['predicted_follower_trip_id'], s['station'], bool(row['direction'])
                        arrival = path['arrivals'][target]['time_s']
                        duration = path['ready'][target]['ready_time_s']-arrival
                        if arrival < row['prediction_time_s']:
                            raise AssertionError('downstream service was already in progress at the query')
                        b, c = s['original_s']-duration, s['selected_s']-duration
                        sg = support_group(s['phase_n'])
                        stop_parts[sg][0] += b
                        stop_parts[sg][1] += c
                        base_total += b
                        phase_total += c
                        stops[sg].append(dict(episode=[*parts.prior.follower(row), s['station'], int(row['direction'])],
                            base_bias_s=b, phase_bias_s=c, base_service_mae_s=abs(b), phase_service_mae_s=abs(c),
                            phase_n=float(s['phase_n']), selected_service_s=s['selected_s'], truth_service_s=duration,
                            historical_fallback=float(s['historical_fallback'])))
                    close(base_total, base)
                    close(phase_total, phase)
                query_groups[group].append(dict(episode=episode, base_bias_s=base, phase_bias_s=phase,
                    base_service_mae_s=abs(base), phase_service_mae_s=abs(phase)))
            query_rows.append(dict(episode=episode, group=group, base=base, phase=phase, stop_parts=stop_parts))
        budget = budgets(query_rows)
        for arm, field in (('base', 'base_allocated_bias_s'), ('service', 'phase_allocated_bias_s')):
            close(sum(budget['query/'+g]['metrics'][field] for g in QUERY_GROUPS),
                expected['components'][mode][arm]['metrics']['downstream_service_bias_s'])
            close(sum(budget['stop/'+g]['metrics'][field] for g in (*SUPPORT, 'retained_structured')),
                expected['components'][mode][arm]['metrics']['downstream_service_bias_s'])
        modes[mode] = dict(budgets=budget,
            queries={g: parts.prior.describe(query_groups[g]) for g in QUERY_GROUPS},
            stops={g: parts.prior.describe(stops[g]) for g in SUPPORT},
            fallback={g: parts.prior.describe(r) for g, r in fallback.items()})
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes,
        counts=dict(counts), schemas={k: sorted(v) for k, v in schemas.items()}))


def run(args):
    if read(args.v149_root/'readout/summary/status.json')['status'] != 'v149_readonly_confirmation_audit_complete':
        raise AssertionError('requires complete prior V149 audit')
    model = read(args.v146_root/'model.json')
    if (model['status'], model['lower_extension_s'], model['upper_extension_s']) != ('sealed', 3., 0.):
        raise AssertionError('V146 reference changed')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v150_service_support.py', '-v'], check=True)
    geometry = frozen.candidate.model.load_geometry(args.runtime)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v149_root, args.out_root, geometry, s, d) for s, d in frozen.CASES]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 32 == 0:
                print('V150_READONLY_PROGRESS', n, len(jobs), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    result = {m: {section: parts.prior.aggregate(cells, lambda c: c['modes'][m][section])
        for section in ('budgets', 'queries', 'stops', 'fallback')} for m in frozen.MODES}
    schemas = {k: sorted(set().union(*(c['schemas'][k] for c in cells))) for k in cells[0]['schemas']}
    write(args.out_root/'summary/status.json', dict(status='v150_readonly_service_support_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_point_source='d7996d60b4',
        cells=len(cells), saved_paths=2*len(cells), results=result, schemas=schemas,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        saved_points_support_and_component_budgets_reproduced=True,
        original_joint_gate=read(args.v149_root/'summary/status.json')['joint_prediction_gate'],
        new_rollouts=0, fitted_parameters=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, window_s=3600., interval_tails=[.1, .1],
        limitations='Post-readout diagnosis, no predictor selection or admission. Nine samples is an '
            'existing descriptive support convention, not a sufficiency guarantee or causal n effect. Conditional empty '
            'cells remain missing; episode-weighted budgets include observed zero contributions from '
            'other groups. Opaque fallback has unknown service error, not measured zero error. '
            'Same-time event timestamps do not establish callback order; native launch_observed_s '
            'is direct observation evidence only for the native forecast where it was logged.'))
    print('V150_READONLY_SERVICE_SUPPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v149-root', 'v146-root', 'runtime', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
