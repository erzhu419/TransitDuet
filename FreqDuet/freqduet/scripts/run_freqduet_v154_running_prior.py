#!/usr/bin/env python3
"""Develop one frozen running-prior shadow on archived V151 calibration days."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import numpy as np

import audit_freqduet_v153_running_lag as prior
import audit_freqduet_v141_ready_eta as native
import freqduet_phase_running_prior as candidate


frozen, parts, stats = prior.frozen, prior.parts, prior.stats
read, write, close = prior.read, prior.write, prior.close


def build(root, geometry):
    actors = {}
    for seed in frozen.ACTORS:
        groups = defaultdict(lambda: defaultdict(dict))
        for day in frozen.CALIBRATION:
            for mode in frozen.MODES:
                name = f'{seed}_{day}_{mode}'
                receipt = read(root/'paths'/(name+'.json'))
                if any(receipt[k] for k in ('actor_updates', 'critic_updates', 'dual_updates', 'main_policy_changed')):
                    raise AssertionError('calibration archive changed deployment')
                terminal = read(root/'terminal'/(name+'.json'))
                dispatch = {r['trip_id']: r['actual_s'] for r in terminal['dispatches']}
                with np.load(root/'records'/(name+'.npz')) as a:
                    fields = a['event_fields'].tolist()
                    arrivals = [dict(zip(fields, r.tolist())) for r in a['arrivals']]
                    departures = {stats.key(r): r for r in (dict(zip(fields, v.tolist())) for v in a['departures'])}
                samples = defaultdict(list)
                for a in arrivals:
                    trip, station, direction = stats.key(a)
                    route = geometry[direction]
                    previous = departures.get((trip, route['ids'][route['index'][station]-1], direction))
                    start = dispatch[trip] if previous is None else previous['time_s']
                    duration = a['time_s']-start
                    if duration < 0.:
                        raise AssertionError('negative archived running duration')
                    samples[direction, station, int(start//3600)].append(duration)
                for k, values in samples.items():
                    groups[k][day][mode] = dict(mean=fmean(values), second=fmean(v*v for v in values))
        actors[str(seed)] = [dict(direction=k[0], station=k[1], hour=k[2],
            days=[dict(day=d, mean=fmean(v['mean'] for v in modes.values()),
                second=fmean(v['second'] for v in modes.values())) for d, modes in sorted(days.items())])
            for k, days in sorted(groups.items())]
    return dict(status='sealed_development_moments', actors=actors,
        days=list(frozen.CALIBRATION), modes_shared=True, confirmation_days_read=False)


def prepare(root, seed, day, mode, geometry):
    path = native.prior.load_path(root, root, seed, day, mode)
    history, services, indexed = native.frozen.prepare(path, geometry)
    durations = native.ServiceResiduals([(k[2], k[1], r['time_s'],
        r['time_s']-path['ready'][k]['ready_time_s']) for k, r in path['departures'].items()])
    path['cutoff'] = read(root/'paths'/f'{seed}_{day}_{mode}.json')['contract']['clock']['demand_end_time_s']
    path['_observed_ready'] = frozen.candidate.ObservedReady(path['ready'])
    return path, (history, services, durations, indexed)


def fallback(path, row, geometry, objects, phases):
    history, services, durations, indexed = objects
    stop = native.frozen.predict(path, row, indexed, geometry, history, services)
    gap = stop['gap_s'] if stop['applied'] else row['online']['gap_s']
    ready = dict(applied=False, status='unchanged_v140')
    if stop['status'] == 'ready_without_prior_command':
        k = tuple(stop['current_stop_key'])
        route = geometry[k[2]]
        ready = native.predict_ready(history, durations, route, k[2], route['index'][k[1]],
            route['index'][row['station_id']], row['prediction_time_s'], path['ready'][k]['ready_time_s'])
    saved = dict(v140_s=gap, v141_s=ready['gap_s'] if ready['applied'] else gap,
        v140_status=stop['status'], v141_status=ready['status'])
    values, _ = frozen.candidate.frozen.forecasts(path, row, saved, geometry, objects, phases)
    return values['joint']


def case(root, out, runtime, seed, day, running, services):
    geometry = native.model.load_geometry(runtime)
    modes, counts = {}, Counter()
    service_cells = candidate.exclude_day(services['actors'][str(seed)], day)
    if any(d['day'] == day for c in service_cells for d in c['days']):
        raise AssertionError('held day leaked into service prior')
    for mode in frozen.MODES:
        path, objects = prepare(root/'calibration', seed, day, mode, geometry)
        phases = frozen.candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        service = frozen.candidate.ServiceLaw(service_cells, path['arrivals'], path['ready'], path['cutoff'])
        law = candidate.RunningLaw(running['actors'][str(seed)], day, prior.RunningPrefix(path, geometry))
        if any(d['day'] == day for rows in law.prior.values() for d in rows):
            raise AssertionError('held day leaked into running prior')
        groups = defaultdict(list)
        for k, row in path['trace'].items():
            before, detail = frozen.candidate.forecast(path, row, geometry, objects, phases, service, True)
            after, proposed = candidate.forecast(path, row, geometry, objects, phases, service, law)
            if before is None:
                if after is not None:
                    raise AssertionError('running prior changed unsupported V151 support')
                before = fallback(path, row, geometry, objects, phases)
            if after is None:
                after = before
            label, truth = stats.target_label(row, path['ready'])
            counts[mode+'/'+label] += 1
            if label != 'resolved' or before is None:
                continue
            e0, e1 = before-truth, after-truth
            r = dict(episode=(*stats.follower(row), *k[1:]), reference_mae_s=abs(e0), shadow_mae_s=abs(e1),
                reference_bias_s=e0, shadow_bias_s=e1, delta_full_mae_s=abs(e1)-abs(e0))
            if 'running_s' in proposed:
                packet, _, status = prior.prior.structured_packet(path, row, geometry, objects, phases, service, True)
                close(sum(packet.values()), before)
                actual = parts.actual_parts(path, row, geometry[bool(row['direction'])])['running']
                b, a = packet['running']-actual, proposed['running_s']-actual
                r.update(reference_running_bias_s=b, shadow_running_bias_s=a,
                    reference_running_mae_s=abs(b), shadow_running_mae_s=abs(a),
                    delta_running_mae_s=abs(a)-abs(b), service_clock_feedback_s=(after-before)-(a-b),
                    changed_leg_fraction=float(proposed['changed_legs'] > 0))
                for f in ('mean_prior_days', 'mean_leg_dispersion_s'):
                    if proposed[f] is not None:
                        r[f] = proposed[f]
            now = row['prediction_time_s']
            scopes = ['all', 'clearance' if now >= path['cutoff'] else 'active']
            if int(now//3600) in (2, 13):
                scopes.append('hour'+str(int(now//3600)))
            if proposed.get('unsupported_legs', 0) or proposed.get('single_day_legs', 0):
                scopes.append('low_support')
            for g in scopes:
                groups[g].append(r)
            counts[mode+'/partial_legs_retained'] += proposed.get('partial_legs_retained', 0)
            counts[mode+'/unsupported_legs'] += proposed.get('unsupported_legs', 0)
        modes[mode] = {g: stats.describe(r) for g, r in groups.items()}
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes, counts=dict(counts),
        held_day_excluded_from_both_priors=True))


def run(args):
    write(args.out_root/'protocol.json', dict(stage='v154_archived_calibration_lodo_development',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], actors=list(frozen.ACTORS),
        days=list(frozen.CALIBRATION), held_out_unit='entire demand day across both modes',
        historical_weight='n independent prior days / (n+1); one native-prefix anchor',
        partial_running='retain V151 geometry-scaled residual', window_s=3600., tails=[.1, .1],
        calibration='all eight leave-day-out folds; no hyperparameter search', confirmation_days_read=False,
        primary='both modes: full MAE improvement all/hour2; full MAE non-harm clearance/hour13; '
            'running MAE non-harm all/hour2/clearance/hour13; low-support full MAE non-harm',
        uncertainty='empirical mixture dispersion per complete future leg, not calibrated ETA interval'))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v154_running_prior.py', '-v'], check=True)
    services = read(args.v151_root/'model.json')
    if (services['status'], services['execution_source'], services['calibration_days']) != (
            'sealed', '1b783ea8f3', list(frozen.CALIBRATION)):
        raise AssertionError('V151 service reference differs')
    running = build(args.v151_root/'calibration', native.model.load_geometry(args.runtime))
    write(args.out_root/'model.json', running)
    cases = list(frozen.CAL_CASES)
    case(args.v151_root, args.out_root, args.runtime, *cases[0], running, services)
    print('V154_FIRST_LODO_PAIR_COMPLETE', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v151_root, args.out_root, args.runtime, s, d, running, services)
            for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 16 == 0:
                print('V154_DEVELOPMENT_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    results = {m: stats.aggregate(cells, lambda c: c['modes'][m]) for m in frozen.MODES}
    def passes(mode, scope, field, strict=False):
        ci = results[mode].get(scope, {}).get('metrics', {}).get(field, {}).get('ci95')
        return ci is not None and (ci[1] < 0. if strict else ci[1] <= 0.)
    gate = all(all(passes(m, g, 'delta_full_mae_s', g in ('all', 'hour2')) for g in
        ('all', 'hour2', 'clearance', 'hour13', 'low_support')) and all(
        passes(m, g, 'delta_running_mae_s') for g in ('all', 'hour2', 'clearance', 'hour13')) for m in frozen.MODES)
    write(args.out_root/'summary/status.json', dict(status='v154_lodo_development_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cases), paths=2*len(cases),
        results=results, development_gate=gate, independent_confirmation=False, controller_admitted=False,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        confirmation_days_read=False, held_day_excluded_from_both_priors=True, running_prior_constructed=True,
        v151_model_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1],
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Archived calibration-day LODO development, not fresh confirmation or deployed V151 '
            'efficacy. V151-LODO uses seven service-prior days in each fold, not the sealed eight-day '
            'confirmation model. Folds share prior days; crossed evaluation-day CIs are descriptive '
            'conditional on this calibration archive, not independent model-retraining inference. '
            'Empirical component dispersion is not a calibrated prediction interval; V146 stays fixed. '
            'Missing crossed subgroup support remains inconclusive, never zero-filled. No policy switch.'))
    print('V154_LODO_DEVELOPMENT_COMPLETE', gate, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v151-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
