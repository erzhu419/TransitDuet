#!/usr/bin/env python3
"""Calibrate independent-day service laws; confirm one frozen shadow candidate."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import freqduet_historical_service_eta as candidate
import run_freqduet_v149_phase_clock as prior


ACTORS, MODES = prior.ACTORS, prior.MODES
CALIBRATION = tuple(range(2751001, 2751009))
CONFIRMATION = tuple(range(2752001, 2752017))
CAL_CASES = tuple(itertools.product(ACTORS, CALIBRATION))
CASES = tuple(itertools.product(ACTORS, CONFIRMATION))
SAMPLERS = {d: 1510000000+i for i, d in enumerate((*CALIBRATION, *CONFIRMATION))}
ARMS = ('v141', 'v149', 'history', 'events', 'joint')
read, write = prior.read, prior.write


def registration():
    return dict(stage='v151_historical_service_independent_confirmation', actors=list(ACTORS),
        calibration_days=list(CALIBRATION), confirmation_days=list(CONFIRMATION),
        sampler_seeds={str(d): SAMPLERS[d] for d in SAMPLERS}, modes=list(MODES), arms=list(ARMS),
        calibration_paths=2*len(CAL_CASES), confirmation_paths=2*len(CASES), checkpoint_ep=199,
        historical_law='per-actor/stop/direction/published arrival phase empirical distribution; '
            'equal calibration-day mass, equal available control-mode mass within each day; '
            'shared prior across old/handoff, no mode input to predictor',
        current_day='one extra empirical-day mass from strict completed same-phase prefix, '
            'one-hour window; original all-past fallback only if that window is empty',
        residual='condition the weighted duration mixture on duration>=observed elapsed age '
            'before normalizing; no extrapolation or zero fill beyond support; retain V149 point on abstention',
        events='launch phase precedes all bus updates; same-clock ready requires earlier saved callback '
            'of same physical follower; same-clock historical labels and commands remain excluded',
        window_s=3600., native_interval_tails=[.1, .1], v146_endpoint_extensions_s=[3., 0.],
        calibration_selection='all registered calibration days, no score-based selection or hyperparameter search',
        confirmation_gate='joint only: both modes clearance MAE and absolute-cell-bias CI95 upper<0 '
            'relative to both V149 and V141; all-day/hour13 MAE upper<=0 and same-identity '
            'source-switch MAE burden upper<=0 relative to both; complete crossed labels required',
        inferential_unit='queries within follower-stop episode, then actor/day;5000 crossed draws,seed20260803',
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        controller_admitted=False, confirmation_single_use=True)


def register(args):
    plan, target = registration(), args.out_root/'protocol.json'
    if target.exists():
        if any(read(target)[k] != v for k, v in plan.items()):
            raise AssertionError('V151 preregistration changed')
        return
    archive = read(args.v144_root/'details/admission.json')
    used = {d for r in archive['metadata'] for d in r['days']}
    used.update(r['identity'][1] for r in archive['bundles'] if r['identity'] is not None)
    used.update(read(args.v145_root/'protocol.json')['demand_days'])
    for days in read(args.v146_root/'protocol.json')['days'].values():
        used.update(days)
    used.update(read(args.v149_root/'protocol.json')['days'])
    fresh = set(CALIBRATION) | set(CONFIRMATION)
    if used & fresh or set(CALIBRATION) & set(CONFIRMATION):
        raise AssertionError('V151 calibration/confirmation days are not independent')
    for path in Path(__file__).parent.glob('*.py'):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in fresh:
                raise AssertionError(f'fresh day already occurs in {path.name}:{node.lineno}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
                if node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args):
                    if any(d in range(*(a.value for a in node.args)) for d in fresh):
                        raise AssertionError(f'fresh day overlaps {path.name}:{node.lineno}')
    v146 = read(args.v146_root/'model.json')
    if (v146['status'], v146['lower_extension_s'], v146['upper_extension_s']) != ('sealed', 3., 0.):
        raise AssertionError('V146 reference changed')
    plan.update(registration_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    write(target, plan)
    print('V151_REGISTERED_8_CALIBRATION_16_CONFIRMATION_DAYS', flush=True)


def require(args):
    plan = read(args.out_root/'protocol.json')
    if any(plan[k] != v for k, v in registration().items()):
        raise AssertionError('registered V151 design differs')
    if read(args.out_root/'preflight/done.json')['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('source differs from completed V151 preflight')


def collect_pair(args, index):
    seed, day = CAL_CASES[index]
    target = args.out_root/'calibration_pairs'/f'{seed}_{day}.json'
    for mode in MODES:
        prior.reference.collect(args.out_root/'calibration', args.runtime, seed, day, mode, SAMPLERS[day])
    write(target, dict(seed=seed, day=day, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT']))


def preflight(args):
    register(args)
    proof = read(args.v145_root/'preflight/done.json')
    if (proof['execution_source'], proof['status']) != ('6eaa4a389b', 'consumed_actions_clocks_points_and_bounds_exact'):
        raise AssertionError('collector equivalence proof differs')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v151_*.py', '-v'], check=True)
    evaluate_pair(args.v146_root/'confirmation', args.runtime, ACTORS[0], 2552001, [], verify=True)
    # Two new paths are calibration data only, never confirmation or a selection screen.
    collect_pair(args, 0)
    write(args.out_root/'preflight/done.json', dict(status='complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        consumed_baseline_reconstructed=True, fresh_calibration_smoke_paths=2,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False))
    print('V151_PREFLIGHT_COMPLETE', flush=True)


def seal(args):
    require(args)
    if (args.out_root/'model.json').exists():
        model = read(args.out_root/'model.json')
        if model['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
            raise AssertionError('sealed prior source changed')
        return
    groups = {s: defaultdict(lambda: defaultdict(dict)) for s in ACTORS}
    for seed, day in CAL_CASES:
        cell = read(args.out_root/'calibration_pairs'/f'{seed}_{day}.json')
        if cell != dict(seed=seed, day=day, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT']):
            raise AssertionError('calibration pair source differs')
        for mode in MODES:
            name = f'{seed}_{day}_{mode}'
            root = args.out_root/'calibration'
            receipt = read(root/'paths'/(name+'.json'))
            if any(receipt[k] for k in ('actor_updates', 'critic_updates', 'dual_updates', 'main_policy_changed')):
                raise AssertionError('calibration changed the deployment')
            cutoff = receipt['contract']['clock']['demand_end_time_s']
            ready = {prior.prior.key(r): r for r in prior.prior.prior.trace(root/'ready_labels'/(name+'.jsonl.gz'))}
            with np.load(root/'records'/(name+'.npz')) as record:
                fields = record['event_fields'].tolist()
                arrivals = [dict(zip(fields, r.tolist())) for r in record['arrivals']]
            samples = defaultdict(list)
            for r in arrivals:
                k = prior.prior.key(r)
                duration = ready[k]['ready_time_s']-r['time_s']
                if duration < 0.:
                    raise AssertionError('negative calibration service duration')
                samples[k[2], k[1], r['time_s'] >= cutoff].append(duration)
            for k, values in samples.items():
                groups[seed][k][day][mode] = values
    actors = {str(s): [dict(direction=k[0], station=k[1], clearance=k[2],
        days=[dict(day=d, modes=[v[m] for m in MODES if m in v]) for d, v in sorted(days.items())])
        for k, days in sorted(groups[s].items())] for s in ACTORS}
    write(args.out_root/'model.json', dict(status='sealed', actors=actors,
        calibration_days=list(CALIBRATION), confirmation_days_read=False,
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT']))
    write(args.out_root/'model_seal/status.json', dict(status='v151_prior_sealed',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], calibration_days=list(CALIBRATION),
        calibration_paths=2*len(CAL_CASES), actor_updates=0, critic_updates=0, dual_updates=0,
        confirmation_days_read=False, main_policy_changed=False,
        service_cells={s: len(v) for s, v in actors.items()}))
    print('V151_INDEPENDENT_DAY_PRIOR_SEALED', flush=True)


def record(episode, points, truth):
    errors = {a: v-truth for a, v in points.items()}
    return dict(episode=episode, **{a+'_mae_s': abs(e) for a, e in errors.items()},
        **{a+'_bias_s': e for a, e in errors.items()},
        **{a+'_delta_mae_s': abs(errors[a])-abs(errors['v149']) for a in ('history', 'events', 'joint')},
        joint_vs_v141_delta_mae_s=abs(errors['joint'])-abs(errors['v141']))


def evaluate_pair(root, runtime, seed, day, cells, save=False, verify=False):
    geometry = candidate.frozen.model.load_geometry(runtime)
    paths, predictions, modes, counts = {}, {}, {}, Counter()
    for mode in MODES:
        path, objects = prior.prior.prepare(root, seed, day, mode, geometry)
        path['_observed_ready'] = candidate.ObservedReady(path['ready'])
        law = candidate.ServiceLaw(cells, path['arrivals'], path['ready'], path['cutoff'])
        phases = candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        groups, selected, shadows = defaultdict(list), {}, []
        for k, row in path['trace'].items():
            old, _ = candidate.frozen.forecasts(path, row, path['shadows'][k], geometry, objects, phases)
            points, details = dict(v141=old['base'], v149=old['joint']), {}
            if verify:
                p, _ = candidate.forecast(path, row, geometry, objects, phases)
                if old['joint'] is None:
                    if p is not None:
                        raise AssertionError('unchanged reconstruction changed unavailable forecast')
                else:
                    prior.prior.close(old['joint'] if p is None else p, old['joint'])
            for arm, use_history, use_events in (('history', True, False), ('events', False, True), ('joint', True, True)):
                value, detail = candidate.forecast(path, row, geometry, objects, phases,
                    law if use_history else None, use_events)
                points[arm] = old['joint'] if value is None else value
                counts[mode+'/'+arm+'/'+detail['status']] += 1
                details[arm] = detail
            if save:
                shadows.append(dict(query=list(k), time_s=row['prediction_time_s'], points=points, details=details))
            status, truth = prior.prior.prior.target_label(row, path['ready'])
            counts[mode+'/label/'+status] += 1
            if status != 'resolved' or points['v141'] is None:
                continue
            if any(v is None or not math.isfinite(v) or v < 0. for v in points.values()):
                raise AssertionError('V151 lost a finite baseline point')
            r = record([*prior.prior.prior.follower(row), *k[1:]], points, truth)
            now = row['prediction_time_s']
            labels = ['all', 'clearance' if now >= path['cutoff'] else 'active']
            if int(now//3600) == 13:
                labels.append('hour13')
            for label in labels:
                groups[label].append(r)
            selected[k] = points, truth
        modes[mode] = {g: prior.prior.prior.describe(r) for g, r in groups.items()}
        for d in modes[mode].values():
            metrics = d['metrics']
            for arm in ('history', 'events', 'joint'):
                metrics[arm+'_delta_absolute_bias_s'] = abs(metrics[arm+'_bias_s'])-abs(metrics['v149_bias_s'])
            metrics['joint_vs_v141_delta_absolute_bias_s'] = abs(metrics['joint_bias_s'])-abs(metrics['v141_bias_s'])
        paths[mode], predictions[mode] = path, selected
        if save:
            prior.reference.capture.identity.write_trace(root/'candidate_shadows'/f'{seed}_{day}_{mode}.jsonl.gz', shadows)
    switches, denominator = [], 0
    for k in predictions['old'].keys() & predictions['handoff'].keys():
        b, c = [paths[m]['trace'][k] for m in MODES]
        if prior.prior.prior.follower(b) != prior.prior.prior.follower(c):
            continue
        denominator += 1
        if prior.prior.prior.source(b, paths['old']['shadows'][k]) == prior.prior.prior.source(c, paths['handoff']['shadows'][k]):
            continue
        vb, tb = predictions['old'][k]
        vc, tc = predictions['handoff'][k]
        switches.append(record(k, {a: vc[a]-vb[a] for a in ARMS}, tc-tb))
    fields = set(record([0], dict.fromkeys(ARMS, 0.), 0.))-{'episode'}
    burden = dict(queries=denominator, switch_queries=len(switches), metrics={
        f: sum(r[f] for r in switches)/denominator for f in fields}) if denominator else dict(metrics={})
    return dict(seed=seed, day=day, modes=modes, counts=dict(counts),
        paired=dict(switch_burden=burden), execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])


def confirm_pair(args, index):
    seed, day = CASES[index]
    target = args.out_root/'cells'/f'{seed}_{day}.json'
    if target.exists():
        if read(target)['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
            raise AssertionError('confirmation cell source differs')
        return
    model = read(args.out_root/'model.json')
    if (model['status'], model['calibration_days'], model['execution_source']) != (
            'sealed', list(CALIBRATION), os.environ['FREQDUET_DIAGNOSTIC_COMMIT']):
        raise AssertionError('confirmation requires the sealed calibration-only model')
    for mode in MODES:
        prior.reference.collect(args.out_root/'confirmation', args.runtime, seed, day, mode, SAMPLERS[day])
        prior.reference.evaluate_path(args.out_root/'confirmation', args.runtime, seed, day, mode)
    write(target, evaluate_pair(args.out_root/'confirmation', args.runtime, seed, day,
        model['actors'][str(seed)], save=True))


def run(args):
    require(args)
    fn = collect_pair if args.command == 'calibrate' else confirm_pair
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(fn, args, i) for i in range(args.start, args.end)]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=len(jobs), unit=args.command+'_pair')), flush=True)
    write(args.out_root/(args.command+'_shards')/f'{args.start}-{args.end}'/'done.json', dict(status='complete'))
    print('V151_'+args.command.upper()+'_SHARD_COMPLETE', args.start, args.end, flush=True)


def aggregate(args):
    require(args)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in CASES]
    if any(c['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] for c in cells):
        raise AssertionError('confirmation sources differ')
    modes = {m: prior.prior.prior.aggregate(cells, lambda c: c['modes'][m]) for m in MODES}
    paired = prior.prior.prior.aggregate(cells, lambda c: c['paired'])
    counts = dict(sum((Counter(c['counts']) for c in cells), Counter()))

    def upper(result, group, field, strict):
        ci = result.get(group, {}).get('metrics', {}).get(field, {}).get('ci95')
        return ci is not None and (ci[1] < 0. if strict else ci[1] <= 0.)

    complete = not any(n for k, n in counts.items() if any(s in k for s in
        ('missing_same_trip_ready', 'same_trip_bus_mismatch', 'missing_follower_identity')))
    gate = complete and all(all(
        upper(modes[m], 'clearance', prefix+'_delta_mae_s', True) and
        upper(modes[m], 'clearance', prefix+'_delta_absolute_bias_s', True) and
        all(upper(modes[m], g, prefix+'_delta_mae_s', False) for g in ('all', 'hour13')) and
        upper(paired, 'switch_burden', prefix+'_delta_mae_s', False)
        for prefix in ('joint', 'joint_vs_v141')) for m in MODES)
    write(args.out_root/'summary/status.json', dict(status='v151_independent_confirmation_complete',
        registration=read(args.out_root/'protocol.json'), execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        modes=modes, paired=paired, counts=counts, labels_complete=complete, joint_prediction_gate=gate,
        calibration_paths=2*len(CAL_CASES), confirmation_paths=2*len(CASES),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        limitations='Prediction-only; no realized controller gain. Empirical tails do not extrapolate '
            'beyond observed duration support. Historical prior is actor-specific, shared between modes, '
            'and calibrated on eight distinct days; raw samples are not independent days. '
            'V146 current-service interval reference is unchanged, not recalibrated for new points or full ETA. '
            'Only the fixed joint arm is gated; component arms are descriptive, not a best-arm search.'))
    print('V151_INDEPENDENT_CONFIRMATION_COMPLETE', gate, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'calibrate', 'seal', 'confirm', 'aggregate'))
    for name in ('out-root', 'runtime', 'v144-root', 'v145-root', 'v146-root', 'v149-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    dict(preflight=preflight, calibrate=run, seal=seal, confirm=run, aggregate=aggregate)[args.command](args)
