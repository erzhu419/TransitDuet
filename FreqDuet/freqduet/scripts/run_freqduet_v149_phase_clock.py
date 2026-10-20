#!/usr/bin/env python3
"""Preregister and validate fixed shadow predictors on fresh frozen-control days."""

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

import audit_freqduet_v148_eta_components as prior
import freqduet_phase_clock_eta as candidate
import run_freqduet_v145_independent_intervals as reference


ACTORS, MODES = reference.ACTORS, reference.MODES
DAYS = tuple(range(2651001, 2651033))
SAMPLERS = {day: 1490000000+i for i, day in enumerate(DAYS)}
CASES = tuple(itertools.product(ACTORS, DAYS))
ARMS = ('base', 'service', 'clock', 'joint')
read, write = reference.read, reference.write


def registration():
    return dict(stage='v149_phase_clock_independent_validation', actors=list(ACTORS),
        days=list(DAYS), sampler_seeds={str(d): SAMPLERS[d] for d in DAYS}, modes=list(MODES),
        cases=len(CASES), paths=2*len(CASES), arms=list(ARMS), checkpoint_ep=199,
        service='completed same-stop means conditioned on observed arrival before/after published demand cutoff; '
            'query-time frozen coefficients, sequential predicted arrival phase, native mean if unsupported',
        clock='same physical position and strict stop/motion prefix across all AVL speed sources; '
            'retain frozen point if common components unavailable',
        window_s=3600., all_past_fallback='only if same-phase one-hour pool empty',
        current_service_points='unchanged', native_interval_tails=[.1, .1],
        v146_endpoint_extensions_s=[3., 0.], actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, fitted_parameters=0,
        inferential_unit='query within follower-stop episode, then equal actor/day cell; 5000 crossed draws',
        bootstrap_seed=20260803,
        joint_gate='both modes: paired clearance MAE and absolute-cell-bias CI95 upper<0; '
            'all-day and hour13 MAE CI95 upper<=0; same-identity source-switch MAE burden CI95 upper<=0',
        component_checks='service clearance MAE CI95 upper<0 in both modes; clock switch burden upper<0; '
            'descriptive mechanism checks only, no best-arm selection or controller admission',
        missing='unresolved eligible same-vehicle labels block gate; missing crossed cells inconclusive',
        confirmation_single_use=True, controller_admitted=False)


def register(args):
    plan, target = registration(), args.out_root/'protocol.json'
    if target.exists():
        old = read(target)
        if any(old[k] != v for k, v in plan.items()):
            raise AssertionError('V149 preregistration changed')
        return
    used = set()
    archive = read(args.v144_root/'details/admission.json')
    for row in archive['metadata']:
        used.update(row['days'])
    used.update(row['identity'][1] for row in archive['bundles'] if row['identity'] is not None)
    used.update(read(args.v145_root/'protocol.json')['demand_days'])
    for days in read(args.v146_root/'protocol.json')['days'].values():
        used.update(days)
    if used.intersection(DAYS):
        raise AssertionError('confirmation demand days already consumed')
    for path in Path(__file__).parent.glob('*.py'):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in DAYS:
                raise AssertionError(f'day already in {path.name}:{node.lineno}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
                if node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args):
                    if any(d in range(*(a.value for a in node.args)) for d in DAYS):
                        raise AssertionError(f'day overlaps {path.name}:{node.lineno}')
    calibration = read(args.v146_root/'model.json')
    if (calibration['status'], calibration['lower_extension_s'], calibration['upper_extension_s']) != ('sealed', 3., 0.):
        raise AssertionError('V146 calibration is not frozen reference')
    plan.update(registration_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], prior_days_checked=len(used))
    write(target, plan)
    print('V149_REGISTERED_32_FRESH_DAYS', flush=True)


def preflight(args):
    register(args)
    proof = read(args.v145_root/'preflight/done.json')
    if (proof['execution_source'], proof['status']) != ('6eaa4a389b', 'consumed_actions_clocks_points_and_bounds_exact'):
        raise AssertionError('unchanged collector lacks its established equivalence proof')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v149_*.py', '-v'], check=True)
    # A consumed pair tests baseline reconstruction, not predictor admission.
    evaluate_pair(args.v146_root/'confirmation', args.runtime, ACTORS[0], 2552001)
    write(args.out_root/'preflight/done.json', dict(status='complete', new_days_read=False,
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT']))
    print('V149_PREFLIGHT_COMPLETE', flush=True)


def require_preflight(args):
    plan = read(args.out_root/'protocol.json')
    if any(plan[k] != v for k, v in registration().items()):
        raise AssertionError('V149 registered design changed')
    if read(args.out_root/'preflight/done.json')['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('V149 source changed after preflight')


def record(episode, values, truth):
    errors = {a: v-truth for a, v in values.items()}
    return dict(episode=episode, **{a+'_mae_s': abs(v) for a, v in errors.items()},
        **{a+'_bias_s': v for a, v in errors.items()},
        **{a+'_delta_mae_s': abs(errors[a])-abs(errors['base']) for a in ARMS[1:]})


def evaluate_pair(root, runtime, seed, day, save_shadows=False):
    geometry = candidate.model.load_geometry(runtime)
    paths, predictions, modes, counts = {}, {}, {}, Counter()
    for mode in MODES:
        path, objects = prior.prepare(root, seed, day, mode, geometry)
        phases = candidate.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        selected, groups, shadows = {}, defaultdict(list), []
        for k, row in path['trace'].items():
            values, detail = candidate.forecasts(path, row, path['shadows'][k], geometry, objects, phases)
            if save_shadows:
                shadows.append(dict(query=list(k), time_s=row['prediction_time_s'], points=values, **detail))
            counts[mode+'/prediction/'+detail['status']] += 1
            for support in ('phase_supported_stops', 'phase_unsupported_stops'):
                counts[mode+'/'+support] += detail[support]
            # Same-vehicle future labels are attached only after every point is fixed.
            status, truth = prior.prior.target_label(row, path['ready'])
            counts[mode+'/label/'+status] += 1
            if status != 'resolved' or values['base'] is None:
                continue
            if any(v is None or not math.isfinite(v) or v < 0. for v in values.values()):
                raise AssertionError('candidate lost a finite baseline point')
            for arm in ARMS[1:]:
                counts[mode+'/'+arm+'_changed_queries'] += bool(abs(values[arm]-values['base']) > 1e-7)
            r = record([*prior.prior.follower(row), *k[1:]], values, truth)
            now = row['prediction_time_s']
            labels = ['all', 'clearance' if now >= path['cutoff'] else 'active']
            if int(now//3600) == 13:
                labels.append('hour13')
            for label in labels:
                groups[label].append(r)
            selected[k] = values, truth
        paths[mode], predictions[mode] = path, selected
        if save_shadows:
            reference.capture.identity.write_trace(root/'candidate_shadows'/f'{seed}_{day}_{mode}.jsonl.gz', shadows)
        modes[mode] = {g: prior.prior.describe(rows) for g, rows in groups.items()}
        for stats in modes[mode].values():
            metrics = stats['metrics']
            for arm in ARMS[1:]:
                metrics[arm+'_delta_absolute_bias_s'] = abs(metrics[arm+'_bias_s'])-abs(metrics['base_bias_s'])
    switches, matched = [], []
    for k in predictions['old'].keys() & predictions['handoff'].keys():
        b, c = [paths[m]['trace'][k] for m in MODES]
        if prior.prior.follower(b) != prior.prior.follower(c):
            counts['paired/follower_identity_changed'] += 1
            continue
        vb, tb = predictions['old'][k]
        vc, tc = predictions['handoff'][k]
        r = record(k, {a: vc[a]-vb[a] for a in ARMS}, tc-tb)
        matched.append(r)
        if prior.prior.source(b, paths['old']['shadows'][k]) != prior.prior.source(c, paths['handoff']['shadows'][k]):
            switches.append(r)
    fields = set(record([0], dict.fromkeys(ARMS, 0.), 0.))-{'episode'}
    burden = dict(queries=len(matched), switch_queries=len(switches), metrics={
        f: sum(r[f] for r in switches)/len(matched) for f in fields}) if matched else dict(metrics={})
    return dict(seed=seed, day=day, modes=modes, counts=dict(counts),
        paired=dict(switch_burden=burden, switch_conditional=prior.prior.describe(switches),
            matched=prior.prior.describe(matched)),
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])


def run_one(root, runtime, index):
    seed, day = CASES[index]
    target = root/'cells'/f'{seed}_{day}.json'
    if target.exists():
        if read(target)['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
            raise AssertionError('cached cell source differs')
        return
    for mode in MODES:
        reference.collect(root/'confirmation', runtime, seed, day, mode, SAMPLERS[day])
        if not (root/'confirmation/validation_cells'/f'{seed}_{day}_{mode}.json').exists():
            reference.evaluate_path(root/'confirmation', runtime, seed, day, mode)
    write(target, evaluate_pair(root/'confirmation', runtime, seed, day, save_shadows=True))


def run(args):
    require_preflight(args)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(run_one, args.out_root, args.runtime, i) for i in range(args.start, args.end)]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(
                dict(current=n, total=args.end-args.start, unit='fresh_frozen_pair')), flush=True)
    write(args.out_root/'shards'/f'{args.start}-{args.end}.json', dict(status='complete'))
    if all((args.out_root/'cells'/f'{s}_{d}.json').exists() for s, d in CASES):
        aggregate(args)


def aggregate(args):
    require_preflight(args)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in CASES]
    modes = {m: prior.prior.aggregate(cells, lambda c: c['modes'][m]) for m in MODES}
    paired = prior.prior.aggregate(cells, lambda c: c['paired'])
    counts = dict(sum((Counter(c['counts']) for c in cells), Counter()))

    def upper(result, group, field, strict):
        ci = result.get(group, {}).get('metrics', {}).get(field, {}).get('ci95')
        return ci is not None and (ci[1] < 0. if strict else ci[1] <= 0.)

    labels_complete = not any(n for k, n in counts.items() if any(x in k for x in
        ('missing_same_trip_ready', 'same_trip_bus_mismatch', 'missing_follower_identity')))
    joint = labels_complete and all(
        upper(modes[m], 'clearance', 'joint_delta_mae_s', True) and
        upper(modes[m], 'clearance', 'joint_delta_absolute_bias_s', True) and
        all(upper(modes[m], g, 'joint_delta_mae_s', False) for g in ('all', 'hour13')) for m in MODES)
    joint = joint and upper(paired, 'switch_burden', 'joint_delta_mae_s', False)
    write(args.out_root/'summary/status.json', dict(status='v149_independent_validation_complete',
        registration=read(args.out_root/'protocol.json'), modes=modes, paired=paired, counts=counts,
        joint_prediction_gate=joint, labels_complete=labels_complete,
        service_mechanism_check=all(upper(modes[m], 'clearance', 'service_delta_mae_s', True) for m in MODES),
        clock_mechanism_check=upper(paired, 'switch_burden', 'clock_delta_mae_s', True),
        new_demand_days=len(DAYS), new_rollouts=2*len(CASES), actor_updates=0,
        critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        limitations='Binary published demand/clearance phase, not future queue/load knowledge. '
            'Prediction-only validation under fixed archived controls; no realized control gain. '
            'Current-stage V146 intervals are unchanged, not full-ETA uncertainty. '
            'No-switch zero burden is an observed zero contribution, not imputed conditional error. '
            'Only the fixed joint candidate has an admission gate; component arms are diagnostics.'))
    print('V149_INDEPENDENT_VALIDATION_COMPLETE', joint, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'v144-root', 'v145-root', 'v146-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(CASES))
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    dict(preflight=preflight, run=run, aggregate=aggregate)[args.command](args)
