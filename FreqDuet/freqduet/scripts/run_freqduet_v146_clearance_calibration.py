#!/usr/bin/env python3
"""Preregister fresh calibration and sealed confirmation of clearance intervals."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import run_freqduet_v145_independent_intervals as reference
from freqduet_clearance_calibration import fit, cell_stats, SCOPES
from freqduet_interval_validation import crossed, mean
from scripts.run_freqduet_protocol_v2_matrix import hierarchical_bootstrap


ACTORS, MODES = reference.ACTORS, reference.MODES
DAYS = dict(calibration=tuple(range(2551001, 2551033)), confirmation=tuple(range(2552001, 2552033)))
SAMPLERS = {cohort: {d: base+i for i, d in enumerate(DAYS[cohort])}
    for cohort, base in (('calibration', 1461000000), ('confirmation', 1462000000))}
ITEMS = {c: tuple(itertools.product(ACTORS, days, MODES)) for c, days in DAYS.items()}
read, write = reference.read, reference.write


def registration():
    return dict(stage='v146_clearance_calibration', actors=list(ACTORS), modes=list(MODES),
        days={c: list(v) for c, v in DAYS.items()}, samplers={c: {str(d): s for d, s in v.items()}
            for c, v in SAMPLERS.items()}, paths_per_cohort=768, checkpoint_ep=199,
        point_sources=['4df22f4fa6', '9adf0bc09c'], collector_source='6eaa4a389b',
        rank_source='418a3f001a', window_s=3600., lower_tail=.1, upper_tail=.1,
        candidate='two shared nonnegative endpoint offsets on native-finite post-demand service intervals',
        calibration_quantile=.9, minimum_calibration_days_per_actor_mode=9,
        calibration_weights='equal day/nonempty actor-mode cell/episode/query; unresolved finite labels block fit',
        interval_score='width+10*(lower_shortfall+upper_excess), nominal coverage0.8',
        primary_scopes=['service_clearance', 'service_hour13', 'service_all'],
        coverage_gate='one-sided crossed95 lower>=0.8 in both modes and every primary scope',
        score_gate='crossed95 upper<0 for paired candidate-base clearance interval score in both modes',
        bootstrap_draws=5000, bootstrap_seed=20260803,
        confirmation_requires_seal=True, unchanged_native_support=True,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)


def register(args):
    target, plan = args.out_root/'protocol.json', registration()
    if target.exists():
        old = read(target)
        if any(old[k] != v for k, v in plan.items()):
            raise AssertionError('V146 preregistration changed')
        return old
    new_days = set(itertools.chain.from_iterable(DAYS.values()))
    if len(new_days) != 64:
        raise AssertionError('calibration and confirmation days overlap')
    archive = read(args.v144_root/'details/admission.json')
    used = set(read(args.v145_root/'protocol.json')['demand_days'])
    for row in archive['metadata']:
        used.update(row['days'])
    for row in archive['bundles']:
        if row['identity'] is not None:
            used.add(row['identity'][1])
    if used & new_days:
        raise AssertionError('new V146 demand days have prior archive usage')
    for path in Path(__file__).parent.glob('*.py'):
        if path.name == Path(__file__).name:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in new_days:
                raise AssertionError(f'day already in {path.name}:{node.lineno}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
                if node.args and all(isinstance(v, ast.Constant) and isinstance(v.value, int) for v in node.args):
                    if any(d in range(*(v.value for v in node.args)) for d in new_days):
                        raise AssertionError(f'day overlaps generator {path.name}:{node.lineno}')
    plan.update(registration_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], prior_days_checked=len(used))
    write(target, plan)
    print('V146_REGISTERED_32_CALIBRATION_32_CONFIRMATION_DAYS', flush=True)
    return plan


def preflight(args):
    register(args)
    proof = read(args.v145_root/'preflight/done.json')
    if (proof['execution_source'], proof['status']) != (
        '6eaa4a389b', 'consumed_actions_clocks_points_and_bounds_exact'):
        raise AssertionError('unchanged V145 collector needs its original exact-replay proof')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v146_*.py', '-v'], check=True)
    write(args.out_root/'preflight/done.json', dict(status='v146_tests_and_frozen_collector_proof_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        collector_reference=str(args.v145_root/'preflight/done.json'), new_days_read=False))
    print('V146_PREFLIGHT_COMPLETE', flush=True)


def require_preflight(args):
    plan = read(args.out_root/'protocol.json')
    if any(plan[k] != v for k, v in registration().items()):
        raise AssertionError('V146 protocol differs from registration')
    if read(args.out_root/'preflight/done.json')['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('V146 execution source differs from preflight')


def load_rows(root, name):
    with gzip.open(root/'intervals'/(name+'.jsonl.gz'), 'rt') as f:
        return [json.loads(line) for line in f]


def sealed_model(root):
    model = read(root/'model.json')
    if model['status'] != 'sealed' or model['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('confirmation needs this source\'s sealed calibration-only model')
    return model


def run_one(root, runtime, cohort, index, model):
    seed, day, mode = ITEMS[cohort][index]
    name, phase = f'{seed}_{day}_{mode}', root/cohort
    marker = phase/'completed'/(name+'.done')
    if marker.exists():
        return index
    receipt = reference.collect(phase, runtime, seed, day, mode, SAMPLERS[cohort][day])
    if not (phase/'validation_cells'/(name+'.json')).exists():
        rows, _, _ = reference.evaluate_path(phase, runtime, seed, day, mode)
    else:
        rows = load_rows(phase, name)
    if cohort == 'confirmation':
        cutoff = receipt['contract']['clock']['demand_end_time_s']
        write(phase/'calibrated_cells'/(name+'.json'), dict(seed=seed, day=day, mode=mode,
            stats=cell_stats(rows, model, cutoff), cutoff_s=cutoff,
            main_policy_changed=False, original_point_files_untouched=True))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return index


def run_batch(args):
    require_preflight(args)
    model = sealed_model(args.out_root) if args.cohort == 'confirmation' else None
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        tasks = [pool.submit(run_one, args.out_root, args.runtime, args.cohort, i, model)
            for i in range(args.start, args.end)]
        for n, task in enumerate(as_completed(tasks), 1):
            task.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=args.end-args.start,
                unit=args.cohort+'_frozen_path', elapsed_seconds=time.monotonic()-started)), flush=True)
    write(args.out_root/args.cohort/'shards'/f'{args.start}-{args.end}.json', dict(status='complete'))


def wait_for_cohort(root, cohort):
    markers = [root/cohort/'completed'/f'{s}_{d}_{m}.done' for s, d, m in ITEMS[cohort]]
    started = time.monotonic()
    while not all(p.exists() for p in markers):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=sum(p.exists() for p in markers),
            total=len(markers), unit=cohort+'_completed_path', elapsed_seconds=time.monotonic()-started)), flush=True)
        time.sleep(30)


def seal(args):
    require_preflight(args)
    if (args.out_root/'model.json').exists():
        model = read(args.out_root/'model.json')
    else:
        wait_for_cohort(args.out_root, 'calibration')
        phase, cells = args.out_root/'calibration', []
        for seed, day, mode in ITEMS['calibration']:
            name = f'{seed}_{day}_{mode}'
            receipt = read(phase/'paths'/(name+'.json'))
            cutoff = receipt['contract']['clock']['demand_end_time_s']
            rows = [r for r in load_rows(phase, name) if r['kind'] == 'service' and r['time_s'] >= cutoff]
            cells.append(dict(seed=seed, day=day, mode=mode, cutoff=cutoff, rows=rows))
        model = fit(cells, ACTORS, MODES, DAYS['calibration'])
        model.update(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            calibration_days=list(DAYS['calibration']), confirmation_outcomes_read=False,
            point_parameters_changed=False, native_window_s=3600., native_tails=[.1, .1])
        write(args.out_root/'model.json', model)
    if model['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('sealed model source changed')
    write(args.out_root/'seal_summary/status.json', model)
    if model['status'] == 'sealed':
        write(args.out_root/'seal_summary/done.json', model)
        print('V146_CALIBRATION_SEALED', flush=True)
    else:
        print('V146_CALIBRATION_INCONCLUSIVE_CONFIRMATION_NOT_RELEASED', flush=True)


def contrast(cells, scope):
    values = [c['stats'][scope]['paired']['score_delta'] for c in cells]
    result = dict(cells=len(cells), contributing_cells=sum(v is not None for v in values),
        mean=mean([v for v in values if v is not None]), ci95=None, improvement_confirmed=False)
    if any(v is None for v in values):
        return result
    frame = pd.DataFrame([dict(train_seed=c['seed'], eval_seed=c['day'], delta=v) for c, v in zip(cells, values)])
    draws = hierarchical_bootstrap(frame, 'delta', draws=5000)
    result.update(ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
        improvement_confirmed=bool(np.percentile(draws, 97.5) < 0.))
    return result


def aggregate(args):
    require_preflight(args)
    model = sealed_model(args.out_root)
    wait_for_cohort(args.out_root, 'confirmation')
    phase = args.out_root/'confirmation'
    cells = [read(phase/'calibrated_cells'/f'{s}_{d}_{m}.json') for s, d, m in ITEMS['confirmation']]
    if [(c['seed'], c['day'], c['mode']) for c in cells] != list(ITEMS['confirmation']):
        raise AssertionError('confirmation grid differs from preregistration')
    results = {}
    for mode in MODES:
        selected = [c for c in cells if c['mode'] == mode]
        results[mode] = {}
        for scope in SCOPES:
            entry = {arm: crossed([dict(seed=c['seed'], day=c['day'], stats=c['stats'][scope][arm])
                for c in selected], ACTORS, DAYS['confirmation']) for arm in ('base', 'candidate')}
            if any(c['stats'][scope]['base']['finite_n'] != c['stats'][scope]['candidate']['finite_n']
                for c in selected):
                raise AssertionError('confirmation changed native support')
            entry['paired_interval_score'] = contrast(selected, scope)
            results[mode][scope] = entry
    coverage = all(results[m][s]['candidate']['coverage_confirmed'] for m in MODES
        for s in registration()['primary_scopes'])
    score = all(results[m]['service_clearance']['paired_interval_score']['improvement_confirmed'] for m in MODES)
    write(args.out_root/'summary/status.json', dict(status='v146_independent_confirmation_complete',
        registration=read(args.out_root/'protocol.json'), calibration=model, results=results,
        primary_coverage_confirmed=coverage, primary_interval_score_improved=score,
        primary_joint_pass=coverage and score, new_rollouts=1536, new_demand_days=64,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        controller_admitted=False, point_parameters_changed=False,
        limitations='Two empirical endpoint offsets, not an IID conformal guarantee. Fixed archived actors '
            'and simulator demand days; target is current-stage native-finite intervals, not full ETA or '
            'control efficacy. Missing labels/support are explicit, incomplete crossed grids inconclusive. '
            'Confirmation is single-use; no adjustment of tails, window, model or admission thresholds.'))
    print('V146_INDEPENDENT_CONFIRMATION_COMPLETE', coverage and score, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'run', 'seal', 'aggregate'))
    for name in ('out-root', 'runtime', 'v145-root', 'v144-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cohort', choices=tuple(DAYS), default='calibration')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=768)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    {'preflight': preflight, 'run': run_batch, 'seal': seal, 'aggregate': aggregate}[args.command](args)
