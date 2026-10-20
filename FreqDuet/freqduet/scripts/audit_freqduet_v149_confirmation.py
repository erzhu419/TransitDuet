#!/usr/bin/env python3
"""Read saved V149 points; explain coverage and remaining error without refitting."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import audit_freqduet_v148_eta_components as components
import run_freqduet_v149_phase_clock as frozen


read, write, close = components.read, components.write, components.close
ARMS = frozen.ARMS


def service_parts(parts, delta, structured):
    result = dict(parts)
    if structured:
        result['downstream_service'] += delta
    elif abs(delta) > 1e-7:
        raise AssertionError('cannot allocate a service change inside opaque fallback')
    return result


def burden(rows, denominator):
    if denominator == 0:
        return dict(queries=0, metrics={})
    return dict(queries=denominator, metrics={a+'_mae_s':
        sum(r[a] for r in rows)/denominator for a in ARMS})


def audit_cell(root, out, geometry, seed, day):
    expected = read(root/'cells'/f'{seed}_{day}.json')
    if expected['execution_source'] != 'd7996d60b4':
        raise AssertionError('V149 cell differs from frozen execution source')
    paths, selected, stats, counts, maxima = {}, {}, {}, Counter(), defaultdict(float)
    for mode in frozen.MODES:
        name = f'{seed}_{day}_{mode}'
        receipt = read(root/'confirmation/paths'/(name+'.json'))
        if (receipt['seed'], receipt['day'], receipt['mode'], receipt['sampler_seed']) != (
                seed, day, mode, frozen.SAMPLERS[day]):
            raise AssertionError('saved path differs from registered actor/day/sampler')
        if any(receipt[k] for k in ('actor_updates', 'critic_updates', 'dual_updates', 'main_policy_changed')):
            raise AssertionError('V149 changed frozen deployment')
        path, objects = components.prepare(root/'confirmation', seed, day, mode, geometry)
        points = {tuple(r['query']): r for r in components.prior.trace(
            root/'confirmation/candidate_shadows'/(name+'.jsonl.gz'))}
        if points.keys() != path['trace'].keys() or points.keys() != path['shadows'].keys():
            raise AssertionError('saved candidates lost frozen query support')
        rows, valid = dict(base=[], service=[]), {}
        for k, row in path['trace'].items():
            saved, forecast = path['shadows'][k], points[k]
            status, truth = components.prior.target_label(row, path['ready'])
            if status != 'resolved' or forecast['points']['base'] is None:
                continue
            values, now = forecast['points'], row['prediction_time_s']
            close(values['base'], saved['v141_s'])
            for key, delta in (('clock_minus_base_s', values['clock']-values['base']),
                    ('joint_minus_service_s', values['joint']-values['service'])):
                maxima[mode+'/'+key] = max(maxima[mode+'/'+key], abs(delta))
                counts[mode+'/'+key+'/changed_gt_existing_1e-7'] += int(abs(delta) > 1e-7)
            base_structured = (row['online']['component_used'] or
                saved['v140_status'] == 'passive_stop_component' or
                saved['v141_status'] == 'passive_unissued_ready')
            counts[mode+'/valid/'+forecast['status']] += 1
            counts[mode+'/newly_structured'] += int(forecast['status'] == 'structured' and not base_structured)
            valid[k] = forecast, truth, bool(base_structured)
            if now < path['cutoff']:
                continue
            counts[mode+'/clearance/'+forecast['status']] += 1
            counts[mode+'/clearance/queries_with_unsupported_phase_stops'] += int(forecast['phase_unsupported_stops'] > 0)
            parts, structured, _ = components.decompose(path, row, saved, truth, geometry, *objects)
            adjusted = service_parts(parts, values['service']-values['base'], structured)
            episode = [*components.prior.follower(row), *k[1:]]
            for arm, packet in (('base', parts), ('service', adjusted)):
                rows[arm].append(components.error_record(episode, values[arm]-truth, packet, structured))
        stats[mode] = {arm: components.prior.describe(r) for arm, r in rows.items()}
        for arm in rows:
            for metric in ('bias', 'mae'):
                close(stats[mode][arm]['metrics']['full_'+metric+'_s'],
                    expected['modes'][mode]['clearance']['metrics'][arm+'_'+metric+'_s'])
            if stats[mode][arm]['queries'] != expected['modes'][mode]['clearance']['queries']:
                raise AssertionError('clearance decomposition changed support')
        paths[mode], selected[mode] = path, valid
    groups, denominator = defaultdict(list), 0
    for k in selected['old'].keys() & selected['handoff'].keys():
        b, c = [paths[m]['trace'][k] for m in frozen.MODES]
        if components.prior.follower(b) != components.prior.follower(c):
            continue
        denominator += 1
        if components.prior.source(b, paths['old']['shadows'][k]) == components.prior.source(c, paths['handoff']['shadows'][k]):
            continue
        sb, tb, structured_b = selected['old'][k]
        sc, tc, structured_c = selected['handoff'][k]
        values = {a: abs(sc['points'][a]-sb['points'][a]-(tc-tb)) for a in ARMS}
        label = 'structured_both' if structured_b and structured_c else 'retained_fallback'
        groups[label].append(values)
        groups['all_switches'].append(values)
        for mode, p, structured in (('old', sb, structured_b), ('handoff', sc, structured_c)):
            counts['switch/'+mode+'/'+p['status']] += 1
            if not structured:
                counts['switch/opaque/'+mode+'/'+p['status']] += 1
                counts['switch/opaque_source/'+mode+'/'+paths[mode]['trace'][k]['evidence']['follower_source']] += 1
    paired = {g: burden(groups[g], denominator) for g in ('all_switches', 'structured_both', 'retained_fallback')}
    for arm in ARMS:
        close(paired['all_switches']['metrics'][arm+'_mae_s'],
            expected['paired']['switch_burden']['metrics'][arm+'_mae_s'])
    counts['switch/total'] = len(groups['all_switches'])
    for g in ('structured_both', 'retained_fallback'):
        counts['switch/'+g] = len(groups[g])
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day,
        components=stats, paired=paired, counts=dict(counts), maxima=dict(maxima)))


def margins(cells):
    result = {}
    for mode, scope in itertools.product(frozen.MODES, ('clearance', 'all', 'hour13')):
        for field in ('service_delta_mae_s', 'joint_delta_mae_s', 'service_delta_absolute_bias_s'):
            by_axis = {}
            for axis in ('seed', 'day'):
                groups = defaultdict(list)
                for c in cells:
                    groups[c[axis]].append(c['modes'][mode][scope]['metrics'][field])
                by_axis[axis] = {str(k): fmean(v) for k, v in sorted(groups.items())}
            result[mode+'/'+scope+'/'+field] = by_axis
    return result


def run(args):
    status = read(args.v149_root/'summary/status.json')
    calibration = read(args.v146_root/'model.json')
    if status['status'] != 'v149_independent_validation_complete' or not status['labels_complete']:
        raise AssertionError('requires complete V149 independent readout')
    if (calibration['status'], calibration['lower_extension_s'], calibration['upper_extension_s']) != ('sealed', 3., 0.):
        raise AssertionError('V146 calibration changed')
    if any(status['registration'][k] != v for k, v in frozen.registration().items()):
        raise AssertionError('V149 preregistration changed')
    cells = [read(args.v149_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    if [(c['seed'], c['day']) for c in cells] != list(frozen.CASES):
        raise AssertionError('incomplete crossed actor/day grid')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v149_confirmation.py', '-v'], check=True)
    geometry = frozen.candidate.model.load_geometry(args.runtime)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(audit_cell, args.v149_root, args.out_root, geometry, s, d) for s, d in frozen.CASES]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 32 == 0:
                print('V149_READONLY_PROGRESS', n, len(jobs), flush=True)
    audits = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    decomposition = {m: components.prior.aggregate(audits, lambda c: c['components'][m]) for m in frozen.MODES}
    paired = components.prior.aggregate(audits, lambda c: c['paired'])
    compact = dict(status='v149_readonly_confirmation_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_source='d7996d60b4',
        cells=len(cells), saved_paths=2*len(cells), original_gate=status['joint_prediction_gate'],
        components=decomposition, paired=paired, margins=margins(cells),
        counts=dict(sum((Counter(c['counts']) for c in audits), Counter())),
        maxima={k: max(c['maxima'].get(k, 0.) for c in audits) for k in audits[0]['maxima']},
        endpoint_and_switch_burden_reproduced=True, new_rollouts=0, fitted_parameters=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Post-readout explanatory audit, not a new confirmation. Existing 1e-7 '
            'accounting tolerance classifies numerical identity only; the original raw gate is unchanged. '
            'Component MAE allocations are signed additive allocations, not causal removal benefits.')
    write(args.out_root/'summary/status.json', compact)
    print('V149_READONLY_CONFIRMATION_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v149-root', 'v146-root', 'runtime', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
