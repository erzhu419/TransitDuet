#!/usr/bin/env python3
"""Read frozen V151 points and explain their bias shifts without another rollout."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import run_freqduet_v151_historical_service as frozen


parts = frozen.prior.prior
stats = parts.prior
read, write, close = parts.read, parts.write, parts.close
SUPPORTED = {'structured', 'historical_current_service',
    'observed_same_clock_ready', 'observed_same_clock_launch'}
CHANNELS = ('event_shift_s', 'current_residence_shift_s', 'downstream_shift_s', 'boundary_shift_s')
MECHANISMS = ('downstream_only', 'current_and_downstream', 'observed_ready',
    'observed_launch', 'recovered_tail', 'retained_reference')


def attribution(points, initial_shift=None):
    delta = points['joint']-points['events']
    result = dict(event_shift_s=points['events']-points['v149'],
        current_residence_shift_s=0. if initial_shift is None else initial_shift,
        downstream_shift_s=0. if initial_shift is None else delta-initial_shift,
        boundary_shift_s=delta if initial_shift is None else 0.)
    close(sum(result.values()), points['joint']-points['v149'])
    return result


def describe(rows):
    result = stats.describe(rows)
    if rows:
        m = result['metrics']
        m['joint_delta_absolute_bias_s'] = abs(m['joint_bias_s'])-abs(m['v149_bias_s'])
    return result


def budgets(rows):
    labels = ['scope/active', 'scope/clearance', *('mechanism/'+m for m in MECHANISMS)]
    return {label: stats.describe([dict(episode=r['episode'], fraction=float(label in r['labels']),
        **{f: r['values'][f] if label in r['labels'] else 0. for f in CHANNELS}) for r in rows])
        for label in labels}


def case(root, out, runtime, seed, day, model):
    expected = read(root/'cells'/f'{seed}_{day}.json')
    geometry, modes, counts = frozen.candidate.frozen.model.load_geometry(runtime), {}, Counter()
    for mode in frozen.MODES:
        path, objects = parts.prepare(root/'confirmation', seed, day, mode, geometry)
        path['_observed_ready'] = frozen.candidate.ObservedReady(path['ready'])
        law = frozen.candidate.ServiceLaw(model['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
        name = f'{seed}_{day}_{mode}.jsonl.gz'
        saved = {tuple(r['query']): r for r in stats.trace(root/'confirmation/candidate_shadows'/name)}
        groups, hours, mechanisms, ledger = defaultdict(list), defaultdict(list), defaultdict(list), []
        if saved.keys() != path['trace'].keys():
            raise AssertionError('readout lost frozen query support')
        for k, row in path['trace'].items():
            s = saved[k]
            status, truth = stats.target_label(row, path['ready'])
            if status != 'resolved' or s['points']['v141'] is None:
                continue
            points, detail = s['points'], s['details']
            es, js = detail['events']['status'], detail['joint']['status']
            initial = None
            if es in SUPPORTED and js in SUPPORTED:
                pe, se, _ = frozen.candidate.position(path, row, objects[-1], geometry, objects, events=True)
                pj, sj, support = frozen.candidate.position(path, row, objects[-1], geometry, objects, law, True)
                if pe is None or pj is None or (se, sj) != (es, js):
                    raise AssertionError('saved structured initial-state support changed')
                if any(pe[f] != pj[f] for f in ('origin', 'target', 'progress', 'direction', 'now')):
                    raise AssertionError('joint history changed physical origin on common support')
                if support is not None:
                    close(support['mean'], detail['joint']['current_support']['mean'])
                initial = pj['initial_s']-pe['initial_s']
                mechanism = {'structured': 'downstream_only', 'historical_current_service': 'current_and_downstream',
                    'observed_same_clock_ready': 'observed_ready', 'observed_same_clock_launch': 'observed_launch'}[js]
            else:
                mechanism = 'recovered_tail' if js in SUPPORTED else 'retained_reference'
            shifts = attribution(points, initial)
            r = frozen.record([*stats.follower(row), *k[1:]], points, truth)
            fields = ('v149_mae_s', 'joint_mae_s', 'v149_bias_s', 'joint_bias_s',
                'joint_delta_mae_s', 'history_delta_mae_s', 'events_delta_mae_s')
            values = {f: r[f] for f in fields}
            values.update(shifts, joint_minus_history_mae_s=abs(points['joint']-truth)-abs(points['history']-truth))
            values['mae_interaction_s'] = (values['joint_delta_mae_s']-
                values['history_delta_mae_s']-values['events_delta_mae_s'])
            now = row['prediction_time_s']
            phase = 'clearance' if now >= path['cutoff'] else 'active'
            labels = ['all', phase]
            if int(now//3600) == 13:
                labels.append('hour13')
            record = dict(episode=r['episode'], **values)
            for label in labels:
                groups[label].append(record)
            hours['hour/'+str(int(now//3600))].append(dict(episode=r['episode'],
                **{f: values[f] for f in ('v149_mae_s', 'joint_mae_s', 'v149_bias_s', 'joint_bias_s', 'joint_delta_mae_s')}))
            mechanisms[phase+'/'+mechanism].append(dict(episode=r['episode'],
                **{f: values[f] for f in ('v149_mae_s', 'joint_mae_s', 'v149_bias_s', 'joint_bias_s', 'joint_delta_mae_s')}))
            ledger.append(dict(episode=r['episode'], values=shifts,
                labels=['scope/'+phase, 'mechanism/'+mechanism]))
            counts[mode+'/'+mechanism] += 1
        summary = {g: describe(r) for g, r in groups.items()}
        for g, d in summary.items():
            before = expected['modes'][mode][g]
            if d['queries'] != before['queries']:
                raise AssertionError('diagnostic population differs from frozen confirmation')
            for field in ('v149_mae_s', 'joint_mae_s', 'v149_bias_s', 'joint_bias_s',
                    'joint_delta_mae_s', 'joint_delta_absolute_bias_s'):
                close(d['metrics'][field], before['metrics'][field])
            close(sum(d['metrics'][f] for f in CHANNELS),
                d['metrics']['joint_bias_s']-d['metrics']['v149_bias_s'])
        budget = budgets(ledger)
        for prefix in ('scope/', 'mechanism/'):
            close(sum(d['metrics'][f] for g, d in budget.items() if g.startswith(prefix) for f in CHANNELS),
                summary['all']['metrics']['joint_bias_s']-summary['all']['metrics']['v149_bias_s'])
        modes[mode] = dict(scopes=summary, hours={h: stats.describe(r) for h, r in hours.items()},
            mechanisms={g: stats.describe(r) for g, r in mechanisms.items()}, budgets=budget)
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes, counts=dict(counts)))


def margins(root):
    cells = [read(root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    result = {}
    for mode in frozen.MODES:
        for scope in ('all', 'clearance', 'hour13'):
            for axis in ('seed', 'day'):
                groups = defaultdict(list)
                for c in cells:
                    groups[c[axis]].append(c['modes'][mode][scope]['metrics'])
                result[mode+'/'+scope+'/'+axis] = {str(k): {f: fmean(r[f] for r in rows) for f in
                    ('joint_delta_mae_s', 'joint_delta_absolute_bias_s', 'joint_bias_s')} for k, rows in groups.items()}
    return result


def run(args):
    parent, model = read(args.v151_root/'summary/status.json'), read(args.v151_root/'model.json')
    if parent['execution_source'] != '1b783ea8f3' or not parent['labels_complete']:
        raise AssertionError('requires complete frozen V151 confirmation')
    if (model['status'], model['calibration_days'], model['execution_source']) != (
            'sealed', list(frozen.CALIBRATION), '1b783ea8f3'):
        raise AssertionError('V151 prior seal differs')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v151_confirmation.py', '-v'], check=True)
    case(args.v151_root, args.out_root, args.runtime, *frozen.CASES[0], model)
    print('V151_READOUT_FIRST_CASE_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v151_root, args.out_root, args.runtime, s, d, model) for s, d in frozen.CASES[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V151_READOUT_PROGRESS', n, len(frozen.CASES), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    results = {m: {section: stats.aggregate(cells, lambda c: c['modes'][m][section])
        for section in ('scopes', 'hours', 'mechanisms', 'budgets')} for m in frozen.MODES}
    write(args.out_root/'summary/status.json', dict(status='v151_readonly_confirmation_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen_source='1b783ea8f3',
        cases=len(cells), saved_paths=2*len(cells), results=results, margins=margins(args.v151_root),
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        original_joint_gate=parent['joint_prediction_gate'], original_metrics_and_bias_budgets_reproduced=True,
        new_rollouts=0, calibration_refit=False, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, window_s=3600., native_interval_tails=[.1, .1],
        limitations='Consumed confirmation readout only, no new gate or parameter selection. '
            'Channels are exact prediction-point shifts versus event-only, not true component errors '
            'or causal passenger benefits. Unsupported/changed structural support remains a whole-point '
            'boundary channel, not an invented component decomposition. Conditional incomplete '
            'crossed groups have no CI. Bias is per-cell mean error; absolute bias is not MAE.'))
    print('V151_READONLY_CONFIRMATION_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v151-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
