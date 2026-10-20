#!/usr/bin/env python3
"""Frozen V154 running prior on the disjoint, previously audited V145 archive."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v154_running_prior as frozen


read, write, stats = frozen.read, frozen.write, frozen.stats


def archive_cases(parent, running, services):
    reg = parent['registration']
    if (parent['status'], reg['registration_source']) != (
            'v145_independent_interval_validation_complete', '6eaa4a389b'):
        raise AssertionError('requires complete original V145 archive')
    if reg['archived_actors'] != list(frozen.frozen.ACTORS) or reg['modes'] != ['old', 'handoff']:
        raise AssertionError('external archive actors/modes differ')
    if reg['checkpoint_ep'] != 199 or reg['window_s'] != 3600.:
        raise AssertionError('external archive deployment/window differs')
    days = reg['demand_days']
    if set(days) & (set(running['days']) | set(services['calibration_days'])):
        raise AssertionError('archive validation days overlap either prior')
    return list(itertools.product(reg['archived_actors'], days))


def report(results):
    missing, violations = [], []
    for mode in frozen.frozen.MODES:
        requirements = [(g, 'delta_full_mae_s', g in ('all', 'hour2')) for g in
            ('all', 'hour2', 'clearance', 'hour13', 'low_support')]
        requirements += [(g, 'delta_running_mae_s', False) for g in ('all', 'hour2', 'clearance', 'hour13')]
        for scope, field, strict in requirements:
            ci = results[mode].get(scope, {}).get('metrics', {}).get(field, {}).get('ci95')
            key = '/'.join((mode, scope, field))
            if ci is None:
                missing.append(key)
            elif ci[1] > 0. or (strict and ci[1] == 0.):
                violations.append(key)
    return dict(fixed_joint_gate=not missing and not violations,
        insufficient_crossed_support=missing, unmet_interval_requirements=violations)


def run(args):
    running, services = read(args.v154_root/'model.json'), read(args.v151_root/'model.json')
    original = read(args.v154_root/'summary/status.json')
    archive = read(args.archive_root/'summary/status.json')
    if (original['status'], original['execution_source']) != ('v154_lodo_development_complete', '5be4cea534'):
        raise AssertionError('requires complete frozen V154 development')
    if running['status'] != 'sealed_development_moments' or running['days'] != list(frozen.frozen.CALIBRATION):
        raise AssertionError('V154 running moments differ')
    if (services['status'], services['execution_source']) != ('sealed', '1b783ea8f3'):
        raise AssertionError('V151 service prior differs')
    cases = archive_cases(archive, running, services)
    plan = dict(stage='v155_frozen_archive_revalidation', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        running_source='5be4cea534', service_source='1b783ea8f3', archive_source='6eaa4a389b',
        actors=archive['registration']['archived_actors'], days=archive['registration']['demand_days'],
        paths=2*len(cases), full_eight_day_priors=True, archived_days_disjoint_from_both_priors=True,
        comparison='V154 fixed running shadow vs sealed V151 service reference on same frozen paths',
        primary='unchanged V154 full/running-MAE conditions including low-support; absent group untested',
        fresh_independent_confirmation=False, calibration_refit=False, new_rollouts=0,
        main_policy_changed=False, v146_extensions_s=[3., 0.], window_s=3600., tails=[.1, .1])
    write(args.out_root/'protocol.json', plan)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v15[45]_*.py', '-v'], check=True)
    # Check required archive inputs before scoring; do not start any collector.
    for seed, day in cases:
        for mode in frozen.frozen.MODES:
            for directory, suffix in (('paths', '.json'), ('traces', '.jsonl.gz'),
                    ('ready_labels', '.jsonl.gz'), ('records', '.npz'), ('terminal', '.json')):
                p = args.archive_root/directory/f'{seed}_{day}_{mode}{suffix}'
                if not p.is_file():
                    raise FileNotFoundError(p)
    frozen.case(args.archive_root, args.out_root, args.runtime, *cases[0], running, services)
    print('V155_FIRST_ARCHIVE_PAIR_COMPLETE', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(frozen.case, args.archive_root, args.out_root, args.runtime, s, d, running, services)
            for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V155_ARCHIVE_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    results = {m: stats.aggregate(cells, lambda c: c['modes'][m]) for m in frozen.frozen.MODES}
    write(args.out_root/'summary/status.json', dict(status='v155_frozen_archive_revalidation_complete',
        registration=plan, results=results, **report(results),
        pairs=len(cases), paths=2*len(cases), controller_admitted=False,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        calibration_refit=False, new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, v151_model_changed=False, v146_changed=False,
        limitations='Previously consumed V145 interval-audit archive, not brand-new scenario confirmation. '
            'No V154/V155 outcomes used to alter weights or thresholds. Both priors fixed from disjoint '
            'eight V151 calibration days. Same-day strict completed prefixes remain allowed. '
            'Intervals are crossed actor/archive-day diagnostics conditional on fixed priors. '
            'Unobserved low-support groups are not evidence of no harm. Predictive efficacy is not '
            'realized control gain; main and V146 unchanged.'))
    print('V155_FROZEN_ARCHIVE_REVALIDATION_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v151-root', 'v154-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
