#!/usr/bin/env python3
"""Read-only V155 clearance accounting and fixed history-thinning stress."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v155_archive_running_prior as prior


frozen = prior.frozen
candidate, parts, stats = frozen.candidate, frozen.parts, frozen.stats
read, write, close = prior.read, prior.write, frozen.close
accounting = frozen.prior.prior
ARMS = ('full', 'single', 'none')


def retain_days(cells, days):
    return [dict(c, days=[d for d in c['days'] if d['day'] in days]) for c in cells]


def shadow_packet(reference, before, after, running):
    close(sum(reference.values()), before)
    packet = dict(reference)
    packet['running'] = running
    packet['downstream_service'] += (after-before)-(running-reference['running'])
    close(sum(packet.values()), after)
    return packet


def full_record(episode, before, after, truth):
    b, a = before-truth, after-truth
    return dict(episode=episode, reference_mae_s=abs(b), shadow_mae_s=abs(a),
        reference_bias_s=b, shadow_bias_s=a, delta_full_mae_s=abs(a)-abs(b))


def case(args, seed, day, running, services):
    geometry = frozen.native.model.load_geometry(args.runtime)
    expected = read(args.v155_root/'cells'/f'{seed}_{day}.json')
    modes, counts = {}, Counter()
    cells = running['actors'][str(seed)]
    histories = dict(full=cells, single=retain_days(cells, [min(running['days'])]), none=retain_days(cells, []))
    for mode in frozen.frozen.MODES:
        path, objects = frozen.prepare(args.archive_root, seed, day, mode, geometry)
        phases = frozen.frozen.candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        service = frozen.frozen.candidate.ServiceLaw(services['actors'][str(seed)],
            path['arrivals'], path['ready'], path['cutoff'])
        prefix = frozen.prior.RunningPrefix(path, geometry)
        laws = {arm: candidate.RunningLaw(cells, day, prefix) for arm, cells in histories.items()}
        groups = {a: defaultdict(list) for a in ARMS}
        for k, row in path['trace'].items():
            label, truth = stats.target_label(row, path['ready'])
            counts[mode+'/'+label] += 1
            if label != 'resolved':
                continue
            before, _ = frozen.frozen.candidate.forecast(path, row, geometry, objects, phases, service, True)
            if before is None:
                before = frozen.fallback(path, row, geometry, objects, phases)
            if before is None:
                continue
            episode = (*stats.follower(row), *k[1:])
            now = row['prediction_time_s']
            scopes = ['all', 'clearance' if now >= path['cutoff'] else 'active']
            if int(now//3600) in (2, 13):
                scopes.append('hour'+str(int(now//3600)))
            reference = actual = None
            for arm, law in laws.items():
                after, detail = candidate.forecast(path, row, geometry, objects, phases, service, law)
                after = before if after is None else after
                if arm == 'none':
                    close(after, before)
                r = full_record(episode, before, after, truth)
                labels = list(scopes)
                if detail.get('status') == 'observed_same_clock_launch' and 'clearance' in scopes:
                    labels.append('clearance_launch')
                counts[mode+'/'+arm+'/unsupported_legs'] += detail.get('unsupported_legs', 0)
                counts[mode+'/'+arm+'/single_day_legs'] += detail.get('single_day_legs', 0)
                if 'running_s' in detail:
                    if reference is None:
                        reference, _, _ = accounting.structured_packet(path, row, geometry, objects, phases, service, True)
                        if reference is None:
                            raise AssertionError('lost common structured component support')
                        close(sum(reference.values()), before)
                        actual = parts.actual_parts(path, row, geometry[bool(row['direction'])])
                        close(sum(actual.values()), truth)
                    b, a = reference['running']-actual['running'], detail['running_s']-actual['running']
                    r.update(reference_running_bias_s=b, shadow_running_bias_s=a,
                        reference_running_mae_s=abs(b), shadow_running_mae_s=abs(a),
                        delta_running_mae_s=abs(a)-abs(b))
                    if detail['mean_prior_days'] is not None:
                        r['mean_prior_days'] = detail['mean_prior_days']
                    if 'clearance' in scopes:
                        packet = shadow_packet(reference, before, after, detail['running_s'])
                        errors = accounting.component_errors(dict(v149=reference, joint=packet), actual)
                        errors = {f.replace('v149_', 'reference_').replace('joint_', 'shadow_'): v for f, v in errors.items()}
                        groups[arm]['clearance_components'].append(dict(episode=episode, **errors))
                for scope in labels:
                    groups[arm][scope].append(r)
        summaries = {a: {g: stats.describe(r) for g, r in rows.items()} for a, rows in groups.items()}
        for scope, before in expected['modes'][mode].items():
            reproduced = summaries['full'][scope]
            if reproduced['queries'] != before['queries']:
                raise AssertionError('V155 full-prior cohort differs')
            for field in ('reference_mae_s', 'shadow_mae_s', 'delta_full_mae_s',
                    'reference_bias_s', 'shadow_bias_s', 'delta_running_mae_s'):
                close(reproduced['metrics'][field], before['metrics'][field])
        modes[mode] = summaries
    write(args.out_root/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes,
        counts=dict(counts), original_v155_reproduced=True, zero_prior_reproduces_v151=True))


def run(args):
    parent = read(args.v155_root/'summary/status.json')
    running, services = read(args.v154_root/'model.json'), read(args.v151_root/'model.json')
    if (parent['status'], parent['registration']['source']) != (
            'v155_frozen_archive_revalidation_complete', '107db12b04'):
        raise AssertionError('requires complete frozen V155')
    if (services['status'], services['execution_source'], running['status']) != (
            'sealed', '1b783ea8f3', 'sealed_development_moments'):
        raise AssertionError('frozen prior inputs differ')
    cases = prior.archive_cases(read(args.archive_root/'summary/status.json'), running, services)
    write(args.out_root/'protocol.json', dict(stage='v156_clearance_accounting_support_stress',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], arms=list(ARMS),
        single_prior_day=min(running['days']), histories='all eight / earliest calibration day / none',
        service_prior='unchanged full eight-day V151 in every arm', weights='unchanged n/(n+1)',
        original_v155_gate=parent['fixed_joint_gate'], v155_gate_reinterpreted=False,
        clearance_identity='full MAE = sum(component MAE) - 2*cancellation, before averaging',
        low_support='controlled running-history deletion, not naturally observed low-support cohort',
        new_rollouts=0, calibration_refit=False, window_s=3600., tails=[.1, .1]))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v15[46]_*.py', '-v'], check=True)
    case(args, *cases[0], running, services)
    print('V156_FIRST_ARCHIVE_PAIR_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args, s, d, running, services) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V156_READONLY_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    results = {m: {a: stats.aggregate(cells, lambda c: c['modes'][m][a]) for a in ARMS}
        for m in frozen.frozen.MODES}
    write(args.out_root/'summary/status.json', dict(status='v156_clearance_support_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cases), paths=2*len(cases),
        results=results, original_v155_gate=parent['fixed_joint_gate'], v155_gate_reinterpreted=False,
        single_prior_day=min(running['days']), original_v155_reproduced=True,
        zero_prior_reproduces_v151=True, exact_component_error_accounting=True,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        calibration_refit=False, new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, v151_model_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], controller_admitted=False,
        limitations='Consumed V145 archive diagnosis, not independent confirmation. Full/single/none '
            'differ only in available running-prior days; service prior and weights stay fixed. '
            'Earliest calibration day fixed without score selection. Artificial history deletion '
            'tests response to missing priors, not natural sparse-history robustness or day heterogeneity. '
            'Zero-day arm is a fallback identity, not an independent no-harm result. Clearance component '
            'analysis uses common structured support; opaque reference endpoints stay in full ETA '
            'metrics but not invented component errors. Component MAEs and cancellation use episode '
            'weighting before crossed actor/day summaries; sparse groups stay inconclusive. '
            'No new predictor or controller promotion; original V155 gate remains unchanged.'))
    print('V156_CLEARANCE_SUPPORT_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v151-root', 'v154-root', 'v155-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
