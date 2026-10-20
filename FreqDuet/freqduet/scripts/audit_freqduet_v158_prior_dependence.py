#!/usr/bin/env python3
"""Read-only prior-day reliability and frozen full-ETA error dependence."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import numpy as np
import pandas as pd

import audit_freqduet_v157_launch_running as launch


audit = launch.audit
SCOPES = ('all', 'active', 'hour2', 'clearance', 'hour13')
COARSE = ('running', 'service', 'residence')
IDENTIFIED = ('running', 'service', 'holding', 'execution_wait')


def day_moments(days):
    means = [d['mean'] for d in days]
    center = fmean(means)
    return dict(prior_days=len(days), mean=center, minimum=min(means), maximum=max(means),
        between_day_variance_s2=fmean((m-center)**2 for m in means),
        within_day_variance_s2=fmean(max(0., d['second']-d['mean']**2) for d in days))


def error_record(episode, errors, full_error):
    audit.close(sum(errors.values()), full_error)
    record = dict(episode=episode, full_bias_s=full_error, full_mae_s=abs(full_error),
        full_mse_s2=full_error**2,
        cancellation_s=(sum(abs(v) for v in errors.values())-abs(full_error))/2.)
    record.update({'e_'+k: v for k, v in errors.items()})
    record.update({'m_'+a+'__'+b: errors[a]*errors[b]
        for a, b in combinations_with_replacement(errors, 2)})
    return record


def error_views(packet, actual, joint):
    prediction, truth = audit.accounting.coarse(packet), audit.accounting.coarse(actual)
    coarse = {c: prediction[c]-truth[c] for c in COARSE}
    views = dict(coarse=coarse)
    if joint is None:
        views['holding_identified'] = {c: coarse[c] if c in COARSE else packet[c]-actual[c]
            for c in IDENTIFIED}
    return views


def prior_records(path, row, law, geometry, cache):
    values = []
    now, trip = row['prediction_time_s'], audit.stats.follower(row)[0]
    for (direction, station), call in law.calls.items():
        days = call['prior_day_moments']
        if not days:
            continue
        hour = int(call['entry_time_s']//3600)
        key = direction, station, hour
        if key not in cache:
            cache[key] = day_moments(days)
        m = cache[key]
        route = geometry[direction]
        previous_station = route['ids'][route['index'][station]-1]
        departure = path['departures'].get((trip, previous_station, direction))
        start = path['dispatch'][trip]['actual_s'] if departure is None else departure['time_s']
        end = path['arrivals'][trip, station, direction]['time_s']
        truth = max(0., end-max(now, start))
        selected = law.prefix.selected(direction, station, now)
        audit.close(selected['mean_s'], call['native_mean_s'])
        if selected['latest_age_s'] <= 0.:
            raise AssertionError('prior diagnosis used an unfinished prefix')
        native, frozen = call['native_mean_s'], call['selected']['mean']
        values.append(dict(prior_days=m['prior_days'],
            between_day_variance_s2=m['between_day_variance_s2'],
            within_day_variance_s2=m['within_day_variance_s2'],
            prior_day_mean_span_s=m['maximum']-m['minimum'],
            current_minus_prior_mean_s=native-m['mean'],
            current_prior_gap_squared_s2=(native-m['mean'])**2,
            prior_leg_bias_s=m['mean']-truth, prior_leg_mae_s=abs(m['mean']-truth),
            native_leg_bias_s=native-truth, native_leg_mae_s=abs(native-truth),
            frozen_leg_bias_s=frozen-truth, frozen_leg_mae_s=abs(frozen-truth),
            frozen_leg_mse_s2=(frozen-truth)**2,
            frozen_mixture_variance_s2=call['selected']['dispersion_s']**2,
            future_leg_outside_day_mean_range=float(not m['minimum'] <= truth <= m['maximum']),
            entry_hour_mismatch_fraction=float(int(start//3600) != hour),
            current_prefix_n=selected['count'], latest_completion_age_s=selected['latest_age_s'],
            old_entry_fraction=selected['before_hour_fraction']))
    return {f: fmean(v[f] for v in values) for f in values[0]} if values else None


def case(args, seed, day, running, services):
    geometry = audit.frozen.native.model.load_geometry(args.runtime)
    expected = audit.read(args.v156_root/'cells'/f'{seed}_{day}.json')
    modes, counts = {}, Counter()
    for mode in audit.frozen.frozen.MODES:
        path, objects = audit.frozen.prepare(args.archive_root, seed, day, mode, geometry)
        phases = audit.frozen.frozen.candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        service = audit.frozen.frozen.candidate.ServiceLaw(services['actors'][str(seed)],
            path['arrivals'], path['ready'], path['cutoff'])
        law = launch.RecordedRunning(running['actors'][str(seed)], day,
            audit.frozen.prior.RunningPrefix(path, geometry))
        groups, endpoints, prior_cache = defaultdict(list), defaultdict(list), {}
        for k, row in path['trace'].items():
            label, truth = audit.stats.target_label(row, path['ready'])
            if label != 'resolved':
                continue
            before, _ = audit.candidate.frozen.forecast(path, row, geometry, objects, phases, service, True)
            if before is None:
                before = audit.frozen.fallback(path, row, geometry, objects, phases)
            if before is None:
                continue
            law.calls.clear()
            after, detail = audit.candidate.forecast(path, row, geometry, objects, phases, service, law)
            after = before if after is None else after
            now = row['prediction_time_s']
            scopes = ['all', 'clearance' if now >= path['cutoff'] else 'active']
            if int(now//3600) in (2, 13):
                scopes.append('hour'+str(int(now//3600)))
            episode = (*audit.stats.follower(row), *k[1:])
            endpoint = audit.full_record(episode, before, after, truth)
            if 'running_s' in detail:
                reference, joint, _ = audit.accounting.structured_packet(path, row, geometry, objects, phases, service, True)
                audit.close(sum(reference.values()), before)
                actual = audit.parts.actual_parts(path, row, geometry[bool(row['direction'])])
                audit.close(sum(actual.values()), truth)
                shadow = audit.shadow_packet(reference, before, after, detail['running_s'])
                endpoint['delta_running_mae_s'] = abs(shadow['running']-actual['running'])-abs(reference['running']-actual['running'])
                reliability = prior_records(path, row, law, geometry, prior_cache)
                counts[mode+'/structured'] += 1
                counts[mode+'/complete_forecast_legs'] += len(law.calls)
                counts[mode+'/joint_residence_not_pure_holding'] += int(joint is not None)
                for name, packet in (('v151', reference), ('eight_day', shadow)):
                    for cohort, errors in error_views(packet, actual, joint).items():
                        r = error_record(episode, errors, sum(packet.values())-truth)
                        for scope in scopes:
                            groups[name+'/'+cohort+'/'+scope].append(r)
                if reliability is not None:
                    for scope in scopes:
                        groups['prior/'+scope].append(dict(episode=episode, **reliability))
            else:
                counts[mode+'/opaque_full_endpoint'] += 1
            for scope in scopes:
                endpoints[scope].append(endpoint)
        endpoint_stats = {g: audit.stats.describe(rows) for g, rows in endpoints.items()}
        for scope, reproduced in endpoint_stats.items():
            old = expected['modes'][mode]['full'][scope]
            if old['queries'] != reproduced['queries']:
                raise AssertionError('V156 full endpoint support changed')
            for field in ('reference_mae_s', 'shadow_mae_s', 'delta_full_mae_s',
                    'reference_bias_s', 'shadow_bias_s', 'delta_running_mae_s'):
                audit.close(reproduced['metrics'][field], old['metrics'][field])
        modes[mode] = dict(groups={g: audit.stats.describe(rows) for g, rows in groups.items()})
    audit.write(args.out_root/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes,
        counts=dict(counts), v156_endpoints_reproduced=True))


def matrices(metrics, channels):
    mu = np.array([metrics['e_'+c] for c in channels])
    raw = np.zeros((len(channels), len(channels)))
    for i, a in enumerate(channels):
        for j in range(i, len(channels)):
            raw[i, j] = raw[j, i] = metrics['m_'+a+'__'+channels[j]]
    return mu, raw


def covariance_report(means, seconds):
    # Equal episode means inside each cell; equal actor/day cells outside.
    grand = means.mean(axis=(0, 1))
    raw = seconds.mean(axis=(0, 1))
    cell_products = np.einsum('sdi,sdj->sdij', means, means).mean(axis=(0, 1))
    within = raw-cell_products
    actor = means.mean(axis=1)-grand
    day = means.mean(axis=0)-grand
    interaction = means-grand-actor[:, None, :]-day[None, :, :]
    actor_cov = actor.T@actor/len(actor)
    day_cov = day.T@day/len(day)
    interaction_cov = np.einsum('sdi,sdj->ij', interaction, interaction)/(means.shape[0]*means.shape[1])
    total = raw-np.outer(grand, grand)
    np.testing.assert_allclose(total, within+actor_cov+day_cov+interaction_cov, rtol=0., atol=1e-7)
    scales = np.sqrt(np.maximum(0., np.diag(total)))
    rho = [[float(total[i, j]/(scales[i]*scales[j])) if scales[i]*scales[j] > 0. else None
        for j in range(len(scales))] for i in range(len(scales))]
    full_variance, bias = float(total.sum()), float(grand.sum())
    audit.close(float(raw.sum()), full_variance+bias*bias)
    return dict(bias_s=grand.tolist(), second_moment_s2=raw.tolist(), covariance_s2=total.tolist(),
        correlation=rho, within_cell_covariance_s2=within.tolist(),
        actor_mean_covariance_s2=actor_cov.tolist(), day_mean_covariance_s2=day_cov.tolist(),
        actor_day_interaction_covariance_s2=interaction_cov.tolist(),
        full_variance_s2=full_variance, diagonal_variance_s2=float(np.trace(total)),
        offdiagonal_variance_s2=float(total.sum()-np.trace(total)), full_mse_s2=float(raw.sum()))


def dependence(cells, mode, group, channels):
    actors, days = sorted({c['seed'] for c in cells}), sorted({c['day'] for c in cells})
    entries = [c['modes'][mode]['groups'].get(group) for c in cells]
    result = dict(channels=list(channels), contributing_cells=sum(e is not None for e in entries))
    if any(e is None for e in entries):
        return dict(result, decision='inconclusive_crossed_support')
    size = len(channels)
    mu, raw = np.zeros((len(actors), len(days), size)), np.zeros((len(actors), len(days), size, size))
    for c, entry in zip(cells, entries):
        i, j = actors.index(c['seed']), days.index(c['day'])
        mu[i, j], raw[i, j] = matrices(entry['metrics'], channels)
    result.update(covariance_report(mu, raw))
    frames = pd.DataFrame([dict(train_seed=c['seed'], eval_seed=c['day'], **e['metrics']) for c, e in zip(cells, entries)])
    # The existing bootstrap resets its RNG, so all moment draws share actor/day indices.
    fields = ['e_'+c for c in channels]+['m_'+a+'__'+b for a, b in combinations_with_replacement(channels, 2)]
    draws = {f: audit.stats.hierarchical_bootstrap(frames, f, draws=5000) for f in fields}
    covariance_ci, correlation_ci = {}, {}
    for a, b in combinations_with_replacement(channels, 2):
        cov = draws['m_'+a+'__'+b]-draws['e_'+a]*draws['e_'+b]
        va = np.maximum(0., draws['m_'+a+'__'+a]-draws['e_'+a]**2)
        vb = np.maximum(0., draws['m_'+b+'__'+b]-draws['e_'+b]**2)
        valid = va*vb > 0.
        key = a+'__'+b
        covariance_ci[key] = np.percentile(cov, [2.5, 97.5]).tolist()
        correlation_ci[key] = dict(ci95=np.percentile(cov/np.sqrt(va*vb), [2.5, 97.5]).tolist()
            if valid.all() else None, defined_draws=int(valid.sum()))
    return dict(result, covariance_ci95=covariance_ci, correlation_ci95=correlation_ci)


def inventory(model):
    rows = defaultdict(list)
    for actor in model['actors'].values():
        for c in actor:
            if c['days']:
                m = day_moments(c['days'])
                rows[str(c['hour'])].append({k: v for k, v in m.items() if k not in ('mean', 'minimum', 'maximum')})
    return {h: dict(actor_segment_hour_cells=len(v), metrics={f: fmean(r[f] for r in v) for f in v[0]})
        for h, v in rows.items()}


def run(args):
    parent = audit.read(args.v156_root/'summary/status.json')
    running, services = audit.read(args.v154_root/'model.json'), audit.read(args.v151_root/'model.json')
    if (parent['status'], parent['execution_source'], services['execution_source'], running['status']) != (
            'v156_clearance_support_audit_complete', '8d1e201a13', '1b783ea8f3', 'sealed_development_moments'):
        raise AssertionError('requires frozen V156 and sealed priors')
    cases = audit.prior.archive_cases(audit.read(args.archive_root/'summary/status.json'), running, services)
    audit.write(args.out_root/'protocol.json', dict(stage='v158_readonly_prior_dependence',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cases=len(cases), scopes=SCOPES,
        models=['v151', 'eight_day'], window_s=3600., tails=[.1, .1], v146_extensions_s=[3., 0.],
        no_fit=True, new_rollouts=0, prior_selection=False, forecast_weights_unchanged=True,
        purpose='Prior-day dispersion and current-prefix shift; centered and uncentered error dependence. '
            'Coarse residence includes unseparated current-ready residual; pure holding only when identified.'))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v158_prior_dependence.py', '-v'], check=True)
    case(args, *cases[0], running, services)
    print('V158_FIRST_SAVED_PAIR_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args, s, d, running, services) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V158_READONLY_PROGRESS', n, len(cases), flush=True)
    cells = [audit.read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    results = {m: audit.stats.aggregate(cells, lambda c: c['modes'][m]['groups']) for m in audit.frozen.frozen.MODES}
    correlations = {m: {name+'/'+cohort+'/'+scope: dependence(cells, m, name+'/'+cohort+'/'+scope, channels)
        for name in ('v151', 'eight_day') for cohort, channels in (('coarse', COARSE), ('holding_identified', IDENTIFIED))
        for scope in SCOPES} for m in audit.frozen.frozen.MODES}
    audit.write(args.out_root/'summary/status.json', dict(status='v158_prior_dependence_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cases), paths=2*len(cases),
        results=results, dependence=correlations, frozen_prior_inventory_by_entry_hour=inventory(running),
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        original_v156_endpoints_reproduced=True, new_rollouts=0, calibration_refit=False,
        main_policy_changed=False, v151_model_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], controller_admitted=False,
        limitations='Consumed V145 paths; no fresh confirmation or fitted uncertainty model. '
            'Crossed actor/day inference conditions on the same eight frozen prior days. '
            'Daily-mean dispersion includes finite-sample mean noise; moments alone cannot separate it '
            'from latent day variation. Complete-leg diagnostics average legs within query, then queries '
            'within follower-target episode; partial legs remain in full ETA but not prior-leg metrics. '
            'Day-mean ranges are not predictive intervals and mixture variance is not calibrated coverage. '
            'Coarse and holding-identified cohorts have different support; no unmatched correlation '
            'comparison. Mean shifts, covariance and raw error products are separate; correlations '
            'are descriptive, not causal holding effects. Within-cell covariance still includes '
            'stop/state/horizon variation, not only conditional random noise. Missing support has no crossed CI. '
            'No prior-day ranking, window/quantile/weight tuning, interval construction or gate change.'))
    print('V158_PRIOR_DEPENDENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v151-root', 'v154-root', 'v156-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
