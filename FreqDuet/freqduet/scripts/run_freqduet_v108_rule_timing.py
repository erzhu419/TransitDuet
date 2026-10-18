#!/usr/bin/env python3
"""Passive timing accounting on consumed V107 paths, without controller tuning."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v107_rule_confirmation as prior


pilot, confirmation = prior.pilot, prior.confirmation
timing = prior.prior.frozen.timing
coordinates = prior.prior.frozen.coordinate_moments
SEEDS = (12011, 933013)
ITEMS = [item for item in prior.ITEMS if item[0] in SEEDS]
CONTRASTS = {name: pair for name, pair in prior.CONTRASTS.items()
             if pair[0] in prior.PRIMARY and pair[1] not in prior.PRIMARY}
CONTRASTS['bilateral025_vs_forward025'] = ('bilateral025', 'forward025')
KINDS = ('arrival', 'departure')


def scoped_moments(rule, kind):
    full = rule.summary[kind]
    result = {'full': full}
    result.update({'band/' + k: v for k, v in full['four_hour_bands'].items()})
    groups = defaultdict(list)
    for row in getattr(rule, kind + 's'):
        groups[(row['station_id'], row['direction'])].append(row)
    for (station, direction), rows in groups.items():
        if len(rows) > 1:
            result[f'stop/{station}/{int(direction)}'] = timing.timing_series(rows)
    return {k: {m: v[m] for m in ('samples', 'mean_s', 'variance_s2', 'cv_squared', 'covariance_s2')}
            for k, v in result.items()}


def moment_effect(base, candidate):
    mb, mc = base['mean_s'], candidate['mean_s']
    result = {k + '_variance_at_base_mean':
        (candidate['covariance_s2'][k] - base['covariance_s2'][k]) / mb ** 2 for k in timing.COMPONENTS}
    result['mean_normalization'] = candidate['variance_s2'] * (1 / mc ** 2 - 1 / mb ** 2)
    np.testing.assert_allclose(sum(result.values()), candidate['cv_squared'] - base['cv_squared'], atol=1e-10, rtol=0)
    return dict(result, cv_squared=candidate['cv_squared'] - base['cv_squared'],
        raw_variance_s2=candidate['variance_s2'] - base['variance_s2'], mean_headway_s=mc - mb,
        base_samples=base['samples'], candidate_samples=candidate['samples'])


def full_effect(base, candidate, kind):
    b, c = base['coordinates'][kind], candidate['coordinates'][kind]
    delta = (np.asarray(c['covariance']) - b['covariance']) / b['mean'] ** 2
    parts = dict(release_diagonal=float(delta[0, 0]), holding_diagonal=float(delta[1, 1]),
        other_diagonal=float(delta[2, 2]), release_holding_cross=float(2 * delta[0, 1]),
        release_other_cross=float(2 * delta[0, 2]), holding_other_cross=float(2 * delta[1, 2]))
    measured = timing.component_effect(base['series'][kind], candidate['series'][kind])
    np.testing.assert_allclose(sum(parts.values()) + measured['mean_normalization'], measured['cv_squared'], atol=1e-10, rtol=0)
    return dict(parts, **measured, **timing.order_changes(
        base['series'][kind]['orders'], candidate['series'][kind]['orders']))


def check_cached(actual, cached, item):
    if tuple(cached[k] for k in ('seed', 'day', 'method', 'rep')) != item:
        raise AssertionError('V108 reference is not the matching V107 path')
    for key in (*prior.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(actual['results'][key], cached['results'][key], 'V108_cached_' + key)
    for key in ('contract', 'coverage'):
        pilot.joint.assert_same_state(actual[key], cached[key], 'V108_cached_' + key)


def run_item(root, old, index):
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        seed, day, method, rep = item = ITEMS[index]
        rule, credit = prior.prior.TradeoffAudit(seed, method), pilot.CreditObserver()
        row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000 + seed + 100 * day + rep,
            [0.] * 4, f'v108_{seed}_{day}_{method}_{rep}', probability_rule=rule,
            trace_observer=rule, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
        result = dict(seed=seed, day=day, method=method, rep=rep, results=row,
            contract=rule.contract, coverage=rule.coverage)
        check_cached(result, json.loads((old / 'pairs' / f'p{prior.ITEMS.index(item)}.json').read_text()), item)
        np.testing.assert_allclose(rule.summary['arrival']['cv_squared'], row['cv_squared'], atol=1e-10, rtol=0)
        result.update(series=rule.summary, scopes={k: scoped_moments(rule, k) for k in KINDS},
            coordinates={k: coordinates(getattr(rule, k + 's')) for k in KINDS})
        rule.save(root / 'records' / f'p{index}.npz')
        pilot.write_json(path, result)
    print('V108_TIMING_ITEM_DONE', index, flush=True)


def paired_rows(rows):
    services, full, scoped, unmatched = [], [], [], []
    for seed, day in itertools.product(SEEDS, prior.DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        methods = {m: [r for r in cell if r['method'] == m] for m in prior.METHODS}
        means = {m: {k: float(np.mean([r['results'][k] for r in rs])) for k in prior.METRICS}
                 for m, rs in methods.items()}
        service, total, blocks = {}, {}, {}
        for name, (a, b) in CONTRASTS.items():
            service[name] = {k: means[a][k] - means[b][k] for k in prior.METRICS}
            for kind in KINDS:
                # Deterministic paths are shared across the three control samplers.
                pairs = [(methods[b][rep % len(methods[b])], methods[a][rep % len(methods[a])])
                         for rep in range(max(len(methods[a]), len(methods[b])))]
                effects = [full_effect(base, candidate, kind) for base, candidate in pairs]
                key = name + '/' + kind
                total[key] = {k: float(np.mean([e[k] for e in effects])) for k in effects[0]}
                keys = [set(r['scopes'][kind]) for pair in pairs for r in pair]
                common, union = set.intersection(*keys), set.union(*keys)
                if common != union:
                    unmatched.append(dict(seed=seed, day=day, contrast=name, kind=kind, scopes=sorted(union - common)))
                blocks[key] = {}
                for scope in sorted(common):
                    effects = [moment_effect(base['scopes'][kind][scope], candidate['scopes'][kind][scope])
                               for base, candidate in pairs]
                    blocks[key][scope] = {k: float(np.mean([e[k] for e in effects])) for k in effects[0]}
                if kind == 'arrival':
                    np.testing.assert_allclose(total[key]['cv_squared'], service[name]['cv_squared'], atol=1e-10, rtol=0)
        services.append(dict(seed=seed, day=day, effects=service))
        full.append(dict(seed=seed, day=day, effects=total))
        scoped.append(dict(seed=seed, day=day, effects=blocks))
    return services, full, scoped, unmatched


def scoped_means(rows):
    result = {}
    for contrast in rows[0]['effects']:
        scopes = set.union(*(set(r['effects'][contrast]) for r in rows))
        result[contrast] = {}
        for scope in sorted(scopes):
            selected = [r for r in rows if scope in r['effects'][contrast]]
            metrics = selected[0]['effects'][contrast][scope]
            result[contrast][scope] = dict(actor_day_cells=len(selected),
                means={k: float(np.mean([r['effects'][contrast][scope][k] for r in selected])) for k in metrics})
    return result


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V108 timing rows differ from the fixed consumed-path grid')
    services, full, scoped, unmatched = paired_rows(rows)
    pilot.write_json(root / 'summary/status.json', dict(status='frozen_rule_timing_diagnosis_complete',
        effects=confirmation.describe(services, 'effects'), timing_effects=confirmation.describe(full, 'effects'),
        service_axes=prior.prior.factorial.prior.prior.axis_summary(services, 'effects', prior.KEY_METRICS),
        timing_axes=prior.prior.factorial.prior.prior.axis_summary(full, 'effects', tuple(full[0]['effects'][next(iter(full[0]['effects']))])),
        scoped_descriptive=scoped_means(scoped), unmatched_scopes=unmatched,
        archived_actors=list(SEEDS), demand_days=list(prior.DAYS), replayed_paths=len(ITEMS),
        cached_v107_endpoints_reproduced=True, stochastic_repeats=3,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, gain_refitted=False,
        scope='two source-order actors, consumed V107 days; passive covariance accounting, not causal mediation or new confirmation'))
    print('V108_TIMING_DIAGNOSIS_COMPLETE', flush=True)


def smoke(root, old, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v108_*.py', '-v'], check=True)
    indices = [i for i, (seed, day, method, rep) in enumerate(ITEMS) if day == prior.DAYS[0] and rep == 0]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        list(pool.map(run_item, itertools.repeat(root), itertools.repeat(old), indices))
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_and_ten_exact_cached_paths_passed',
        tested_paths=len(indices), formal_paths=len(ITEMS), fresh_confirmation_labels=0, main_policy_changed=False))
    print('V108_SMOKE_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v107-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v107_root, a.workers)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.v107_root), range(a.start, a.end)))
        pilot.write_json(a.out_root / 'shards' / f'{a.start}-{a.end}.json', dict(status='complete'))


if __name__ == '__main__':
    main()
