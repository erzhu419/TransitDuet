#!/usr/bin/env python3
"""Four-component accounting on existing V108 primitives; no new rollout."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v108_rule_timing as prior
import run_freqduet_v92_elapsed_review as elapsed


pilot, confirmation = prior.pilot, prior.confirmation
CELLS = list(itertools.product(prior.SEEDS, prior.prior.DAYS))
KEY_METRICS = ('cv_squared', 'raw_variance_s2', 'mean_normalization',
    *(k + '_allocated' for k in elapsed.COMPONENTS),
    'realized_holding__observed_nonholding_stop_cross', 'realized_holding__remaining_elapsed_cross')


def assert_parent(actual, parent):
    np.testing.assert_allclose([actual['samples'], actual['mean'], actual['variance'], actual['cv_squared']],
        [parent['samples'], parent['mean_s'], parent['variance_s2'], parent['cv_squared']], atol=1e-8, rtol=1e-12)
    allocation = actual['covariance'].sum(1)
    np.testing.assert_allclose([allocation[0], allocation[1], allocation[2] + allocation[3]],
        [parent['covariance_s2'][k] for k in prior.timing.COMPONENTS], atol=1e-8, rtol=1e-12)


def scoped_series(records, split, kind):
    full = elapsed.series(records, split, kind)
    times = {elapsed.key(r): r['time_s'] for r in records[kind + 's']}
    groups = defaultdict(lambda: ([], []))
    for key, edge in full['edges'].items():
        parts = np.asarray(split[key, kind]) - split[edge['predecessor'], kind]
        for scope in (f'stop/{key[1]}/{int(key[2])}', f'band/{int(times[key] // 14400)}'):
            groups[scope][0].append(edge['headway'])
            groups[scope][1].append(parts)
    return dict(full=full, **{k: elapsed.covariance_moments(*v) for k, v in groups.items()})


def read_path(old, index):
    row = json.loads((old / 'pairs' / f'p{index}.json').read_text())
    if tuple(row[k] for k in ('seed', 'day', 'method', 'rep')) != prior.ITEMS[index]:
        raise AssertionError('V109 primitive reference differs from the V108 frozen grid')
    records = elapsed.read_records(old / 'records' / f'p{index}.npz')
    split = elapsed.split_stop_time(records)
    scopes = {kind: scoped_series(records, split, kind) for kind in prior.KINDS}
    collapse = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.]])
    for kind, blocks in scopes.items():
        if set(blocks) != set(row['scopes'][kind]):
            raise AssertionError('V109 changed the V108 stop/clock support')
        for scope, measured in blocks.items():
            assert_parent(measured, row['scopes'][kind][scope])
        np.testing.assert_allclose(collapse @ blocks['full']['covariance'] @ collapse.T,
            row['coordinates'][kind]['covariance'], atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(scopes['arrival']['full']['cv_squared'], row['results']['cv_squared'], atol=1e-10, rtol=0)
    return dict(method=row['method'], rep=row['rep'], scopes=scopes)


def moment_effect(base, candidate):
    delta = (candidate['covariance'] - base['covariance']) / base['mean'] ** 2
    result = {name + '_allocated': float(delta[i].sum()) for i, name in enumerate(elapsed.COMPONENTS)}
    result.update({name + '_diagonal': float(delta[i, i]) for i, name in enumerate(elapsed.COMPONENTS)})
    result.update({elapsed.COMPONENTS[i] + '__' + elapsed.COMPONENTS[j] + '_cross': float(2 * delta[i, j])
                   for i in range(4) for j in range(i + 1, 4)})
    result.update(mean_normalization=candidate['variance'] * (1 / candidate['mean'] ** 2 - 1 / base['mean'] ** 2),
        cv_squared=candidate['cv_squared'] - base['cv_squared'],
        raw_variance_s2=candidate['variance'] - base['variance'], mean_headway_s=candidate['mean'] - base['mean'],
        base_samples=base['samples'], candidate_samples=candidate['samples'])
    np.testing.assert_allclose(sum(result[name + '_allocated'] for name in elapsed.COMPONENTS)
        + result['mean_normalization'], result['cv_squared'], atol=1e-10, rtol=0)
    return result


def average_effects(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def compare_paths(paths):
    methods = {m: sorted([r for r in paths if r['method'] == m], key=lambda r: r['rep'])
               for m in prior.prior.METHODS}
    full, scoped = {}, {}
    for contrast, (candidate, base) in prior.CONTRASTS.items():
        b, c = methods[base], methods[candidate]
        pairs = [(b[i % len(b)], c[i % len(c)]) for i in range(max(len(b), len(c)))]
        for kind in prior.KINDS:
            name = contrast + '/' + kind
            keys = [set(r['scopes'][kind]) for pair in pairs for r in pair]
            if any(k != keys[0] for k in keys[1:]):
                raise AssertionError('V109 cell lost a previously complete V108 scope')
            scoped[name] = {scope: average_effects([moment_effect(a['scopes'][kind][scope], z['scopes'][kind][scope])
                for a, z in pairs]) for scope in sorted(keys[0])}
            full[name] = average_effects([elapsed.effects(a['scopes'][kind]['full'], z['scopes'][kind]['full']) for a, z in pairs])
            full[name].update({k: scoped[name]['full'][k] for k in ('raw_variance_s2', 'mean_headway_s')})
    return full, scoped


def analyze_cell(root, old, index):
    path = root / 'details' / f'c{index}.json'
    if path.exists():
        return json.loads(path.read_text())
    seed, day = CELLS[index]
    indices = [i for i, item in enumerate(prior.ITEMS) if item[:2] == (seed, day)]
    full, scoped = compare_paths([read_path(old, i) for i in indices])
    result = dict(seed=seed, day=day, effects=full, scoped=scoped, existing_paths_read=len(indices))
    pilot.write_json(path, result)
    print('V109_ELAPSED_CELL_DONE', index, flush=True)
    return result


def review(root, old):
    rows = [analyze_cell(root, old, i) for i in range(len(CELLS))]
    if [(r['seed'], r['day']) for r in rows] != CELLS:
        raise AssertionError('V109 review differs from the complete consumed actor/day grid')
    parent = json.loads((old / 'summary/status.json').read_text())
    stats = confirmation.describe(rows, 'effects')
    for contrast in prior.CONTRASTS:
        for kind in prior.KINDS:
            name = contrast + '/' + kind
            for metric in ('cv_squared', 'mean_normalization'):
                actual, reference = stats[name][metric], parent['timing_effects'][name][metric]
                for count in ('n_training_seeds', 'n_demand_days'):
                    if actual[count] != reference[count]:
                        raise AssertionError('V109 changed parent bootstrap axes')
                np.testing.assert_allclose([actual['mean'], *actual['ci95']],
                    [reference['mean'], *reference['ci95']], atol=1e-10, rtol=0)
    pilot.write_json(root / 'summary/status.json', dict(status='stop_elapsed_readonly_review_complete',
        timing_effects=stats,
        timing_axes=prior.prior.prior.factorial.prior.prior.axis_summary(rows, 'effects', KEY_METRICS),
        scoped_descriptive=prior.scoped_means([dict(seed=r['seed'], day=r['day'], effects=r['scoped']) for r in rows]),
        archived_actors=list(prior.SEEDS), demand_days=list(prior.prior.DAYS), existing_paths_read=sum(r['existing_paths_read'] for r in rows),
        all_v108_scopes_moments_covariance_and_crossed_intervals_reproduced=True,
        new_rollouts=0, checkpoint_loads=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, controller_admitted=False, original_gate_changed=False,
        components=elapsed.COMPONENTS, scope='two archived actors and consumed V108 paths; covariance accounting, not causal mediation',
        remaining_definition='other elapsed minus observed nonholding stop time; includes movement and unobserved terminal/clock effects'))
    print('V109_STOP_ELAPSED_REVIEW_COMPLETE', flush=True)


def smoke(root, old):
    for pattern in ('test_v92_*.py', 'test_v109_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1] / 'tests'), '-p', pattern, '-v'], check=True)
    for seed in prior.SEEDS:
        analyze_cell(root, old, CELLS.index((seed, prior.prior.DAYS[0])))
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_and_eighteen_parent_path_reads_passed',
        existing_paths_read=18, new_rollouts=0, main_policy_changed=False))
    print('V109_SMOKE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'review'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v108-root', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v108_root)
    else:
        review(a.out_root, a.v108_root)
