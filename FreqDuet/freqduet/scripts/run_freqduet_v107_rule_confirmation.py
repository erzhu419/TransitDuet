#!/usr/bin/env python3
"""Independent actor/day confirmation of both frozen quarter-gain rules."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v106_service_tradeoff as prior


pilot, confirmation = prior.pilot, prior.confirmation
SEEDS = tuple(confirmation.VALIDATION_SEEDS)
DAYS = tuple(range(2071001, 2071017))
PRIMARY = ('forward025', 'bilateral025')
METHODS = ('native', 'preserve', 'zero', *PRIMARY)
METRICS = prior.METRICS
KEY_METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s', 'holding_s_per_trip')
ITEMS = [(seed, day, method, rep) for seed, day, method in itertools.product(SEEDS, DAYS, METHODS)
         for rep in (range(3) if method in ('native', 'preserve') else (0,))]
CONTRASTS = {method + '_vs_' + base: (method, base)
             for base in ('native', 'preserve', 'zero') for method in METHODS if method != base}
CONTRASTS['forward025_vs_bilateral025'] = ('forward025', 'bilateral025')


def joint_gate(stats):
    return stats['passenger_person_s']['ci95'][1] < 0 and all(stats[k]['ci95'][1] <= 0
        for k in ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))


def confirmation_gates(stats):
    result = {}
    for method in PRIMARY:
        passed = {base: joint_gate(stats[method + '_vs_' + base]) for base in ('native', 'preserve')}
        result[method] = dict(vs_native=passed['native'], vs_preserve=passed['preserve'],
                             joint_both_controls=all(passed.values()))
    return result


def run_item(root, index):
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        pilot.write_json(path, prior.replay(root, *ITEMS[index]))
    print('V107_CONFIRMATION_ITEM_DONE', index, flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V107 rows differ from the frozen actor/day/method/sampler grid')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for row in cell[1:]:
            pilot.joint.assert_same_state(row['contract'], cell[0]['contract'], 'V107_shared_physical_contract')
        means = {method: {k: float(np.mean([r['results'][k] for r in cell if r['method'] == method]))
                         for k in METRICS} for method in METHODS}
        absolute.append(dict(seed=seed, day=day, values=means))
        paired.append(dict(seed=seed, day=day, effects={name: {k: means[a][k] - means[b][k]
            for k in METRICS} for name, (a, b) in CONTRASTS.items()}))
    stats = confirmation.describe(paired, 'effects')
    levels = confirmation.describe(absolute, 'values')
    gates = confirmation_gates(stats)
    pilot.write_json(root / 'summary/status.json', dict(status='frozen_quarter_gain_confirmation_complete',
        effects=stats, absolute=levels, gates=gates,
        axes=prior.factorial.prior.prior.axis_summary(paired, 'effects', KEY_METRICS),
        coverage={method: {k: sum(r['coverage'][k] for r in rows if r['method'] == method)
                          for k in rows[0]['coverage']} for method in PRIMARY},
        archived_actors=list(SEEDS), demand_days=list(DAYS), development_actors=list(prior.SEEDS),
        development_days=list(prior.DAYS), new_rollouts=len(ITEMS), stochastic_repeats=3,
        deterministic_paths_per_actor_day=1, primary_methods=list(PRIMARY),
        bootstrap='5000 paired crossed actor/day draws after sampler averaging; shared deterministic controls',
        primary_rule='each co-primary rule must pass all original service constraints versus both native and preserve; '
                     'existing two-sided 95% intervals supply one-sided 97.5% upper bounds for each joint claim',
        action_bins_s=pilot.step.ACTIONS.tolist(), upper_regime='unchanged learned B200 for every arm',
        model_updates=0, actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='frozen V106 quarter-gain rules, other ten archived actors and new demand days; '
              'not newly trained actors, frequency-separation evidence, exact fixed-upper comparison or automatic promotion'))
    print('V107_CONFIRMATION_COMPLETE', gates, flush=True)


def smoke(root, old, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v107_*.py', '-v'], check=True)
    seed, day = prior.SEEDS[0], prior.DAYS[0]
    cases = [(seed, day, method, 0) for method in METHODS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        actual = [f.result() for f in [pool.submit(prior.replay, root, *case) for case in cases]]
    for case, measured in zip(cases, actual):
        cached = json.loads((old / 'pairs' / f'p{prior.ITEMS.index(case)}.json').read_text())
        for k in (*METRICS, 'action_counts'):
            pilot.joint.assert_same_state(measured['results'][k], cached['results'][k], 'V107_cached_' + k)
        pilot.joint.assert_same_state(measured['contract'], cached['contract'], 'V107_cached_contract')
        pilot.joint.assert_same_state(measured['coverage'], cached['coverage'], 'V107_cached_coverage')
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_and_five_cached_control_replays_passed',
        smoke_rollouts=len(cases), fresh_confirmation_labels=0, formal_rollouts=len(ITEMS),
        actor_updates=0, dual_updates=0, main_policy_changed=False))
    print('V107_SMOKE_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v106-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v106_root, a.workers)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), range(a.start, a.end)))
        pilot.write_json(a.out_root / 'shards' / f'{a.start}-{a.end}.json', dict(status='complete'))


if __name__ == '__main__':
    main()
