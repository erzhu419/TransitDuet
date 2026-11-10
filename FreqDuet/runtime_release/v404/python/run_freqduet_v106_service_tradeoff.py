#!/usr/bin/env python3
"""Frozen-controller development screen of full-day service tradeoffs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v98_upper_holding_factorial as factorial


pilot, frozen, confirmation = factorial.pilot, factorial.frozen, factorial.confirmation
SEEDS = (12037, 933083)
DAYS = tuple(range(2061001, 2061005))
RULES = {kind + suffix: (kind, gain) for kind in ('forward', 'bilateral')
         for suffix, gain in (('025', .25), ('050', .5), ('100', 1.))}
METHODS = ('native', 'preserve', 'zero', *RULES)
ITEMS = [(s, d, method, rep) for s, d, method in itertools.product(SEEDS, DAYS, METHODS)
         for rep in (range(3) if method in ('native', 'preserve') else (0,))]
METRICS = frozen.METRICS


def rule_action(context, kind, gain, feasible):
    if not context.evidence_valid:
        desired = 0.
    elif kind == 'bilateral' and context.follower_evidence_valid:
        desired = gain * (context.follower_departure_gap_s - context.forward_headway_s) / 2
    else:
        desired = gain * (context.target_headway_s - context.forward_headway_s)
    desired = np.clip(desired, 0., pilot.step.ACTIONS[-1])
    choices = np.flatnonzero(feasible)
    # ACTIONS is ascending: equal-distance ties use the shorter command.
    return int(choices[np.argmin(np.abs(pilot.step.ACTIONS[choices] - desired))])


class TradeoffAudit(factorial.FactorialAudit):
    def __init__(self, seed, method):
        super().__init__(seed, 'learned_' + method)
        self.method = method
        self.coverage = dict(decisions=0, missing_forward=0, bilateral=0, forward_fallback=0)

    def __call__(self, policy, states, features, theta):
        if self.method in ('preserve', 'zero'):
            return super().__call__(policy, states, features, theta)
        native = frozen.source.experiment.LogitCross.__call__(self, policy, states, features, theta)
        if self.method == 'native':
            return native
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        kind, gain = RULES[self.method]
        self.coverage['decisions'] += 1
        self.coverage['missing_forward'] += not context.evidence_valid
        self.coverage['bilateral'] += kind == 'bilateral' and context.evidence_valid and context.follower_evidence_valid
        self.coverage['forward_fallback'] += kind == 'bilateral' and context.evidence_valid and not context.follower_evidence_valid
        selected = rule_action(context, kind, gain, policy.feasible_action_mask(states)[0].numpy())
        law = torch.zeros_like(native)
        law[0, selected] = 1.
        return law


def replay(root, seed, day, method, rep):
    rule, credit = TradeoffAudit(seed, method), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000 + seed + 100 * day + rep,
        [0.] * 4, f'v106_{seed}_{day}_{method}_{rep}', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return dict(seed=seed, day=day, method=method, rep=rep, results=row,
                contract=rule.contract, coverage=rule.coverage)


def run_item(root, index):
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        pilot.write_json(path, replay(root, *ITEMS[index]))
    print('V106_TRADEOFF_ITEM_DONE', index, flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V106 rows differ from the registered grid')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for r in cell[1:]:
            pilot.joint.assert_same_state(r['contract'], cell[0]['contract'], 'shared_V106_physical_contract')
        means = {m: {k: float(np.mean([r['results'][k] for r in cell if r['method'] == m]))
            for k in METRICS} for m in METHODS}
        absolute.append(means)
        effects = {m + '_vs_' + base: {k: means[m][k] - means[base][k] for k in METRICS}
                   for base in ('native', 'preserve', 'zero') for m in METHODS if m != base}
        paired.append(dict(seed=seed, day=day, effects=effects))
    stats = confirmation.describe(paired, 'effects')
    gates = {m: (stats[m + '_vs_preserve']['passenger_person_s']['ci95'][1] < 0 and
        all(stats[m + '_vs_preserve'][k]['ci95'][1] <= 0 for k in
            ('cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete'))) for m in RULES}
    pilot.write_json(root / 'summary/status.json', dict(status='full_day_service_tradeoff_screen_complete',
        effects=stats, absolute={m: {k: float(np.mean([r[m][k] for r in absolute])) for k in METRICS} for m in METHODS},
        axes=factorial.prior.prior.axis_summary(paired, 'effects', METRICS),
        rule_vs_preserve_joint_diagnostic=gates,
        coverage={m: {k: sum(r['coverage'][k] for r in rows if r['method'] == m)
                       for k in rows[0]['coverage']} for m in RULES},
        archived_actors=list(SEEDS), demand_days=list(DAYS), new_rollouts=len(ITEMS),
        stochastic_repeats=3, deterministic_paths_per_actor_day=1,
        bootstrap='5000 crossed actor/day draws after sampler averaging; shared deterministic controls',
        action_bins_s=pilot.step.ACTIONS.tolist(), upper_regime='unchanged learned B200 for every arm',
        model_updates=0, actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='development tradeoff screen, all gains reported; no winner selection or promotion; '
              'zero is learned-upper/zero-holding, not an exact fixed-headway upper baseline'))
    print('V106_TRADEOFF_COMPLETE', gates, flush=True)


def smoke(root, v104, v96, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v106_*.py', '-v'], check=True)
    old = json.loads((v104 / 'references/r0.json').read_text())
    preserved = json.loads((v96 / 'pairs/p0.json').read_text())
    cases = [(old['seed'], old['day'], 'native', 0),
             (preserved['seed'], preserved['day'], 'preserve', preserved['rep'])]
    cases += [(old['seed'], old['day'], method, rep)
              for method in ('zero', 'forward050', 'bilateral050') for rep in (0, 1)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        actual = [f.result() for f in [pool.submit(replay, root, *case) for case in cases]]
    for measured, cached in ((actual[0]['results'], old['results']),
                             (actual[1]['results'], preserved['results']['preserve'])):
        for key in (*METRICS, 'action_counts'):
            pilot.joint.assert_same_state(measured[key], cached[key], 'V106_cached_' + key)
    for a, b in ((2, 3), (4, 5), (6, 7)):
        for key in (*METRICS, 'action_counts'):
            pilot.joint.assert_same_state(actual[a]['results'][key], actual[b]['results'][key], 'sampler_invariant_' + key)
        pilot.joint.assert_same_state(actual[a]['contract'], actual[b]['contract'], 'sampler_invariant_contract')
    if actual[2]['results']['holding_s_per_trip'] != 0:
        raise AssertionError('zero holding control executed holding')
    for index in (4, 6):
        if actual[index]['coverage']['decisions'] == 0:
            raise AssertionError('smoke did not exercise the rule')
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_cached_native_preserve_and_rule_sampler_invariance_passed',
        smoke_rollouts=len(cases), formal_rollouts=len(ITEMS), training_updates=0, main_policy_changed=False))
    print('V106_SMOKE_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v104-root', type=Path)
    p.add_argument('--v96-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v104_root, a.v96_root, a.workers)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), range(a.start, a.end)))
        pilot.write_json(a.out_root / 'shards' / f'{a.start}-{a.end}.json', dict(status='complete'))


if __name__ == '__main__':
    main()
