#!/usr/bin/env python3
"""Fresh-day pilot of fixed observed nonholding-disturbance feedforward."""

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

import run_freqduet_v106_service_tradeoff as prior
import run_freqduet_v107_rule_confirmation as confirmation


pilot = prior.pilot
SEEDS = prior.SEEDS
DAYS = tuple(range(2101001, 2101005))
CANDIDATE = 'bilateral025_nonholding_ff'
METHODS = ('native', 'preserve', 'zero', 'bilateral025', CANDIDATE)
ITEMS = [(s, d, m, rep) for s, d, m in itertools.product(SEEDS, DAYS, METHODS)
         for rep in (range(3) if m in ('native', 'preserve') else (0,))]
METRICS = prior.METRICS
CONTRASTS = {CANDIDATE + '_vs_' + base: (CANDIDATE, base) for base in METHODS if base != CANDIDATE}
CONTRASTS['bilateral025_vs_preserve'] = ('bilateral025', 'preserve')


def desired_holding(context, nonholding_gap_s):
    if not context.evidence_valid:
        return 0.
    error = ((context.follower_departure_gap_s - context.forward_headway_s) / 2
             if context.follower_evidence_valid else context.target_headway_s - context.forward_headway_s)
    return .25 * error - .75 * nonholding_gap_s


def quantized_action(desired, feasible):
    desired = np.clip(desired, 0., pilot.step.ACTIONS[-1])
    choices = np.flatnonzero(feasible)
    return int(choices[np.argmin(np.abs(pilot.step.ACTIONS[choices] - desired))])


class DisturbanceAudit(prior.TradeoffAudit):
    def __init__(self, seed, method):
        super().__init__(seed, 'bilateral025' if method == CANDIDATE else method)
        self.variant = method
        self.feedforward = dict(decisions=0, negative_gap=0, positive_gap=0, changed_commands=0,
            nonholding_gap_sum_s=0., requested_adjustment_sum_s=0., absolute_adjustment_sum_s=0.,
            selected_command_delta_sum_s=0., low_saturated=0, high_saturated=0)

    def __call__(self, policy, states, features, theta):
        baseline = super().__call__(policy, states, features, theta)
        if self.variant != CANDIDATE:
            return baseline
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        if not context.evidence_valid:
            return baseline
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        _, _, gap = self.observable_coordinates(bus, context)
        desired = desired_holding(context, gap)
        selected = quantized_action(desired, policy.feasible_action_mask(states)[0].numpy())
        old = int(baseline[0].argmax())
        self.feedforward['decisions'] += 1
        self.feedforward['negative_gap'] += gap < 0
        self.feedforward['positive_gap'] += gap > 0
        self.feedforward['changed_commands'] += selected != old
        self.feedforward['nonholding_gap_sum_s'] += gap
        self.feedforward['requested_adjustment_sum_s'] += -.75 * gap
        self.feedforward['absolute_adjustment_sum_s'] += abs(.75 * gap)
        self.feedforward['selected_command_delta_sum_s'] += float(pilot.step.ACTIONS[selected] - pilot.step.ACTIONS[old])
        self.feedforward['low_saturated'] += desired <= 0
        self.feedforward['high_saturated'] += desired >= float(pilot.step.ACTIONS[-1])
        law = torch.zeros_like(baseline)
        law[0, selected] = 1.
        return law


def replay(root, seed, day, method, rep):
    rule, credit = DisturbanceAudit(seed, method), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000 + seed + 100 * day + rep,
        [0.] * 4, f'v110_{seed}_{day}_{method}_{rep}', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    rule.save(root / 'records' / f'{seed}_{day}_{method}_{rep}.npz')
    return dict(seed=seed, day=day, method=method, rep=rep, results=row,
                contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward)


def run_item(root, index):
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        pilot.write_json(path, replay(root, *ITEMS[index]))
    print('V110_DISTURBANCE_ITEM_DONE', index, flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V110 pilot rows differ from the frozen grid')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for row in cell[1:]:
            pilot.joint.assert_same_state(row['contract'], cell[0]['contract'], 'V110_shared_physical_contract')
        means = {m: {k: float(np.mean([r['results'][k] for r in cell if r['method'] == m])) for k in METRICS}
                 for m in METHODS}
        absolute.append(dict(seed=seed, day=day, values=means))
        paired.append(dict(seed=seed, day=day, effects={name: {k: means[a][k] - means[b][k] for k in METRICS}
            for name, (a, b) in CONTRASTS.items()}))
    stats = prior.confirmation.describe(paired, 'effects')
    gates = {base: confirmation.joint_gate(stats[CANDIDATE + '_vs_' + base])
             for base in ('native', 'preserve', 'bilateral025')}
    primary = [r for r in rows if r['method'] == CANDIDATE]
    pilot.write_json(root / 'summary/status.json', dict(status='observed_disturbance_development_pilot_complete',
        effects=stats, absolute=prior.confirmation.describe(absolute, 'values'),
        axes=prior.factorial.prior.prior.axis_summary(paired, 'effects', confirmation.KEY_METRICS),
        diagnostic_gates=gates, joint_all_controls=all(gates.values()),
        feedforward_totals={k: sum(r['feedforward'][k] for r in primary) for k in primary[0]['feedforward']},
        coverage={k: sum(r['coverage'][k] for r in primary) for k in primary[0]['coverage']},
        archived_development_actors=list(SEEDS), demand_days=list(DAYS), new_rollouts=len(ITEMS),
        feedback_gain=.25, feedforward_gain=.75, gain_refitted=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, deterministic_paths_per_actor_day=1, stochastic_repeats=3,
        scope='single fresh-day structural development pilot; no adaptive coefficient/day search or promotion; '
              'zero retains learned upper, not an external fixed-upper baseline'))
    print('V110_DISTURBANCE_PILOT_COMPLETE', gates, flush=True)


def smoke(root, old, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v110_*.py', '-v'], check=True)
    seed, day = SEEDS[0], prior.DAYS[0]
    cases = [(seed, day, m, 0) for m in METHODS if m != CANDIDATE]
    cases += [(seed, day, CANDIDATE, rep) for rep in (0, 1)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        actual = [f.result() for f in [pool.submit(replay, root, *case) for case in cases]]
    for case, current in zip(cases[:4], actual[:4]):
        cached = json.loads((old / 'pairs' / f'p{prior.ITEMS.index(case)}.json').read_text())
        for key in (*METRICS, 'action_counts'):
            pilot.joint.assert_same_state(current['results'][key], cached['results'][key], 'V110_cached_' + key)
        for key in ('contract', 'coverage'):
            pilot.joint.assert_same_state(current[key], cached[key], 'V110_cached_' + key)
    for key in (*METRICS, 'action_counts'):
        pilot.joint.assert_same_state(actual[4]['results'][key], actual[5]['results'][key], 'V110_sampler_invariant_' + key)
    for key in ('contract', 'coverage', 'feedforward'):
        pilot.joint.assert_same_state(actual[4][key], actual[5][key], 'V110_sampler_invariant_' + key)
    if actual[4]['feedforward']['changed_commands'] == 0:
        raise AssertionError('V110 smoke did not exercise observed-disturbance compensation')
    pilot.write_json(root / 'smoke/done.json', dict(status='tests_cached_controls_and_causal_candidate_sampler_invariance_passed',
        smoke_rollouts=len(cases), fresh_development_labels=0, formal_rollouts=len(ITEMS), main_policy_changed=False))
    print('V110_SMOKE_COMPLETE', flush=True)


if __name__ == '__main__':
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
