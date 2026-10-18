#!/usr/bin/env python3
"""Frozen-policy independent pilot of departure-settled incremental response."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from freqduet_incremental_compensation import ACTIONS, Compensation
import run_freqduet_v110_observed_disturbance as prior


pilot, confirmation = prior.pilot, prior.confirmation
SEEDS = (12011, 933013)
DAYS = tuple(range(2131001, 2131005))
PRIMARY = 'ledger_incremental'
ABLATION = 'bounded_innovation'
CONTROLS = ('native', 'preserve', 'bilateral025')
METHODS = (*CONTROLS, PRIMARY, ABLATION)
ITEMS = [(s, d, m, rep) for s, d, m in itertools.product(SEEDS, DAYS, METHODS)
         for rep in (range(3) if m in ('native', 'preserve') else (0,))]
METRICS = prior.METRICS
CONTRASTS = {m + '_vs_' + base: (m, base) for m in (PRIMARY, ABLATION) for base in CONTROLS}
CONTRASTS[PRIMARY + '_vs_' + ABLATION] = (PRIMARY, ABLATION)


class IncrementalAudit(prior.DisturbanceAudit):
    def __init__(self, seed, method):
        super().__init__(seed, 'bilateral025' if method in (PRIMARY, ABLATION) else method)
        self.incremental_method = method
        self.budget = Compensation(retain=method == PRIMARY)
        self.decisions = []

    def attach(self, runner):
        super().attach(runner)
        self.settle_departure = self.recorder.record_departure

        def departure(recorder, station, direction, time, trip):
            result = self.settle_departure(recorder, station, direction, time, trip)
            key = int(trip), int(station), bool(direction)
            if key in self.budget.pending:
                self.budget.settle(key, self.departures[-1]['realized_holding'])
            return result

        self.recorder.record_departure = departure

    def detach(self):
        self.recorder.record_departure = self.settle_departure
        super().detach()

    def __call__(self, policy, states, features, theta):
        baseline = super().__call__(policy, states, features, theta)
        if self.incremental_method not in (PRIMARY, ABLATION):
            return baseline
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
        predecessor = int(bus.forward_predecessor_trip_id) if context.evidence_valid else None
        residual = self.observable_coordinates(bus, context)[2] if context.evidence_valid else None
        error = (((context.follower_departure_gap_s-context.forward_headway_s)/2
            if context.follower_evidence_valid else context.target_headway_s-context.forward_headway_s)
            if context.evidence_valid else 0.)
        held = float(bus.episode_hold_vehicle_seconds)-self.launches[key[0]]['holding_counter_at_launch']
        selected, row = self.budget.request(key, predecessor, residual, error,
            policy.feasible_action_mask(states)[0].numpy(), ACTIONS[int(baseline[0].argmax())], held)
        row.update(policy_time_s=float(self.env.current_time), follower_valid=bool(context.follower_evidence_valid))
        self.decisions.append(row)
        law = torch.zeros_like(baseline)
        law[0, selected] = 1.
        return law

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        if self.incremental_method not in (PRIMARY, ABLATION):
            self.accounting = {}
            return
        if self.budget.pending or len(self.decisions) != len(records):
            raise AssertionError('incremental actions lack completed execution receipts')
        for row, sampled in zip(self.decisions, records):
            if (row['trip_id'], row['station_id'], row['direction']) != tuple(
                    sampled[k] for k in ('trip_id', 'station_id', 'direction')):
                raise AssertionError('decision identity differs from sampled action')
            if row['chosen_s'] != ACTIONS[sampled['action']] or row['executed_s'] != row['chosen_s']:
                raise AssertionError('completed holding differs from the chosen seven-bin action')
            if row['chosen_s'] > max(row['feedback_error_s'], 0.):
                raise AssertionError('incremental response overshot the observed gap balance')
        errors = self.budget.conservation_errors()
        if max(errors.values()) > 1e-8:
            raise AssertionError('executed compensation budget is not conserved')
        rows = self.decisions
        self.accounting = dict(decisions=len(rows), settled=len(rows),
            changed_commands=sum(r['chosen_s'] != r['shadow_s'] for r in rows),
            balanced=sum(r['feedback_error_s'] <= 0 for r in rows),
            balanced_nonzero=sum(r['feedback_error_s'] <= 0 and r['chosen_s'] > 0 for r in rows),
            cap_actions=sum(r['chosen_s'] == ACTIONS[-1] for r in rows),
            retained_nonzero=sum(r['innovation_s'] == 0 and r['unserved_s'] != 0 and
                r['reset_reason'] == 'continued' for r in rows),
            executed_s=sum(r['executed_s'] for r in rows), shadow_s=sum(r['shadow_s'] for r in rows),
            signed_extra_s=sum(r['settled_extra_s'] for r in rows),
            absolute_innovation_s=sum(abs(r['innovation_s']) for r in rows),
            discarded_absolute_s=sum(abs(r['discarded_s']) for r in rows),
            **{reason: sum(r['reset_reason'] == reason for r in rows)
               for reason in ('new_reference', 'missing_forward', 'single_step', 'continued')},
            target_conservation_error=errors['target'], delivered_conservation_error=errors['delivered'])


def replay(root, seed, day, method, rep):
    rule, credit = IncrementalAudit(seed, method), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day+rep,
        [0.]*4, f'v113_{seed}_{day}_{method}_{rep}', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    name = f'{seed}_{day}_{method}_{rep}'
    rule.save(root / 'records' / (name+'.npz'))
    trace = root / 'traces' / (name+'.jsonl.gz')
    trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace, 'wt', encoding='ascii') as output:
        for decision in rule.decisions:
            output.write(json.dumps(decision, allow_nan=False)+'\n')
    return dict(seed=seed, day=day, method=method, rep=rep, results=row,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)


def run_item(root, index):
    if not (root / 'smoke/done.json').exists():
        raise AssertionError('independent pilot must wait for consumed-path smoke')
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        pilot.write_json(path, replay(root, *ITEMS[index]))
    print('V113_INCREMENTAL_ITEM_DONE', index, flush=True)


def smoke(root, old, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v113_*.py', '-v'], check=True)
    day = prior.DAYS[0]
    cases = [(s, day, 'bilateral025', 0) for s in prior.SEEDS]
    cases += [(prior.SEEDS[0], day, m, 0) for m in ('native', 'preserve')]
    cases += [(prior.SEEDS[0], day, m, rep) for m in (PRIMARY, ABLATION) for rep in (0, 1)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        actual = [f.result() for f in [pool.submit(replay, root, *case) for case in cases]]
    for case, current in zip(cases[:4], actual[:4]):
        cached = json.loads((old / 'pairs' / f'p{prior.ITEMS.index(case)}.json').read_text())
        for key in ('results', 'contract', 'coverage', 'feedforward'):
            pilot.joint.assert_same_state(current[key], cached[key], 'V113_cached_'+key)
        # Compare all timing and launch arrays before opening independent days.
        with np.load(root / 'records' / ('_'.join(map(str, case))+'.npz')) as new:
            with np.load(old / 'records' / ('_'.join(map(str, case))+'.npz')) as before:
                for key in before.files:
                    np.testing.assert_array_equal(new[key], before[key])
    for a, b in ((4, 5), (6, 7)):
        for key in ('results', 'contract', 'coverage', 'accounting'):
            pilot.joint.assert_same_state(actual[a][key], actual[b][key], 'V113_sampler_invariance_'+key)
        if actual[a]['accounting']['changed_commands'] == 0 or actual[a]['accounting']['balanced'] == 0:
            raise AssertionError('smoke failed to exercise changed and balanced decisions')
    pilot.write_json(root / 'smoke/done.json', dict(status='multistop_tests_exact_controls_and_execution_accounting_passed',
        smoke_rollouts=len(cases), independent_rollouts=len(ITEMS), fresh_demand_labels=0,
        actor_updates=0, dual_updates=0, main_policy_changed=False))
    print('V113_SMOKE_COMPLETE', flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V113 rows differ from the frozen independent grid')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for row in cell[1:]:
            pilot.joint.assert_same_state(row['contract'], cell[0]['contract'], 'V113_shared_contract')
        means = {m: {k: float(np.mean([r['results'][k] for r in cell if r['method'] == m]))
            for k in METRICS} for m in METHODS}
        absolute.append(dict(seed=seed, day=day, values=means))
        paired.append(dict(seed=seed, day=day, effects={name: {k: means[a][k]-means[b][k] for k in METRICS}
            for name, (a, b) in CONTRASTS.items()}))
    stats = prior.prior.confirmation.describe(paired, 'effects')
    gates = {base: confirmation.joint_gate(stats[PRIMARY+'_vs_'+base]) for base in CONTROLS}
    pilot.write_json(root / 'summary/status.json', dict(status='incremental_compensation_independent_pilot_complete',
        effects=stats, absolute=prior.prior.confirmation.describe(absolute, 'values'),
        axes=prior.prior.factorial.prior.prior.axis_summary(paired, 'effects', confirmation.KEY_METRICS),
        actor_day=paired, diagnostic_gates=gates, joint_all_controls=all(gates.values()),
        accounting={m: {k: sum(r['accounting'][k] for r in rows if r['method'] == m)
            for k in next(r['accounting'] for r in rows if r['method'] == m)} for m in (PRIMARY, ABLATION)},
        archived_actors=list(SEEDS), new_demand_days=list(DAYS), rollouts=len(ITEMS),
        feedback_gain=.25, innovation_gain=.75, gains_refitted=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='two archived actors and four new demand days; frozen-policy structural pilot, not admission',
        limitations='Executed compensation is relative to the quarter-gain shadow on visited states, not causal attribution. '
            'The current-gap bound uses the existing follower forecast and does not guarantee full-day regularity. '
            'No external fixed-upper baseline, newly trained actor, or frequency-separation claim.'))
    print('V113_INCREMENTAL_PILOT_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v110-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v110_root, a.workers)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), range(a.start, a.end)))
        pilot.write_json(a.out_root / 'shards' / f'{a.start}-{a.end}' / 'done.json', dict(status='complete'))
