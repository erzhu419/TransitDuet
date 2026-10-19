#!/usr/bin/env python3
"""Frozen-policy pilot of causal, non-borrowing holding authorization."""

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

from freqduet_funded_quota import ACTIONS, FundedQuota
from freqduet_incremental_compensation import bounded_action
import run_freqduet_v113_incremental_compensation as prior


pilot, confirmation = prior.pilot, prior.confirmation
SEEDS = prior.SEEDS
DAYS = tuple(range(2151001, 2151005))
PRIMARY = 'funded_quota'
CONTROLS = (*prior.CONTROLS, prior.ABLATION)
METHODS = (*CONTROLS, PRIMARY)
ITEMS = [(s, d, m, rep) for s, d, m in itertools.product(SEEDS, DAYS, METHODS)
         for rep in (range(3) if m in ('native', 'preserve') else (0,))]
METRICS = prior.METRICS
CONTRASTS = {PRIMARY+'_vs_'+base: (PRIMARY, base) for base in CONTROLS}
CONTRASTS.update({prior.ABLATION+'_vs_'+base: (prior.ABLATION, base) for base in prior.CONTROLS})


class FundedAudit(prior.IncrementalAudit):
    def __init__(self, seed):
        super().__init__(seed, 'bilateral025')
        self.budget = FundedQuota()

    def attach(self, runner):
        super().attach(runner)
        self.unobserved_step = self.env.step

        def step(*args, **kwargs):
            result = self.unobserved_step(*args, **kwargs)
            for bus in self.env.bus_all:
                trip = int(bus.trip_id)
                if trip in self.budget.trips:
                    held = float(bus.episode_hold_vehicle_seconds)-self.launches[trip]['holding_counter_at_launch']
                    self.budget.check_actual(trip, held)
            return result

        self.env.step = step

    def detach(self):
        self.env.step = self.unobserved_step
        super().detach()

    def __call__(self, policy, states, features, theta):
        baseline = prior.prior.DisturbanceAudit.__call__(self, policy, states, features, theta)
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
        row.update(policy_time_s=float(self.env.current_time), bus_id=int(bus.bus_id),
            onboard_passengers=len(bus.passengers), follower_valid=bool(context.follower_evidence_valid))
        self.decisions.append(row)
        law = torch.zeros_like(baseline)
        law[0, selected] = 1.
        return law

    def finish(self, runner, records, holds):
        prior.prior.DisturbanceAudit.finish(self, runner, records, holds)
        if self.budget.pending or len(self.decisions) != len(records):
            raise AssertionError('funded actions lack completed receipts')
        for row, sampled in zip(self.decisions, records):
            if tuple(row[k] for k in ('trip_id', 'station_id', 'direction')) != tuple(
                    sampled[k] for k in ('trip_id', 'station_id', 'direction')):
                raise AssertionError('funded trace differs from the sampled stop identity')
            if row['chosen_s'] != ACTIONS[sampled['action']] or row['executed_s'] != row['chosen_s']:
                raise AssertionError('funded command differs from its actual execution')
            if row['chosen_s'] > min(max(row['feedback_error_s'], 0.), row['available_s']):
                raise AssertionError('funded command exceeds current evidence or authorization')
            if row['increment_s'] <= 0 and row['chosen_s'] > row['shadow_s']:
                raise AssertionError('old credit overrode a nonpositive current innovation')
        audit = self.budget.audit()
        if audit['conservation_error'] != 0 or audit['minimum_prefix_margin_s'] < 0:
            raise AssertionError('funded authorization is not conserved at every observed prefix')
        rows = self.decisions
        self.accounting = dict(**audit, decisions=len(rows), settled=len(rows),
            changed_commands=sum(r['chosen_s'] != r['shadow_s'] for r in rows),
            above_shadow=sum(r['chosen_s'] > r['shadow_s'] for r in rows),
            below_shadow=sum(r['chosen_s'] < r['shadow_s'] for r in rows),
            negative_innovation=sum(r['increment_s'] < 0 for r in rows),
            nonpositive_innovation_above_shadow=sum(r['increment_s'] <= 0 and r['chosen_s'] > r['shadow_s'] for r in rows),
            balanced=sum(r['feedback_error_s'] <= 0 for r in rows),
            balanced_nonzero=sum(r['feedback_error_s'] <= 0 and r['chosen_s'] > 0 for r in rows),
            cap_actions=sum(r['chosen_s'] == ACTIONS[-1] for r in rows),
            quota_limited=sum(ACTIONS[bounded_action(r['raw_request_s'], r['feedback_error_s'], [True]*7)] > r['available_s']
                for r in rows), requested_increment_sum_s=sum(r['increment_s'] for r in rows))


def replay(root, seed, day, method, rep):
    if method != PRIMARY:
        return prior.replay(root, seed, day, method, rep)
    rule, credit = FundedAudit(seed), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day+rep,
        [0.]*4, f'v115_{seed}_{day}_{method}_{rep}', probability_rule=rule,
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
    return dict(seed=seed, day=day, method=method, rep=rep, results=row, contract=rule.contract,
        coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)


def smoke(root, old, workers):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v115_*.py', '-v'], check=True)
    seed, other, day = SEEDS[0], SEEDS[1], prior.DAYS[0]
    cases = [(seed, day, m, 0) for m in CONTROLS]+[(other, day, 'bilateral025', 0)]
    cases += [(seed, day, PRIMARY, r) for r in (0, 1)]+[(other, day, PRIMARY, 0)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        actual = [f.result() for f in [pool.submit(replay, root, *case) for case in cases]]
    for case, current in zip(cases[:5], actual[:5]):
        cached = json.loads((old / 'pairs' / f'p{prior.ITEMS.index(case)}.json').read_text())
        pilot.joint.assert_same_state(current, cached, 'V115_exact_cached_control')
        name = '_'.join(map(str, case))+'.npz'
        with np.load(root / 'records' / name) as new:
            with np.load(old / 'records' / name) as before:
                for key in before.files:
                    np.testing.assert_array_equal(new[key], before[key])
    prior.assert_sampler_invariance(actual[5], actual[6])
    for index in (5, 7):
        counts = actual[index]['accounting']
        if (not counts['below_shadow'] or not counts['above_shadow'] or not counts['negative_innovation']
                or not counts['quota_limited'] or counts['prefix_checks'] <= counts['decisions']):
            raise AssertionError('smoke did not exercise savings, funded reuse and in-flight prefix checks')
    pilot.write_json(root / 'smoke/done.json', dict(status='funded_quota_tests_exact_controls_and_all_prefixes_passed',
        smoke_rollouts=len(cases), independent_rollouts=len(ITEMS), fresh_demand_labels=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False))
    print('V115_FUNDED_SMOKE_COMPLETE', flush=True)


def run_item(root, index):
    if not (root / 'smoke/done.json').exists():
        raise AssertionError('independent funded pilot must wait for prefix smoke')
    path = root / 'pairs' / f'p{index}.json'
    if not path.exists():
        pilot.write_json(path, replay(root, *ITEMS[index]))
    print('V115_FUNDED_ITEM_DONE', index, flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ('seed', 'day', 'method', 'rep')) for r in rows] != ITEMS:
        raise AssertionError('V115 rows differ from the frozen independent grid')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for r in cell[1:]:
            pilot.joint.assert_same_state(r['contract'], cell[0]['contract'], 'V115_shared_physical_contract')
        means = {m: {k: float(np.mean([r['results'][k] for r in cell if r['method'] == m]))
            for k in METRICS} for m in METHODS}
        absolute.append(dict(seed=seed, day=day, values=means))
        paired.append(dict(seed=seed, day=day, effects={name: {k: means[a][k]-means[b][k] for k in METRICS}
            for name, (a, b) in CONTRASTS.items()}))
    stats = prior.prior.prior.confirmation.describe(paired, 'effects')
    gates = {base: confirmation.joint_gate(stats[PRIMARY+'_vs_'+base]) for base in CONTROLS}
    accounts = [r['accounting'] for r in rows if r['method'] == PRIMARY]
    accounting = {k: sum(a[k] for a in accounts) for k in accounts[0]}
    accounting['minimum_prefix_margin_s'] = min(a['minimum_prefix_margin_s'] for a in accounts)
    accounting['conservation_error'] = max(a['conservation_error'] for a in accounts)
    pilot.write_json(root / 'summary/status.json', dict(status='funded_quota_independent_pilot_complete',
        effects=stats, absolute=prior.prior.prior.confirmation.describe(absolute, 'values'),
        axes=prior.prior.prior.factorial.prior.prior.axis_summary(paired, 'effects', confirmation.KEY_METRICS),
        actor_day=paired, diagnostic_gates=gates, joint_all_controls=all(gates.values()), accounting=accounting,
        archived_actors=list(SEEDS), new_demand_days=list(DAYS), rollouts=len(ITEMS),
        feedback_gain=.25, innovation_gain=.75, gains_refitted=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope='frozen structural pilot, two archived actors and four new demand days; no promotion',
        limitations='Authorization uses quantized quarter-rule shadows on visited states; its prefix bound '
            'does not imply domination of the separate bilateral trajectory. Current balance uses follower estimates. '
            'No external fixed-upper or frequency-separation claim.'))
    print('V115_FUNDED_PILOT_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v113-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v113_root, a.workers)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), range(a.start, a.end)))
        pilot.write_json(a.out_root / 'shards' / f'{a.start}-{a.end}' / 'done.json', dict(status='complete'))
