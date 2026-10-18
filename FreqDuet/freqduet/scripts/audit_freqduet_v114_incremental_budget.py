#!/usr/bin/env python3
"""Read consumed V113 traces: carried budgets, pulse response and actuator arithmetic."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_incremental_compensation import ACTIONS, Compensation, bounded_action
import run_freqduet_v113_incremental_compensation as prior


METHODS = (prior.PRIMARY, prior.ABLATION)
COMPONENTS = ('feedback_minus_shadow_s', 'effective_innovation_s', 'carry_s',
              'zero_clipping_s', 'cap_clipping_s', 'balance_bound_s', 'quantization_s')


def decision_math(row):
    error = row['feedback_error_s']
    feedback = .25 * error
    innovation = row['increment_s'] if error > 0 else 0.
    carry = row['unserved_s'] - innovation
    raw = row['raw_request_s']
    clipped = min(max(raw, 0.), ACTIONS[-1])
    balanced = min(clipped, max(error, 0.))
    chosen = ACTIONS[bounded_action(raw, error, [True]*7)]
    pulse = ACTIONS[bounded_action(feedback+innovation, error, [True]*7)]
    shadow = ACTIONS[min(range(7), key=lambda i: abs(ACTIONS[i]-min(max(feedback, 0.), ACTIONS[-1])))]
    if chosen != row['chosen_s'] or shadow != row['shadow_s'] or raw != feedback+row['unserved_s']:
        raise AssertionError('trace differs from the unmasked seven-bin V113 law')
    result = dict(feedback_minus_shadow_s=feedback-shadow, effective_innovation_s=innovation,
        carry_s=carry, zero_clipping_s=max(-raw, 0.), cap_clipping_s=min(ACTIONS[-1]-raw, 0.),
        balance_bound_s=balanced-clipped, quantization_s=chosen-balanced,
        command_delta_s=chosen-shadow, chosen_s=chosen, pulse_shadow_s=pulse,
        carry_command_delta_s=chosen-pulse, raw_increment_s=row['increment_s'],
        positive_increment_s=max(innovation, 0.), negative_increment_s=min(innovation, 0.))
    if not math.isclose(sum(result[k] for k in COMPONENTS), result['command_delta_s'], abs_tol=1e-9):
        raise AssertionError('nonlinear actuator arithmetic does not sum to the actual command delta')
    return result


def cohorts(row, values):
    def sign(x):
        return 'positive' if x > 0 else 'negative' if x < 0 else 'zero'
    names = ['all', 'carry_'+sign(values['carry_s']), 'innovation_'+sign(values['effective_innovation_s'])]
    if row['feedback_error_s'] <= 0:
        names.append('balanced')
    if row['chosen_s'] == ACTIONS[-1]:
        names.append('cap45')
    if row['innovation_s'] == 0 and values['carry_s'] > 0:
        names.append('positive_carry_without_new_innovation')
    return names


def accumulate(totals, row, values):
    for group in cohorts(row, values):
        cell = totals[group]
        cell['count'] += 1
        for k, v in values.items():
            cell[k] += v


def path_audit(root, seed, day, method):
    index = prior.ITEMS.index((seed, day, method, 0))
    pair = json.loads((root / 'pairs' / f'p{index}.json').read_text())
    if tuple(pair[k] for k in ('seed', 'day', 'method', 'rep')) != (seed, day, method, 0):
        raise AssertionError('V113 consumed path identity differs')
    name = f'{seed}_{day}_{method}_0'
    with np.load(root / 'records' / (name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        arrays = {kind: [dict(zip(fields, row.tolist())) for row in record[kind]]
                  for kind in ('arrivals', 'departures')}
        trips = len(record['launches'])
    def key(r):
        return int(r['trip_id']), int(r['station_id']), bool(r['direction'])
    arrivals = {key(r): r for r in arrays['arrivals']}
    departures = {key(r): r for r in arrays['departures']}
    rule = Compensation(retain=method == prior.PRIMARY)
    groups, decisions, counts = defaultdict(lambda: defaultdict(float)), {}, Counter()
    balanced_nonzero = 0
    with gzip.open(root / 'traces' / (name+'.jsonl.gz'), 'rt', encoding='ascii') as source:
        for line in source:
            row = json.loads(line)
            identity = key(row)
            if identity in decisions:
                raise AssertionError('a controlled stop was counted twice')
            arrival, departure = arrivals[identity], departures[identity]
            if (arrival['realized_holding'] != row['executed_before_s'] or
                    departure['realized_holding']-arrival['realized_holding'] != row['executed_s'] or
                    row['executed_s'] != row['chosen_s'] or
                    not arrival['time_s'] <= row['policy_time_s'] <= departure['time_s']):
                raise AssertionError('trace does not match completed action timing and actual execution')
            _, rebuilt = rule.request(identity, row['predecessor_trip_id'], row['residual_s'],
                row['feedback_error_s'], [True]*7, row['shadow_s'], row['executed_before_s'])
            rule.settle(identity, departure['realized_holding'])
            prior.pilot.joint.assert_same_state(rebuilt, {k: row[k] for k in rebuilt}, 'V114_budget_reconstruction')
            values = decision_math(row)
            accumulate(groups, row, values)
            decisions[identity] = values
            counts[str(int(row['chosen_s']))] += 1
            balanced_nonzero += row['feedback_error_s'] <= 0 and row['chosen_s'] > 0
    if rule.pending or len(decisions) != pair['accounting']['decisions']:
        raise AssertionError('V114 reconstructed budget lacks completed decisions')
    actual = groups['all']['chosen_s']
    if actual != pair['accounting']['executed_s'] or balanced_nonzero != pair['accounting']['balanced_nonzero']:
        raise AssertionError('V114 trace totals differ from V113 accounting')
    expected_counts = {str(int(a)): counts[str(int(a))] for a in ACTIONS}
    prior.pilot.joint.assert_same_state(expected_counts, pair['results']['action_counts'], 'V114_executed_bins')
    if not math.isclose(actual/trips, pair['results']['holding_s_per_trip'], abs_tol=1e-6):
        raise AssertionError('V113 endpoint holding does not match its completed action trace')
    errors = rule.conservation_errors()
    if max(errors.values()) > 1e-8:
        raise AssertionError('reconstructed execution budget is not conserved')
    return dict(seed=seed, day=day, method=method, groups=dict(groups), launched_trips=trips,
                balanced_nonzero=balanced_nonzero, conservation_errors=errors), decisions


def matched_commands(ledger, pulse):
    if ledger.keys() != pulse.keys():
        raise AssertionError('the two cleared paths have different controlled trip/stop support')
    direct = sum(v['carry_command_delta_s'] for v in ledger.values())
    propagated = sum(ledger[k]['pulse_shadow_s']-pulse[k]['chosen_s'] for k in ledger)
    actual = sum(ledger[k]['chosen_s']-pulse[k]['chosen_s'] for k in ledger)
    if not math.isclose(direct+propagated, actual, abs_tol=1e-9):
        raise AssertionError('same-prefix response plus between-path prefix change does not sum')
    return dict(matched_decisions=len(ledger), actual_holding_delta_s=actual,
        same_ledger_prefix_carry_response_s=direct, between_path_prefix_difference_s=propagated)


def review(root, out):
    parent = json.loads((root / 'summary/status.json').read_text())
    if parent['status'] != 'incremental_compensation_independent_pilot_complete' or parent['rollouts'] != len(prior.ITEMS):
        raise AssertionError('V114 needs the completed frozen V113 pilot')
    combined = {m: defaultdict(lambda: defaultdict(float)) for m in METHODS}
    paths, matched = [], []
    for seed in prior.SEEDS:
        for day in prior.DAYS:
            decisions = {}
            for method in METHODS:
                summary, decisions[method] = path_audit(root, seed, day, method)
                paths.append(summary)
                for group, values in summary['groups'].items():
                    for k, v in values.items():
                        combined[method][group][k] += v
            matched.append(dict(seed=seed, day=day, **matched_commands(decisions[prior.PRIMARY], decisions[prior.ABLATION])))
    for method in METHODS:
        counts = combined[method]['all']
        expected = parent['accounting'][method]
        if counts['count'] != expected['decisions'] or counts['chosen_s'] != expected['executed_s']:
            raise AssertionError('V114 coverage differs from the full V113 aggregate')
    prior.pilot.write_json(out / 'summary/status.json', dict(status='v113_consumed_incremental_budget_audit_complete',
        combined={m: dict(v) for m, v in combined.items()}, actor_day=paths, matched_actor_day=matched,
        matched_totals={k: sum(r[k] for r in matched) for k in matched[0] if k not in ('seed', 'day')},
        consumed_traces=len(paths), new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        v113_primary_closed=True, ablation_promoted=False,
        scope='read-only command arithmetic on two actors and four consumed days; no controller fitting',
        limitations='Cohorts overlap except within each carry/innovation sign partition. '
            'Same-prefix pulse shadows are not executed counterfactuals; between-path prefix differences '
            'include feedback and history propagation, not an isolated causal mediator. '
            'Action arithmetic does not allocate passenger-time or CV loss, and traces lack per-decision passenger load.'))
    print('V114_INCREMENTAL_BUDGET_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v113-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v114_*.py', '-v'], check=True)
    review(a.v113_root, a.out_root)
