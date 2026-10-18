#!/usr/bin/env python3
"""Read consumed V110 action/timing records; no rollout or model update."""

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import run_freqduet_v110_observed_disturbance as prior
import run_freqduet_v92_elapsed_review as elapsed


pilot = prior.pilot
CELLS = list(itertools.product(prior.SEEDS, prior.DAYS))


def balanced_response(residual):
    context = SimpleNamespace(evidence_valid=True, follower_evidence_valid=True,
        target_headway_s=360., forward_headway_s=360., follower_departure_gap_s=360.)
    desired = prior.desired_holding(context, residual)
    action = prior.quantized_action(desired, np.ones(len(pilot.step.ACTIONS), dtype=bool))
    return dict(cumulative_nonholding_gap_s=residual, bilateral_error_s=0.,
        original_command_s=0., ff_desired_s=float(desired), ff_command_s=float(pilot.step.ACTIONS[action]))


def path_accounting(records, results, trips):
    departures = {elapsed.key(r): r for r in records['departures']}
    histogram, stations = Counter(), defaultdict(lambda: [0, 0., 0.])
    previous, carried = {}, {}
    persistence = dict(adjacent_same_predecessor=0, negative_then_negative=0,
        negative_then_negative_after_own_holding=0, holding_at_repeated_negative_s=0.)
    for row in records['arrivals']:
        k = elapsed.key(row)
        departure = departures[k]
        hold = departure['realized_holding'] - row['realized_holding']
        if hold not in pilot.step.ACTIONS:
            raise AssertionError('observed stop holding is outside the frozen seven bins')
        histogram[str(int(hold))] += 1
        if departure['time_s'] - row['time_s'] < hold:
            raise AssertionError('executed holding exceeds stop occupancy')
        np.testing.assert_allclose(row['release'] + row['realized_holding'] + row['other_elapsed'],
            row['time_s'], atol=1e-9, rtol=0)
        station = stations[str((k[1], k[2]))]
        station[0] += 1
        station[1] += hold
        station[2] += int(hold == pilot.step.ACTIONS[-1])
        # Arrival-to-arrival contrasts, not reconstructed ready/follower contexts.
        if k[1:] in previous:
            before = previous[k[1:]]
            residual = row['other_elapsed'] - before['other_elapsed']
            if k[0] in carried:
                old_key, old_before, old_residual = carried[k[0]]
                if abs(k[1] - old_key[1]) == 1 and before['trip_id'] == old_before:
                    persistence['adjacent_same_predecessor'] += 1
                    if old_residual < 0 and residual < 0:
                        persistence['negative_then_negative'] += 1
                        if row['realized_holding'] > 0:
                            persistence['negative_then_negative_after_own_holding'] += 1
                            persistence['holding_at_repeated_negative_s'] += hold
            carried[k[0]] = k, before['trip_id'], residual
        previous[k[1:]] = row
    actions = {str(int(a)): int(results['action_counts'].get(str(int(a)), 0)) for a in pilot.step.ACTIONS}
    # Non-observed initial trips can add zero-hold events, but no positive dose.
    for a in pilot.step.ACTIONS[1:]:
        if histogram[str(int(a))] != actions[str(int(a))]:
            raise AssertionError('executed positive-bin counts do not reproduce commanded actions')
    if histogram['0'] < actions['0']:
        raise AssertionError('recorded zero holds omit observed zero commands')
    total = sum(int(a) * n for a, n in histogram.items())
    np.testing.assert_allclose(total / trips, results['holding_s_per_trip'], atol=1e-6, rtol=0)
    moments = elapsed.prior.timing_series(records['arrivals'])
    np.testing.assert_allclose(moments['cv_squared'], results['cv_squared'], atol=1e-10, rtol=0)
    decisions = sum(actions.values())
    dose = dict(executed_holding_s=float(total), holding_s_per_trip=float(total / trips),
        decisions=decisions, mean_command_s=total / decisions,
        zero_fraction=actions['0'] / decisions, cap45_fraction=actions['45'] / decisions,
        arrival_events=len(records['arrivals']), unobserved_zero_events=histogram['0'] - actions['0'])
    return dict(dose=dose, action_counts=actions, arrival_persistence=persistence,
        station_descriptive={k: dict(arrivals=v[0], holding_s=v[1], cap45_count=v[2]) for k, v in stations.items()})


def read_path(old, index):
    seed, day, method, rep = prior.ITEMS[index]
    row = json.loads((old / 'pairs' / f'p{index}.json').read_text())
    if tuple(row[k] for k in ('seed', 'day', 'method', 'rep')) != prior.ITEMS[index]:
        raise AssertionError('V111 changed the complete consumed V110 grid')
    path = old / 'records' / f'{seed}_{day}_{method}_{rep}.npz'
    records = elapsed.read_records(path)
    with np.load(path) as data:
        trips = len(data['launches'])
    measured = path_accounting(records, row['results'], trips)
    return dict(seed=seed, day=day, method=method, rep=rep, **measured)


def review(root, old):
    paths = [read_path(old, i) for i in range(len(prior.ITEMS))]
    cells = [dict(seed=s, day=d, values={m: {
        k: float(np.mean([r['dose'][k] for r in paths if (r['seed'], r['day'], r['method']) == (s, d, m)]))
        for k in paths[0]['dose']} for m in prior.METHODS}) for s, d in CELLS]
    parent = json.loads((old / 'summary/status.json').read_text())
    dose = prior.prior.confirmation.describe(cells, 'values')
    for method in prior.METHODS:
        np.testing.assert_allclose(dose[method]['holding_s_per_trip']['mean'],
            parent['absolute'][method]['holding_s_per_trip']['mean'], atol=1e-6, rtol=0)
    totals = {m: {str(int(a)): sum(r['action_counts'][str(int(a))] for r in paths if r['method'] == m)
        for a in pilot.step.ACTIONS} for m in prior.METHODS}
    persistence = {m: {k: sum(r['arrival_persistence'][k] for r in paths if r['method'] == m)
        for k in paths[0]['arrival_persistence']} for m in prior.METHODS}
    for row in paths:
        pilot.write_json(root / 'details' / f"{row['seed']}_{row['day']}_{row['method']}_{row['rep']}.json", row)
    ff = parent['feedforward_totals']
    result = dict(status='v110_readonly_action_accounting_complete', absolute_dose=dose,
        observed_action_counts=totals, arrival_to_arrival_persistence=persistence,
        ff_requests=dict(valid_forward=ff['decisions'],
            signed_mean_s=ff['requested_adjustment_sum_s'] / ff['decisions'],
            absolute_mean_s=ff['absolute_adjustment_sum_s'] / ff['decisions'],
            shadow_command_delta_mean_s=ff['selected_command_delta_sum_s'] / ff['decisions'],
            low_request_fraction=ff['low_saturated'] / ff['decisions'],
            high_request_fraction=ff['high_saturated'] / ff['decisions']),
        balanced_gap_law_examples=[balanced_response(r) for r in (-60., 0., 60.)],
        actor_day_dose=cells, existing_paths_read=len(paths), new_rollouts=0, checkpoint_loads=0,
        model_updates=0, main_policy_changed=False, controller_admitted=False,
        v110_candidate_closed=True, gain_refitted=False,
        scope='consumed two-actor/four-day paths; executed dose accounting and arrival-state association, not mediation',
        limitations='Saved NPZ lacks exact ready/follower/feasibility contexts. Arrival residual persistence cannot '
            'quantify repeated compensation at decision time; analytic balanced-gap examples are not rollout effects.')
    pilot.write_json(root / 'summary/status.json', result)
    print('V111_READONLY_ACCOUNTING_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v110-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v111_*.py', '-v'], check=True)
    review(a.out_root, a.v110_root)
