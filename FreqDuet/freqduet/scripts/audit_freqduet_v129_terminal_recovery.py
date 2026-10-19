#!/usr/bin/env python3
"""Read-only delay propagation and recovery accounting on V128 terminal records."""

import argparse
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys


GAP_S = 360.
ITEMS = ((12011, 2231001), (12011, 2231004), (933013, 2231001), (933013, 2231004))
LOWER = dict(raw='bilateral025', split='split_bilateral025')


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, label):
    if abs(actual-expected) > 1e-6:
        raise AssertionError(f'{label}: {actual} != {expected}')


def transition(reference, actual, visible, previous=None):
    supply = max(0., visible-reference)
    inherited, deficit, recovery, planned_slack = 0., 0., 0., 0.
    if previous is not None:
        planned_gap = reference-previous['reference_s']
        inherited = max(0., previous['delay_s']+GAP_S-planned_gap)
        deficit = max(0., inherited-previous['delay_s'])
        recovery = max(0., previous['delay_s']-inherited)
        planned_slack = max(0., planned_gap-GAP_S)
    delay = actual-reference
    residual = delay-max(inherited, supply)
    if min(delay, residual) < -1e-6:
        raise AssertionError('actual launch precedes its recorded plan, vehicle, or spacing floor')
    innovation = max(0., supply-inherited)
    extra = max(0., inherited-supply)
    order_floor = reference if previous is None else previous['actual_s']
    return dict(reference_s=reference, actual_s=actual, delay_s=delay, supply_floor_s=supply,
        inherited_s=inherited, short_plan_addition_s=deficit, planned_recovery_s=recovery,
        planned_slack_s=planned_slack, new_supply_s=innovation, unexplained_s=max(0., residual),
        hard_spacing_wait_s=extra, immediate_release_advance_s=actual-max(reference, visible, order_floor))


def turnover_split(row, dispatch, launches, returns, clocks):
    prior = dispatch['prior_trip_id']
    if prior < 0:
        return dict(prior_trip_id=prior, prior_trip_delay_s=0., new_supply_without_prior_lateness_s=row['new_supply_s'],
            prior_lateness_exposure_s=0.)
    old, returned = launches[prior], returns[prior]
    prior_delay = old['actual_s']-clocks[prior]
    visibility_lag = dispatch['first_dispatch_visible_s']-returned['return_s']
    cycle = returned['holding_s']+returned['nonholding_elapsed_s']+visibility_lag
    close(cycle, dispatch['first_dispatch_visible_s']-old['actual_s'], 'same-vehicle cycle')
    close(prior_delay+cycle-(row['reference_s']-clocks[prior]),
        dispatch['first_dispatch_visible_s']-row['reference_s'], 'turnover lateness')
    without = max(0., row['supply_floor_s']-prior_delay-row['inherited_s'])
    return dict(prior_trip_id=prior, prior_trip_delay_s=prior_delay,
        prior_holding_s=returned['holding_s'], prior_nonholding_s=returned['nonholding_elapsed_s'],
        planned_cycle_s=row['reference_s']-clocks[prior], new_supply_without_prior_lateness_s=without,
        prior_lateness_exposure_s=row['new_supply_s']-without)


def summarize_direction(rows):
    edges = rows[1:]
    summed = {k: sum(r[k] for r in edges) for k in ('short_plan_addition_s', 'planned_recovery_s',
        'planned_slack_s', 'new_supply_s', 'unexplained_s', 'new_supply_without_prior_lateness_s',
        'prior_lateness_exposure_s')}
    close(rows[0]['delay_s']+summed['short_plan_addition_s']+summed['new_supply_s']
        +summed['unexplained_s']-summed['planned_recovery_s'], rows[-1]['delay_s'], 'direction delay balance')
    length, longest = 0, 0
    for row in rows:
        length = length+1 if row['inherited_s'] > 0 and row['new_supply_s'] == 0 and row['unexplained_s'] == 0 else 0
        longest = max(longest, length)
    thirds = [rows[len(rows)*i//3:len(rows)*(i+1)//3] for i in range(3)]
    return dict(trips=len(rows), first_delay_s=rows[0]['delay_s'], final_delay_s=rows[-1]['delay_s'],
        maximum_delay_s=max(r['delay_s'] for r in rows), balance=summed,
        net_recovery_s=sum(max(0., a['delay_s']-b['delay_s']) for a, b in zip(rows, edges)),
        slack_intervals=sum(r['planned_slack_s'] > 0 for r in edges),
        longest_inherited_chain=longest,
        thirds=[dict(first_trip=part[0]['trip_id'], last_trip=part[-1]['trip_id'],
            mean_delay_s=fmean(r['delay_s'] for r in part)) for part in thirds],
        largest_supply_innovations=sorted(rows, key=lambda r: r['new_supply_s'], reverse=True)[:2])


def paired_returns(record, baseline):
    current_returns = {r['trip_id']: r for r in record['returns']}
    old_returns = {r['trip_id']: r for r in baseline['returns']}
    old_dispatches = {r['trip_id']: r for r in baseline['dispatches']}
    rows = []
    for dispatch in record['dispatches']:
        prior, old = dispatch['prior_trip_id'], old_dispatches[dispatch['trip_id']]
        if prior < 0 or (dispatch['bus_id'], prior) != (old['bus_id'], old['prior_trip_id']):
            continue
        c, b = current_returns[prior], old_returns[prior]
        changes = {k: c[k]-b[k] for k in ('launch_s', 'holding_s', 'nonholding_elapsed_s')}
        changes['visibility_lag_s'] = (dispatch['first_dispatch_visible_s']-c['return_s'])\
            -(old['first_dispatch_visible_s']-b['return_s'])
        changes['visible_s'] = dispatch['first_dispatch_visible_s']-old['first_dispatch_visible_s']
        close(changes['visible_s'], sum(changes[k] for k in
            ('launch_s', 'holding_s', 'nonholding_elapsed_s', 'visibility_lag_s')), 'paired return propagation')
        rows.append(changes)
    return dict(matched_returned_vehicles=len(rows),
        means={k: fmean(r[k] for r in rows) for k in rows[0]} if rows else None)


def analyze_path(record, baseline, receipt):
    clocks = {int(k): v for k, v in record['reference_clocks'].items()}
    launches = {r['trip_id']: r for r in record['dispatches']}
    returns = {r['trip_id']: r for r in record['returns']}
    original = {r['trip_id']: r for r in baseline['dispatches']}
    if launches.keys() != original.keys() or launches.keys() != clocks.keys():
        raise AssertionError('recorded trip rosters differ')
    grouped, rows = {}, []
    for dispatch in record['dispatches']:
        trip, direction = dispatch['trip_id'], dispatch['direction']
        group = grouped.setdefault(str(int(direction)), [])
        visible = dispatch['first_dispatch_visible_s']
        if visible is None:
            if dispatch['prior_trip_id'] >= 0:
                raise AssertionError('returned vehicle has no recorded visibility clock')
            visible = clocks[trip]
        row = transition(clocks[trip], dispatch['actual_s'], visible, group[-1] if group else None)
        row.update(trip_id=trip, **turnover_split(row, dispatch, launches, returns, clocks))
        gate_clock = clocks[trip]+row['inherited_s']
        row['sticky_clock_shortfall_s'] = max(0., gate_clock-dispatch['eligible_s'])
        row['hard_wait_inside_native_s'] = min(row['hard_spacing_wait_s'], row['sticky_clock_shortfall_s'])
        row['qualification_shift_s'] = dispatch['eligible_s']-clocks[trip]
        row['native_readiness_s'] = dispatch['actual_s']-dispatch['eligible_s']
        old = original[trip]
        row['actual_shift_s'] = dispatch['actual_s']-old['actual_s']
        row['reference_shift_s'] = clocks[trip]-old['eligible_s']
        row['same_vehicle_prior'] = (dispatch['bus_id'], dispatch['prior_trip_id']) == (old['bus_id'], old['prior_trip_id'])
        group.append(row)
        rows.append(row)
    mean_keys = ('delay_s', 'supply_floor_s', 'inherited_s', 'hard_spacing_wait_s', 'hard_wait_inside_native_s',
        'unexplained_s', 'immediate_release_advance_s',
        'qualification_shift_s', 'native_readiness_s', 'actual_shift_s', 'reference_shift_s')
    means = {k: fmean(r[k] for r in rows) for k in mean_keys}
    for k, v in [('delay_s', 'inclusive_terminal_delay_s'), ('native_readiness_s', 'readiness_delay_s'),
            ('qualification_shift_s', 'qualification_shift_s')]:
        close(means[k], receipt['values'][v], 'V128 '+v)
    close(means['delay_s'], means['qualification_shift_s']+means['native_readiness_s'], 'terminal clocks')
    old_delay = fmean(r['actual_s']-r['eligible_s'] for r in baseline['dispatches'])
    close(means['actual_shift_s']-means['reference_shift_s'], means['delay_s']-old_delay, 'paired clock balance')
    return dict(means=means, trips=len(rows), baseline_delay_s=old_delay,
        reconstruction_exact=all(r['unexplained_s'] == 0 for r in rows),
        maximum_unexplained_s=max(r['unexplained_s'] for r in rows),
        sticky_clock_shortfall_trips=sum(r['sticky_clock_shortfall_s'] > 0 for r in rows),
        maximum_sticky_clock_shortfall_s=max(r['sticky_clock_shortfall_s'] for r in rows),
        remapped_vehicle_or_prior=sum(not r['same_vehicle_prior'] for r in rows),
        immediate_advance_trips=sum(r['immediate_release_advance_s'] > 0 for r in rows),
        paired_returns=paired_returns(record, baseline),
        directions={k: summarize_direction(v) for k, v in grouped.items()})


def run(root, v128, v127):
    parent = read(v128/'summary/status.json')
    paths = []
    for seed, day in ITEMS:
        for lower, method in LOWER.items():
            label = f'{seed}_{day}_{lower}_spacing'
            receipt = next(r for r in parent['terminal_receipts'] if (r['seed'], r['day'], r['method'])
                == (seed, day, lower+'_spacing'))
            result = analyze_path(read(v128/'terminal'/(label+'.json')),
                read(v127/'turnover'/f'{seed}_{day}_{method}.json'), receipt)
            paths.append(dict(seed=seed, day=day, lower=lower, **result))
    summary = dict(status='v129_read_only_terminal_recovery_complete', actor_day=paths,
        means_by_lower={lower: {k: fmean(r['means'][k] for r in paths if r['lower'] == lower)
            for k in paths[0]['means']} for lower in LOWER},
        consumed_treated_paths=8, consumed_control_paths=8, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Read-only path accounting. Vehicle visibility, prior-trip durations and allocations '
            'are endogenous realized outcomes, not an executable counterfactual. Immediate release advance '
            'is conditional on the recorded past and chosen vehicle; it does not establish downstream '
            'passenger or CV improvement. Prior-lateness exposure is algebraic, not causal mediation. '
            'Published clocks are the V128 query-time commitments, not immutable original timetables. '
            'Any positive reconstruction residual remains unexplained, not assigned to supply or spacing.')
    out = root/'summary/status.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('V129_TERMINAL_RECOVERY_COMPLETE', dict(paths=len(paths),
        exact_paths=sum(r['reconstruction_exact'] for r in paths)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v128-root', 'v127-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v129_*.py', '-v'], check=True)
    run(args.out_root, args.v128_root, args.v127_root)
