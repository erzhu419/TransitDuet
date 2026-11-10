"""Server-only V314 trace accounting and descriptive four-day factorial readout."""

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path
import pickle
from statistics import mean

DAYS = (2077001, 2077002, 2077003, 2077004)
ARMS = ('fixed', 'plan', 'budget', 'joint')
STREAMS = ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms')
METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
    'public_launch_delay_total_s', 'readiness_delay_total_s', 'holding_s_per_trip',
    'waiting_person_s', 'onboard_person_s', 'holding_person_s',
    'trips_unlaunched', 'trips_incomplete')
CONTRASTS = (('plan', 'fixed'), ('budget', 'fixed'), ('joint', 'fixed'),
             ('joint', 'plan'), ('joint', 'budget'))


def read(path):
    return json.loads(path.read_text())


def raw(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def short(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return repr(value)[:180]


def first_diff(expected, actual, path=''):
    """Require every historical field; new per-record metadata is permitted."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return dict(path=path, reason='type', expected='dict', actual=short(actual))
        for key, value in expected.items():
            where = f'{path}[{key!r}]'
            if key not in actual:
                return dict(path=where, reason='missing_historical_field')
            difference = first_diff(value, actual[key], where)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            return dict(path=path, reason='sequence_type_or_length',
                expected_type=type(expected).__name__, expected_length=len(expected),
                actual_type=type(actual).__name__,
                actual_length=len(actual) if isinstance(actual, (list, tuple)) else None)
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_diff(left, right, f'{path}[{index}]')
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return dict(path=path, reason='value', expected=short(expected), actual=short(actual))
    return None


def neutral_check(paths, results):
    matrix = results/'protocol_v6_v277_metric_alignment/matrix_repaired'
    checks = []
    for offset, day in enumerate(DAYS):
        index = 16+offset
        old = read(matrix/'pairs'/f'p{index}.json')
        assert (old['index'], old['day'], old['arm']) == (index, day, 'arrival_cv_current')
        current = read(paths/f'd{day}_fixed/summary/results.json')
        archived = raw(matrix/'traces'/f'p{index}.pkl.gz')
        trace = raw(paths/f'd{day}_fixed/raw/path.pkl.gz')
        fields = {}
        for field in STREAMS:
            difference = None
            if field == 'uniforms' and archived[field].keys() != trace[field].keys():
                difference = dict(path=field, reason='logical_event_key_set',
                    old_events=len(archived[field]), new_events=len(trace[field]))
            difference = difference or first_diff(archived[field], trace[field], field)
            fields[field] = dict(exact=difference is None, first_divergence=difference)
        endpoint_difference = first_diff(old['results'], current['results'], 'results')
        contract_difference = first_diff(old['contract'], current['contract'], 'contract')
        checks.append(dict(day=day, old_pair=f'p{index}.json', old_trace=f'p{index}.pkl.gz',
            fields=fields, endpoints_exact=endpoint_difference is None,
            endpoint_first_divergence=endpoint_difference,
            contract_exact=contract_difference is None,
            contract_first_divergence=contract_difference))
    exact = all(r['endpoints_exact'] and r['contract_exact'] and
                all(f['exact'] for f in r['fields'].values()) for r in checks)
    return dict(neutral_exact=exact, historical_matrix=str(matrix), by_day=checks)


def budget_account(trace, summary):
    """Replay irrevocable submissions, including those before same-tick publish."""
    pending = {p['time_s']:{(c['bus'], c['trip'], c['station'], c['action_s'])
                          for c in p['already_submitted']} for p in trace['publish_ticks']}
    timeline = []
    for index, query in enumerate(trace['queries']):
        key = query['bus'], query['trip'], query['station'], query['executed_request_s']
        priority = 0 if key in pending.get(query['time_s'], set()) else 2
        timeline.append((query['time_s'], priority, index, 'command', query))
    for index, event in enumerate(trace['slow_events']):
        timeline.append((event['time_s'], 1 if event['kind'] == 'publish' else 3,
                         index, event['kind'], event))
    remaining, closed, errors = {}, set(), []
    issued = tightened = spent = unused = commands = residence = 0
    issues = tightenings = 0
    returns = {(r['bus_id'], r['trip_id']):r['physical_time_s'] for r in trace['returns']}
    for time_s, _, _, kind, row in sorted(timeline):
        if kind == 'publish':
            for value in row['budgets']:
                key = value['bus'], value['trip']
                old, new = remaining.get(key), value['remaining_s']
                if key in closed or old != value['old_remaining_s'] or new < 0:
                    errors.append(dict(kind='issue_state', bus=key[0], trip=key[1], time_s=time_s))
                if old is None:
                    issues += 1
                    issued += new
                else:
                    tightenings += 1
                    tightened += old-new
                    if new > old:
                        errors.append(dict(kind='quota_increased', bus=key[0], trip=key[1]))
                remaining[key] = new
        elif kind == 'command':
            key = row['bus'], row['trip']
            if key in remaining:
                action = row['executed_request_s']
                if action not in row['legal'] or action < 0 or action > remaining[key]:
                    errors.append(dict(kind='command_over_budget', bus=key[0], trip=key[1], time_s=time_s))
                remaining[key] -= action
                spent += action
                residence += max(1, action)
                commands += 1
        else:
            assert kind == 'budget_return'
            key = row['bus'], row['trip']
            if remaining.get(key) != row['unused_s'] or returns.get(key) != time_s:
                errors.append(dict(kind='return_state', bus=key[0], trip=key[1], time_s=time_s))
            unused += row['unused_s']
            remaining.pop(key, None)
            closed.add(key)
    expected = dict(budget_issues=issues, budget_tightenings=tightenings,
        budgeted_commands=commands, budgeted_command_s=spent,
        budgeted_residence_s=residence, closed_budgets=len(closed),
        remaining_budget_count=len(remaining))
    difference = first_diff(expected, summary['slow'], 'slow')
    if difference:
        errors.append(difference)
    if remaining or issued != tightened+spent+unused:
        errors.append(dict(kind='budget_conservation', remaining_cycles=len(remaining)))
    return dict(passed=not errors, first_error=errors[0] if errors else None,
        error_count=len(errors), initial_quota_s=issued, tightening_s=tightened,
        committed_s=spent, zero_tick_inclusive_residence_s=residence, unused_s=unused,
        **expected)


def probe_account(trace, summary):
    seen, entries, errors = set(), [], []
    for query, profile in zip(trace['queries'], trace['graph_queries']):
        if 'quota_probe' not in profile:
            continue
        probe = profile['quota_probe']
        cycle = query['bus'], query['trip']
        aligned = all(query[k] == profile[k] for k in ('time_s', 'bus', 'trip', 'station'))
        if not aligned or cycle in seen or probe['budget_action_s'] != query['executed_request_s']:
            errors.append(dict(kind='probe_execution_alignment', bus=cycle[0], trip=cycle[1]))
        seen.add(cycle)
        if probe['supported']:
            if probe['changed'] != (probe['without_own_budget_s'] != probe['budget_action_s']):
                errors.append(dict(kind='probe_changed_label', bus=cycle[0], trip=cycle[1]))
        elif probe['changed'] is not None or probe['without_own_budget_s'] is not None:
            errors.append(dict(kind='unsupported_probe_as_label', bus=cycle[0], trip=cycle[1]))
        entries.append(dict(bus=cycle[0], trip=cycle[1], station=query['station'],
                            time_s=query['time_s'], **probe))
    counts = dict(attempted=len(entries), supported=sum(e['supported'] is True for e in entries),
        changed=sum(e['changed'] is True for e in entries),
        extra_evaluations=sum(e['evals'] for e in entries))
    difference = first_diff(counts, summary['quota_probes'], 'quota_probes')
    if difference:
        errors.append(difference)
    changed = [e for e in entries if e['changed'] is True]
    return dict(**counts, passed=not errors, first_error=errors[0] if errors else None,
        supported_action_delta_s=sum(e['budget_action_s']-e['without_own_budget_s']
                                     for e in entries if e['supported']),
        changed_examples=changed[:2], unsupported_examples=[e for e in entries if not e['supported']][:1])


def path_account(summary, trace):
    queries, profiles = trace['queries'], trace['graph_queries']
    chain = (len(queries) == len(profiles) == len(trace['holds']) and
        all(all(q[k] == p[k] for k in ('time_s', 'bus', 'trip', 'station')) and
            q['executed_request_s'] == p['shadow_action_s'] and
            q['executed_request_s'] in q['legal'] for q,p in zip(queries,profiles)))
    clocks = trace['public_clocks']
    public_exact = (len(clocks) == len(trace['dispatches']) == 262 and
                    all(r['public_s'] == clocks[r['trip']] for r in trace['dispatches']))
    execution = summary['execution']
    budget, probes = budget_account(trace, summary), probe_account(trace, summary)
    contract = summary['contract']
    checks = dict(complete=summary['status'] == 'complete_development_path',
        service_cleared=execution['done_reason'] == 'service_cleared',
        physical_fleet_12=execution['physical_vehicles'] == contract['physical_vehicle_count'] == 12,
        fixed_pool=contract['fleet_inventory_mode'] == 'fixed_pool',
        commands_closed=execution['commanded_s'] == execution['executed_s'] and execution['unexecuted_s'] == 0,
        action_receipts_aligned=chain, public_clock_unchanged=public_exact,
        budget_conserved=budget['passed'], probe_receipts_aligned=probes['passed'],
        no_learning=summary['training_updates'] == 0,
        one_start=summary['physical_starts'] == 1,
        no_policy_change=not summary['main_policy_changed'],
        no_confirmation_expansion=not summary['confirmation_expanded'])
    return dict(passed=all(checks.values()), checks=checks, budget=budget, probes=probes)


def plan_authority(trace, base):
    base_by_trip = {r['trip']:r for r in base['dispatches']}
    publications = {p['trip']:p for event in trace['slow_events'] if event['kind'] == 'publish'
                    for p in event['plans']}
    rows = []
    for row in trace['dispatches']:
        if row['scheduled_s'] == row['public_s']:
            continue
        other = base_by_trip[row['trip']]
        ready = row['ready_return_s']+1
        effective = math.ceil(row['scheduled_s'])
        binding = row['actual_s'] == effective and ready <= effective
        own_clock_effect = row['actual_s']-max(row['public_s'], ready)
        delta = row['actual_s']-other['actual_s']
        published = row['trip'] in publications and publications[row['trip']]['release_s'] == row['scheduled_s']
        material = published and binding and abs(own_clock_effect) >= 5 and abs(delta) >= 5
        rows.append(dict(trip=row['trip'], bus=row['bus'], same_bus=row['bus'] == other['bus'],
            scheduled_s=row['scheduled_s'], public_s=row['public_s'], actual_s=row['actual_s'],
            ready_s=ready if math.isfinite(ready) else None, release_binding=binding,
            release_matches_publication=published, own_clock_effect_s=own_clock_effect,
            paired_actual_delta_s=delta, material=material))
    material = [r for r in rows if r['material']]
    return dict(changed_release_trips=len(rows), binding_trips=sum(r['release_binding'] for r in rows),
        material_actual_changes=len(material), material_same_bus=sum(r['same_bus'] for r in material),
        paired_absolute_dispatch_change_s=sum(abs(r['paired_actual_delta_s']) for r in rows),
        material_examples=material[:3])


def crn_pair(left, right):
    common = left.keys() & right.keys()
    mismatch = next((key for key in sorted(common) if left[key]['uniform'] != right[key]['uniform']), None)
    return dict(common_events=len(common), left_only=len(left.keys()-right.keys()),
        right_only=len(right.keys()-left.keys()), exact=mismatch is None and bool(common),
        first_mismatch=None if mismatch is None else list(mismatch))


def aggregate(args):
    arms = ('fixed',) if args.mode == 'neutral' else ARMS
    missing = []
    for day, arm in itertools.product(DAYS, arms):
        directory = args.paths/f'd{day}_{arm}'
        for relative in ('summary/results.json', 'raw/path.pkl.gz'):
            if not (directory/relative).is_file():
                missing.append(f'd{day}_{arm}/{relative}')
    if missing:
        return dict(status='incomplete_paths', missing=missing, neutral_exact=False,
                    training_admissible=False, decision='stop_no_retry_no_expansion')
    neutral = neutral_check(args.paths, args.results)
    if args.mode == 'neutral':
        return dict(status='neutral_exact' if neutral['neutral_exact'] else 'neutral_divergence',
            **neutral, training_updates=0, physical_starts_in_aggregation=0,
            decision='allow_twelve_intervention_paths' if neutral['neutral_exact'] else 'stop_before_interventions')
    cube, accounts, authority, alignments, contracts = {}, {}, {}, {}, {}
    for day in DAYS:
        traces = {}
        for arm in ARMS:
            label = f'd{day}_{arm}'
            row = read(args.paths/label/'summary/results.json')
            assert (row['day'], row['arm'], row['actor'], row['sampler']) == (day, arm, 12011, 27011001)
            cube[day,arm] = row
            traces[arm] = raw(args.paths/label/'raw/path.pkl.gz')
            accounts[label] = path_account(row, traces[arm])
        contracts[str(day)] = all(cube[day,arm]['contract'] == cube[day,'fixed']['contract'] for arm in ARMS)
        for arm, base in CONTRASTS:
            authority[f'd{day}_{arm}_vs_{base}'] = plan_authority(traces[arm], traces[base])
        for left, right in itertools.combinations(ARMS, 2):
            alignments[f'd{day}_{left}_vs_{right}'] = crn_pair(traces[left]['uniforms'], traces[right]['uniforms'])
        clocks = traces['fixed']['public_clocks']
        contracts[str(day)] &= all(t['public_clocks'] == clocks for t in traces.values())
    metrics = (*METRICS, 'journey_s_per_passenger')
    endpoints = {}
    for (day, arm), row in cube.items():
        values = {m:row['results'][m] for m in METRICS}
        values['journey_s_per_passenger'] = 60*values['journey_min']
        endpoints[day,arm] = values
    effects = {}
    for arm, base in CONTRASTS:
        effects[arm+'_vs_'+base] = {}
        for metric in metrics:
            by_day = {str(d):endpoints[d,arm][metric]-endpoints[d,base][metric] for d in DAYS}
            effects[arm+'_vs_'+base][metric] = dict(mean=mean(by_day.values()), by_day=by_day,
                minimum=min(by_day.values()), maximum=max(by_day.values()))
    plan_days = [d for d in DAYS if authority[f'd{d}_joint_vs_fixed']['material_actual_changes'] > 0]
    budget_days = [d for d in DAYS if accounts[f'd{d}_joint']['probes']['changed'] > 0]
    preflight_path = args.paths.parent/'preflight01/summary/results.json'
    gates = dict(preflight_passed=preflight_path.is_file() and read(preflight_path)['status'] == 'passed',
        implementation_correct=all(a['passed'] for a in accounts.values()),
        matched_physical_contracts=all(contracts.values()), neutral_exact=neutral['neutral_exact'],
        common_event_randomness_exact=all(v['exact'] for v in alignments.values()),
        joint_material_plan_at_least_two_days=len(plan_days) >= 2,
        joint_genuine_budget_at_least_two_days=len(budget_days) >= 2)
    admissible = all(gates.values())
    by_path = []
    for day, arm in itertools.product(DAYS, ARMS):
        row, label = cube[day,arm], f'd{day}_{arm}'
        by_path.append(dict(day=day, arm=arm, endpoints=endpoints[day,arm],
            wall_s=row['wall_s'], callback_latency_s=row['callback_latency_s'],
            slow=row['slow'], accounting=accounts[label]))
    return dict(status='sixteen_development_paths_complete', neutral_exact=neutral['neutral_exact'],
        neutral=neutral, by_path=by_path,
        arm_means={arm:{m:mean(endpoints[d,arm][m] for d in DAYS) for m in metrics} for arm in ARMS},
        paired_differences=effects, plan_authority=authority, event_uniform_alignment=alignments,
        matched_contracts_by_day=contracts, gates=gates, joint_plan_days=plan_days,
        joint_budget_days=budget_days, training_admissible=admissible,
        decision='freeze_separate_upper_training_protocol' if admissible else 'close_candidate_no_retuning',
        physical_paths=16, physical_starts_in_aggregation=0, training_updates=0,
        main_policy_changed=False, confirmation_expanded=False,
        limits='Four consumed development days, descriptive paired differences; no independent confirmation or learned-upper benefit. Plan binding is an actual-dispatch authority screen, not a passenger-benefit claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('neutral', 'full'), required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = aggregate(args)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'results.json').open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    print('V314_AGGREGATE_COMPLETE', value['status'], flush=True)
