"""Server-side four-day execution contrasts; consumed days are development only."""

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import traceback

import numpy as np

DAYS = [2077001, 2077002, 2077003, 2077004]
NEW_ARMS = ['learned_P', 'learned_D', 'manual_joint_D']
ARMS = ['learned_G', *NEW_ARMS, 'manual_joint_G']
CONTRASTS = [('learned_D', arm) for arm in
             ('learned_P', 'learned_G', 'manual_joint_D', 'manual_joint_G')]
CONTRASTS += [('learned_P', 'learned_G'), ('manual_joint_D', 'manual_joint_G')]
PROTECTIONS = ['onboard_s_per_passenger', 'journey_s_per_passenger',
               'readiness_s_per_trip', 'public_delay_s_per_day']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, bounded=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if bounded:
        assert len(encoded.encode()) <= 48*1024, 'Summary exceeds the local-fetch bound'
    with path.open('x') as stream:
        stream.write(encoded+'\n')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(values):
    data = np.asarray(values, dtype=float)
    assert data.shape == (4,) and np.isfinite(data).all()
    draws = np.asarray(list(itertools.product(range(4), repeat=4)))
    return dict(mean=float(data.mean()), by_day=dict(zip(map(str, DAYS), data.tolist())),
        descriptive_ci95=np.quantile(data[draws].mean(axis=1), [.025, .975]).tolist())


def endpoints(row, trip_count, units):
    result, passengers = row['results'], int(row['contract']['passengers_generated'])
    assert passengers > 0 and trip_count == 262
    waiting, onboard = result['waiting_person_s'], result['onboard_person_s']
    assert waiting+onboard == result['passenger_person_s']
    exposure = result['stream_audit']['exposure']
    assert exposure['waiting_person_s'] == waiting and exposure['onboard_person_s'] == onboard
    j0 = ((waiting+onboard)/units['passenger_scale']
          + units['public_delay_weight']*result['public_launch_delay_total_s']/units['public_delay_scale'])
    return dict(waiting_s_per_passenger=waiting/passengers,
        onboard_s_per_passenger=onboard/passengers,
        journey_s_per_passenger=(waiting+onboard)/passengers,
        readiness_s_per_trip=result['readiness_delay_total_s']/trip_count,
        public_delay_s_per_day=result['public_launch_delay_total_s'],
        cv_squared=result['cv_squared'], dispatch_cv_squared=result['dispatch_cv_squared'],
        holding_s_per_trip=row['execution']['executed_s']/trip_count,
        holding_person_s_per_passenger=result['holding_person_s']/passengers,
        J0=j0, J1=j0+result['cv_squared']/units['cv_scale'])


def benefit_gate(metrics):
    waiting = metrics['waiting_s_per_passenger']
    checks = dict(waiting_each_day_nonharm=all(v <= 0 for v in waiting['by_day'].values()),
        waiting_mean_improves=waiting['mean'] < 0,
        waiting_ci_upper_improves=waiting['descriptive_ci95'][1] < 0)
    checks.update({key+'_ci_upper_nonharm':metrics[key]['descriptive_ci95'][1] <= 0
                   for key in PROTECTIONS})
    return dict(passed=all(checks.values()), checks=checks,
        waiting_improved_days=sum(v < 0 for v in waiting['by_day'].values()),
        waiting_unchanged_days=sum(v == 0 for v in waiting['by_day'].values()))


def actual_dose(left, right):
    """Observed path differences, without imposing a 5 s threshold on D minus P."""
    a = {r['trip']:r for r in left['dispatches']}
    b = {r['trip']:r for r in right['dispatches']}
    assert a.keys() == b.keys() and len(a) == 262
    scheduled = [trip for trip in a if a[trip]['scheduled_s'] != b[trip]['scheduled_s']]
    actual = [a[trip]['actual_s']-b[trip]['actual_s'] for trip in a]
    binding = [trip for trip in scheduled if
        a[trip]['actual_s'] == math.ceil(a[trip]['scheduled_s']) and
        a[trip]['ready_return_s']+1 <= a[trip]['actual_s']]
    queries = lambda trace:{(q['bus'], q['trip'], q['station']):q for q in trace['queries']}
    qa, qb = queries(left), queries(right)
    assert len(qa) == len(left['queries']) and len(qb) == len(right['queries'])
    common = qa.keys() & qb.keys()
    return dict(scheduled_release_changed_trips=len(scheduled),
        actual_dispatch_changed_trips=sum(v != 0 for v in actual),
        candidate_release_binding_changed_trips=len(binding),
        actual_dispatch_delta_sum_s=sum(actual), actual_dispatch_abs_delta_sum_s=sum(map(abs, actual)),
        common_holding_events=len(common), candidate_only_events=len(qa.keys()-qb.keys()),
        comparator_only_events=len(qb.keys()-qa.keys()),
        holding_action_changed_events=sum(qa[k]['executed_request_s'] != qb[k]['executed_request_s'] for k in common),
        holding_clock_changed_events=sum(qa[k]['time_s'] != qb[k]['time_s'] for k in common),
        interpretation='Factual trajectories differ in state; these are executed doses, not same-state causal holding effects.')


def path_for(config, paths, arm, day):
    if arm == 'learned_G':
        return Path(config['reference_G_root'])/f'd{day}'
    if arm == 'manual_joint_G':
        return Path(config['reference_manual_G_root'])/f'd{day}_joint'
    return paths/f'{arm}_d{day}'


def aggregate(config, paths, out):
    assert config['days'] == DAYS and config['arms'] == NEW_ARMS and config['seed'] == 314011
    gate_config = config['gates']
    assert gate_config == dict(waiting_daily_nonharm=True, waiting_mean_strict_improvement=True,
        waiting_ci_upper_strict_improvement=True, protection_metrics=PROTECTIONS,
        protection_ci_upper_max=0, cv_is_diagnostic=True)
    protocol = read(config['frozen_config'])
    assert protocol['reward_units']['cv_weight'] == 0 and protocol['candidate_seed'] == config['seed']
    assert protocol['evaluation_days'] == DAYS and protocol['fast_actor'] == 12011
    assert protocol['fast_sampler'] == 27011001
    base = module_at('_v324_v314_accounting', Path(config['base_source'])/'aggregate.py')
    learning = module_at('_v324_v316_credit', Path(config['frozen_source'])/'aggregate_learning.py')
    checkpoint = read(Path(config['checkpoint'])/'summary/results.json')
    assert checkpoint['status'] == 'complete_upper_batch' and checkpoint['learner_seed'] == config['seed']
    assert checkpoint['completed_batches'] == 8 and checkpoint['successful_updates'] == 1536
    assert checkpoint['protocol_id'] == protocol['protocol_id']
    by_path, accounting, doses, alignments, contracts = {}, {}, {}, {}, {}
    detailed_accounts, raw_differences = {}, {f'{a}_minus_{b}':[] for a,b in CONTRASTS}
    all_valid = True
    for day in DAYS:
        rows, traces = {}, {}
        for arm in ARMS:
            directory = path_for(config, paths, arm, day)
            row, trace = read(directory/'summary/results.json'), base.raw(directory/'raw/path.pkl.gz')
            assert row['day'] == day
            if arm == 'learned_G':
                assert row['status'] == 'complete_live_directional_shadow_day' and row['binding']['passed']
            elif arm == 'manual_joint_G':
                assert row['status'] == 'complete_development_path' and row['arm'] == 'joint'
            else:
                assert row['status'] == 'complete_directional_execution_day' and row['arm'] == arm
            if arm.startswith('learned_'):
                assert row['learner_seed'] == config['seed'] and row['checkpoint'] == config['checkpoint']
                assert row['upper_deterministic'] is True and row['behavior_epsilon'] == 0
            result = row['results']
            assert result['scenario_seed'] == day and result['train_seed'] == protocol['fast_actor']
            assert result['sampler_seed'] == protocol['fast_sampler'] and result['mode'] == 'stochastic'
            checks = dict(no_unserved_trips=result['trips_unlaunched'] == result['trips_incomplete'] == 0,
                all_physical_returns=len(trace['returns']) == 262,
                no_remaining_budgets=row['slow']['remaining_budget_count'] == 0,
                no_dual_updates=result['dual_updates'] == 0,
                base_actors_unchanged=result['base_actors_unchanged'] is True)
            physical = base.path_account(dict(row, status='complete_development_path'), trace)
            credit = None
            if arm.startswith('learned_'):
                credit = learning.credit_check(row, trace,
                    base.raw(directory/'raw/upper_transitions.pkl.gz'), protocol)
            valid = physical['passed'] and all(checks.values()) and (credit is None or credit['passed'])
            key = f'{arm}_d{day}'
            detailed_accounts[key] = dict(physical=physical, credit=credit, additional=checks)
            accounting[key] = dict(passed=bool(valid), physical=physical['passed'],
                credit=None if credit is None else credit['passed'], **checks)
            all_valid = all_valid and valid
            values = endpoints(row, len(trace['dispatches']), protocol['reward_units'])
            by_path[key] = dict(day=day, arm=arm, passengers=row['contract']['passengers_generated'],
                trips=len(trace['dispatches']), endpoints=values,
                physical_starts=row['physical_starts'], planned_changes=row['slow']['planned_changes'],
                budget_issues=row['slow']['budget_issues'],
                upper_mode_counts=None if row.get('upper') is None else row['upper']['mode_counts'])
            rows[arm], traces[arm] = row, trace
        for arm in ARMS:
            key = f'{arm}_d{day}'
            contracts[key] = (rows[arm]['contract'] == rows['learned_G']['contract'] and
                              traces[arm]['public_clocks'] == traces['learned_G']['public_clocks'])
            all_valid = all_valid and contracts[key]
        for candidate, comparator in CONTRASTS:
            name = f'{candidate}_minus_{comparator}'
            left, right = by_path[f'{candidate}_d{day}']['endpoints'], by_path[f'{comparator}_d{day}']['endpoints']
            raw_differences[name].append({k:left[k]-right[k] for k in left})
            key = f'{name}_d{day}'
            alignments[key] = base.crn_pair(traces[candidate]['uniforms'], traces[comparator]['uniforms'])
            doses[key] = actual_dose(traces[candidate], traces[comparator])
            all_valid = all_valid and alignments[key]['exact']
    comparisons = {}
    for name, rows in raw_differences.items():
        metrics = {key:paired([row[key] for row in rows]) for key in rows[0]}
        comparisons[name] = dict(metrics=metrics, outcome_gate=benefit_gate(metrics))
    main_names = [f'{a}_minus_{b}' for a,b in CONTRASTS[:4]]
    passed = bool(all_valid and all(comparisons[name]['outcome_gate']['passed'] for name in main_names))
    save(out/'details/accounting.json', detailed_accounts)
    save(out/'details/executed_dose.json', doses)
    dose_fields = ('scheduled_release_changed_trips', 'actual_dispatch_changed_trips',
                   'candidate_release_binding_changed_trips', 'holding_action_changed_events')
    dose_summary = {name:{str(day):{key:doses[f'{name}_d{day}'][key] for key in dose_fields}
                         for day in DAYS} for name in raw_differences}
    result = dict(status='complete_directional_execution_aggregate',
        decision='development_candidate_passed_no_automatic_confirmation' if passed else
                 'close_directional_execution_candidate_no_confirmation',
        implementation_correct=bool(all_valid), development_candidate_passed=passed,
        required_comparisons=main_names, comparisons=comparisons, by_path=by_path,
        accounting=accounting, matched_contracts=contracts, common_randomness=alignments,
        actual_dose=dose_summary, checkpoint=config['checkpoint'], learner_seed=config['seed'],
        new_physical_starts=sum(by_path[f'{arm}_d{day}']['physical_starts'] for arm in NEW_ARMS for day in DAYS),
        reused_reference_paths=8, aggregation_physical_starts=0,
        training_updates=0, main_policy_changed=False, confirmation_expanded=False,
        metric_semantics=dict(passenger_denominator='same-day contract.passengers_generated',
            trip_denominator=262, passenger_times='complete-stream person-seconds / generated passengers',
            readiness='total readiness delay seconds / 262 completed trips',
            public_delay='total seconds per full day against immutable public timetable',
            difference='candidate minus comparator; negative improves',
            interval='256 exact empirical resamples of four paired days; equal day weights',
            J0='(W+R)/3600000 + public_delay/94320', J1='J0 + CV_squared/0.05'),
        gates=dict(waiting='each day <= 0; mean < 0; descriptive CI upper < 0',
            protections={key:'descriptive CI upper <= 0' for key in PROTECTIONS},
            cv='reported only; no new hard gate', original_V316_J1_gate_unchanged=True),
        limitations=[
            'Four reused development days and one frozen preselected actor; not independent confirmation.',
            'A frozen network can choose different modes on changed trajectories; comparisons include that closed-loop response.',
            'D minus P isolates directional LF input within this implementation; P minus G includes aggregation/filter differences.',
            'Every arm retains LF and optimized holding, so this is not a no-frequency or timetable-by-holding ablation.',
            'Passing the new development screen does not revise historical V316 gate outcomes or promote the main policy.'])
    save(out/'summary/results.json', result, bounded=True)
    print('V324_AGGREGATE', result['decision'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'paths', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        aggregate(read(args.config), args.paths, args.out)
    except Exception:
        save(args.out/'summary/failure.json', dict(status='failed_hold', error=traceback.format_exc()[-7000:],
            physical_starts=0, training_updates=0, automatic_retry=False,
            main_policy_changed=False, confirmation_expanded=False), bounded=True)
        raise
