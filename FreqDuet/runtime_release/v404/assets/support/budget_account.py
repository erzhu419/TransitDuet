"""Read eight frozen-policy budget-setting paths and report all tradeoffs."""
import argparse
import importlib.util
import math
from pathlib import Path
import traceback

import basis_worker as worker
from gate_default import read, raw, save

DAYS = (2077001, 2077002, 2077003, 2077004)
SETTINGS = (('r025', .25), ('r075', .75))
METRICS = ('waiting_s_per_passenger', 'onboard_s_per_passenger',
           'journey_s_per_passenger', 'readiness_s_per_trip',
           'public_delay_s_per_day', 'cv_squared', 'J0',
           'holding_s_per_trip', 'holding_person_s_per_passenger')
MATCHED = Path('/home/erzhu419/scheduleurm_work/FreqDuet-v350-live-basis/matched01/aggregate')
BASELINES = Path('/home/erzhu419/scheduleurm_work/FreqDuet-v351-budget-interface/audit01/inputs')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def operational(row):
    keys = ('planned_changes', 'advances', 'delays', 'budget_issues',
            'budget_tightenings', 'budgeted_commands', 'budgeted_command_s',
            'budgeted_residence_s', 'closed_budgets', 'remaining_budget_count')
    return dict(slow={k:row['slow'][k] for k in keys},
                executed_holding_s=row['execution']['executed_s'],
                upper_mode_counts=(row.get('upper') or {}).get('mode_counts'))


def quota_account(trace, rho):
    rows = [b for e in trace['slow_events'] if e['kind'] == 'publish' for b in e['budgets']]
    for row in rows:
        proposed = math.ceil(max(rho*row['normal_command_s'],
                                 row['normal_command_s']-row['pressure_s']))
        expected = proposed if row['old_remaining_s'] is None else min(row['old_remaining_s'], proposed)
        assert expected == row['remaining_s'], (rho, row)
    return dict(passed=True, rho=rho, recorded_issues_and_tightenings=len(rows))


def aggregate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    config = read(args.config)
    assert config['protocol_id'] == 'v352_budget_interface_source01'
    assert config['upper_fourier_k'] == 4 and config['candidate_seed'] == 314011
    assert config['evaluation_days'] == list(DAYS)
    assert args.gate.resolve() == Path(config['default_gate_result']).resolve()
    gate = read(args.gate)
    assert gate['completed'] and gate['engineering_passed'] and gate['default_exact']
    assert Path(gate['default_path']).resolve() == Path(config['default_path']).resolve()
    assert gate['physical_starts'] == gate['training_updates'] == 0
    support = {name:module('_v352_'+name, MATCHED/'support'/(name+'.py'))
               for name in ('v324', 'v326', 'physical', 'credit')}
    prior, original, physical, credit = (support[k] for k in ('v324', 'v326', 'physical', 'credit'))
    v350 = module('_v352_v350', Path(config['source_v350'])/'aggregate_matched.py')
    frozen = read(config['reference_learning_config'])
    assert all(config[key] == frozen[key] for key in v350.RECIPE)
    established = read(MATCHED/'summary/results.json')
    assert established['completed'] and established['implementation_correct']
    by_path, validation, details, doses, differences = {}, {}, {}, {}, {}
    for day in DAYS:
        baseline_dir = BASELINES/f's314011_d{day}'
        baseline, baseline_trace = read(baseline_dir/'summary.json'), raw(baseline_dir/'path.pkl.gz')
        manual = read(MATCHED/'reference'/f'manual_d{day}.json')
        references = {}
        for label, row in (('K4_r050', baseline), ('manual_joint', manual)):
            assert row['day'] == day and row['training_updates'] == 0
            assert established['validation'][f'K4_s314011_d{day}' if label == 'K4_r050'
                                             else f'manual_d{day}']['passed']
            metrics = prior.endpoints(row, 262, config['reward_units'])
            established_metrics = established['by_path'][f'K4_s314011_d{day}' if label == 'K4_r050'
                                                        else f'manual_d{day}']
            assert all(metrics[key] == value for key, value in established_metrics.items())
            references[label] = {k:metrics[k] for k in METRICS}
            by_path[f'{label}_d{day}'] = dict(day=day, endpoints=references[label],
                                             operation=operational(row))
        for label, rho in SETTINGS:
            name = f'{label}_d{day}'
            folder = args.paths/name
            row = read(folder/'summary/results.json')
            trace = raw(folder/'raw/path.pkl.gz')
            saved = raw(folder/'raw/upper_transitions.pkl.gz')
            assert row['budget_retain_fraction'] == rho and saved['identity']['budget_retain_fraction'] == rho
            assert row['upper_fourier_k'] == 4
            assert row['budget_configuration']['retain_fraction'] == rho
            assert row['budget_configuration']['shared_manager'] is True
            v350.summary_gate(row, config, day, 314011, Path(config['reference_checkpoint']))
            assert row['contract'] == baseline['contract'] == manual['contract']
            assert trace['public_clocks'] == baseline_trace['public_clocks']
            accounted = dict(physical=original.physical_account(physical, row, trace),
                credit=credit.credit_check(row, trace, saved, config),
                action_semantics=original.actual_action_account(worker, row, trace, saved),
                quota=quota_account(trace, rho),
                common_randomness=physical.crn_pair(trace['uniforms'], baseline_trace['uniforms']))
            assert all(accounted[k]['passed'] for k in ('physical', 'credit', 'action_semantics', 'quota'))
            assert accounted['common_randomness']['exact']
            details[name] = accounted
            validation[name] = dict(passed=True, raw_physical_credit_action_quota=True,
                                    common_randomness=accounted['common_randomness'])
            values = prior.endpoints(row, len(trace['dispatches']), config['reward_units'])
            metrics = {k:values[k] for k in METRICS}
            budget = accounted['physical']['budget']
            by_path[name] = dict(day=day, rho=rho, endpoints=metrics, operation=operational(row),
                budget_account={k:budget[k] for k in ('initial_quota_s', 'tightening_s', 'committed_s', 'unused_s')})
            doses[name+'_minus_K4_r050'] = prior.actual_dose(trace, baseline_trace)
            for reference, other in references.items():
                differences.setdefault(label+'_minus_'+reference, []).append(
                    {key:metrics[key]-other[key] for key in METRICS})
            del trace, saved
        print('V352_AGGREGATE_DAY', day, flush=True)
    comparisons = {name:{metric:prior.paired([row[metric] for row in rows]) for metric in METRICS}
                   for name, rows in differences.items()}
    save(args.out/'details/accounting.json', details, limit=1024*1024)
    save(args.out/'summary/results.json', dict(protocol_id=config['protocol_id'],
        status='complete_budget_setting_sensitivity', completed=True, implementation_correct=True,
        decision='report_prespecified_budget_tradeoffs_no_selection',
        candidate_seed=314011, candidate_checkpoint=config['reference_checkpoint'],
        settings=[.25, .5, .75], paired_demand_days=4, by_path=by_path,
        comparisons=comparisons, validation=validation, executed_differences=doses,
        default_gate=str(args.gate.resolve()), default_reproduction_passed=True,
        variant_physical_starts=8, default_physical_starts=1, total_experiment_physical_starts=9,
        physical_starts=0, training_updates=0, model_evaluations=0, actor_calls=0, critic_calls=0,
        main_policy_changed=False, confirmation_expanded=False, automatic_retry=False,
        metric_semantics=dict(difference='named setting minus named reference; negative costs improve',
            passengers='same generated passengers; waiting/onboard include full clearance tail',
            readiness='readiness total seconds / 262 trips', holding='actual executed command seconds / 262 trips',
            J0='(waiting_person_s+onboard_person_s)/3600000 + public_launch_delay_total_s/94320',
            CI='256 exhaustive empirical four-day paired resamples, equal day weights'),
        limitations=[
            'Four consumed development days and one frozen learner seed; no independent confirmation or setting selection.',
            'Executed action differences occur on different factual states; they are not same-state causal effects.',
            'Rho controls retained holding-command time, not vehicle counts or driver rosters.',
            'Manual reference uses already validated V350 summary; original K4 raw traces are used for paired execution and random-stream checks.',
            'Reported tradeoffs do not revise historical promotion gates or establish complete LF/HF specialization.']))
    print('V352_AGGREGATE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'paths', 'gate', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        aggregate(args)
    except Exception:
        save(args.out/'summary/failure.json', dict(completed=False, error=traceback.format_exc()[-6000:],
            physical_starts=0, training_updates=0, model_evaluations=0,
            main_policy_changed=False, confirmation_expanded=False, automatic_retry=False))
        raise
