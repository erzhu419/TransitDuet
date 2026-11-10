"""Frozen server-side development screening for three final upper learners."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
           'public_launch_delay_total_s', 'readiness_delay_total_s', 'holding_s_per_trip',
           'waiting_person_s', 'onboard_person_s', 'holding_person_s', 'trips_unlaunched', 'trips_incomplete')
COMPONENTS = ('waiting_person_s', 'onboard_person_s', 'public_delay_s', 'arrival_cv_squared')


def read(path):
    return json.loads(path.read_text())


def credit_check(row, trace, saved, config):
    upper, transitions = saved['upper'], saved['transitions']
    prefix = np.array([upper['prefix'][k] for k in COMPONENTS])
    delta = np.array([[t['components'][k] for k in COMPONENTS] for t in transitions])
    endpoint = np.array([row['results'][k] for k in
        ('waiting_person_s', 'onboard_person_s', 'public_launch_delay_total_s', 'cv_squared')])
    units = config['reward_units']
    weights = np.array([1/units['passenger_scale'], 1/units['passenger_scale'],
        units['public_delay_weight']/units['public_delay_scale'], units['cv_weight']/units['cv_scale']])
    choices = {c['time_s']:c for c in trace['upper_choices']}
    publications = [e for e in trace['slow_events'] if e['kind'] == 'publish']
    checks = dict(identity=saved['protocol'] == config and saved['upper'] == row['upper'] and
        all(saved['identity'][k] == row[k] for k in ('learner_seed', 'day', 'split', 'checkpoint')),
        components=np.allclose(prefix+delta.sum(axis=0), endpoint, rtol=0, atol=1e-7),
        rewards=np.allclose([t['reward'] for t in transitions], -delta@weights, rtol=0, atol=1e-7),
        sum_reward=np.isclose(sum(t['reward'] for t in transitions), upper['expected_reward'], rtol=0, atol=1e-7),
        one_terminal=transitions[-1]['done'] and sum(t['done'] for t in transitions) == 1,
        interval_chain=transitions[0]['start_s'] == 7200 and
            transitions[-1]['end_s'] == row['results']['stream_audit']['horizon_s'] and
            all(a['end_s'] == b['start_s'] for a,b in zip(transitions, transitions[1:])) and
            all(t['end_s'] > t['start_s'] and np.isclose(t['duration_steps']*900,
                t['end_s']-t['start_s'], rtol=0, atol=1e-7) for t in transitions),
        choice_count=len(choices) == len(transitions) == upper['actions'] == upper['transitions'],
        choices_execute=all(e['learned_action'] == choices[e['time_s']]['action'] and
            (not e['plans'] or e['learned_action'] in (1, 3)) for e in publications) and
            all(int(t['action'][0]) == choices[t['start_s']]['action'] for t in transitions))
    return dict(passed=bool(all(checks.values())), checks={k:bool(v) for k,v in checks.items()},
                conservation_max_error=float(np.max(np.abs(prefix+delta.sum(axis=0)-endpoint))))


def paired(values, days):
    draws = np.asarray(list(itertools.product(range(4), repeat=4)))
    data = np.asarray(values, dtype=float)
    return dict(mean=float(data.mean()), by_day=dict(zip(map(str, days), data.tolist())),
                descriptive_ci95=np.quantile(data[draws].mean(axis=1), [.025, .975]).tolist())


def aggregate(args):
    config = read(args.config)
    seeds, days = config['learning_seeds'], config['evaluation_days']
    assert len(seeds) == 3 and len(days) == 4
    units = config['reward_units']
    def endpoints(result):
        values = {k:result[k] for k in METRICS}
        values['journey_s_per_passenger'] = 60*result['journey_min']
        values['system_cost'] = ((result['waiting_person_s']+result['onboard_person_s'])/units['passenger_scale']
            + units['public_delay_weight']*result['public_launch_delay_total_s']/units['public_delay_scale']
            + units['cv_weight']*result['cv_squared']/units['cv_scale'])
        return values
    spec = importlib.util.spec_from_file_location('v314_frozen_accounting', Path(config['base_source'])/'aggregate.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    manual = read(args.reference.parent/'aggregate_full/summary/results.json')
    assert manual['status'] == 'sixteen_development_paths_complete' and manual['neutral_exact']
    checkpoints, rows, accounts, contrasts, authority, crn, contracts = {}, {}, {}, {}, {}, {}, {}
    for seed in seeds:
        checkpoint = args.paths.parent/'checkpoints'/f's{seed}'/f"b{len(config['train_batches'])}"
        receipt = read(checkpoint/'summary/results.json')
        assert receipt['status'] == 'complete_upper_batch' and receipt['learner_seed'] == seed
        assert receipt['protocol_id'] == config['protocol_id']
        assert receipt['completed_batches'] == len(config['train_batches'])
        assert receipt['consumed_days'] == [d for batch in config['train_batches'] for d in batch]
        assert receipt['successful_updates'] == len(config['train_batches'])*config['updates_per_batch']
        assert receipt['target_audit']['outside_clip'] == 0
        checkpoints[str(seed)] = dict(path=str(checkpoint.resolve()), successful_updates=receipt['successful_updates'],
            completed_batches=receipt['completed_batches'], effective_alpha=receipt['effective_alpha'],
            final_batch_target_audit=receipt['target_audit'])
    for day in days:
        reference = {arm:(read(args.reference/f'd{day}_{arm}/summary/results.json'),
            base.raw(args.reference/f'd{day}_{arm}/raw/path.pkl.gz')) for arm in base.ARMS}
        for seed in seeds:
            label = f's{seed}_d{day}'
            folder = args.paths/label
            row, trace = read(folder/'summary/results.json'), base.raw(folder/'raw/path.pkl.gz')
            assert (row['status'], row['learner_seed'], row['day'], row['split']) == ('complete_upper_day', seed, day, 'development')
            assert row['checkpoint'] == checkpoints[str(seed)]['path'] and row['protocol_id'] == config['protocol_id']
            assert row['training_updates'] == 0 and row['native_actors_unchanged'] and row['independent_upper_unchanged']
            normalized = dict(row, status='complete_development_path')  # Do not mutate archived receipts.
            accounts[label] = dict(physical=base.path_account(normalized, trace),
                credit=credit_check(row, trace, base.raw(folder/'raw/upper_transitions.pkl.gz'), config))
            values = endpoints(row['results'])
            rows[label] = dict(day=day, learner_seed=seed, endpoints=values,
                upper=row['upper'], slow=row['slow'], wall_s=row['wall_s'], worker_training_updates=0)
            for arm, (other, old_trace) in reference.items():
                key = f'{label}_vs_{arm}'
                contracts[key] = row['contract'] == other['contract'] and trace['public_clocks'] == old_trace['public_clocks']
                crn[key] = base.crn_pair(trace['uniforms'], old_trace['uniforms'])
                authority[key] = base.plan_authority(trace, old_trace)
                other_values = endpoints(other['results'])
                contrasts.setdefault(f's{seed}_vs_{arm}', []).append({
                    k:values[k]-other_values[k] for k in values})
    metrics = (*METRICS, 'journey_s_per_passenger', 'system_cost')
    effects = {key:{m:paired([v[m] for v in values], days) for m in metrics} for key,values in contrasts.items()}
    facts = {}
    for seed in seeds:
        counts = {arm:[d for d in days if authority[f's{seed}_d{d}_vs_{arm}']['material_actual_changes'] > 0]
                  for arm in ('fixed', 'joint')}
        facts[str(seed)] = dict(material_plan_days=counts,
            material_plan_at_least_two_days={arm:len(found) >= 2 for arm,found in counts.items()},
            physical_and_credit_checks=all(accounts[f's{seed}_d{d}']['physical']['passed'] and
                accounts[f's{seed}_d{d}']['credit']['passed'] for d in days))
    correct = (all(f['physical_and_credit_checks'] for f in facts.values()) and
               all(contracts.values()) and all(r['exact'] for r in crn.values()))
    screen, candidate = config['screening'], config['candidate_seed']
    gates = dict(implementation_correct=correct,
        all_seeds_have_real_terminal_authority=all(
            len(facts[str(s)]['material_plan_days']['fixed']) >= screen['minimum_plan_days'] for s in seeds),
        all_seeds_improve_journey_against_safe_controls=all(
            effects[f's{s}_vs_{arm}']['journey_s_per_passenger']['mean'] < 0
            for s in seeds for arm in screen['safe_comparators']),
        all_seeds_improve_frozen_system_cost=all(
            effects[f's{s}_vs_{arm}']['system_cost']['mean'] < 0
            for s in seeds for arm in screen['cost_comparators']),
        candidate_preserves_baseline_service=all(
            effects[f's{candidate}_vs_{screen["candidate_nonharm_against"]}'][metric]['mean'] <= 0
            for metric in ('cv_squared','readiness_delay_s','public_launch_delay_total_s')))
    passed = all(gates.values())
    return dict(status='twelve_final_development_facts_complete', protocol_id=config['protocol_id'],
        candidate_seed=config.get('candidate_seed'), checkpoints=checkpoints, by_path=rows,
        paired_differences=effects, candidate_facts=facts, accounting=accounts,
        plan_authority=authority, common_randomness=crn, matched_public_and_physical_contracts=contracts,
        implementation_correct=correct, screening_gates=gates, screening_passed=passed,
        total_successful_upper_updates=sum(r['successful_updates'] for r in checkpoints.values()),
        worker_training_updates=0, physical_starts_in_aggregation=0, main_policy_changed=False,
        confirmation_expanded=False,
        decision='freeze_unique_candidate_confirmation_protocol' if passed else 'close_learning_candidate_no_extension',
        limits='Four consumed days; 256 exact empirical bootstrap resamples give descriptive intervals only. This is a frozen development screen, not independent confirmation, policy promotion, or evidence of frequency separation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'paths', 'reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'results.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
