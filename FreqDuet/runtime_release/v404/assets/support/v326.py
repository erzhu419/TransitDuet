"""Frozen two-action evaluation and online-Q readout; no policy replacement."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np
import torch

DAYS = [2077001, 2077002, 2077003, 2077004]
SEEDS = [314011, 314037, 314049]
CANDIDATE = 314011
REFERENCES = ('V325', 'manual_joint_G', 'learned_G')
MANDATORY = REFERENCES[:2]
PROTECTIONS = ('onboard_s_per_passenger', 'journey_s_per_passenger',
               'readiness_s_per_trip', 'public_delay_s_per_day', 'J0')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, bounded=True):
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


def outcome_gate(metrics):
    waiting = metrics['waiting_s_per_passenger']
    checks = dict(waiting_each_day_nonharm=all(v <= 0 for v in waiting['by_day'].values()),
        waiting_mean_improves=waiting['mean'] < 0,
        waiting_ci_upper_improves=waiting['descriptive_ci95'][1] < 0)
    checks.update({key:metrics[key]['descriptive_ci95'][1] <= 0 for key in PROTECTIONS})
    return dict(passed=all(checks.values()), checks=checks)


def reference_folder(config, reference, day):
    if reference == 'V325':
        return Path(config['reference_v325_root'])/f'learned_G_budget_on_d{day}'
    if reference == 'manual_joint_G':
        return Path(config['reference_manual_G_root'])/f'd{day}_joint'
    return Path(config['reference_G_root'])/f'd{day}'


def physical_account(base, row, trace):
    checked = base.path_account(dict(row, status='complete_development_path'), trace)
    checked['additional'] = dict(no_unserved_trips=row['results']['trips_unlaunched'] == row['results']['trips_incomplete'] == 0,
        all_returns=len(trace['returns']) == 262, no_remaining_budgets=row['slow']['remaining_budget_count'] == 0,
        no_dual_updates=row['results']['dual_updates'] == 0,
        base_actors_unchanged=row['results']['base_actors_unchanged'] is True)
    checked['passed'] = checked['passed'] and all(checked['additional'].values())
    return checked


def actual_action_account(worker, row, trace, saved):
    choices, transitions = trace['upper_choices'], saved['transitions']
    events = [e for e in trace['slow_events'] if e['kind'] == 'publish']
    assert saved['action_semantics'] == row['action_semantics'] and row['action_semantics']['passed']
    assert row['action_semantics']['action_values'] == row['upper']['action_values'] == [2, 3]
    assert len(choices) == len(transitions) == len(events) > 0
    worker.transition_actions(transitions)
    for choice, event, transition in zip(choices, events, transitions):
        action = choice['action']
        assert action in (2, 3) and event['learned_action'] == int(transition['action'][0]) == action
        assert choice['mode'] == ('budget' if action == 2 else 'joint')
        assert choice['time_s'] == event['time_s'] == transition['start_s']
        assert 'permission' not in event and 'plan_frequency' not in event
        assert not event['plans'] or action == 3
        pi, mu = np.asarray(choice['pi']), np.asarray(choice['mu'])
        assert pi.shape == mu.shape == (2,) and np.isfinite(pi).all() and np.isfinite(mu).all()
        assert (pi >= 0).all() and choice['behavior_epsilon'] == 0
        np.testing.assert_allclose(pi.sum(), 1., rtol=0, atol=1e-6)
        expected = np.zeros(2)
        expected[action-2] = 1.
        np.testing.assert_array_equal(mu, expected)
        assert action == 2+int(pi.argmax())
    assert row['slow']['arm'] == choices[-1]['mode']
    return dict(passed=True, records=len(choices), canonical_actions=[2, 3],
        actor_transition_execution_equal=True, previous_action_actual=True, permission_override=False)


def load_online_networks(worker, checkpoint, config, seed):
    trainer, saved = worker.load_training_checkpoint(checkpoint, config, seed)
    policy = torch.load(checkpoint/'policy.pt', map_location='cpu', weights_only=False)
    expected_days = [day for batch in config['train_batches'] for day in batch]
    for envelope in (saved, policy):
        assert envelope['protocol'] == config and envelope['learner_seed'] == seed
        assert envelope['completed_batches'] == 8 and envelope['successful_updates'] == 1536
        assert envelope['consumed_days'] == expected_days
    parameters = saved['trainer_state']['policy']
    assert parameters.keys() == policy['policy'].keys()
    assert all(torch.equal(parameters[key], policy['policy'][key]) for key in parameters)
    assert trainer.critic_aggregation == 'twin_min' and trainer.successful_updates == 1536
    assert all(torch.equal(trainer.q_net.state_dict()[key], value)
               for key, value in saved['trainer_state']['q_net'].items())
    worker.action_library(trainer)
    trainer.policy_net.eval()
    trainer.q_net.eval()
    return trainer


def q_diagnostic(trainer, row, trace, saved):
    choices, transitions = trace['upper_choices'], saved['transitions']
    state = torch.as_tensor(np.stack([t['state'] for t in transitions]), dtype=torch.float32, device=trainer.device)
    updates, replay_size = trainer.successful_updates, len(trainer.replay_buffer)
    with torch.no_grad():
        # Preserve the worker's one-event actor forward shape when binding pi.
        pi = np.stack([trainer.policy_net.dist_info(s[None])[0][0].cpu().numpy() for s in state])
        online_heads = trainer._discrete_q_values(trainer.q_net, state)
        assert online_heads.shape == (2, len(choices), 2)
        q = online_heads.min(dim=0).values.cpu().numpy()
        twin_disagreement = (online_heads[0].argmax(dim=1) != online_heads[1].argmax(dim=1)).sum().item()
    np.testing.assert_allclose(pi, np.asarray([c['pi'] for c in choices]), rtol=0, atol=1e-7)
    selected = np.asarray([c['action']-2 for c in choices], dtype=int)
    np.testing.assert_array_equal(selected, pi.argmax(axis=1))
    assert q.shape == pi.shape == (len(choices), 2) and np.isfinite(q).all()
    gap = q.max(axis=1)-q[np.arange(len(q)), selected]
    disagreement = selected != q.argmax(axis=1)
    assert trainer.successful_updates == updates == 1536 and len(trainer.replay_buffer) == replay_size
    first = np.flatnonzero(disagreement)
    return dict(decisions=len(choices), argmax_disagreements=int(disagreement.sum()),
        disagreements_with_q_gap_over_1e_6=int((disagreement & (gap > 1e-6)).sum()),
        selected_q_gap_mean=float(gap.mean()), selected_q_gap_max=float(gap.max()),
        twin_critic_argmax_disagreements=int(twin_disagreement),
        joint_preferred_by_q_but_budget_executed=int(((selected == 0) & (q[:,1] > q[:,0]+1e-6)).sum()),
        mode_counts=row['upper']['mode_counts'], actual_plan_changes=row['slow']['planned_changes'],
        first_disagreement=None if not len(first) else dict(time_s=choices[int(first[0])]['time_s'],
            actual_action=int(selected[first[0]]+2), critic_action=int(q[first[0]].argmax()+2),
            pi=pi[first[0]].tolist(), online_twin_min_q=q[first[0]].tolist()),
        checks=dict(final_policy_reproduced=True, final_online_critic_loaded=True,
                    canonical_action_execution_bound=True, diagnostic_updates=0))


def aggregate(config_path, paths, out):
    worker = module_at('_v326_current_worker', Path(__file__).parent/'worker.py')
    config = worker.read_config(config_path)
    worker.require_preflight(config)
    assert config['learning_seeds'] == SEEDS and config['evaluation_days'] == DAYS
    assert config['candidate_seed'] == CANDIDATE and config['action_values'] == [2, 3]
    assert len(config['train_batches']) == 8 and config['updates_per_batch'] == 192
    assert config['reward_units']['cv_weight'] == 0 and config['trainer_settings']['critic_aggregation'] == 'twin_min'
    prior = module_at('_v326_v324_helpers', Path(config['source_v324'])/'aggregate.py')
    base = module_at('_v326_v314_accounting', Path(config['base_source'])/'aggregate.py')
    old_learning = module_at('_v326_credit_accounting', Path(config['frozen_config']).parent/'aggregate_learning.py')
    checkpoints, trainers = {}, {}
    for seed in SEEDS:
        folder = paths.parent/'checkpoints'/f's{seed}'/'b8'
        receipt = read(folder/'summary/results.json')
        assert receipt['status'] == 'complete_upper_batch' and receipt['learner_seed'] == seed
        assert receipt['protocol_id'] == config['protocol_id'] and receipt['successful_updates'] == 1536
        assert receipt['completed_batches'] == 8 and receipt['target_audit']['outside_clip'] == 0
        trainers[seed] = load_online_networks(worker, folder, config, seed)
        checkpoints[str(seed)] = dict(path=str(folder.resolve()), completed_batches=8,
            successful_updates=1536, consumed_training_days=len(receipt['consumed_days']))
    by_path, reference_values, detailed_accounts, contracts, crn, doses, authority, diagnostics = {}, {}, {}, {}, {}, {}, {}, {}
    differences = {f's{seed}_minus_{reference}':[] for seed in SEEDS for reference in REFERENCES}
    correct = True
    for day in DAYS:
        refs, traces = {}, {}
        fixed_folder = Path(config['reference_manual_G_root'])/f'd{day}_fixed'
        fixed_row, fixed_trace = read(fixed_folder/'summary/results.json'), base.raw(fixed_folder/'raw/path.pkl.gz')
        assert fixed_row['status'] == 'complete_development_path' and fixed_row['arm'] == 'fixed'
        for reference in REFERENCES:
            folder = reference_folder(config, reference, day)
            row, trace = read(folder/'summary/results.json'), base.raw(folder/'raw/path.pkl.gz')
            assert row['day'] == day
            if reference == 'V325':
                assert row['status'] == 'complete_budget_permission_day' and row['learner_seed'] == CANDIDATE
                assert row['budget_permission']['mapping_checked']
            elif reference == 'learned_G':
                assert row['status'] == 'complete_live_directional_shadow_day' and row['binding']['passed']
                assert row['learner_seed'] == CANDIDATE
            else:
                assert row['status'] == 'complete_development_path' and row['arm'] == 'joint'
            key = f'{reference}_d{day}'
            physical = physical_account(base, row, trace)
            detailed_accounts[key] = dict(physical=physical)
            correct = correct and physical['passed']
            reference_values[key] = prior.endpoints(row, len(trace['dispatches']), config['reward_units'])
            refs[reference], traces[reference] = row, trace
        for seed in SEEDS:
            key = f's{seed}_d{day}'
            folder = paths/key
            row, trace = read(folder/'summary/results.json'), base.raw(folder/'raw/path.pkl.gz')
            saved = base.raw(folder/'raw/upper_transitions.pkl.gz')
            assert row['status'] == 'complete_upper_day' and row['split'] == 'development'
            assert row['day'] == day and row['learner_seed'] == seed and row['protocol_id'] == config['protocol_id']
            assert row['checkpoint'] == checkpoints[str(seed)]['path']
            assert row['upper_deterministic'] is True and row['behavior_epsilon'] == 0
            assert row['training_updates'] == 0 and row['native_actors_unchanged'] and row['independent_upper_unchanged']
            assert row['results']['scenario_seed'] == day and row['results']['train_seed'] == config['fast_actor'] == 12011
            assert row['results']['sampler_seed'] == config['fast_sampler'] == 27011001
            physical = physical_account(base, row, trace)
            credit = old_learning.credit_check(row, trace, saved, config)
            semantics = actual_action_account(worker, row, trace, saved)
            detailed_accounts[key] = dict(physical=physical, credit=credit, actual_action_semantics=semantics)
            correct = correct and physical['passed'] and credit['passed'] and semantics['passed']
            values = prior.endpoints(row, len(trace['dispatches']), config['reward_units'])
            by_path[key] = dict(day=day, learner_seed=seed, passengers=row['contract']['passengers_generated'],
                endpoints=values, upper_modes=row['upper']['mode_counts'], planned_changes=row['slow']['planned_changes'],
                physical_starts=row['physical_starts'])
            contracts[key+'_fixed_authority'] = row['contract'] == fixed_row['contract'] and trace['public_clocks'] == fixed_trace['public_clocks']
            authority[key] = base.plan_authority(trace, fixed_trace)
            diagnostics[key] = q_diagnostic(trainers[seed], row, trace, saved)
            for reference in REFERENCES:
                name = f's{seed}_minus_{reference}'
                pair_key = name+f'_d{day}'
                contracts[pair_key] = row['contract'] == refs[reference]['contract'] and trace['public_clocks'] == traces[reference]['public_clocks']
                crn[pair_key] = base.crn_pair(trace['uniforms'], traces[reference]['uniforms'])
                doses[pair_key] = prior.actual_dose(trace, traces[reference])
                other = reference_values[f'{reference}_d{day}']
                differences[name].append({metric:values[metric]-other[metric] for metric in values})
    correct = bool(correct and all(contracts.values()) and all(value['exact'] for value in crn.values()))
    comparisons = {name:{metric:prior.paired([row[metric] for row in rows]) for metric in rows[0]}
                   for name, rows in differences.items()}
    candidate_gates = {reference:outcome_gate(comparisons[f's{CANDIDATE}_minus_{reference}']) for reference in MANDATORY}
    coverage = {str(seed):[day for day in DAYS if authority[f's{seed}_d{day}']['material_actual_changes'] > 0] for seed in SEEDS}
    seed_guard = {str(seed):{reference:{metric:comparisons[f's{seed}_minus_{reference}'][metric]['mean'] <= 0
        for metric in ('waiting_s_per_passenger', 'J0')} for reference in MANDATORY} for seed in SEEDS}
    all_seed_guard = all(value for by_ref in seed_guard.values() for fields in by_ref.values() for value in fields.values())
    all_coverage = all(len(days) >= 2 for days in coverage.values())
    passed = correct and all_coverage and all_seed_guard and all(g['passed'] for g in candidate_gates.values())
    save(out/'details/accounting.json', detailed_accounts, bounded=False)
    save(out/'details/attribution.json', dict(contracts=contracts, common_randomness=crn,
        actual_dose=doses, plan_authority=authority, reference_endpoints=reference_values), bounded=False)
    diagnostic_by_seed = {}
    for seed in SEEDS:
        rows = [diagnostics[f's{seed}_d{day}'] for day in DAYS]
        count = sum(r['decisions'] for r in rows)
        modes = Counter()
        for row in rows:
            modes.update(row['mode_counts'])
        diagnostic_by_seed[str(seed)] = dict(decisions=count, modes=dict(modes),
            argmax_disagreements=sum(r['argmax_disagreements'] for r in rows),
            disagreements_with_q_gap_over_1e_6=sum(r['disagreements_with_q_gap_over_1e_6'] for r in rows),
            twin_critic_argmax_disagreements=sum(r['twin_critic_argmax_disagreements'] for r in rows),
            selected_q_gap_mean=sum(r['selected_q_gap_mean']*r['decisions'] for r in rows)/count,
            selected_q_gap_max=max(r['selected_q_gap_max'] for r in rows),
            joint_preferred_by_q_but_budget_executed=sum(r['joint_preferred_by_q_but_budget_executed'] for r in rows))
    save(out/'summary/diagnostic.json', dict(status='complete_two_action_actor_critic_diagnostic',
        by_seed=diagnostic_by_seed, by_path=diagnostics, checkpoints=checkpoints,
        final_online_twin_min_critic=True, actor_probabilities_and_execution_bound=True,
        canonical_actions=[2, 3], diagnostic_updates=0, physical_starts=0,
        limits='Q rankings are model predictions, not counterfactual traffic outcomes. Counts are decisions, not independent days; no greedy-Q replacement or selection.'))
    result = dict(status='complete_two_action_upper_aggregate',
        decision='development_candidate_passed_no_automatic_confirmation' if passed else 'close_two_action_upper_candidate_no_confirmation',
        implementation_correct=correct, development_candidate_passed=passed, candidate_seed=CANDIDATE,
        candidate_checkpoint=checkpoints[str(CANDIDATE)]['path'], candidate_gates=candidate_gates,
        all_seed_mean_guard_passed=all_seed_guard, all_seed_mean_guard=seed_guard,
        all_seed_plan_coverage_passed=all_coverage, material_plan_days=coverage,
        comparisons=comparisons, by_path=by_path, checkpoints=checkpoints,
        authority={key:{name:value[name] for name in ('changed_release_trips','binding_trips','material_actual_changes')}
                   for key, value in authority.items()},
        implementation_checks=dict(all_public_physical_contracts_matched=all(contracts.values()),
            all_common_randomness_exact=all(v['exact'] for v in crn.values()),
            all_actual_action_labels_bound=True, all_final_policy_and_online_critic_identities_bound=True),
        total_successful_upper_updates=4608, completed_training_days=96,
        evaluation_physical_starts=sum(row['physical_starts'] for row in by_path.values()),
        aggregation_physical_starts=0, diagnostic_updates=0, main_policy_changed=False, confirmation_expanded=False,
        metric_semantics=dict(difference='new learner minus named reference; negative improves',
            passenger_denominator='same-day generated passengers; full waiting/onboard tail', trip_denominator=262,
            bootstrap='256 paired four-day empirical resamples; equal day weights; each seed reported separately',
            references='V325 and learned_G use the frozen 314011 policy for every seed; manual_joint_G is shared',
            J0='(W+R)/3600000 + public_delay/94320', J1='J0 + CV_squared/0.05; diagnostic only'),
        limits=[
            'Four consumed development days and three separately reported learning seeds; not twelve independent demand days or confirmation.',
            '314011/b8 is preselected and cannot be replaced by a favorable seed or checkpoint.',
            'All seeds need waiting/J0 mean nonharm against both mandatory references; only the preselected candidate has the interval-based outcome gate.',
            'The original >=5 s publication-bound dispatch check against fixed requires two days per seed; it certifies plan authority, not isolated plan benefit.',
            'Budget versus joint learns plan permission over a persistent budget base; G/reward/fast controller remain unchanged.',
            'No waiting-reward, no-frequency or full layer ablation was added, and historical gate decisions are unchanged.'])
    save(out/'summary/results.json', result)
    print('V326_AGGREGATE', result['decision'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'paths', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        aggregate(args.config, args.paths, args.out)
    except Exception:
        save(args.out/'summary/failure.json', dict(status='failed_hold', error=traceback.format_exc()[-7000:],
            physical_starts=0, training_updates=0, automatic_retry=False,
            main_policy_changed=False, confirmation_expanded=False))
        raise
