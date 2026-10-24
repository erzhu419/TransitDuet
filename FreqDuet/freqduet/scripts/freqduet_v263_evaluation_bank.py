"""Frozen V253/V254 E units and S-locked contrasts; no evaluation selection."""

import copy
import json

import torch

import freqduet_v259_selection_bank as selection


core, SPEC, METRICS = selection.core, selection.SPEC, selection.METRICS
COLLECTION = 'protocol_v6_v263_evaluation_bank_collection'
S_SOURCE, S_SEAL_SOURCE = '357e60fd70', '65bdd96938'
LOCK_FIELDS = ('context', 'contrast_id', 'native_bin', 'chosen_branch', 'chosen_bin', 'plan')
action_plan = selection.action_plan


def evaluation_unit(context, future):
    assert context in range(4) and future in range(8), 'only the 32 frozen E units'
    row = next(r for r in SPEC['futures'] if
        (r['phase'], r['context'], r['future']) == ('evaluation', context, future))
    return copy.deepcopy(dict(SPEC['contexts'][context], **{k:v for k,v in row.items() if k != 'context'}))


def check_unit(unit):
    assert unit == evaluation_unit(unit['context'], unit['future']), 'not a frozen E unit'


def unit_key(unit):
    check_unit(unit)
    return f"c{unit['context']}_evaluation_f{unit['future']}"


def check_lock(lock):
    assert lock['context'] in range(4)
    assert lock['contrast_id'] == f"c{lock['context']}:{lock['chosen_branch']}_vs_native_b{lock['native_bin']}"
    feasible = [lock['native_bin']]+[int(b[1:]) for b in lock['plan']['physical'] if b != 'native']
    assert lock['plan'] == action_plan(lock['native_bin'], feasible)
    assert lock['chosen_branch'] in lock['plan']['physical']
    assert lock['chosen_bin'] == (lock['native_bin'] if lock['chosen_branch'] == 'native' else int(lock['chosen_branch'][1:]))


def choices_from_seal(seal):
    summary = seal['summary']
    assert summary['source'] == S_SEAL_SOURCE and summary['collector_source'] == S_SOURCE
    assert summary['status'] == 'v262_all_S_complete_contrasts_sealed_E_closed'
    assert summary['complete_units'] == 32 and summary['failed_starts'] == 0
    assert summary['contrasts_sealed'] and summary['offline_choices_made']
    assert not summary['evaluation_authorized'] and summary['evaluation_outcomes_opened'] == 0
    locks = [copy.deepcopy({k:r[k] for k in LOCK_FIELDS}) for r in summary['contrasts']]
    assert [r['context'] for r in locks] == list(range(4))
    for lock in locks:
        check_lock(lock)
    return locks


def evaluation_audit(base, repeated, event, joint, unit, context_reference, lock):
    check_unit(unit); check_lock(lock)
    unit, lock, context_reference = copy.deepcopy((unit, lock, context_reference))
    assert lock['context'] == unit['context']
    assert context_reference['unit'] == selection.selection_unit(unit['context'], 0)
    smoke_type = core.atomic_audit(base, repeated, event, joint)
    same = joint.pilot.joint.assert_same_state

    class EvaluationAudit(smoke_type):
        def __init__(self, seed, branch, uniforms, reference=None):
            assert seed == unit['actor'] and uniforms.seed == unit['prefix_sampler']
            assert isinstance(uniforms, event.EventUniforms)
            assert branch in lock['plan']['physical'], 'masked bins and aliases never start'
            if reference is None:
                assert branch == 'native', 'same-unit E native must precede alternatives'
            else:
                same(reference['unit'], unit, 'V263_reference_E_unit')
                same(reference['locked_contrast'], lock, 'V263_reference_S_lock')
            base.__init__(self, seed, 'v107')
            self.branch, self.uniforms, self.reference = branch, uniforms, reference
            self.fork_anchor, self.future_receipt, self.requests = None, None, []

        def __call__(self, policy, states, features, theta):
            native = base.__call__(self, policy, states, features, theta)
            row = self.pulse_capture.queries[-1]
            is_anchor = self.fork_anchor is None and row['eligible']
            selected = row['native_bin']
            if is_anchor:
                assert self.pulse_capture.anchor['query'] == row
                assert row['query_index'] == len(self.queries)-1
                same(action_plan(row['native_bin'], row['feasible_bins']), lock['plan'], 'V263_frozen_S_mask')
                chosen = row['native_bin'] if self.branch == 'native' else int(self.branch[1:])
                assert chosen in row['feasible_bins']
                anchor = core.checked_snapshot(self, self.atomic_runner, policy, states, features, theta,
                    native, self.uniforms, repeated, joint)
                assert anchor['scenario']['seed'] == unit['demand_day']
                same(anchor, context_reference['anchor'], 'V263_S0_to_E_precommand_prefix')
                if self.reference is not None:
                    same(anchor, self.reference['anchor'], 'V263_common_E_precommand_prefix')
                before_native = native.clone()
                startup = repeated.fork_future(self.atomic_runner, policy, unit['root'])
                self.uniforms.seed = unit['streams']['lower_event_and_sampling']
                after = core.checked_snapshot(self, self.atomic_runner, policy, states, features, theta,
                    native, self.uniforms, repeated, joint)
                for key in anchor.keys()-{'rng','lower_sampling','upper_sampling','scenario','event_uniforms'}:
                    same(anchor[key], after[key], 'V263_preserved_'+key)
                same(anchor['event_uniforms']['counts'], dict(self.uniforms.counts), 'V263_kept_occurrences')
                same(anchor['event_uniforms']['records'], self.uniforms.records, 'V263_kept_prefix_uniforms')
                assert torch.equal(native, before_native)
                receipt = dict(startup=startup, root=unit['root'], event_seed=self.uniforms.seed,
                    event_counts=dict(self.uniforms.counts))
                if self.reference is not None:
                    same(receipt, self.reference['future'], 'V263_common_E_future_start')
                self.fork_anchor, self.future_receipt = anchor, receipt
                selected = chosen
            value = native
            if selected != row['native_bin']:
                value = torch.zeros_like(native)
                value[0,selected] = 1.
                self.queries[-1]['command_s'] = core.ACTIONS[selected]
            self.requests.append(dict(copy.deepcopy(row), selected_bin=selected,
                is_anchor=is_anchor, explicit_native_alias=False))
            return value

        def branch_data(self, results):
            return dict(super().branch_data(results), unit=copy.deepcopy(unit), complete=True,
                locked_contrast=copy.deepcopy(lock))
    return EvaluationAudit


def validate_path(data, unit, lock, same, reference=None):
    check_unit(unit); check_lock(lock)
    assert lock['context'] == unit['context']
    same(data['unit'], unit, 'V263_path_unit')
    same(data['locked_contrast'], lock, 'V263_preserved_S_choice')
    assert data['complete'] and not (data['selector_defined'] or data['main_policy_changed'] or data['v238_changed'])
    q, future = data['anchor']['query'], data['future']
    assert q['eligible'] and q == next(r for r in data['payloads']['pulse']['queries'] if r['eligible'])
    assert q['clock']['query_s'] >= 7200. and q['native_holding_s'] > 0 and not q['terminal_masked']
    assert q['forward_s'] is not None and q['backward_s'] is not None
    plan = action_plan(q['native_bin'], q['feasible_bins'])
    assert plan == lock['plan'] and data['branch'] in plan['physical']
    assert data['anchor']['scenario']['seed'] == unit['demand_day']
    assert data['anchor']['event_uniforms']['seed'] == unit['prefix_sampler']
    assert future['root'] == future['startup']['scenario_seed'] == unit['root']
    assert future['event_seed'] == unit['streams']['lower_event_and_sampling']
    assert future['startup']['scenario_version'] == data['anchor']['scenario']['version']
    same(data['anchor']['physical'], future['startup']['physical'], 'V263_current_window')
    same(data['anchor']['event_uniforms']['counts'], future['event_counts'], 'V263_event_occurrences')
    if reference is not None:
        same(data['anchor'], reference['anchor'], 'V263_unit_prefix')
        same(future, reference['future'], 'V263_unit_future')
        same(data['contract'], reference['contract'], 'V263_same_E_authority_and_population')
    assert data['ledger']['physical_vehicles'] == 12 and data['ledger']['incomplete_trips'] == 0
    assert data['completion'] == dict(trips_unlaunched=0, trips_incomplete=0)
    assert data['ledger']['executed_holding_s'] == data['ledger']['commanded_holding_s']
    assert data['results']['base_actors_unchanged'] and data['results']['training_updates'] == data['results']['dual_updates'] == 0
    measured = core.execution_chain(data['requests'], data['actions'], data['holds'], data['payloads']['v248']['departures'])
    same(measured, data['execution'], 'V263_recomputed_execution')
    assert measured['complete'] and measured['sampled_actions_exact'] and measured['issued_commands_exact']
    anchors = [r for r in data['requests'] if r['is_anchor']]
    assert len(anchors) == 1 and anchors[0]['query_index'] == q['query_index']
    chosen = q['native_bin'] if data['branch'] == 'native' else int(data['branch'][1:])
    assert anchors[0]['selected_bin'] == chosen
    assert measured['anchor_executed_s'] == data['holding_summary']['anchor_executed_s'] == core.ACTIONS[chosen]
    assert measured['changed_requests'] == (0 if data['branch'] == 'native' else 1)
    return dict(passed=True, plan=plan, locked_contrast=copy.deepcopy(lock),
        actual_anchor_holding_s=measured['anchor_executed_s'], actual_execution_complete=True,
        queries=len(data['requests']), metrics={k:data['results'][k] for k in METRICS})


def validate_unit(branches, unit, lock, same, alignment):
    native = branches['native']
    q = native['anchor']['query']
    plan = action_plan(q['native_bin'], q['feasible_bins'])
    assert plan == lock['plan'] and set(branches) == set(plan['physical'])
    obligation = native['anchor']['v235']['queries'][-1]['view']['own_association']['obligation']
    supply = selection.anchor_supply(native)
    paths = {}
    for name, data in branches.items():
        assert data['branch'] == name
        path = validate_path(data, unit, lock, same, native)
        a, h = data['holding_summary']['anchor_executed_s'], native['holding_summary']['anchor_executed_s']
        paths[name] = dict(path, coupling=alignment(native['uniforms'], data['uniforms']),
            delta_vs_native={k:data['results'][k]-native['results'][k] for k in METRICS},
            local_regret=2*(a-h)*(a+h-(q['backward_s']-q['forward_s'])),
            holding=selection.holding_differences(native['holding_summary'], data['holding_summary']),
            supply=selection.supply_differences(supply, selection.anchor_supply(data),
                native['payloads']['v230']['metadata']['native_dt_s'], obligation))
    return dict(passed=True, unit=unit, plan=plan, paths=paths, query_obligation=obligation,
        locked_contrast=copy.deepcopy(lock), primary_branch=lock['chosen_branch'],
        primary_contrast_id=lock['contrast_id'], all_arms_retained=True, selection_recomputed=False,
        scope='S-locked primary only; all legal E outcomes retained without reselection.')


def authorization(source, locks):
    assert [r['context'] for r in locks] == list(range(4))
    for lock in locks:
        check_lock(lock)
    return dict(source=source, manifest_source='eaba2513da', protocol_source='4667fd4668',
        S_seal_source=S_SEAL_SOURCE, locked_contrasts=copy.deepcopy(locks), phase='evaluation',
        units=[unit_key(evaluation_unit(c,f)) for c in range(4) for f in range(8)],
        physical_start_cap=224, per_unit_start_cap=7, max_workers=8,
        failures_consume_slots=True, automatic_retry=False, extra_reference_paths=0,
        evaluation_authorized=True, training_authorized=False, reselection_authorized=False,
        main_policy_changed=False, v238_changed=False)


def require_authorization(root, source, locks):
    assert root.name == COLLECTION
    path = root/'evaluation_authorization.json'
    assert path.is_file(), 'real E needs separate explicit authorization'
    assert json.loads(path.read_text()) == authorization(source, locks)


def claim_slot(root, unit, branch, source, locks):
    check_unit(unit)
    require_authorization(root, source, locks)
    key = unit_key(unit)
    checks, started = root/'checks'/key, root/'started'/key
    assert branch in locks[unit['context']]['plan']['physical'], 'masked/alias labels never start'
    if branch == 'native':
        assert not list(started.glob('*.json')), 'native first; no restart'
    else:
        native = json.loads((checks/'native.json').read_text())
        assert native['plan'] == locks[unit['context']]['plan']
        order = native['plan']['physical']
        for earlier in order[:order.index(branch)]:
            prior = json.loads((checks/(earlier+'.json')).read_text())
            assert prior['passed'] and prior['unit'] == unit and prior['source'] == source
            assert prior['locked_contrast'] == locks[unit['context']]
    started.mkdir(parents=True, exist_ok=True)
    with (started/(branch+'.json')).open('x') as stream:
        json.dump(dict(source=source, unit=unit, branch=branch,
            locked_contrast=locks[unit['context']], retry_authorized=False), stream)
    assert len(list(started.glob('*.json'))) <= 7
    assert len(list((root/'started').glob('*/*.json'))) <= 224
