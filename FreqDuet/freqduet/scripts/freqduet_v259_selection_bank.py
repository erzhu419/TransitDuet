"""Frozen V253/V254 selection units; no evaluation collector or action selector."""

import copy
import json

import torch

import freqduet_v257_atomic_smoke as core
import seal_freqduet_v254_action_manifest as manifest
from preregister_freqduet_v253_action_supply import holding_differences, supply_differences


COLLECTION = 'protocol_v6_v259_selection_bank_collection'
SPEC = manifest.manifest('eaba2513da', list(core.STREAMS))
METRICS = ('passenger_person_s','journey_min','waiting_person_s','onboard_person_s',
    'cv_squared','readiness_delay_s','readiness_delay_total_s','public_launch_delay_total_s',
    'holding_s_per_trip','trips_unlaunched','trips_incomplete')


def selection_unit(context, future):
    assert context in range(4) and future in range(8), 'only the 32 frozen S units'
    row = next(r for r in SPEC['futures'] if
               (r['phase'],r['context'],r['future']) == ('selection',context,future))
    return copy.deepcopy(dict(SPEC['contexts'][context], **{k:v for k,v in row.items() if k != 'context'}))


def check_unit(unit):
    assert unit == selection_unit(unit['context'],unit['future']), 'not a frozen S unit'


def unit_key(unit):
    check_unit(unit)
    return f"c{unit['context']}_selection_f{unit['future']}"


def action_plan(native_bin, feasible):
    original = core.smoke_plan(native_bin,feasible)
    return dict(physical=[b for b in original['physical'] if b != 'forced_native'],
                aliases=original['aliases'],masked=original['masked'],max_starts=7)


def selection_audit(base, repeated, event, joint, unit, context_reference=None):
    check_unit(unit)
    unit = copy.deepcopy(unit)
    if context_reference is not None:
        assert context_reference['context'] == unit['context']
    smoke_type = core.atomic_audit(base,repeated,event,joint)
    same = joint.pilot.joint.assert_same_state

    class SelectionAudit(smoke_type):
        def __init__(self, seed, branch, uniforms, reference=None):
            assert seed == unit['actor'] and uniforms.seed == unit['prefix_sampler']
            assert isinstance(uniforms,event.EventUniforms)
            if reference is None:
                assert branch == 'native', 'unit native must precede every fixed bin'
            else:
                same(reference['unit'],unit,'V259_reference_S_unit')
                q = reference['anchor']['query']
                assert branch in action_plan(q['native_bin'],q['feasible_bins'])['physical']
            # Reuse the accepted six-observer/V255 constructor, not smoke's c0-only checks.
            base.__init__(self,seed,'v107')
            self.branch,self.uniforms,self.reference = branch,uniforms,reference
            self.fork_anchor,self.future_receipt,self.requests = None,None,[]

        def __call__(self, policy, states, features, theta):
            native = base.__call__(self,policy,states,features,theta)
            row = self.pulse_capture.queries[-1]
            is_anchor = self.fork_anchor is None and row['eligible']
            selected = row['native_bin']
            if is_anchor:
                assert self.pulse_capture.anchor['query'] == row
                assert row['query_index'] == len(self.queries)-1
                chosen = row['native_bin'] if self.branch == 'native' else int(self.branch[1:])
                assert chosen in row['feasible_bins']
                anchor = core.checked_snapshot(self,self.atomic_runner,policy,states,features,theta,
                                               native,self.uniforms,repeated,joint)
                assert anchor['scenario']['seed'] == unit['demand_day']
                if self.reference is not None:
                    same(anchor,self.reference['anchor'],'V259_common_precommand_prefix')
                if context_reference is not None:
                    same(anchor,context_reference['anchor'],'V259_context_prefix_across_S_futures')
                before_native = native.clone()
                startup = repeated.fork_future(self.atomic_runner,policy,unit['root'])
                self.uniforms.seed = unit['streams']['lower_event_and_sampling']
                after = core.checked_snapshot(self,self.atomic_runner,policy,states,features,theta,
                                              native,self.uniforms,repeated,joint)
                for key in anchor.keys()-{'rng','lower_sampling','upper_sampling','scenario','event_uniforms'}:
                    same(anchor[key],after[key],'V259_preserved_'+key)
                same(anchor['event_uniforms']['counts'],dict(self.uniforms.counts),'V259_kept_occurrences')
                same(anchor['event_uniforms']['records'],self.uniforms.records,'V259_kept_prefix_uniforms')
                assert torch.equal(native,before_native)
                receipt = dict(startup=startup,root=unit['root'],event_seed=self.uniforms.seed,
                               event_counts=dict(self.uniforms.counts))
                if self.reference is not None:
                    same(receipt,self.reference['future'],'V259_common_future_start')
                self.fork_anchor,self.future_receipt = anchor,receipt
                selected = chosen
            value = native
            if selected != row['native_bin']:
                value = torch.zeros_like(native)
                value[0,selected] = 1.
                self.queries[-1]['command_s'] = core.ACTIONS[selected]
            self.requests.append(dict(copy.deepcopy(row),selected_bin=selected,
                is_anchor=is_anchor,explicit_native_alias=False))
            return value

        def branch_data(self, results):
            return dict(super().branch_data(results),unit=copy.deepcopy(unit),complete=True)
    return SelectionAudit


def validate_path(data, unit, same, reference=None):
    check_unit(unit)
    same(data['unit'],unit,'V259_path_unit')
    assert data['complete'] and not (data['selector_defined'] or data['main_policy_changed'] or data['v238_changed'])
    q,future = data['anchor']['query'],data['future']
    assert q['eligible'] and q == next(r for r in data['payloads']['pulse']['queries'] if r['eligible'])
    assert q['clock']['query_s'] >= 7200. and q['native_holding_s'] > 0 and not q['terminal_masked']
    assert q['forward_s'] is not None and q['backward_s'] is not None
    plan = action_plan(q['native_bin'],q['feasible_bins'])
    assert data['branch'] in plan['physical'], 'masked bins/aliases/forced-native are not physical S arms'
    assert data['anchor']['scenario']['seed'] == unit['demand_day']
    assert data['anchor']['event_uniforms']['seed'] == unit['prefix_sampler']
    assert future['root'] == future['startup']['scenario_seed'] == unit['root']
    assert future['event_seed'] == unit['streams']['lower_event_and_sampling']
    assert future['startup']['scenario_version'] == data['anchor']['scenario']['version']
    same(data['anchor']['physical'],future['startup']['physical'],'V259_current_window')
    same(data['anchor']['event_uniforms']['counts'],future['event_counts'],'V259_event_occurrences')
    if reference is not None:
        same(data['anchor'],reference['anchor'],'V259_unit_prefix')
        same(future,reference['future'],'V259_unit_future')
        same(data['contract'],reference['contract'],'V259_same_future_authority_and_population')
    assert data['ledger']['physical_vehicles'] == 12 and data['ledger']['incomplete_trips'] == 0
    assert data['completion'] == dict(trips_unlaunched=0,trips_incomplete=0)
    assert data['ledger']['executed_holding_s'] == data['ledger']['commanded_holding_s']
    assert data['results']['base_actors_unchanged']
    assert data['results']['training_updates'] == data['results']['dual_updates'] == 0
    measured = core.execution_chain(data['requests'],data['actions'],data['holds'],data['payloads']['v248']['departures'])
    same(measured,data['execution'],'V259_recomputed_execution')
    assert measured['complete'] and measured['sampled_actions_exact'] and measured['issued_commands_exact']
    anchors = [r for r in data['requests'] if r['is_anchor']]
    assert len(anchors) == 1 and anchors[0]['query_index'] == q['query_index']
    chosen = q['native_bin'] if data['branch'] == 'native' else int(data['branch'][1:])
    assert anchors[0]['selected_bin'] == chosen
    assert measured['anchor_executed_s'] == data['holding_summary']['anchor_executed_s'] == core.ACTIONS[chosen]
    assert measured['changed_requests'] == (0 if data['branch'] == 'native' else 1)
    return dict(passed=True,plan=plan,actual_anchor_holding_s=measured['anchor_executed_s'],
        actual_execution_complete=True,queries=len(data['requests']),
        metrics={k:data['results'][k] for k in METRICS})


def anchor_supply(data):
    key = data['anchor']['query']['cycle_key']
    cycles = [c for c in data['payloads']['v235']['cycles'] if c['cycle_key'] == key]
    assert len(cycles) == 1
    c = cycles[0]
    result = dict(cycle_key=key,launch_s=c['launch_s'],return_s=c['return_s'],
        H_s=c['executed_holding_s'],other_elapsed_s=None if c['return_s'] is None else
            c['return_s']-c['launch_s']-c['executed_holding_s'],
        next_trip=None,next_launch_s=None,next_public_s=None)
    if c['return_s'] is not None:
        events = [e for e in data['payloads']['legacy']['events'] if e['kind'] == 'launch'
            and e['bus_id'] == key[0] and e['event_seq'] > c['return_event_seq']]
        if events:
            first = min(events,key=lambda e:e['event_seq'])
            rows = [d for d in data['payloads']['legacy']['dispatches'] if
                    d['trip'] == first['trip_id'] and d['bus'] == key[0]]
            assert len(rows) == 1 and rows[0]['actual_s'] == first['time_s']
            result.update(next_trip=first['trip_id'],next_launch_s=rows[0]['actual_s'],
                          next_public_s=rows[0]['public_baseline_s'])
    return result


def validate_unit(branches, unit, same, alignment):
    native = branches['native']
    q = native['anchor']['query']
    plan = action_plan(q['native_bin'],q['feasible_bins'])
    assert set(branches) == set(plan['physical'])
    obligation = native['anchor']['v235']['queries'][-1]['view']['own_association']['obligation']
    reference_supply = anchor_supply(native)
    rows = {}
    for name,data in branches.items():
        row = validate_path(data,unit,same,native)
        a,h = data['holding_summary']['anchor_executed_s'],native['holding_summary']['anchor_executed_s']
        rows[name] = dict(row,coupling=alignment(native['uniforms'],data['uniforms']),
            delta_vs_native={k:data['results'][k]-native['results'][k] for k in METRICS},
            local_regret=2*(a-h)*(a+h-(q['backward_s']-q['forward_s'])),
            holding=holding_differences(native['holding_summary'],data['holding_summary']),
            supply=supply_differences(reference_supply,anchor_supply(data),
                                     native['payloads']['v230']['metadata']['native_dt_s'],obligation))
    assert any(r['actual_anchor_holding_s'] != native['holding_summary']['anchor_executed_s'] for r in rows.values())
    return dict(passed=True,unit=unit,plan=plan,paths=rows,query_obligation=obligation,
        offline_choices_made=False,evaluation_outcomes_opened=0,
        scope='All legal S arms retained;no outcome or supply-identity filtering.')


def authorization(source):
    return dict(source=source,manifest_source='eaba2513da',protocol_source='4667fd4668',
        phase='selection',units=[unit_key(selection_unit(c,f)) for c in range(4) for f in range(8)],
        physical_start_cap=224,per_unit_start_cap=7,max_workers=8,
        failures_consume_slots=True,automatic_retry=False,extra_reference_paths=0,
        evaluation_authorized=False,training_authorized=False,
        main_policy_changed=False,v238_changed=False)


def require_authorization(root, source):
    assert root.name == COLLECTION
    path = root/'selection_authorization.json'
    assert path.is_file(), 'S collection needs separate explicit authorization'
    assert json.loads(path.read_text()) == authorization(source)


def claim_slot(root, unit, branch, source):
    require_authorization(root,source)
    stem = unit_key(unit)
    checks,started = root/'checks'/stem,root/'started'/stem
    if branch == 'native':
        assert not list(started.glob('*.json')), 'native first;no restart or replacement'
    else:
        native = json.loads((checks/'native.json').read_text())
        assert native['passed'] and native['source'] == source and native['unit'] == unit
        plan = native['plan']['physical']
        assert branch in plan, 'alias/masked labels cannot consume a slot'
        for earlier in plan[:plan.index(branch)]:
            prior = json.loads((checks/(earlier+'.json')).read_text())
            assert prior['passed'] and prior['source'] == source and prior['unit'] == unit
    started.mkdir(parents=True,exist_ok=True)
    marker = started/(branch+'.json')
    with marker.open('x') as stream:
        json.dump(dict(source=source,unit=unit,branch=branch,retry_authorized=False),stream)
    assert len(list(started.glob('*.json'))) <= 7
    assert len(list((root/'started').glob('*/*.json'))) <= 224
