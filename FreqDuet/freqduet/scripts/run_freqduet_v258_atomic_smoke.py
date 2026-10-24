#!/usr/bin/env python3
"""Run only the V253/V254 c0 smoke, native first; no formal or training entrypoint."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import traceback
import unittest

import freqduet_v257_atomic_smoke as atomic


COLLECTION = 'protocol_v6_v258_atomic_smoke_collection'
METRICS = ('passenger_person_s', 'journey_min', 'waiting_person_s', 'onboard_person_s',
    'cv_squared', 'readiness_delay_s', 'readiness_delay_total_s',
    'public_launch_delay_total_s', 'holding_s_per_trip', 'trips_unlaunched', 'trips_incomplete')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def read(path):
    return json.loads(path.read_text())


def save_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_record(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def slot(branch):
    return 'c0_smoke_f0_'+branch


def check_file(root, branch):
    return root/'checks'/(slot(branch)+'.json')


def accepted(root, branch, source, native_gate=None):
    value = read(check_file(root, branch))
    if branch == 'native' and native_gate is not None:
        assert value['source'] == '140949bcf8' and not value['passed']
        assert value['failure'] == 'AssertionError: checkpoint value differs: V258_archived_authority.scenario_tape_id'
        assert native_gate['source'] == '22bd7999c5' and native_gate['data_source'] == '140949bcf8'
        assert native_gate['status'] == 'v258_completed_native_readonly_recheck_passed' and native_gate['passed']
        assert native_gate['original_check_preserved'] and native_gate['original_failed_start_debit'] == 1
        return native_gate
    assert value['source'] == source and value['branch'] == branch and value['passed']
    return value


def claim_slot(root, branch, source, native_gate=None):
    assert root.name == COLLECTION
    if branch == 'native':
        assert not list((root/'started').glob('*.json')), 'native must be first'
    else:
        native = accepted(root, 'native', source, native_gate)
        order = native['plan']['physical']
        assert branch in order, 'masked bins/native aliases never start'
        for previous in order[:order.index(branch)]:
            accepted(root, previous, source, native_gate)
    assert len(list((root/'started').glob('*.json'))) < 8
    marker = root/'started'/(slot(branch)+'.json')
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x') as stream:
        json.dump(dict(slot=slot(branch), branch=branch, source=source, context=0,
            actor=12037, demand_day=2151001, prefix_sampler=402112137,
            future_root=3540999999, retry_authorized=False), stream)


def interfaces():
    import run_freqduet_v256_passive_pulse as passive
    assert 'source_aa8616bcfb' in passive.__file__
    assert 'source_263593f7e8' in atomic.__file__
    frozen, pulse, old, cycles, requests = passive.runtime_interfaces()
    legacy, joint = old.legacy, old.legacy.passive.prior
    base = pulse.recording_audit(frozen.observer.recording_audit(
        requests.recording_audit(cycles.recording_audit(old.RecordingAudit))), 'observer_only')
    return legacy, joint, atomic.atomic_audit(base, joint.repeated, legacy.pilot.event, joint)


def archived_authority(data, old, same, tape_class):
    keys = ('protocol_version','clock','fleet_target','physical_vehicle_count','fleet_inventory_mode','dt_s')
    new_contract,old_contract = data['contract'],old['physical_contract']
    assert set(new_contract) == set(old_contract) == set(keys)|{'scenario_tape_id','passengers_generated'}
    same({k:new_contract[k] for k in keys},{k:old_contract[k] for k in keys},'V258_archived_fixed_authority')
    scenario = data['anchor']['scenario']
    assert old_contract['scenario_tape_id'] == tape_class(seed=scenario['seed'],version=scenario['version']).identifier
    assert new_contract['scenario_tape_id'] == tape_class(seed=data['future']['root'],version=scenario['version']).identifier
    return dict(fixed_authority_exact=True, old_and_new_tape_ids_match_their_registered_seeds=True,
        old_future_passengers_generated=old_contract['passengers_generated'],
        new_future_passengers_generated=new_contract['passengers_generated'])


def archived_prefix(data, v256_root, legacy, same):
    """Compare stored pre-fork fields only; the old future endpoint is not a comparator."""
    archive = v256_root.with_name(v256_root.name+'_observer_repair1')
    pulse = legacy.prior.read(archive/'pulse_sidecars/c0_v107.json.gz')
    prefix, query = data['anchor'], data['anchor']['query']
    same(prefix['pulse']['anchor'], pulse['anchor'], 'V258_archived_first_anchor')
    same(query, pulse['anchor']['query'], 'V258_archived_anchor_query')
    n = query['query_index']
    same(prefix['pulse']['queries'], pulse['queries'][:n+1], 'V258_archived_query_prefix')
    old = legacy.prior.read(archive/'records/c0_v107.json.gz')
    same(prefix['legacy']['events'], old['events'][:len(prefix['legacy']['events'])], 'V258_archived_events')
    fields = dict(commands='upper_commands', callbacks='upper_callbacks', inputs='lower_policy_inputs',
        dispatches='actual_launches', returns='vehicle_returns', arrivals='arrivals', departures='departures')
    for key, name in fields.items():
        values = prefix['legacy'][key]
        same(values, old['trace'][name][:len(values)], 'V258_archived_'+key)
    same(data['actions'][:n], old['trace']['lower_actions'][:n], 'V258_archived_sampled_prefix')
    uniforms = [[list(k),v] for k,v in prefix['event_uniforms']['records'].items()]
    same(uniforms, old['trace']['action_uniforms'][:len(uniforms)], 'V258_archived_uniforms')
    from env.scenario import ScenarioTape
    authority = archived_authority(data,old,same,ScenarioTape)
    from seal_freqduet_v254_action_manifest import PAYLOADS
    lists = dict(v224=('events','prepares','queries'),
        v230=('query_extensions','upper_versions','cycle_events','supply_outcomes'),
        v235=('queries','transitions','observed_links'), v241=('queries',),
        v248=('launches','departures','samples','prepares','queries'))
    for name, keys in lists.items():
        saved = legacy.prior.read(archive/PAYLOADS[name]/'c0_v107.json.gz')
        for key in keys:
            values = prefix[name][key]
            same(values, saved[key][:len(values)], 'V258_archived_'+name+'_'+key)
    return dict(archived_stored_prefix_exact=True, archived_endpoint_compared=False, **authority)


def path_check(data, reference, joint):
    same, q = joint.pilot.joint.assert_same_state, data['anchor']['query']
    assert data['complete'] and q['eligible'] and q['clock']['query_s'] >= 7200.
    assert q == next(r for r in data['payloads']['pulse']['queries'] if r['eligible'])
    future = data['future']
    assert future['root'] == future['startup']['scenario_seed'] == 3540999999
    assert future['event_seed'] == 3550999999
    same(data['anchor']['physical'], future['startup']['physical'], 'V258_retained_current_window')
    same(data['anchor']['event_uniforms']['counts'], future['event_counts'], 'V258_kept_occurrences')
    assert future['startup']['scenario_version'] == data['anchor']['scenario']['version']
    if reference is not None:
        same(data['anchor'], reference['anchor'], 'V258_common_precommand_prefix')
        same(future, reference['future'], 'V258_common_future')
    assert data['ledger']['physical_vehicles'] == 12 and data['ledger']['incomplete_trips'] == 0
    assert data['ledger']['executed_holding_s'] == data['ledger']['commanded_holding_s']
    assert data['completion'] == dict(trips_unlaunched=0, trips_incomplete=0)
    assert data['results']['base_actors_unchanged']
    assert data['results']['training_updates'] == data['results']['dual_updates'] == 0
    assert not (data['selector_defined'] or data['main_policy_changed'] or data['v238_changed'])
    execution = atomic.execution_chain(data['requests'], data['actions'], data['holds'],
                                        data['payloads']['v248']['departures'])
    same(execution, data['execution'], 'V258_recomputed_execution')
    assert execution['complete'] and execution['sampled_actions_exact'] and execution['issued_commands_exact']
    expected = q['native_bin'] if data['branch'] in ('native','forced_native') else int(data['branch'][1:])
    assert expected in q['feasible_bins'] and execution['anchor_executed_s'] == atomic.ACTIONS[expected]
    assert execution['changed_requests'] == (0 if data['branch'] in ('native','forced_native') else 1)
    return dict(passed=True, reference_ready=data['branch'] == 'native',
        anchor_s=q['clock']['query_s'], anchor_query_index=q['query_index'],
        anchor_query_seq=q['clock']['query_seq'], anchor_cycle=q['cycle_key'],
        native_bin=q['native_bin'], feasible_bins=q['feasible_bins'],
        plan=atomic.smoke_plan(q['native_bin'],q['feasible_bins']), future_switches=1,
        future_root=future['root'], event_seed=future['event_seed'],
        current_window_preserved=True, event_occurrences_preserved=True,
        queries=len(data['requests']), actual_anchor_holding_s=execution['anchor_executed_s'],
        changed_requests=execution['changed_requests'], actual_execution_complete=True,
        actual_total_holding_s=data['ledger']['executed_holding_s'],
        metrics={k:data['results'][k] for k in METRICS})


def update_status(root, source, gate=None, native_gate=None):
    rows = {p.stem.removeprefix('c0_smoke_f0_'):read(p) for p in sorted((root/'checks').glob('*.json'))}
    starts = len(list((root/'started').glob('*.json')))
    failures = sum(not r['passed'] for r in rows.values())
    effective = dict(rows)
    if native_gate is not None:
        effective['native'] = accepted(root,'native',source,native_gate)
    native = effective.get('native',{})
    planned = native.get('plan',{}).get('physical', ['native'])
    complete = set(effective) == set(planned) and all(r['passed'] for r in effective.values())
    current_failures = sum(not r['passed'] for r in effective.values())
    if current_failures:
        state = 'stopped'
    elif gate is not None:
        state = 'engineering_passed_formal_closed' if gate['passed'] else 'engineering_failed'
    else:
        state = 'paths_passed_gate_pending' if complete else 'native_passed_remaining_pending'
    result = dict(status='v258_smoke_'+state, source=source, adapter_source='263593f7e8',
        manifest_source='eaba2513da', paths=rows, started_smoke_paths=starts,
        failed_starts=failures, max_smoke_paths=8, remaining_smoke_cap=8-starts,
        automatic_retry=False, formal_starts=0, training_updates=0, main_policy_changed=False,
        v238_changed=False, selector_defined=False, gate=gate, native_readonly_recheck=native_gate,
        engineering_path_passes={b:r['passed'] for b,r in effective.items()},
        interpretation='One consumed c0/day and one smoke future; engineering/descriptive evidence only.')
    if native.get('passed'):
        result['descriptive_delta_vs_native'] = {b:{k:r['metrics'][k]-native['metrics'][k] for k in METRICS}
            for b,r in effective.items() if r['passed']}
    write(root/'summary/status.json',result)
    print('V258_STATUS '+json.dumps(result),flush=True)
    return result


def collect(a, branch, native_gate=None):
    root, source = a.out_root, a.source
    assert not check_file(root,branch).exists(), 'completed slot is read-only; no restart'
    claim_slot(root,branch,source,native_gate)
    print('V258_STARTED '+slot(branch),flush=True)
    rule = None
    output = root/'records'/(slot(branch)+'.pkl.gz')
    result = dict(source=source,branch=branch,passed=False,retry_authorized=False)
    try:
        assert not output.exists()
        legacy,joint,audit_class = interfaces()
        reference = None
        if branch != 'native':
            saved = load_record(root/'records'/(slot('native')+'.pkl.gz'))
            reference = dict(anchor=saved['anchor'],future=saved['future'])
            del saved
        uniforms = legacy.pilot.event.EventUniforms(402112137)
        rule,credit = audit_class(12037,branch,uniforms,reference),legacy.pilot.CreditObserver()
        row,events,_,holds = legacy.pilot.rollout(root,12037,2151001,'stochastic',402112137,
            [0.]*4,'v258_'+branch,**rule.rollout_kwargs(credit))
        row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]),holding_person_s=rule.holding_person_s,
            public_launch_delay_total_s=sum(max(0.,r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches),**rule.dispatch_metrics)
        data = dict(rule.branch_data(row),complete=True,source=source)
        save_record(output,data)
        result.update(path_check(data,reference,joint))
        if branch == 'native':
            result.update(archived_prefix(data,a.v256_root,legacy,joint.pilot.joint.assert_same_state))
        elif branch == 'forced_native':
            native = load_record(root/'records'/(slot('native')+'.pkl.gz'))
            for key in ('payloads','results','execution','actions','holds','ledger','holding_summary','completion','uniforms'):
                joint.pilot.joint.assert_same_state(native[key],data[key],'V258_native_alias_'+key)
            result['native_alias_exact'] = True
    except Exception as exc:
        traceback.print_exc()
        result.update(passed=False,reference_ready=False,failure=type(exc).__name__+': '+str(exc),
            observed_time_s=None if rule is None or not hasattr(rule,'env') else float(rule.env.current_time))
    write(check_file(root,branch),result)
    update_status(root,source,native_gate=native_gate)
    print('V258_PATH_CHECK '+json.dumps(result),flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('tests','native','remaining'))
    for name in ('out-root','v254-root','v256-root','v257-summary'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True)
    a = p.parse_args()
    assert a.out_root.name == COLLECTION
    manifest = read(a.v254_root/'manifest.json')
    assert manifest['source'] == 'eaba2513da' and manifest['physical_start_caps']['smoke'] == 8
    parent = read(a.v257_summary)
    assert (parent['source'],parent['status'],parent['tests']) == ('263593f7e8','v257_atomic_smoke_synthetic_passed',11)
    passive = read(a.v256_root/'summary/status.json')
    assert passive['source'] == 'aa8616bcfb' and passive['status'] == 'v256_repair_passive_pair_passed'
    preflight = a.out_root/'integration_preflight.json'
    if a.command == 'tests':
        assert not preflight.exists()
        interfaces()
        # Resolve proof paths before any smoke debit; nested paths broke the old collector.
        proof = a.out_root.parent/'runtime_optimization_20261014'
        assert (proof/'benchmark_summary.json').is_file()
        assert (proof/'training_equivalence/summary.json').is_file()
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                                    pattern='test_v258_atomic_smoke.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.testsRun > 0 and result.wasSuccessful()
        write(preflight,dict(source=a.source,status='v258_real_driver_synthetic_passed',
            tests=result.testsRun,simulator_starts=0,future_starts=0))
        print('V258_DRIVER_SYNTHETIC_PASSED',flush=True)
        return
    ready = read(preflight)
    assert ready['source'] == a.source and ready['status'] == 'v258_real_driver_synthetic_passed'
    auth = dict(source=a.source,protocol_source='4667fd4668',manifest_source='eaba2513da',
        adapter_source='263593f7e8',context=0,actor=12037,demand_day=2151001,
        smoke_root=3540999999,max_smoke_starts=8,max_workers=1,
        order='native then forced-native then legal nonalias bins in bin order',
        authorization='User approved real single-query smoke; native gate before other legal paths.',
        failures_consume_slots=True,automatic_retry=False,extra_reference_paths=0,
        formal_authorized=False,training_authorized=False,main_policy_changed=False,v238_changed=False)
    auth_file = a.out_root/'execution_authorization.json'
    if auth_file.exists():
        assert read(auth_file) == auth
    else:
        assert a.command == 'native'
        write(auth_file,auth)
    if a.command == 'native':
        collect(a,'native')
        return
    native = accepted(a.out_root,'native',a.source)
    assert native['reference_ready'] and native['archived_stored_prefix_exact']
    for branch in native['plan']['physical'][1:]:
        collect(a,branch)
    legacy,joint,_ = interfaces()
    paths = {b:load_record(a.out_root/'records'/(slot(b)+'.pkl.gz')) for b in native['plan']['physical']}
    gate = atomic.validate_smoke(paths,joint.pilot.joint.assert_same_state,legacy.pilot.event.alignment)
    update_status(a.out_root,a.source,gate)


if __name__ == '__main__':
    main()
