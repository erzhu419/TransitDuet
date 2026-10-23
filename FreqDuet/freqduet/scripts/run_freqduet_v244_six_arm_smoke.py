#!/usr/bin/env python3
"""Execute only V240's six authorized c0 smoke slots; no formal entrypoint."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import traceback
import unittest

import freqduet_v243_six_arm_smoke as adapter
from preregister_freqduet_v240_paired_coordination import CONTEXTS, SMOKE_SEED, registration, reserved_slots


ARMS = tuple(adapter.ARM_RULES)
METRICS = ('passenger_person_s', 'journey_min', 'waiting_person_s', 'onboard_person_s',
    'cv_squared', 'readiness_delay_s', 'readiness_delay_total_s',
    'public_launch_delay_total_s', 'holding_s_per_trip', 'trips_unlaunched', 'trips_incomplete')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def read(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def claim_slot(root, arm, source):
    assert arm in ARMS, 'only six registered c0 smoke arms are authorized'
    if arm != 'v107':
        first = json.loads((root/'checks/smoke_c0_v107.json').read_text())
        assert first['source'] == source and first['reference_ready'] and first['passed']
    marker = root/'started'/f'smoke_c0_{arm}.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x') as stream:
        json.dump(dict(slot=marker.stem, context=0, arm=arm, future_seed=SMOKE_SEED,
            source=source, started=True, retry_authorized=False), stream)


def archived_prefix(data, archive, legacy, same):
    """Use V242's accepted stored fields; no additional reference rollout."""
    old = {k: legacy.prior.read(archive/folder/'c0_v107.json.gz') for k, folder in
        dict(legacy='records', v224='sidecars', v230='resource_sidecars', v235='cycle_sidecars').items()}
    anchor, prefix = data['anchor'], data['anchor']['legacy']
    first = next(q for q in old['legacy']['fresh_upper'] if q['time_s'] >= 7200.)
    same(first, anchor['fresh_query'], 'V244_first_registered_native_anchor')
    same(prefix['events'], old['legacy']['events'][:len(prefix['events'])], 'V244_archived_native_events')
    pairs = dict(commands='upper_commands', callbacks='upper_callbacks', inputs='lower_policy_inputs',
        dispatches='actual_launches', returns='vehicle_returns', arrivals='arrivals', departures='departures')
    for name, key in pairs.items():
        same(prefix[name], old['legacy']['trace'][key][:len(prefix[name])], 'V244_archived_'+name)
    uniforms = [[list(k), v] for k, v in anchor['event_uniforms']['records'].items()]
    same(uniforms, old['legacy']['trace']['action_uniforms'][:len(uniforms)], 'V244_archived_uniform_prefix')
    n = len(prefix['queries'])
    same(data['trace']['lower_actions'][:n], old['legacy']['trace']['lower_actions'][:n], 'V244_archived_sampled_prefix')
    for name, fields in dict(v224=('prepares', 'queries'),
        v230=('query_extensions', 'upper_versions', 'cycle_events', 'supply_outcomes'),
        v235=('queries', 'transitions', 'observed_links')).items():
        for field in fields:
            rows = anchor[name][field]
            same(rows, old[name][field][:len(rows)], 'V244_archived_'+name+'_'+field)
    same(anchor['v230']['metadata'], old['v230']['metadata'], 'V244_initial_resource_contract')
    same(anchor['v235']['metadata'], old['v235']['metadata'], 'V244_initial_cycle_contract')
    return dict(archived_stored_prefix_exact=True, archived_rng_states_available=False)


def path_check(data, reference, same):
    assert data['complete'] and data['context'] == 0 and data['future_seed'] == SMOKE_SEED
    anchor, future = data['anchor'], data['future']
    assert anchor['fresh_query']['time_s'] >= 7200.
    assert anchor['fresh_query'] == next(q for q in data['fresh_upper'] if q['time_s'] >= 7200.)
    assert future['scenario_seed'] == future['startup']['scenario_seed'] == SMOKE_SEED
    assert future['event_seed'] == SMOKE_SEED+10000000
    assert future['startup']['scenario_version'] == anchor['scenario']['version']
    same(anchor['physical'], future['startup']['physical'], 'V244_retained_current_window')
    same(anchor['event_uniforms']['counts'], future['event_counts'], 'V244_retained_event_occurrences')
    if reference is not None:
        same(anchor, reference['anchor'], 'V244_complete_common_prefix')
        same(future, reference['future'], 'V244_complete_common_future_start')
    assert data['ledger']['physical_vehicles'] == 12
    assert data['completion'] == dict(trips_unlaunched=0, trips_incomplete=0)
    assert data['ledger']['incomplete_trips'] == 0
    assert data['ledger']['executed_holding_s'] == data['ledger']['commanded_holding_s']
    assert data['results']['base_actors_unchanged'] and data['results']['training_updates'] == 0
    assert data['results']['dual_updates'] == 0 and not data['main_policy_changed']
    execution = data['execution']
    assert execution['complete'] and execution['sampled_actions_exact'] and execution['issued_commands_exact']
    if adapter.ARM_RULES[data['arm']][1] == 'native':
        assert execution['changed_requests'] == execution['executed_changed_requests'] == 0
    else:
        assert execution['changed_requests'] == execution['executed_changed_requests'] > 0
    assert len(data['requests']) == len(data['trace']['lower_actions']) > 0
    return dict(passed=True, first_native_fresh_anchor_verified=True, anchor_s=anchor['fresh_query']['time_s'],
        common_prefix_exact=reference is not None, common_future_exact=reference is not None,
        retained_current_window_and_H=True, event_occurrences_preserved=True,
        queries=len(data['requests']), execution=execution,
        same_state_supply_blind_differences=sum(p['supply_request']['shadow_bin'] != p['blind_request']['shadow_bin']
                                             for p in data['requests']),
        metrics={k: data['results'][k] for k in METRICS},
        actual_holding_s=data['ledger']['executed_holding_s'])


def collect(root, archive, arm, source):
    slot = 'smoke_c0_'+arm
    check = root/'checks'/f'{slot}.json'
    if check.exists():
        result = json.loads(check.read_text())
        assert result['source'] == source and result['slot'] == slot
        return result
    assert not (root/'started'/f'{slot}.json').exists(), 'consumed smoke attempt cannot restart'
    import run_freqduet_v230_resource_capture as resource
    old = resource.frozen()
    legacy, joint = old.legacy, old.legacy.passive.prior
    same = joint.pilot.joint.assert_same_state
    reference = None
    if arm != 'v107':
        first = json.loads((root/'checks/smoke_c0_v107.json').read_text())
        assert first['source'] == source and first['passed'] and first['reference_ready']
        stored = read(root/'records/smoke_c0_v107.pkl.gz')
        reference = dict(anchor=stored['anchor'], future=stored['future'])
        del stored
    audit_class = adapter.runtime_audit()
    uniforms = legacy.pilot.event.EventUniforms(187000000+CONTEXTS[0][0]+100*CONTEXTS[0][1])
    credit = legacy.pilot.CreditObserver()
    output = root/'records'/f'{slot}.pkl.gz'
    assert not output.exists()
    claim_slot(root, arm, source)
    print('V244_STARTED '+slot, flush=True)
    rule = None
    try:
        seed, day = CONTEXTS[0]
        rule = audit_class(seed, arm, uniforms, reference)
        row, events, _, holds = legacy.pilot.rollout(root, seed, day, 'stochastic', uniforms.seed,
            [0.]*4, 'v244_c0_'+arm, probability_rule=rule, trace_observer=rule,
            event_uniforms=uniforms, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
            public_launch_delay_total_s=sum(max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
            public_early_launch_total_s=sum(max(0., r['public_baseline_s']-r['actual_s']) for r in rule.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches), **rule.dispatch_metrics)
        data = dict(rule.branch_data(), complete=True, source=source, context=0, actor=seed, demand_day=day,
            future_seed=SMOKE_SEED, results=row, tick_s=float(rule.env.time_step),
            trace=legacy.prior.common(rule, events, holds), fresh_upper=rule.fresh_queries,
            observers=dict(v224=dict(rule.handoff.data(), complete=True), v230=rule.resource.data(True),
                           v235=rule.cycle_capture.data(True)),
            observer_scope='Old recorders observe each branch;V243 requests/execution name the applied controls.')
        save(output, data)
        result = dict(slot=slot, source=source, arm=arm, reference_ready=False,
                      **path_check(data, reference, same))
        if arm == 'v107':
            result.update(archived_prefix(data, archive, legacy, same), reference_ready=True)
    except Exception as exc:
        traceback.print_exc()
        result = dict(slot=slot, source=source, arm=arm, passed=False, reference_ready=False,
            failure=type(exc).__name__+': '+str(exc), results=None, retry_authorized=False,
            observed_time_s=None if rule is None or not hasattr(rule, 'env') else float(rule.env.current_time))
        if not output.exists() and rule is not None:
            save(output, dict(complete=False, source=source, arm=arm,
                anchor=rule.fork_anchor, future=rule.future_receipt, requests=rule.issued_requests,
                failure=result['failure']))
    write(check, result)
    print('V244_PATH_CHECK '+json.dumps(result), flush=True)
    return result


def aggregate(root, source):
    checks = {arm: json.loads((root/'checks'/f'smoke_c0_{arm}.json').read_text()) for arm in ARMS}
    consumed = len(list((root/'started').glob('smoke_c0_*.json')))
    summary = dict(status='v244_smoke_stopped', source=source, paths=checks,
        started_smoke_paths=consumed, max_smoke_paths=6, reserved_starts_consumed=2+consumed,
        reserved_starts_remaining=102-consumed, formal_paths=0, formal_authorized=False,
        automatic_retry=False, training_updates=0, main_policy_changed=False)
    if all(r['passed'] and r['source'] == source for r in checks.values()):
        import run_freqduet_v230_resource_capture as resource
        legacy = resource.frozen().legacy
        fields = ('arm', 'anchor', 'future', 'physical_contract', 'ledger', 'completion',
                  'uniforms', 'execution', 'tasks', 'dispatches', 'launch_plans',
                  'requests', 'main_policy_changed')
        data = {}
        for arm in ARMS:
            saved = read(root/'records'/f'smoke_c0_{arm}.pkl.gz')
            data[arm] = {k: saved[k] for k in fields}
            del saved
        gate = adapter.check_smoke(data, legacy.pilot.joint.assert_same_state,
            legacy.pilot.event.alignment, legacy.gate.task_execution, data['v107']['tick_s'])
        write(root/'summary/engineering_gate.json', gate)
        summary['gate'] = gate
        summary['status'] = 'v244_smoke_engineering_passed_formal_closed' if gate['passed'] else 'v244_smoke_engineering_failed'
        base = checks['v107']['metrics']
        summary['descriptive_delta_vs_v107'] = {arm: {k: r['metrics'][k]-base[k] for k in METRICS}
                                              for arm, r in checks.items()}
    write(root/'summary/status.json', summary)
    print('V244_SMOKE_SUMMARY '+json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'arm', 'aggregate'))
    p.add_argument('--arm', choices=ARMS)
    for name in ('out-root', 'archive-root', 'sealed-protocol', 'v243-summary'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert json.loads(a.sealed_protocol.read_text()) == registration('e6788e6109')
    assert 'source_e7ae23d971' in adapter.__file__
    parent = json.loads(a.v243_summary.read_text())
    assert parent['source'] == 'e7ae23d971' and parent['status'] == 'v243_synthetic_six_arm_adapter_passed'
    assert parent['reserved_starts_remaining'] == 102 and not parent['empirical_six_arm_smoke_verified']
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v244_six_arm_smoke.py')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.testsRun > 0 and tested.wasSuccessful()
    if a.command == 'tests':
        return
    auth = dict(source=a.source, protocol_source='e6788e6109', adapter_source='e7ae23d971',
        slots=[s for s in reserved_slots() if s['phase'] == 'smoke'], max_smoke_paths=6,
        authorization='User approved c0/V107 first,then the other five only after reference acceptance.',
        failures_consume_slots=True, automatic_retry=False, extra_references=0,
        formal_authorized=False, training_authorized=False, main_policy_changed=False)
    auth_file = a.out_root/'execution_authorization.json'
    if a.command == 'arm' and a.arm == 'v107' and not auth_file.exists():
        write(auth_file, auth)
    else:
        assert json.loads(auth_file.read_text()) == auth
    if a.command == 'aggregate':
        aggregate(a.out_root, a.source)
    else:
        assert a.arm is not None
        result = collect(a.out_root, a.archive_root, a.arm, a.source)
        if a.arm == 'v107':
            consumed = len(list((a.out_root/'started').glob('smoke_c0_*.json')))
            write(a.out_root/'summary/status.json', dict(source=a.source,
                status='v244_reference_passed_five_pending' if result['passed'] else 'v244_reference_failed_stopped',
                paths=dict(v107=result), started_smoke_paths=consumed, max_smoke_paths=6,
                reserved_starts_consumed=2+consumed, reserved_starts_remaining=102-consumed,
                automatic_retry=False, formal_paths=0, formal_authorized=False,
                training_updates=0, main_policy_changed=False))
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
