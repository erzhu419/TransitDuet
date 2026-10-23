#!/usr/bin/env python3
"""Execute only the four reserved V225 paths, with c0 engineering admission."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import torch

import freqduet_v223_common_reference as law
from freqduet_v224_prepare_capture import validate_saved_capture
from preregister_freqduet_v224_prepare_handoff import CONTEXTS
from preregister_freqduet_v225_action_smoke import (
    FIRST_PATH_CHECKS, descriptive_joint_passed, first_path_passed, registration)
from run_freqduet_v224_prepare_handoff import RecordingAudit, legacy


def exact(a, b, label):
    assert json.loads(json.dumps(a)) == b, 'V225 '+label


def live_prefix(audit, reference, archived):
    """Read-only admission at the first difference, after choosing the live request."""
    i = len(audit.queries)-1
    q, shadow = audit.queries[-1], audit.handoff.queries[-1]
    old = archived['queries'][i]
    exact(shadow, old, 'first shadow query')
    packet = audit.handoff.by_prepare_seq[q['prepare_event_seq']]
    previous = next(p for p in archived['prepares'] if p['clock']['handoff_order'] == old['prepare_order'])
    exact(packet, previous, 'first prepared packet')
    exact(audit.clock.events, reference['events'][:len(audit.clock.events)], 'native event prefix')
    exact(audit.handoff.events, archived['events'][:len(audit.handoff.events)], 'sidecar event prefix')
    arrays = dict(lower_policy_inputs=audit.policy_inputs, upper_commands=audit.commands,
        upper_callbacks=audit.callbacks, issued_holding=audit.credit.holds,
        actual_launches=audit.dispatches, vehicle_returns=audit.returns,
        arrivals=audit.arrivals, departures=audit.departures)
    for name, values in arrays.items():
        exact(values, reference['trace'][name][:len(values)], name+' live prefix')
    for name, values in dict(cycles=audit.cycles, fresh_upper=audit.fresh_queries).items():
        exact(values, reference[name][:len(values)], name+' live prefix')
    counts = {name: len(values) for name, values in arrays.items()}
    return dict(query_index=i, event_seq=q['event_seq'], prepare_event_seq=q['prepare_event_seq'],
        time_s=q['time_s'], native_bin=q['base_bin'], selected_bin=shadow['shadow_bin'],
        prefix_counts=counts, prefix_exact=True, first_shadow_exact=True)


class JointAudit(RecordingAudit):
    def __init__(self, seed, reference, archived):
        # Never invoke V221's old joint/lower forced-cut branch.
        super().__init__(seed, 'upper')
        self.actor = seed
        self.reference, self.archived = reference, archived
        self.execution, self.first = [], None
        self.last_progress = time.monotonic()

    def __call__(self, policy, states, features, theta):
        result = super().__call__(policy, states, features, theta)
        q, row = self.queries[-1], self.handoff.queries[-1]
        with self.handoff.read('v225_apply'):
            chosen = row['shadow_bin']
            assert chosen in q['feasible_bins']
            if chosen != q['base_bin']:
                if self.first is None:
                    self.first = live_prefix(self, self.reference, self.archived)
                result = torch.zeros_like(result)
                result[0, chosen] = 1.
                q['command_s'] = law.ACTIONS[chosen]
            self.execution.append(dict(time_s=q['time_s'], trip_id=q['trip_id'], bus_id=q['bus_id'],
                station_id=q['station_id'], direction=q['direction'], event_seq=q['event_seq'],
                prepare_event_seq=q['prepare_event_seq'], prepare_order=row['prepare_order'],
                query_occurrence=row['query_occurrence'], native_bin=q['base_bin'], selected_bin=chosen,
                feasible_bins=q['feasible_bins'], executed_before_s=q['executed_holding_before_s'],
                command_s=q['command_s']))
            row['executed_shadow'] = True
        if time.monotonic()-self.last_progress >= 45.:
            print('V225_PATH_PROGRESS', json.dumps(dict(actor=self.actor,
                simulation_s=q['time_s'], queries=len(self.execution))), flush=True)
            self.last_progress = time.monotonic()
        return result


def holding_execution(data, sidecar):
    trace, commands = data['trace'], sidecar['execution']
    assert len(commands) == len(sidecar['queries']) == len(trace['lower_actions'])
    prepares = {p['clock']['legacy_prepare_event_seq']: p for p in sidecar['prepares']}
    departures = {(r['trip_id'], r['station_id'], r['direction']): r for r in trace['departures']}
    launches = {r['trip']: r for r in trace['actual_launches']}
    holds = {}
    for r in trace['issued_holding']:
        holds.setdefault(tuple(r[k] for k in ('time_s', 'bus_id', 'trip_id', 'station_id')), []).append(r)
    changed, per_trip = [], Counter()
    for q, shadow, action in zip(commands, sidecar['queries'], trace['lower_actions']):
        packet = prepares[q['prepare_event_seq']]
        assert q['prepare_order'] == packet['clock']['handoff_order'] == shadow['prepare_order']
        assert all(packet['current'][k] == q[k] for k in ('trip_id', 'bus_id', 'direction', 'station_id'))
        phases = tuple(packet['phases'][role]['phase_s'] if packet['phases'][role]['status']
            in ('supported', 'known_zero') else None for role in ('front', 'current', 'follower'))
        calculated = law.command(SimpleNamespace(**packet['context']), q['feasible_bins'], phases)
        exact(calculated, shadow['shadow'], 'live command law')
        assert shadow['executed_shadow'] is True
        assert q['selected_bin'] == shadow['shadow_bin'] == calculated['bin'] == action['action']
        assert q['command_s'] == law.ACTIONS[q['selected_bin']]
        assert (q['time_s'], q['trip_id'], q['station_id'], q['direction']) == tuple(
            action[k] for k in ('time_s', 'trip_id', 'station_id', 'direction'))
        assert action['probabilities'] == [float(i == q['selected_bin']) for i in range(7)]
        if q['selected_bin'] == q['native_bin']:
            continue
        issued = holds[tuple(q[k] for k in ('time_s', 'bus_id', 'trip_id', 'station_id'))]
        assert len(issued) == 1 and issued[0]['action_s'] == q['command_s'], 'issued command mismatch'
        dep = departures[q['trip_id'], q['station_id'], q['direction']]
        assert launches[q['trip_id']]['bus'] == q['bus_id']
        assert dep['time_s'] >= q['time_s']
        actual = dep['realized_holding']-q['executed_before_s']
        assert actual == q['command_s'], 'issued request did not execute as holding'
        changed.append(dict(q, actual_s=actual))
        per_trip[str(q['trip_id'])] += actual-law.ACTIONS[q['native_bin']]
    return dict(queries=len(commands), executed_changes=len(changed),
        increases=sum(q['selected_bin'] > q['native_bin'] for q in changed),
        decreases=sum(q['selected_bin'] < q['native_bin'] for q in changed),
        requested_changed_s=sum(q['command_s'] for q in changed),
        executed_changed_s=sum(q['actual_s'] for q in changed),
        direct_vs_same_state_native_s=sum(per_trip.values()), direct_by_trip_s=dict(per_trip))


def prefix_after_run(data, sidecar, reference):
    first = sidecar['first_change']
    if first is None:
        # Arm/label metadata differs, but every native trace and endpoint must match.
        for key in ('trace', 'events', 'results', 'ledger', 'physical_contract', 'cycles',
                    'tasks', 'responses', 'fresh_upper', 'launch_plans'):
            exact(data[key], reference[key], 'zero-change '+key)
        return
    assert first['prefix_exact'] and first['first_shadow_exact']
    i = first['query_index']
    for name in ('lower_actions', 'action_uniforms'):
        exact(data['trace'][name][:i], reference['trace'][name][:i], name+' prefix')
    # The first random variate and native model input must still match at the override.
    observed, expected = data['trace']['action_uniforms'][i], reference['trace']['action_uniforms'][i]
    expected = [expected[0], dict(expected[1], action_s=law.ACTIONS[first['selected_bin']])]
    exact(observed, expected, 'first uniform identity/value/order/time')
    expected_action = dict(reference['trace']['lower_actions'][i], action=first['selected_bin'],
        probabilities=[float(j == first['selected_bin']) for j in range(7)])
    exact(data['trace']['lower_actions'][i], expected_action, 'first native features/context')


def path_check(ci, data, sidecar, controls):
    reference = controls['upper']
    exact(data['physical_contract'], reference['physical_contract'], 'physical contract')
    recording = validate_saved_capture(sidecar, data)
    # V224's helper validates passive recording; its constant zero execution count
    # is not an execution metric for this new controller.
    recording.pop('shadow_actions_executed')
    execution = holding_execution(data, sidecar)
    prefix_after_run(data, sidecar, reference)
    authority = legacy.authority(data, controls['v107'])
    metrics = tuple(dict.fromkeys((*legacy.METRICS, *registration('schema')['reporting']['metrics'])))
    outcomes = {arm: {k: d['results'][k] for k in metrics}
                for arm, d in dict(controls, joint=data).items()}
    assert all(math.isfinite(v) for row in outcomes.values() for v in row.values())
    cleared = all(r['trips_incomplete'] == r['trips_unlaunched'] == 0 for r in outcomes.values())
    checks = dict.fromkeys(FIRST_PATH_CHECKS, True)
    checks['upper_first_admission_same_reserved_bus_advance'] = authority['passed']
    checks['all_trips_cleared'] = cleared
    engineering = (first_path_passed(checks, execution['executed_changes']) if ci == 0
                   else all(checks.values()))
    effects = {arm: {k: outcomes['joint'][k]-outcomes[arm][k] for k in metrics} for arm in controls}
    trip_holding = lambda d: {str(r['trip_id']): r['holding_s']
                             for r in d['trace']['vehicle_returns']}
    new, old = trip_holding(data), trip_holding(reference)
    assert new.keys() == old.keys()
    total = sum(new[k]-old[k] for k in new)
    return dict(context=ci, actor=CONTEXTS[ci][0], demand_day=CONTEXTS[ci][1],
        engineering_passed=engineering, checks=checks, execution=execution,
        first_change=sidecar['first_change'], upper_authority=authority,
        recording=dict(queries=recording['queries'], prepares=recording['prepares'],
            role_phase_support=recording['role_phase_support'],
            global_rng_unchanged=recording['global_rng_unchanged'],
            observer_native_call_delta=recording['observer_native_call_delta']),
        outcomes=outcomes, joint_minus=effects,
        holding_vs_upper=dict(total_trip_delta_s=total,
            direct_same_state_delta_s=execution['direct_vs_same_state_native_s'],
            native_state_response_s=total-execution['direct_vs_same_state_native_s']),
        descriptive_joint_passed=bool(engineering and descriptive_joint_passed(effects)))


def run_path(root, archive, ci):
    check = root/'checks'/f'c{ci}_joint.json'
    if check.exists():
        return json.loads(check.read_text())
    output = root/'records'/f'c{ci}_joint.json.gz'
    sidepath = root/'sidecars'/f'c{ci}_joint.json.gz'
    audits = []
    original_audit, original_credit = legacy.ReleaseAudit, legacy.pilot.CreditObserver
    try:
        controls = {arm: legacy.prior.read(archive/'records'/f'c{ci}_{arm}.json.gz')
                    for arm in ('v107', 'upper')}
        if not (output.exists() and sidepath.exists()):
            assert not (root/'progress'/f'c{ci}_joint.json').exists(), 'started slot cannot replay'
            archived = legacy.prior.read(archive/'sidecars'/f'c{ci}_upper.json.gz')
            def audit_factory(seed, arm):
                assert arm == 'joint'
                audit = JointAudit(seed, controls['upper'], archived)
                audits.append(audit)
                return audit
            def credit_factory():
                credit = original_credit()
                audits[-1].credit = credit
                return credit
            legacy.ReleaseAudit, legacy.pilot.CreditObserver = audit_factory, credit_factory
            legacy.run_item(root, None, (ci, 'joint'))
            audit = audits[0]
            legacy.passive.trace(sidepath, dict(audit.handoff.data(), complete=True,
                execution=audit.execution, first_change=audit.first))
        result = path_check(ci, legacy.prior.read(output), legacy.prior.read(sidepath), controls)
    except Exception as exc:
        if audits and not sidepath.exists():
            audit = audits[0]
            legacy.passive.trace(sidepath, dict(audit.handoff.data(), complete=False,
                execution=audit.execution, first_change=audit.first))
        result = dict(context=ci, engineering_passed=False, descriptive_joint_passed=None,
                      failure=type(exc).__name__+': '+str(exc))
    finally:
        legacy.ReleaseAudit, legacy.pilot.CreditObserver = original_audit, original_credit
    legacy.passive.write(check, result)
    print('V225_PATH_CHECK', ci, json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--sealed-protocol', type=Path, required=True)
    a = p.parse_args()
    assert json.loads(a.sealed_protocol.read_text()) == registration('e258ea546a'), 'V225 seal changed'
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern=pattern)
        for pattern in ('test_v225_*.py', 'test_v224_*.py'))
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    legacy.passive.write(a.out_root/'preflight.json', dict(tests=tests.testsRun, passed=True,
        source=os.environ['FREQDUET_EXECUTION_COMMIT'], replay_paths=0))
    if a.command == 'tests':
        return
    legacy.pilot.joint.optimized_runtime(a.out_root)
    previous = json.loads((a.archive_root/'summary/status.json').read_text())
    assert previous['status'] == 'v224_remaining_six_passed' and all(r['passed'] for r in previous['paths'])
    auth = dict(protocol_source='e258ea546a', implementation=os.environ['FREQDUET_EXECUTION_COMMIT'],
        authorization='User continued after V225 budget seal:implement live executor and c0-gated four paths.',
        max_started_paths=4, extra_reference_paths=0, max_workers=2, training_updates=0,
        new_futures=0, main_policy_changed=False)
    path = a.out_root/'execution_authorization.json'
    if path.exists():
        assert json.loads(path.read_text()) == auth
    else:
        legacy.passive.write(path, auth)
    results, start = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        for label, ids in (('first_c0', [0]), ('remaining_three', [1, 2, 3])):
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(results), total=4,
                unit='path', phase=label, elapsed_seconds=time.monotonic()-start)), flush=True)
            jobs = [pool.submit(run_path, a.out_root, a.archive_root, ci) for ci in ids]
            for future in as_completed(jobs):
                results.append(future.result())
                results.sort(key=lambda r: r['context'])
                passed = all(r['engineering_passed'] for r in results)
                status = dict(status='v225_'+label+('_passed' if passed else '_stopped'), paths=results,
                    tests=tests.testsRun, source=auth['implementation'], completed_checks=len(results),
                    started_paths=len(list((a.out_root/'progress').glob('*.json'))), max_paths=4,
                    all_engineering_passed=passed,
                    descriptive_joint_passed=(all(r['descriptive_joint_passed'] for r in results)
                                              if len(results) == 4 and passed else None),
                    training_updates=0, new_futures=0, main_policy_changed=False)
                legacy.passive.write(a.out_root/'summary/status.json', status)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(results), total=4,
                    unit='path', phase=label, elapsed_seconds=time.monotonic()-start)), flush=True)
            if not passed:
                return


if __name__ == '__main__':
    main()
