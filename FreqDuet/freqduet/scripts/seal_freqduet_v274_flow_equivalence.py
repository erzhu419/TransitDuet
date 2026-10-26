#!/usr/bin/env python3
"""Seal one unreleased passive path; read metadata only, never construct traffic."""

import argparse
import json
from pathlib import Path


REFERENCE = 'protocol_v6_v271_native_event_graph'
FLOW_SOURCE = '717483c80d'
OLD_SOURCE = '40eddb8fb6'


def protocol(source):
    return dict(
        protocol='V274_native_flow_passive_equivalence', source=source,
        prerequisite=dict(root='protocol_v6_v273_flow_contract', source=FLOW_SOURCE,
                          scheduler_task='t211737', tests=21),
        context=dict(actor_seed=12011, demand_day=2071001, sampler_seed=27011001,
                     control='exact360_fixed_upper_plus_V107_bilateral025',
                     physical_vehicles=12, actions_s=[0, 5, 10, 15, 20, 30, 45]),
        reference=dict(root=REFERENCE, source=OLD_SOURCE, path='paths/noop_fix.json',
                       trace='traces/noop_fix.pkl.gz',
                       graph_trace='traces/graph_queries_noop_fix.pkl.gz',
                       summary='summary/noop_fix.json',
                       reuse_without_rerun=True),
        budget=dict(slot_ids=['flow_passive_c0'], physical_start_cap=1,
                    real_starts_released=0, reference_reruns=0, max_concurrent=1,
                    charge_at_scheduler_submission=True,
                    failures_before_python_also_charged=True, automatic_retry=False,
                    replacements_authorized=0,
                    release_requires='Explicit approval after the passive entry source is sealed.'),
        execution=dict(flow_source=FLOW_SOURCE, native_rule_tensor='return the same original tensor',
                       graph_solver='unchanged V271 no-op shadow, never execute its request',
                       flow_used_by_policy=False, pending_commands='unchanged and irrevocable',
                       runtime_environment='reuse V272 runtime_env_07e565545a.json; change only source search paths',
                       selection='No action selection, fitting, future search or frequency claim.'),
        legacy_equivalence=dict(
            native_trace_exact=['holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms'],
            path_fields_exact=['results', 'contract', 'coverage', 'execution'],
            graph_trace_exact=['completed_visits', 'queries excluding build_s and total_s only'],
            graph_query_timing_exclusions=['build_s', 'total_s'],
            comparison='Existing assert_same_state; exact values/order/keys, no new tolerance or rounding.',
            counters=dict(native_queries=5200, actual_commands=5200,
                          commanded_h_s=38930, executed_h_s=38930),
            wall_clock='Report native, flow and whole-callback mean/p95/max/total; not equality fields.',
            archives='Read old archives only. Write new artifacts under a new direct results child.'),
        flow_records=dict(
            event_fields=['order', 'bus', 'trip', 'direction', 'station', 'arrival_s',
                          'available_s', 'completed_s', 'incoming', 'alighted', 'boarded', 'capacity'],
            query_fields=['query_order', 'time_s', 'bus', 'trip', 'station', 'direction',
                          'arrival_seen_count', 'usable_strict_count',
                          'max_completed_s', 'max_available_s'],
            arrival_roster='Every native exchange, including first/full/zero-seat/same-clock/terminal arrivals; contiguous native order.',
            matching='Match actual bus/trip/direction/station/time to native arrivals and dispatches; zero missing, extra or duplicate events.',
            completion='One completion per arrival, matched to actual departure or terminal return; duplicate terminal callbacks do not duplicate rows.',
            queries='One boundary row per native query, same identity/order; record the observed arrival prefix length at the callback.',
            strict_prefix='Use only that visible prefix AND arrival_s,available_s,completed_s < query_s; incomplete/unavailable visits break the station chain.',
            accounting='incoming-alighted+boarded stays within capacity; alighted and boarded obey the observed seat balance.',
            derived='Server-only completed_flow on recorded query prefixes; bounds and clearance blocks must reproduce the frozen V273 function.',
            full_service='Unfinished blocks keep bounds; full-to-clear boarding is not a point arrival label. No latent queue/destination labels.',
            tied_clocks='Keep native order; same-clock count is zero and zero-duration average rate is null.',
            unsupported='Retain every unsupported event. Naturally absent cases are not_exercised, never trigger extra paths.',
            terminal_endpoint='At service_cleared every recorded visit has one completed label; legacy first-arrival/visit counts reconcile.'),
        acceptance='All legacy equality checks AND complete flow/query/clock/accounting checks. Failure stops this slot, not a zero effect.',
        scheduling=dict(priority='high', manual_dispatch=True,
                        allowed_nodes=['node001', 'node002', 'node003', 'node004', 'node005', 'node006']),
        local_return='Only bounded acceptance JSON, mismatch field/count and latency summary; raw records stay server-side.',
        status='sealed_not_released', traffic_starts=0, training_updates=0,
        main_policy_changed=False, controller_admitted=False,
        scope='Consumed engineering context, not development/confirmation evidence or traffic benefit.')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def seal(a):
    # Missing reference artifacts stop admission before any physical slot release.
    old = a.results_root/REFERENCE
    path = read(old/'paths/noop_fix.json')
    summary = read(old/'summary/noop_fix.json')
    gate = read(a.results_root/'protocol_v6_v273_flow_contract/summary/status.json')
    assert path['source'] == summary['source'] == OLD_SOURCE
    assert (path['seed'], path['demand_day'], path['future']) == (12011, 2071001, 27011001)
    assert summary['status'] == 'noop_native_equivalence_passed'
    assert summary['native_queries'] == summary['actual_commands'] == 5200
    assert path['execution']['commanded_s'] == path['execution']['executed_s'] == 38930
    assert path['execution']['physical_vehicles'] == 12
    assert path['execution']['done_reason'] == 'service_cleared'
    assert gate['source'] == FLOW_SOURCE and gate['tests'] == 21
    assert gate['status'] == 'flow_evidence_and_recourse_contract_checks_passed'
    assert gate['traffic_paths'] == gate['training_updates'] == 0
    artifacts = [old/'paths/noop_fix.json', old/'summary/noop_fix.json',
                 old/'traces/noop_fix.pkl.gz', old/'traces/graph_queries_noop_fix.pkl.gz',
                 a.results_root/'protocol_v6_v272_graph_pilot/runtime_env_07e565545a.json']
    inventory = {}
    for p in artifacts:
        assert p.is_file() and p.stat().st_size > 0, p
        inventory[str(p.relative_to(a.results_root))] = p.stat().st_size
    assert not any((a.out_root/'started').glob('*.json'))
    sealed = protocol(a.source)
    write(a.out_root/'protocol.json', sealed)
    write(a.out_root/'checks/status.json', dict(
        source=a.source, status='passive_protocol_sealed_metadata_ready_not_released',
        reference_source=OLD_SOURCE, flow_source=FLOW_SOURCE, artifact_bytes=inventory,
        reference_native_queries=summary['native_queries'],
        reference_first_arrivals=summary['first_arrivals_without_gap'],
        reference_completed_visits=summary['completed_visits'],
        physical_start_cap=1, real_starts_released=0, physical_slots_consumed=0,
        raw_trace_reads=0, traffic_starts=0, model_fits=0, training_updates=0,
        collector_entry_implemented=False, passive_equivalence_established=False,
        main_policy_changed=False, controller_admitted=False,
        scope='Archive existence and frozen metadata only; actual trace equality remains a future passive gate.'))
    print('V274_PASSIVE_PROTOCOL_SEALED_NO_REPLAY', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    seal(p.parse_args())


if __name__ == '__main__':
    main()
