#!/usr/bin/env python3
"""Aggregate all frozen S units and seal their finite-context offline contrasts."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import unittest


COLLECTOR_SOURCE = '357e60fd70'
ORCHESTRATOR_SOURCE = 'a6762c34e3'


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def mean(values):
    assert values and all(math.isfinite(v) for v in values)
    return math.fsum(values)/len(values)


def lock_context(context, rows, bank, constraints):
    assert len(rows) == 8
    assert [r['unit'] for r in rows] == [bank.selection_unit(context, f) for f in range(8)]
    plan = rows[0]['plan']
    assert all(r['plan'] == plan and set(r['paths']) == set(plan['physical']) for r in rows)
    native_h = rows[0]['paths']['native']['actual_anchor_holding_s']
    native_bin = bank.core.ACTIONS.index(native_h)
    arms = {}
    for branch in plan['physical']:
        paths = [r['paths'][branch] for r in rows]
        dose = paths[0]['actual_anchor_holding_s']
        assert all(p['actual_anchor_holding_s'] == dose for p in paths)
        index = bank.core.ACTIONS.index(dose)
        assert branch == 'native' or branch == 'b'+str(index)
        deltas = {k:[p['delta_vs_native'][k] for p in paths] for k in bank.METRICS}
        means = {k:mean(v) for k,v in deltas.items()}
        eligible = branch != 'native' and means['passenger_person_s'] < 0 and all(means[k] <= 0 for k in constraints)
        arms[branch] = dict(bin=index, holding_s=dose, eligible=eligible, mean_delta=means,
            passenger_gain_futures=sum(v < 0 for v in deltas['passenger_person_s']),
            joint_gain_futures=sum(p['delta_vs_native']['passenger_person_s'] < 0 and
                all(p['delta_vs_native'][k] <= 0 for k in constraints) for p in paths),
            passenger_delta_sd=statistics.stdev(deltas['passenger_person_s']))
    candidates = [b for b in plan['physical'] if arms[b]['eligible']]
    chosen = min(candidates, key=lambda b:(arms[b]['mean_delta']['passenger_person_s'],
        abs(arms[b]['holding_s']-native_h), arms[b]['bin'])) if candidates else 'native'
    selected = [r['paths'][chosen] for r in rows]
    holding = {k:mean([p['holding'][k] for p in selected]) for k in (
        'direct_executed_delta_s', 'later_same_cycle_delta_s', 'other_cycles_delta_s', 'all_cycles_delta_s')}
    assert math.isclose(holding['direct_executed_delta_s']+holding['later_same_cycle_delta_s']+
        holding['other_cycles_delta_s'], holding['all_cycles_delta_s'], rel_tol=0., abs_tol=1e-6)
    supply = {}
    for k in ('return_delta_s', 'first_next_delta_s', 'return_to_next_wait_delta_s', 'public_delay_delta_s'):
        values = [p['supply'][k] for p in selected if p['supply'].get(k) is not None]
        supply[k] = dict(n=len(values), mean=mean(values) if values else None)
    return dict(context=context, actor=rows[0]['unit']['actor'], demand_day=rows[0]['unit']['demand_day'],
        native_bin=native_bin, native_holding_s=native_h, plan=plan, chosen_branch=chosen,
        chosen_bin=arms[chosen]['bin'], chosen_holding_s=arms[chosen]['holding_s'],
        contrast_id=f'c{context}:{chosen}_vs_native_b{native_bin}',
        eligible_branches=candidates, arms=arms, selected_holding_mean=holding,
        selected_supply_status_counts=dict(Counter(p['supply']['status'] for p in selected)),
        selected_supply=supply,
        selected_query_due_matches=sum(p['supply'].get('query_due_obligation_match') is True for p in selected))


def seal(a, bank, constraints, collector):
    out = a.results_root/'protocol_v6_v262_selection_contrasts'
    assert not (out/'sealed_S.json').exists() and not (out/'summary.json').exists()
    collection = a.results_root/'protocol_v6_v259_selection_bank_collection'
    command = [sys.executable, '-u', str(collector/'scripts/run_freqduet_v259_selection_bank.py'),
        'aggregate', '--source', COLLECTOR_SOURCE, '--out-root', str(collection),
        '--v254-root', str(a.results_root/'protocol_v6_v254_action_manifest/seal_eaba2513da'),
        '--v258-root', str(a.results_root/'protocol_v6_v258_atomic_smoke_collection')]
    subprocess.run(command, cwd=collector, check=True)
    aggregate = read(collection/'summary/status.json')
    assert aggregate['source'] == COLLECTOR_SOURCE and aggregate['complete']
    assert not aggregate['missing_units'] and aggregate['physical_starts'] <= 224
    assert not aggregate['evaluation_authorized'] and aggregate['evaluation_outcomes_opened'] == 0
    assert not aggregate['training_updates'] and not aggregate['main_policy_changed'] and not aggregate['v238_changed']
    shard_root = a.results_root/'protocol_v6_v261_selection_remaining'
    admission = read(shard_root/'summary/s0_admission.json')
    assert admission['passed'] and admission['source'] == ORCHESTRATOR_SOURCE
    for i in range(8):
        shard = read(shard_root/'shards'/('shard%d.json'%i))
        assert shard['source'] == ORCHESTRATOR_SOURCE and shard['collector_source'] == COLLECTOR_SOURCE
        assert shard['status'] == 'done' and shard['shard'] == i
        expected = [bank.unit_key(bank.selection_unit(c, f)) for c,f in admission['shards'][i]]
        assert shard['planned_units'] == expected and [u['unit'] for u in shard['units']] == expected
        assert all(u['passed'] and u['returncode'] == 0 for u in shard['units'])
        assert not shard['automatic_retry'] and not shard['evaluation_authorized'] and not shard['training_updates']
    starts, queries, common_events = 0, [], []
    for row in aggregate['units'].values():
        starts += len(row['paths'])
        for path in row['paths'].values():
            assert path['passed'] and path['actual_execution_complete']
            assert path['coupling']['uniform_mismatches'] == 0
            assert path['coupling']['native_only'] == path['coupling']['alternative_only'] == 0
            assert path['coupling']['common_events'] == path['queries']
            assert path['metrics']['trips_unlaunched'] == path['metrics']['trips_incomplete'] == 0
            queries.append(path['queries']); common_events.append(path['coupling']['common_events'])
    assert starts == aggregate['physical_starts']
    contrasts = [lock_context(c, [aggregate['units'][bank.unit_key(bank.selection_unit(c, f))]
        for f in range(8)], bank, constraints) for c in range(4)]
    summary = dict(source=a.source, collector_source=COLLECTOR_SOURCE,
        protocol_source='4667fd4668', manifest_source='eaba2513da',
        status='v262_all_S_complete_contrasts_sealed_E_closed',
        complete_units=32, physical_starts=starts, failed_starts=0, physical_start_cap=224,
        queries_per_path_range=[min(queries), max(queries)],
        common_events_per_path_range=[min(common_events), max(common_events)],
        uniform_mismatches=0, actual_execution_complete=True,
        contrasts=contrasts, constraints=list(constraints),
        selected_context_mean_delta={k:mean([r['arms'][r['chosen_branch']]['mean_delta'][k]
            for r in contrasts]) for k in bank.METRICS},
        contrasts_sealed=True, offline_choices_made=True, online_selector_defined=False,
        evaluation_authorized=False, evaluation_outcomes_opened=0, training_updates=0,
        new_simulator_starts=0, main_policy_changed=False, v238_changed=False,
        limitations='S-selected descriptive effects on four consumed actor/day contexts; no independent-future confirmation or deployable policy claim.',
        next_gate='Implement and synthetically check the frozen all-command E collector; real E paths require separate authorization and no reselection.')
    # The full S results and masks stay together with the locked IDs on the server.
    write_new(out/'sealed_S.json', dict(summary=summary, all_S_units=aggregate['units'],
        selection_authorization=read(collection/'selection_authorization.json')))
    write_new(out/'summary.json', summary)
    small = {k:v for k,v in summary.items() if k != 'contrasts'}
    small['choices'] = [dict(context=r['context'], native_holding_s=r['native_holding_s'],
        chosen_holding_s=r['chosen_holding_s'], eligible_branches=r['eligible_branches'],
        mean_delta=r['arms'][r['chosen_branch']]['mean_delta']) for r in contrasts]
    print('V262_S_SEALED '+json.dumps(small), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    collector = a.results_root/'protocol_v6_v259_selection_bank/source_357e60fd70/FreqDuet/freqduet'
    sys.path.insert(0, str(collector/'scripts'))
    import freqduet_v259_selection_bank as bank
    from preregister_freqduet_v253_action_supply import CONSTRAINTS
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
        pattern='test_v262_selection_contrasts.py')
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    seal(a, bank, CONSTRAINTS, collector)


if __name__ == '__main__':
    main()
