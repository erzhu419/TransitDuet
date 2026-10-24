#!/usr/bin/env python3
"""Accept S0 summaries, then run eight bounded shards of the frozen S collector."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


COLLECTOR_SOURCE = '357e60fd70'
REMAINING = tuple((c, f) for c in range(4) for f in range(1, 8))
SHARDS = tuple(REMAINING[i::8] for i in range(8))


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def audit_s0(a, bank):
    bank.require_authorization(a.collection, COLLECTOR_SOURCE)
    assert sorted(x for shard in SHARDS for x in shard) == list(REMAINING)
    assert len(SHARDS) == 8 and max(map(len, SHARDS)) == 4
    expected, rows = set(), []
    for c in range(4):
        unit = bank.selection_unit(c, 0)
        key = bank.unit_key(unit)
        row = read(a.collection/'units'/(key+'.json'))
        assert row['source'] == COLLECTOR_SOURCE and row['unit'] == unit and row['passed']
        assert not row['offline_choices_made'] and row['evaluation_outcomes_opened'] == 0
        native = read(a.collection/'checks'/key/'native.json')
        plan = native['plan']
        assert row['plan'] == plan and set(row['paths']) == set(plan['physical'])
        assert len(plan['physical']) <= 7 and 'forced_native' not in plan['physical']
        assert not set(plan['physical']) & (set(plan['aliases']) | set(plan['masked']))
        assert {p.stem for p in (a.collection/'checks'/key).glob('*.json')} == set(plan['physical'])
        assert {p.stem for p in (a.collection/'started'/key).glob('*.json')} == set(plan['physical'])
        for branch, path in row['paths'].items():
            check = read(a.collection/'checks'/key/(branch+'.json'))
            marker = read(a.collection/'started'/key/(branch+'.json'))
            assert marker == dict(source=COLLECTOR_SOURCE, unit=unit, branch=branch, retry_authorized=False)
            assert check['source'] == COLLECTOR_SOURCE and check['unit'] == unit
            assert check['branch'] == branch and check['passed'] and not check['retry_authorized']
            for name in ('passed', 'plan', 'actual_anchor_holding_s', 'actual_execution_complete', 'queries', 'metrics'):
                assert check[name] == path[name]
            assert path['actual_execution_complete']
            assert path['metrics']['trips_unlaunched'] == path['metrics']['trips_incomplete'] == 0
            coupling = path['coupling']
            assert coupling['uniform_mismatches'] == 0
            assert coupling['common_events'] == path['queries'] == row['paths']['native']['queries']
            assert coupling['native_only'] == coupling['alternative_only'] == 0
            if branch != 'native':
                assert path['actual_anchor_holding_s'] == bank.core.ACTIONS[int(branch[1:])]
            expected.add(key+'/'+branch+'.json')
        rows.append(dict(context=c, paths=len(row['paths']), plan=plan,
            queries_per_path=native['queries'], common_uniform_mismatches=0,
            collector_prefix_and_future_checks_passed=True, actual_execution_complete=True,
            actual_holding_s={b:p['actual_anchor_holding_s'] for b,p in row['paths'].items()}))
    actual = {str(p.relative_to(a.collection/'started')) for p in (a.collection/'started').glob('*/*.json')}
    assert actual == expected, 'unexpected starts outside the accepted S0 wave'
    result = dict(source=a.source, collector_source=COLLECTOR_SOURCE,
        status='v261_S0_admitted_remaining28', passed=True, contexts=rows,
        physical_starts=len(actual), failed_starts=0, physical_start_cap=224,
        remaining_units=28, shards=SHARDS, max_workers=8,
        evidence='Source-bound unit/check/slot summaries; prefix and future equality enforced at fork by the frozen collector. No raw trace rescan.',
        evaluation_authorized=False, offline_choices_made=False, training_updates=0,
        main_policy_changed=False, v238_changed=False)
    assert not a.admission.exists(), 'preserve the first admission receipt'
    write(a.admission, result)
    print('V261_S0_ADMITTED '+json.dumps(result), flush=True)


def collect_shard(a, bank, collector):
    gate = read(a.admission)
    assert gate['passed'] and gate['source'] == a.source and gate['collector_source'] == COLLECTOR_SOURCE
    assert gate['status'] == 'v261_S0_admitted_remaining28' and gate['shards'] == json.loads(json.dumps(SHARDS))
    bank.require_authorization(a.collection, COLLECTOR_SOURCE)
    units = [bank.selection_unit(c, f) for c, f in SHARDS[a.shard]]
    for unit in units:
        assert not list((a.collection/'started'/bank.unit_key(unit)).glob('*.json')), 'no shard restart'
    receipt = a.admission.parent.parent/'shards'/('shard%d.json'%a.shard)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source=a.source, collector_source=COLLECTOR_SOURCE, shard=a.shard,
        planned_units=[bank.unit_key(u) for u in units], units=[], status='running',
        automatic_retry=False, evaluation_authorized=False, training_updates=0)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    for unit in units:
        command = [sys.executable, '-u', str(collector/'scripts/run_freqduet_v259_selection_bank.py'),
            'collect_unit', '--source', COLLECTOR_SOURCE, '--out-root', str(a.collection),
            '--v254-root', str(a.results_root/'protocol_v6_v254_action_manifest/seal_eaba2513da'),
            '--v258-root', str(a.results_root/'protocol_v6_v258_atomic_smoke_collection'),
            '--preflight-summary', str(a.results_root/'protocol_v6_v259_selection_bank/synthetic_357e60fd70/summary.json'),
            '--context', str(unit['context']), '--future', str(unit['future'])]
        # Independent units continue after a failed child; the failed unit is never retried.
        rc = subprocess.run(command, cwd=collector).returncode
        path = a.collection/'units'/(bank.unit_key(unit)+'.json')
        saved = read(path) if path.exists() else {}
        passed = rc == 0 and saved.get('passed') and saved.get('source') == COLLECTOR_SOURCE and saved.get('unit') == unit
        result['units'].append(dict(unit=bank.unit_key(unit), returncode=rc, passed=bool(passed)))
        write(receipt, result)
        print('V261_UNIT_RECEIPT '+json.dumps(result['units'][-1]), flush=True)
    result['status'] = 'done' if all(r['passed'] for r in result['units']) else 'failed_units_retained'
    write(receipt, result)
    print('V261_SHARD_DONE '+json.dumps(result), flush=True)
    if result['status'] != 'done':
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit_s0', 'collect_shard'))
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--shard', type=int, choices=range(8))
    a = p.parse_args()
    a.collection = a.results_root/'protocol_v6_v259_selection_bank_collection'
    a.admission = a.results_root/'protocol_v6_v261_selection_remaining/summary/s0_admission.json'
    collector = a.results_root/'protocol_v6_v259_selection_bank/source_357e60fd70/FreqDuet/freqduet'
    sys.path.insert(0, str(collector/'scripts'))
    import freqduet_v259_selection_bank as bank
    if a.command == 'audit_s0':
        audit_s0(a, bank)
    else:
        assert a.shard is not None
        collect_shard(a, bank, collector)


if __name__ == '__main__':
    main()
