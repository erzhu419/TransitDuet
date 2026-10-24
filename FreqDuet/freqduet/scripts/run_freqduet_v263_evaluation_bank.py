#!/usr/bin/env python3
"""Synthetic E admission now; real paths require a separately sealed authorization."""

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import unittest

import freqduet_v263_evaluation_bank as bank


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def interfaces():
    import run_freqduet_v259_selection_bank as selection
    assert 'source_357e60fd70' in selection.__file__
    assert 'source_357e60fd70' in bank.selection.__file__
    return selection.interfaces()


def context_reference(a, unit, smoke, seal):
    s0 = bank.selection.selection_unit(unit['context'], 0)
    key = bank.selection.unit_key(s0)
    archived = seal['all_S_units'][key]
    assert archived['passed'] and archived['source'] == bank.S_SOURCE and archived['unit'] == s0
    check = read(a.s_root/'checks'/key/'native.json')
    assert check['passed'] and check['source'] == bank.S_SOURCE and check['unit'] == s0
    data = smoke.load_record(a.s_root/'records'/key/'native.pkl.gz')
    assert data['source'] == bank.S_SOURCE and data['unit'] == s0
    return dict(unit=s0, anchor=data['anchor'])


def collect_path(a, unit, branch, ctx, locks):
    key, lock = bank.unit_key(unit), locks[unit['context']]
    check = a.out_root/'checks'/key/(branch+'.json')
    assert not check.exists(), 'completed paths are read-only'
    bank.claim_slot(a.out_root, unit, branch, a.source, locks)
    print('V263_STARTED '+key+'_'+branch, flush=True)
    result = dict(source=a.source, unit=unit, branch=branch, locked_contrast=lock,
        passed=False, retry_authorized=False)
    try:
        smoke, legacy, joint, base = interfaces()
        native, reference = None, None
        if branch != 'native':
            native = smoke.load_record(a.out_root/'records'/key/'native.pkl.gz')
            reference = {k:native[k] for k in ('unit','anchor','future','locked_contrast')}
        cls = bank.evaluation_audit(base, joint.repeated, legacy.pilot.event, joint, unit, ctx, lock)
        uniforms = legacy.pilot.event.EventUniforms(unit['prefix_sampler'])
        rule, credit = cls(unit['actor'], branch, uniforms, reference), legacy.pilot.CreditObserver()
        row, _, _, _ = legacy.pilot.rollout(a.out_root, unit['actor'], unit['demand_day'],
            'stochastic', unit['prefix_sampler'], [0.]*4, 'v263_'+key+'_'+branch, **rule.rollout_kwargs(credit))
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
            public_launch_delay_total_s=sum(max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches), **rule.dispatch_metrics)
        data = dict(rule.branch_data(row), source=a.source)
        output = a.out_root/'records'/key/(branch+'.pkl.gz')
        assert not output.exists()
        smoke.save_record(output, data)
        result.update(bank.validate_path(data, unit, lock, joint.pilot.joint.assert_same_state, native))
    except Exception as exc:
        traceback.print_exc()
        result['failure'] = type(exc).__name__+': '+str(exc)
    write(check, result)
    print('V263_PATH_CHECK '+json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)
    return result


def collect_unit(a):
    # No S raw record, runtime or slot can be opened by an unauthorized E entry.
    assert a.out_root.name == bank.COLLECTION and (a.out_root/'evaluation_authorization.json').is_file()
    seal = read(a.s_seal)
    locks = bank.choices_from_seal(seal)
    bank.require_authorization(a.out_root, a.source, locks)
    ready = read(a.preflight_summary)
    assert ready['source'] == a.source and ready['status'] == 'v263_evaluation_collector_synthetic_passed'
    assert ready['locked_contrasts'] == locks and not ready['collection_authorized']
    proof = a.out_root.parent/'runtime_optimization_20261014'
    assert (proof/'benchmark_summary.json').is_file() and (proof/'training_equivalence/summary.json').is_file()
    unit = bank.evaluation_unit(a.context, a.future)
    smoke, legacy, joint, _ = interfaces()
    ctx = context_reference(a, unit, smoke, seal)
    native = collect_path(a, unit, 'native', ctx, locks)
    for branch in native['plan']['physical'][1:]:
        collect_path(a, unit, branch, ctx, locks)
    key = bank.unit_key(unit)
    paths = {b:smoke.load_record(a.out_root/'records'/key/(b+'.pkl.gz')) for b in native['plan']['physical']}
    result = bank.validate_unit(paths, unit, locks[unit['context']],
        joint.pilot.joint.assert_same_state, legacy.pilot.event.alignment)
    write(a.out_root/'units'/(key+'.json'), dict(result, source=a.source))
    print('V263_UNIT_DONE '+json.dumps(dict(unit=key, physical_paths=len(paths), passed=True,
        primary_branch=result['primary_branch'], primary_contrast_id=result['primary_contrast_id'],
        all_arms_retained=True, selection_recomputed=False)), flush=True)


def aggregate(a, locks):
    bank.require_authorization(a.out_root, a.source, locks)
    rows, missing, expected_starts = {}, [], 0
    for c in range(4):
        for f in range(8):
            unit = bank.evaluation_unit(c, f)
            key = bank.unit_key(unit)
            path = a.out_root/'units'/(key+'.json')
            if not path.exists():
                missing.append(key)
                continue
            row = read(path)
            assert row['source'] == a.source and row['unit'] == unit and row['passed']
            assert row['locked_contrast'] == locks[c] and row['plan'] == locks[c]['plan']
            assert row['primary_branch'] == locks[c]['chosen_branch'] and row['primary_contrast_id'] == locks[c]['contrast_id']
            assert row['all_arms_retained'] and not row['selection_recomputed']
            assert set(row['paths']) == set(row['plan']['physical'])
            markers = {p.stem for p in (a.out_root/'started'/key).glob('*.json')}
            assert markers == set(row['plan']['physical']) and len(markers) <= 7
            for branch in markers:
                check = read(a.out_root/'checks'/key/(branch+'.json'))
                assert check['passed'] and check['unit'] == unit and check['source'] == a.source
                assert check['locked_contrast'] == locks[c]
            rows[key] = row
            expected_starts += len(markers)
    starts = len(list((a.out_root/'started').glob('*/*.json')))
    assert starts <= 224
    complete = not missing and starts == expected_starts
    result = dict(source=a.source, status='v263_E_complete_pending_frozen_statistics' if complete else 'v263_E_incomplete',
        complete=complete, units=rows, missing_units=missing, physical_starts=starts, physical_start_cap=224,
        locked_contrasts=locks, selection_recomputed=False, training_updates=0,
        main_policy_changed=False, v238_changed=False)
    write(a.out_root/'summary/status.json', result)
    return result


def synthetic(a):
    assert a.out_root.name != bank.COLLECTION and not a.out_root.exists()
    formal = a.s_root.with_name(bank.COLLECTION)
    assert not (formal/'evaluation_authorization.json').exists() and not list((formal/'started').glob('*/*.json'))
    before = (a.s_seal.stat().st_size, a.s_seal.stat().st_mtime_ns)
    locks = bank.choices_from_seal(read(a.s_seal))
    smoke, legacy, joint, base = interfaces()
    for c in range(4):
        ctx = dict(unit=bank.selection.selection_unit(c,0), anchor=None)
        for f in range(8):
            cls = bank.evaluation_audit(base, joint.repeated, legacy.pilot.event, joint,
                bank.evaluation_unit(c,f), ctx, locks[c])
            assert issubclass(cls, base)
    primitives = dict(SAME=joint.pilot.joint.assert_same_state, RNG=joint.pilot.joint.rng_state,
        REPEATED=joint.repeated.fork_future, EVENT=legacy.pilot.event.alignment, RESOURCE=joint.resource_snapshot)
    for name, function in primitives.items():
        os.environ['FREQDUET_V257_'+name] = inspect.getsourcefile(function)
    sys.path.insert(0, str(Path(bank.core.__file__).resolve().parents[1]/'tests'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
        pattern='test_v263_evaluation_bank.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert before == (a.s_seal.stat().st_size, a.s_seal.stat().st_mtime_ns)
    assert not (formal/'evaluation_authorization.json').exists() and not list((formal/'started').glob('*/*.json'))
    summary = dict(source=a.source, status='v263_evaluation_collector_synthetic_'+('passed' if result.wasSuccessful() else 'failed'),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        S_source=bank.S_SOURCE, S_seal_source=bank.S_SEAL_SOURCE, locked_contrasts=locks,
        manifest_source='eaba2513da', adapter_source='263593f7e8',
        evaluation_units=32, evaluation_start_cap=224, per_unit_start_cap=7, max_workers=8,
        runtime_primitives={k:inspect.getsourcefile(v) for k,v in primitives.items()},
        collection_authorized=False, authorization_written=False,
        real_simulator_starts=0, real_future_starts=0, real_evaluation_starts=0,
        raw_payloads_opened=0, S_seal_metadata_unchanged=True,
        selection_recomputed=False, training_updates=0, model_fits=0,
        main_policy_changed=False, v238_changed=False,
        next_gate='Separate explicit E launch authorization only after synthetic admission; all legal bins, S-locked primary, no reselection.')
    write(a.out_root/'summary.json', summary)
    print('V263_SYNTHETIC '+json.dumps(summary), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect_unit', 'aggregate'))
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--preflight-summary', type=Path)
    p.add_argument('--context', type=int, choices=range(4))
    p.add_argument('--future', type=int, choices=range(8))
    a = p.parse_args()
    a.s_root = a.results_root/'protocol_v6_v259_selection_bank_collection'
    a.s_seal = a.results_root/'protocol_v6_v262_selection_contrasts/sealed_S.json'
    assert read(a.results_root/'protocol_v6_v254_action_manifest/seal_eaba2513da/manifest.json') == json.loads(json.dumps(bank.SPEC))
    if a.command == 'tests':
        synthetic(a)
    elif a.command == 'aggregate':
        result = aggregate(a, bank.choices_from_seal(read(a.s_seal)))
        print('V263_E_AGGREGATE '+json.dumps({k:v for k,v in result.items() if k != 'units'}), flush=True)
    else:
        assert a.context is not None and a.future is not None and a.preflight_summary is not None
        collect_unit(a)


if __name__ == '__main__':
    main()
