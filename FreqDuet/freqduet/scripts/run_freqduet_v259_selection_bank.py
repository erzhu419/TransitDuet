#!/usr/bin/env python3
"""Synthetic admission now; real S units require a separate sealed authorization."""

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import unittest

import freqduet_v259_selection_bank as bank


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def interfaces():
    import run_freqduet_v258_atomic_smoke as smoke
    assert 'source_eb7e4ac0dd' in smoke.__file__
    assert 'source_263593f7e8' in bank.core.__file__
    legacy,joint,smoke_type = smoke.interfaces()
    assert tuple(legacy.law.CONTEXTS) == tuple((r['actor'],r['demand_day']) for r in bank.SPEC['contexts'])
    assert legacy.pilot.joint.BASE_EP == 199
    return smoke,legacy,joint,smoke_type.__bases__[0]


def context_reference(a, unit, smoke):
    if unit['future'] > 0:
        first = bank.selection_unit(unit['context'],0)
        key = bank.unit_key(first)
        check = read(a.out_root/'checks'/key/'native.json')
        assert check['passed'] and check['source'] == a.source and check['unit'] == first
        data = smoke.load_record(a.out_root/'records'/key/'native.pkl.gz')
    elif unit['context'] == 0:
        data = smoke.load_record(a.v258_root/'records/c0_smoke_f0_native.pkl.gz')
    else:
        return None
    return dict(context=unit['context'],anchor=data['anchor'])


def collect_path(a, unit, branch, ctx):
    root,key = a.out_root,bank.unit_key(unit)
    check = root/'checks'/key/(branch+'.json')
    assert not check.exists(), 'completed paths are read-only,not restarted'
    bank.claim_slot(root,unit,branch,a.source)
    print('V259_STARTED '+key+'_'+branch,flush=True)
    result = dict(source=a.source,unit=unit,branch=branch,passed=False,retry_authorized=False)
    try:
        smoke,legacy,joint,base = interfaces()
        reference,data_native = None,None
        if branch != 'native':
            data_native = smoke.load_record(root/'records'/key/'native.pkl.gz')
            reference = {k:data_native[k] for k in ('unit','anchor','future')}
        cls = bank.selection_audit(base,joint.repeated,legacy.pilot.event,joint,unit,ctx)
        uniforms = legacy.pilot.event.EventUniforms(unit['prefix_sampler'])
        rule,credit = cls(unit['actor'],branch,uniforms,reference),legacy.pilot.CreditObserver()
        row,_,_,_ = legacy.pilot.rollout(root,unit['actor'],unit['demand_day'],'stochastic',
            unit['prefix_sampler'],[0.]*4,'v259_'+key+'_'+branch,**rule.rollout_kwargs(credit))
        row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]),holding_person_s=rule.holding_person_s,
            public_launch_delay_total_s=sum(max(0.,r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches),**rule.dispatch_metrics)
        data = dict(rule.branch_data(row),source=a.source)
        output = root/'records'/key/(branch+'.pkl.gz')
        assert not output.exists()
        smoke.save_record(output,data)
        result.update(bank.validate_path(data,unit,joint.pilot.joint.assert_same_state,data_native))
    except Exception as exc:
        traceback.print_exc()
        result['failure'] = type(exc).__name__+': '+str(exc)
    write(check,result)
    print('V259_PATH_CHECK '+json.dumps(result),flush=True)
    if not result['passed']:
        raise SystemExit(1)
    return result


def collect_unit(a):
    # This must precede imports that can construct a runner,archive reads or slot claims.
    bank.require_authorization(a.out_root,a.source)
    ready = read(a.preflight_summary)
    assert ready['source'] == a.source and ready['status'] == 'v259_selection_collector_synthetic_passed'
    unit = bank.selection_unit(a.context,a.future)
    smoke,legacy,joint,_ = interfaces()
    proof = a.out_root.parent/'runtime_optimization_20261014'
    assert (proof/'benchmark_summary.json').is_file() and (proof/'training_equivalence/summary.json').is_file()
    ctx = context_reference(a,unit,smoke)
    native = collect_path(a,unit,'native',ctx)
    for branch in native['plan']['physical'][1:]:
        collect_path(a,unit,branch,ctx)
    key = bank.unit_key(unit)
    paths = {b:smoke.load_record(a.out_root/'records'/key/(b+'.pkl.gz')) for b in native['plan']['physical']}
    result = bank.validate_unit(paths,unit,joint.pilot.joint.assert_same_state,legacy.pilot.event.alignment)
    write(a.out_root/'units'/(key+'.json'),dict(result,source=a.source))
    print('V259_UNIT_DONE '+json.dumps(dict(unit=key,physical_paths=len(paths),passed=True,
        evaluation_outcomes_opened=0,offline_choices_made=False)),flush=True)


def aggregate(a):
    bank.require_authorization(a.out_root,a.source)
    rows,missing,expected_starts = {},[],0
    for c in range(4):
        for f in range(8):
            unit = bank.selection_unit(c,f)
            key = bank.unit_key(unit)
            path = a.out_root/'units'/(key+'.json')
            if not path.exists():
                missing.append(key)
                continue
            row = read(path)
            assert row['source'] == a.source and row['unit'] == unit and row['passed']
            assert set(row['paths']) == set(row['plan']['physical'])
            markers = {p.stem for p in (a.out_root/'started'/key).glob('*.json')}
            assert markers == set(row['plan']['physical']) and len(markers) <= 7
            for branch in markers:
                check = read(a.out_root/'checks'/key/(branch+'.json'))
                assert check['passed'] and check['unit'] == unit and check['source'] == a.source
            rows[key] = row
            expected_starts += len(markers)
    starts = len(list((a.out_root/'started').glob('*/*.json')))
    assert starts <= 224
    complete = not missing and starts == expected_starts
    result = dict(source=a.source,status='v259_S_complete_pending_contrast_seal' if complete else 'v259_S_incomplete',
        complete=complete,units=rows,missing_units=missing,physical_starts=starts,physical_start_cap=224,
        offline_choices_made=False,contrasts_sealed=False,evaluation_authorized=False,
        evaluation_outcomes_opened=0,training_updates=0,main_policy_changed=False,v238_changed=False)
    write(a.out_root/'summary/status.json',result)
    return result


def synthetic(a):
    assert a.out_root.name != bank.COLLECTION
    assert not a.out_root.exists()
    smoke,legacy,joint,base = interfaces()
    event = legacy.pilot.event
    primitives = dict(SAME=joint.pilot.joint.assert_same_state,RNG=joint.pilot.joint.rng_state,
        REPEATED=joint.repeated.fork_future,EVENT=event.alignment,RESOURCE=joint.resource_snapshot)
    for name,function in primitives.items():
        os.environ['FREQDUET_V257_'+name] = inspect.getsourcefile(function)
    sys.path.insert(0,str(Path(bank.core.__file__).resolve().parents[1]/'tests'))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    # Bind every frozen unit without running a model,stepping an env or opening raw data.
    for c in range(4):
        for f in range(8):
            cls = bank.selection_audit(base,joint.repeated,event,joint,bank.selection_unit(c,f))
            assert issubclass(cls,base)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                                pattern='test_v259_selection_bank.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = dict(status='v259_selection_collector_synthetic_'+('passed' if result.wasSuccessful() else 'failed'),
        source=a.source,manifest_source='eaba2513da',smoke_source='eb7e4ac0dd',adapter_source='263593f7e8',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        selection_units=32,selection_start_cap=224,per_unit_start_cap=7,
        evaluation_units_reserved=32,evaluation_collector_implemented=False,
        runtime_primitives={k:inspect.getsourcefile(v) for k,v in primitives.items()},
        simulator_starts=0,future_starts=0,selection_starts=0,evaluation_starts=0,
        training_updates=0,raw_payloads_opened=0,model_fits=0,
        collection_authorized=False,authorization_written=False,offline_choices_made=False,
        main_policy_changed=False,v238_changed=False,
        next_gate='Separate S launch authorization only after synthetic acceptance; E remains closed until all S results and contrasts are sealed.')
    write(a.out_root/'summary.json',summary)
    print('V259_SYNTHETIC '+json.dumps(summary),flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('tests','collect_unit','aggregate'))
    for name in ('out-root','v254-root','v258-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-summary',type=Path)
    p.add_argument('--context',type=int,choices=range(4))
    p.add_argument('--future',type=int,choices=range(8))
    p.add_argument('--source',required=True)
    a = p.parse_args()
    assert read(a.v254_root/'manifest.json') == json.loads(json.dumps(bank.SPEC))
    admitted = read(a.v258_root/'summary/status.json')
    assert admitted['source'] == 'eb7e4ac0dd' and admitted['status'] == 'v258_smoke_engineering_passed_formal_closed'
    assert admitted['gate']['passed'] and admitted['started_smoke_paths'] == 8
    if a.command == 'tests':
        synthetic(a)
    elif a.command == 'aggregate':
        result = aggregate(a)
        print('V259_S_AGGREGATE '+json.dumps({k:v for k,v in result.items() if k != 'units'}),flush=True)
    else:
        assert a.context is not None and a.future is not None and a.preflight_summary is not None
        collect_unit(a)


if __name__ == '__main__':
    main()
