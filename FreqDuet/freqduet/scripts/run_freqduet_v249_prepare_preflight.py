#!/usr/bin/env python3
"""Execute V247's two fixed passive slots, one explicitly dispatched arm at a time."""

import argparse
import json
from pathlib import Path
import traceback
import unittest

import freqduet_v248_prepare_timing_capture as observer
from preregister_freqduet_v247_prepare_timing import OLD_PAYLOADS, registration


COLLECTION = 'protocol_v6_v249_prepare_timing_preflight_collection'
GATES = ('all_five_old_payloads_exact', 'recording_complete',
         'clock_domains_separated', 'observer_rng_and_calls_unchanged')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def accepted(row):
    return row.get('passed') is True and all(row.get(k) is True for k in GATES)


def claim_slot(root, arm, source):
    assert root.name == COLLECTION and arm in ('v107', 'upper')
    if arm == 'upper':
        first = json.loads((root/'checks/c0_v107.json').read_text())
        assert first['source'] == source and first['arm'] == 'v107' and accepted(first)
    path = root/'started'/f'c0_{arm}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(dict(slot='c0_'+arm, arm=arm, source=source, protocol_source='e74d381e62',
            observer_source='b83f1c5db4', actor=12037, demand_day=2151001,
            sampler_seed=402112137, retry_authorized=False), stream)


def runtime_interfaces():
    import run_freqduet_v230_resource_capture as resource
    import freqduet_v235_passive_cycle_capture as cycles
    import freqduet_v241_passive_requests as requests
    assert 'source_eec7a20c71' in resource.__file__
    assert 'source_6c3397ff7f' in cycles.__file__
    assert 'source_7fdf968a5f' in cycles.frozen.__file__
    assert 'source_fcb5efdcf0' in requests.__file__
    assert 'source_b4b1d00203' in __import__(requests.shadow_request.__module__).__file__
    assert 'source_b83f1c5db4' in observer.__file__
    old = resource.frozen()
    assert old.legacy.law.CONTEXTS[0] == (12037, 2151001)
    return old, cycles, requests


def collect(a, old, cycles, requests):
    root, arm = a.out_root, a.arm
    check = root/'checks'/f'c0_{arm}.json'
    if check.exists():
        saved = json.loads(check.read_text())
        assert saved['source'] == a.source and saved['arm'] == arm
        return saved
    assert not (root/'started'/f'c0_{arm}.json').exists(), 'started slot cannot restart'
    legacy = old.legacy
    files = {k: root/folder/f'c0_{arm}.json.gz' for k, folder in OLD_PAYLOADS.items()}
    files['v248'] = root/'prepare_timing_sidecars'/f'c0_{arm}.json.gz'
    assert not any(p.exists() for p in files.values())
    assert not (root/'progress'/f'c0_{arm}.json').exists()
    reference = {k: legacy.prior.read(a.archive_root/folder/f'c0_{arm}.json.gz')
                 for k, folder in OLD_PAYLOADS.items()}
    assert all(reference[k]['complete'] for k in OLD_PAYLOADS if k != 'legacy')
    audit_class = observer.recording_audit(requests.recording_audit(cycles.recording_audit(old.RecordingAudit)))
    audits = []

    def factory(seed, name):
        assert seed == 12037 and name == arm
        audit = audit_class(seed, name)
        audits.append(audit)
        return audit

    original = legacy.ReleaseAudit
    claim_slot(root, arm, a.source)
    print('V249_STARTED c0_'+arm, flush=True)
    result = dict(arm=arm, source=a.source, passed=False, retry_authorized=False)
    try:
        legacy.ReleaseAudit = factory
        legacy.run_item(root, a.v216_root, (0, arm))
        assert len(audits) == 1
        audit = audits[0]
        payloads = dict(v224=dict(audit.handoff.data(), complete=True),
            v230=audit.resource.data(True), v235=audit.cycle_capture.data(True),
            v241=audit.request_capture.data(True), v248=audit.prepare_timing_capture.data(True))
        for name, value in payloads.items():
            legacy.passive.trace(files[name], value)
        recorded = {k: legacy.prior.read(files[k]) for k in OLD_PAYLOADS}
        data = legacy.prior.read(files['v248'])
        result.update(observer.validate_capture(reference, recorded, data), recording_complete=True)
        result.update(prepare_to_query_h_increases=sum(
            q['join']['query_holding_s'] > q['join']['prepare_holding_s'] for q in data['queries']),
            same_cycle_same_front_transitions=sum(
                p['change']['status'] == 'same_physical_cycle_and_front' for p in data['prepares']))
        result['passed'] = all(result.get(k) is True for k in GATES)
    except Exception as exc:
        traceback.print_exc()
        result['failure'] = type(exc).__name__+': '+str(exc)
        if audits:
            audit = audits[0]
            for name, value in (
                ('v224', dict(audit.handoff.data(), complete=False)),
                ('v230', audit.resource.data(False)), ('v235', audit.cycle_capture.data(False)),
                ('v241', audit.request_capture.data(False)), ('v248', audit.prepare_timing_capture.data(False))):
                if not files[name].exists():
                    legacy.passive.trace(files[name], value)
    finally:
        legacy.ReleaseAudit = original
    write(check, result)
    rows = [json.loads(p.read_text()) for p in sorted((root/'checks').glob('c0_*.json'))]
    started = len(list((root/'started').glob('*.json')))
    status = dict(status='v249_passive_pair_passed' if len(rows) == 2 and all(accepted(r) for r in rows)
        else 'v249_'+arm+('_passed' if accepted(result) else '_stopped'), source=a.source,
        protocol_source='e74d381e62', observer_source='b83f1c5db4', paths=rows,
        checked_paths=len(rows), started_paths=started, max_started_paths=2,
        remaining_slots=2-started, automatic_retry=False, extra_reference_paths=0,
        new_futures=0, intervention_paths=0, training_updates=0, shadow_actions_executed=0,
        main_policy_changed=False, v238_changed=False)
    write(root/'summary/status.json', status)
    print('V249_PATH_CHECK '+json.dumps(result), flush=True)
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=2, unit='path',
        phase=status['status'])), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests','collect'))
    for name in ('out-root','archive-root','v216-root','v247-root','v248-root'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--source', required=True)
    p.add_argument('--arm', choices=('v107','upper'))
    a = p.parse_args()
    assert a.out_root.name == COLLECTION
    assert json.loads((a.v247_root/'protocol.json').read_text()) == registration('e74d381e62')
    parent = json.loads((a.v248_root/'status.json').read_text())
    assert parent['source'] == 'b83f1c5db4' and parent['status'] == 'v248_independent_prepare_observer_synthetic_passed'
    assert parent['tests_passed'] == 25 and parent['passive_paths'] == 0
    ref = json.loads((a.archive_root/'summary/status.json').read_text())
    assert (ref['source'], ref['status']) == ('c6710744f9', 'v242_passive_pair_passed')
    old, cycles, requests = runtime_interfaces()
    integration = a.out_root/'integration_preflight.json'
    if a.command == 'tests':
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                                  pattern='test_v249_prepare_preflight.py')
        assert suite.countTestCases() > 0
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful()
        assert not integration.exists()
        write(integration, dict(status='v249_interfaces_and_slot_tests_passed', source=a.source,
            tests_passed=result.testsRun, passive_paths=0, observer_source='b83f1c5db4'))
        print('V249_INTEGRATION_PREFLIGHT_PASSED', flush=True)
        return
    assert a.arm is not None
    ready = json.loads(integration.read_text())
    assert ready['source'] == a.source and ready['status'] == 'v249_interfaces_and_slot_tests_passed'
    auth = dict(source=a.source, protocol_source='e74d381e62', observer_source='b83f1c5db4',
        max_started_paths=2, slots=['c0_v107','c0_upper'], max_workers=1,
        authorization='User approved next step:c0/V107 then upper only after all exact gates.',
        failures_consume_slots=True, automatic_retry=False, extra_reference_paths=0,
        intervention_authorized=False, training_authorized=False, main_policy_changed=False)
    path = a.out_root/'execution_authorization.json'
    if path.exists():
        assert json.loads(path.read_text()) == auth
    else:
        write(path, auth)
    if not accepted(collect(a, old, cycles, requests)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
