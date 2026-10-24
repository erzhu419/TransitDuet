#!/usr/bin/env python3
"""Run the two explicitly approved V254 passive slots,one manual task at a time."""

import argparse
import json
from pathlib import Path
import traceback
import unittest


COLLECTION = 'protocol_v6_v256_passive_pulse_collection'
SLOTS = dict(observer_only='passive_c0_v107', forced_native_once='noop_c0_v107')
GATES = ('six_payloads_exact', 'recording_complete', 'clock_domains_preserved',
         'native_calls_and_rng_unchanged')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def accepted(row):
    return row.get('passed') is True and all(row.get(k) is True for k in GATES)


def path_output(root, mode):
    assert root.name == COLLECTION and mode in SLOTS
    # Frozen optimized_runtime resolves its proof relative to out_root.parent.
    return root.with_name(root.name+'_'+mode)


def claim_slot(root, mode, source):
    assert root.name == COLLECTION and mode in SLOTS
    if mode == 'forced_native_once':
        first = json.loads((root/'checks/passive_c0_v107.json').read_text())
        assert first['source'] == source and first['mode'] == 'observer_only' and accepted(first)
    started = root/'started'/f'{SLOTS[mode]}.json'
    started.parent.mkdir(parents=True, exist_ok=True)
    with started.open('x') as stream:
        json.dump(dict(slot=SLOTS[mode], mode=mode, source=source, adapter_source='9f4afff2f3',
            manifest_source='eaba2513da', actor=12037, demand_day=2151001,
            sampler_seed=402112137, engine_arm='v107', automatic_retry=False), stream)


def runtime_interfaces():
    import run_freqduet_v249_prepare_preflight as frozen
    import freqduet_v255_passive_pulse as pulse
    assert 'source_96960db57a' in frozen.__file__
    assert 'source_9f4afff2f3' in pulse.__file__
    assert 'source_b83f1c5db4' in pulse.timing.__file__
    old, cycles, requests = frozen.runtime_interfaces()
    assert pulse.timing is frozen.observer
    return frozen, pulse, old, cycles, requests


def payloads(audit, complete):
    return dict(v224=dict(audit.handoff.data(), complete=complete),
        v230=audit.resource.data(complete), v235=audit.cycle_capture.data(complete),
        v241=audit.request_capture.data(complete), v248=audit.prepare_timing_capture.data(complete))


def collect(a, frozen, pulse, old, cycles, requests):
    root, mode = a.out_root, a.command
    slot = SLOTS[mode]
    check = root/'checks'/f'{slot}.json'
    if check.exists():
        saved = json.loads(check.read_text())
        assert saved['source'] == a.source and saved['mode'] == mode
        return saved
    assert not (root/'started'/f'{slot}.json').exists(), 'started slot cannot restart'
    path_root = path_output(root, mode)
    assert not path_root.exists(), 'path output already exists;inspect without restarting'
    proof = path_root.parent/'runtime_optimization_20261014'
    assert (proof/'benchmark_summary.json').is_file()
    assert (proof/'training_equivalence/summary.json').is_file()
    legacy = old.legacy
    reference = {k: legacy.prior.read(a.archive_root/folder/'c0_v107.json.gz')
                 for k,folder in pulse.PAYLOADS.items()}
    assert all(reference[k]['complete'] for k in pulse.PAYLOADS if k != 'legacy')
    base = frozen.observer.recording_audit(requests.recording_audit(cycles.recording_audit(old.RecordingAudit)))
    audit_class = pulse.recording_audit(base, mode)
    audits = []

    def factory(seed, arm):
        assert seed == 12037 and arm == 'v107'
        audit = audit_class(seed, arm)
        audits.append(audit)
        return audit

    original = legacy.ReleaseAudit
    claim_slot(root, mode, a.source)
    print('V256_STARTED '+slot, flush=True)
    result = dict(source=a.source, slot=slot, mode=mode, passed=False, automatic_retry=False)
    try:
        legacy.ReleaseAudit = factory
        legacy.run_item(path_root, a.v216_root, (0, 'v107'))
        assert len(audits) == 1
        audit = audits[0]
        for key,value in payloads(audit, True).items():
            legacy.passive.trace(path_root/pulse.PAYLOADS[key]/'c0_v107.json.gz', value)
        new_file = path_root/'pulse_sidecars/c0_v107.json.gz'
        legacy.passive.trace(new_file, audit.pulse_capture.data(True))
        recorded = {k: legacy.prior.read(path_root/folder/'c0_v107.json.gz')
                    for k,folder in pulse.PAYLOADS.items()}
        data = legacy.prior.read(new_file)
        result.update(pulse.validate_capture(reference, recorded, data))
        result.update(prepare_occurrences=len(recorded['v248']['prepares']),
            unused_prepares=len(recorded['v248']['prepares'])-
                len({q['prepare_index'] for q in recorded['v248']['queries']}),
            anchor_s=data['anchor']['query']['clock']['query_s'],
            native_bin=data['anchor']['query']['native_bin'],
            anchor_cycle=data['anchor']['query']['cycle_key'],
            prepare_H_s=data['anchor']['query']['clock']['prepare_holding_s'],
            query_H_s=data['anchor']['query']['clock']['query_holding_s'])
        result['passed'] = all(result.get(k) is True for k in GATES)
    except Exception as exc:
        traceback.print_exc()
        result['failure'] = type(exc).__name__+': '+str(exc)
        if audits:
            for key,value in payloads(audits[0], False).items():
                target = path_root/pulse.PAYLOADS[key]/'c0_v107.json.gz'
                if not target.exists():
                    legacy.passive.trace(target, value)
            new_file = path_root/'pulse_sidecars/c0_v107.json.gz'
            if not new_file.exists():
                legacy.passive.trace(new_file, audits[0].pulse_capture.data(False))
    finally:
        legacy.ReleaseAudit = original
    write(check, result)
    rows = [json.loads(p.read_text()) for p in sorted((root/'checks').glob('*.json'))]
    started = len(list((root/'started').glob('*.json')))
    status = dict(status='v256_passive_pair_passed' if len(rows) == 2 and all(accepted(r) for r in rows)
        else 'v256_'+mode+('_passed' if accepted(result) else '_stopped'),
        source=a.source, adapter_source='9f4afff2f3', manifest_source='eaba2513da', paths=rows,
        started_paths=started, max_started_paths=2, remaining_slots=2-started,
        automatic_retry=False, extra_reference_paths=0, future_starts=0,
        smoke_starts=0, formal_starts=0, training_updates=0, model_fits=0,
        main_policy_changed=False, v238_changed=False)
    write(root/'summary/status.json', status)
    print('V256_PATH_CHECK '+json.dumps(result), flush=True)
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=2,
        unit='path', phase=status['status'])), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', *SLOTS))
    for name in ('out-root','archive-root','v216-root','v254-root','v255-root'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert a.out_root.name == COLLECTION
    spec = json.loads((a.v254_root/'manifest.json').read_text())
    assert spec['source'] == 'eaba2513da' and spec['physical_start_caps']['passive'] == 2
    assert spec['preflight']['slots'] == list(SLOTS.values())
    assert a.archive_root.name == spec['preflight']['reference_collection']
    parent = json.loads((a.v255_root/'summary.json').read_text())
    assert (parent['status'],parent['source'],parent['tests']) == ('v255_passive_noop_synthetic_passed','9f4afff2f3',10)
    ref = json.loads((a.archive_root/'summary/status.json').read_text())
    assert (ref['source'],ref['status']) == ('96960db57a','v249_passive_pair_passed')
    frozen, pulse, old, cycles, requests = runtime_interfaces()
    integration = a.out_root/'integration_preflight.json'
    if a.command == 'tests':
        suite = unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromName('test_v255_passive_pulse'),
            unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                pattern='test_v256_passive_pulse.py')])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful()
        assert not integration.exists()
        write(integration, dict(source=a.source, status='v256_frozen_interfaces_and_slots_passed',
            tests=result.testsRun, simulator_starts=0, future_starts=0))
        print('V256_INTEGRATION_PREFLIGHT_PASSED', flush=True)
        return
    ready = json.loads(integration.read_text())
    assert ready['source'] == a.source and ready['status'] == 'v256_frozen_interfaces_and_slots_passed'
    auth = dict(source=a.source, manifest_source='eaba2513da', adapter_source='9f4afff2f3',
        slots=list(SLOTS.values()), max_started_paths=2, max_workers=1,
        authorization='User approved original-future c0/V107 observer-only then native no-op conditional on all exact gates.',
        failures_consume_slots=True, automatic_retry=False, extra_reference_paths=0,
        smoke_authorized=False, formal_authorized=False, training_authorized=False,
        main_policy_changed=False, v238_changed=False)
    auth_file = a.out_root/'execution_authorization.json'
    if auth_file.exists():
        assert json.loads(auth_file.read_text()) == auth
    else:
        write(auth_file, auth)
    if not accepted(collect(a, frozen, pulse, old, cycles, requests)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
