#!/usr/bin/env python3
"""Run only the two authorized V240 passive slots, sequentially and without retries."""

import argparse
import json
from pathlib import Path
import time
import traceback
import unittest

import freqduet_v235_passive_cycle_capture as cycle
import freqduet_v241_passive_requests as requests
from preregister_freqduet_v234_passive_cycle_preflight import PAIR_CHECKS
from preregister_freqduet_v240_paired_coordination import registration, reserved_slots


GATES = (*PAIR_CHECKS, 'original_payloads_exact', 'v235_sidecar_exact',
    'request_occurrences_complete', 'prepared_clocks_preserved', 'frozen_requests_exact',
    'H0_aliases_native', 'observer_rng_and_calls_unchanged')
ARMS = {'v107': 'v107', 'upper_only': 'upper'}
FOLDERS = dict(legacy='records', v224='sidecars', v230='resource_sidecars', v235='cycle_sidecars')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def accepted(row):
    return row.get('passed') is True and all(row.get(k) is True for k in GATES)


def claim_slot(root, arm, source):
    assert arm in ARMS, 'only the two frozen passive slots are authorized'
    if arm == 'upper_only':
        prior = json.loads((root/'checks/passive_c0_v107.json').read_text())
        assert prior['slot'] == 'passive_c0_v107' and prior['source'] == source and accepted(prior)
    slot = 'passive_c0_'+arm
    marker = root/'started'/f'{slot}.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x') as stream:
        json.dump(dict(slot=slot, context=0, engine_arm=ARMS[arm], source=source,
            protocol_source='e6788e6109', started=True, retry_authorized=False), stream)


def collect(root, archive, v216, arm, source):
    slot, engine = 'passive_c0_'+arm, ARMS[arm]
    check = root/'checks'/f'{slot}.json'
    if check.exists():
        result = json.loads(check.read_text())
        assert result['slot'] == slot and result['source'] == source
        return result
    assert not (root/'started'/f'{slot}.json').exists(), 'consumed path cannot restart'
    import run_freqduet_v230_resource_capture as v230
    assert 'source_eec7a20c71' in v230.__file__
    old = v230.frozen()
    legacy, original = old.legacy, old.legacy.ReleaseAudit
    assert legacy.law.CONTEXTS[0] == (12037, 2151001)
    reference = {k: legacy.prior.read(archive/folder/f'c0_{engine}.json.gz')
                 for k, folder in FOLDERS.items()}
    assert all(reference[k]['complete'] for k in ('v224', 'v230', 'v235'))
    files = {k: root/folder/f'c0_{engine}.json.gz' for k, folder in FOLDERS.items()}
    files['requests'] = root/'request_sidecars'/f'c0_{engine}.json.gz'
    assert not any(p.exists() for p in files.values()), 'existing path outputs require read-only recovery'
    assert not (root/'progress'/f'c0_{engine}.json').exists()
    audits = []
    audit_class = requests.recording_audit(cycle.recording_audit(old.RecordingAudit))

    def factory(seed, name):
        assert seed == 12037 and name == engine
        audit = audit_class(seed, name)
        audits.append(audit)
        return audit

    claim_slot(root, arm, source)
    print('V242_STARTED '+slot, flush=True)
    try:
        legacy.ReleaseAudit = factory
        legacy.run_item(root, v216, (0, engine))
        assert len(audits) == 1
        audit = audits[0]
        payloads = dict(v224=dict(audit.handoff.data(), complete=True),
            v230=audit.resource.data(True), v235=audit.cycle_capture.data(True),
            requests=audit.request_capture.data(True))
        for name, payload in payloads.items():
            legacy.passive.trace(files[name], payload)
        recorded = {k: legacy.prior.read(files[k]) for k in FOLDERS}
        # Full old-payload equality plus independent V232 ledger reconstruction.
        old_three = {k: recorded[k] for k in ('legacy', 'v224', 'v230')}
        ref_three = {k: reference[k] for k in old_three}
        result = dict(slot=slot, context=0, arm=arm, engine_arm=engine, source=source,
            **cycle.validate_capture(ref_three, old_three, recorded['v235']))
        result.update(requests.validate_capture(reference, recorded, legacy.prior.read(files['requests'])))
        packet_rows = payloads['requests']['queries']
        result.update(queries=len(packet_rows),
            supply_changed_requests=sum(q['supply_request']['shadow_bin'] != q['native_bin'] for q in packet_rows),
            blind_changed_requests=sum(q['blind_request']['shadow_bin'] != q['native_bin'] for q in packet_rows),
            same_state_supply_blind_differences=sum(q['supply_request']['shadow_bin'] != q['blind_request']['shadow_bin']
                                                     for q in packet_rows),
            shadow_actions_executed=0, main_policy_changed=False)
        result['passed'] = all(result.get(k) is True for k in GATES)
    except Exception as exc:
        traceback.print_exc()
        result = dict(slot=slot, context=0, arm=arm, engine_arm=engine, source=source,
            passed=False, failure=type(exc).__name__+': '+str(exc), retry_authorized=False)
        if audits and audits[0].resource.metadata is not None:
            audit = audits[0]
            for name, payload in (
                ('v224', dict(audit.handoff.data(), complete=False)),
                ('v230', audit.resource.data(False)), ('v235', audit.cycle_capture.data(False)),
                ('requests', audit.request_capture.data(False))):
                if not files[name].exists():
                    legacy.passive.trace(files[name], payload)
    finally:
        legacy.ReleaseAudit = original
    write(check, result)
    print('V242_PATH_CHECK '+json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect'))
    for name in ('out-root', 'archive-root', 'v216-root', 'sealed-protocol', 'v241-summary'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert json.loads(a.sealed_protocol.read_text()) == registration('e6788e6109')
    assert 'source_6c3397ff7f' in cycle.__file__
    assert 'source_7fdf968a5f' in cycle.frozen.__file__
    assert 'source_fcb5efdcf0' in requests.__file__
    assert 'source_b4b1d00203' in __import__(requests.shadow_request.__module__).__file__
    assert 'source_e6788e6109' in __import__(registration.__module__).__file__
    parent = json.loads(a.v241_summary.read_text())
    assert parent['status'] == 'v241_synthetic_passive_requests_passed' and parent['source'] == 'fcb5efdcf0'
    ref_status = json.loads((a.archive_root/'summary/status.json').read_text())
    assert ref_status['status'] == 'v236_passive_pair_passed' and ref_status['source'] == '6972500d67'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v242_passive_requests.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    if a.command == 'tests':
        print('V242_SYNTHETIC_SLOT_GATES_PASSED', flush=True)
        return
    auth = dict(source=a.source, protocol_source='e6788e6109', observer_source='fcb5efdcf0',
        slots=[s for s in reserved_slots() if s['phase'] == 'passive'], max_started_paths=2,
        authorization='User approved V107 first,upper-only only after ALL equality and recording gates.',
        reference_source='6972500d67', reference_root=str(a.archive_root),
        failures_consume_slots=True, automatic_retry=False, max_workers=1,
        smoke_authorized=False, formal_authorized=False, training_authorized=False,
        shadow_execution_authorized=False, main_policy_changed=False)
    auth_file = a.out_root/'execution_authorization.json'
    if auth_file.exists():
        assert json.loads(auth_file.read_text()) == auth
    else:
        write(auth_file, auth)
    start, rows = time.monotonic(), []
    for arm in ARMS:
        row = collect(a.out_root, a.archive_root, a.v216_root, arm, a.source)
        rows.append(row)
        consumed = len(list((a.out_root/'started').glob('*.json')))
        summary = dict(status='v242_'+arm+('_passed' if accepted(row) else '_stopped'),
            source=a.source, protocol_source='e6788e6109', observer_source='fcb5efdcf0',
            tests=tested.testsRun, paths=rows, checked_paths=len(rows),
            started_paths=consumed, max_started_paths=2, reserved_starts_consumed=consumed,
            reserved_starts_remaining=104-consumed, extra_reference_paths=0,
            smoke_paths=0, formal_paths=0, new_futures=0, training_updates=0,
            shadow_actions_executed=0, main_policy_changed=False,
            automatic_retry=False, smoke_authorized=False, formal_authorized=False)
        if len(rows) == 2 and all(accepted(r) for r in rows):
            summary['status'] = 'v242_passive_pair_passed'
        write(a.out_root/'summary/status.json', summary)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=2, unit='path',
            phase=summary['status'], elapsed_seconds=time.monotonic()-start)), flush=True)
        if not accepted(row):
            raise SystemExit(1)
    print('V242_PASSIVE_PAIR_COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
