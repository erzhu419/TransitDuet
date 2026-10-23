#!/usr/bin/env python3
"""V234-authorized c0 V107, then upper only after every passive gate passes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import traceback
import unittest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def claim_slot(root, arm, gate_fields):
    assert arm in ('v107', 'upper'), 'only the two fixed c0 slots are authorized'
    if arm == 'upper':
        prior = json.loads((root/'checks/c0_v107.json').read_text())
        assert prior['slot'] == 'c0_v107' and prior['passed'] is True
        assert all(prior.get(k) is True for k in gate_fields), 'V107 gate incomplete'
    marker = root/'started'/f'c0_{arm}.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Real scheduler duplicate launches must not restart a consumed simulator slot.
    with marker.open('x') as stream:
        json.dump(dict(slot='c0_'+arm, context=0, started=True, retry_authorized=False), stream)


def ledger_summary(capture):
    counts, previous = Counter(), {}
    for row in capture['queries']:
        v = row['view']
        key = tuple(v['cycle_key'])
        p = v['own_association']['obligation']
        counts['unassociated' if p is None else 'associated_'+p['kind']] += 1
        prior = previous.get(key)
        if prior is not None:
            changed = prior['pending_plan_version_refs'] != v['pending_plan_version_refs']
            counts['version_change_intervals_with_positive_H'] += changed and prior['executed_holding_s'] > 0
            old_p = prior['own_association']['obligation']
            counts['association_changes'] += (None if old_p is None else old_p['trip_id']) != (None if p is None else p['trip_id'])
            counts['rank_changes'] += prior['own_association']['distance_rank'] != v['own_association']['distance_rank']
        previous[key] = v
    closed = [c for c in capture['cycles'] if c['status'] == 'returned']
    return dict(query_prefix_holding_s=sum(c['query_increment_s'] for c in closed),
                return_remainder_s=sum(c['after_last_query_s'] for c in closed), **counts)


def collect(root, archive, v216, arm, gate_fields):
    import run_freqduet_v230_resource_capture as v230
    import freqduet_v235_passive_cycle_capture as observer
    assert 'source_eec7a20c71' in v230.__file__
    assert 'source_6c3397ff7f' in observer.__file__
    old = v230.frozen()
    legacy, original = old.legacy, old.legacy.ReleaseAudit
    slot = 'c0_'+arm
    check = root/'checks'/f'{slot}.json'
    if check.exists():
        result = json.loads(check.read_text())
        assert result['slot'] == slot
        return result
    files = {name: root/folder/f'{slot}.json.gz' for name, folder in
             (('legacy', 'records'), ('v224', 'sidecars'), ('v230', 'resource_sidecars'), ('cycle', 'cycle_sidecars'))}
    audits = []
    audit_class = observer.recording_audit(old.RecordingAudit)

    def factory(seed, name):
        assert seed == 12037 and name == arm
        audit = audit_class(seed, name)
        audits.append(audit)
        return audit

    try:
        if not all(p.exists() for p in files.values()):
            assert not (root/'progress'/f'{slot}.json').exists(), 'incomplete path cannot restart'
            claim_slot(root, arm, gate_fields)
            print('V236_STARTED '+slot, flush=True)
            legacy.ReleaseAudit = factory
            legacy.run_item(root, v216, (0, arm))
            assert len(audits) == 1
            audit = audits[0]
            legacy.passive.trace(files['v224'], dict(audit.handoff.data(), complete=True))
            legacy.passive.trace(files['v230'], audit.resource.data(True))
            legacy.passive.trace(files['cycle'], audit.cycle_capture.data(True))
            audits.clear()
        recorded = {k: legacy.prior.read(files[k]) for k in ('legacy', 'v224', 'v230')}
        reference = {k: legacy.prior.read(archive/folder/f'{slot}.json.gz') for k, folder in
                     (('legacy', 'records'), ('v224', 'sidecars'), ('v230', 'resource_sidecars'))}
        capture = legacy.prior.read(files['cycle'])
        result = dict(slot=slot, context=0, arm=arm,
            **observer.validate_capture(reference, recorded, capture), **ledger_summary(capture))
        result['passed'] = all(result[k] is True for k in gate_fields)
    except Exception as exc:
        traceback.print_exc()
        result = dict(slot=slot, context=0, arm=arm, passed=False,
                      failure=type(exc).__name__+': '+str(exc))
        if audits and audits[0].resource.metadata is not None:
            audit = audits[0]
            for name, payload in (('v224', dict(audit.handoff.data(), complete=False)),
                                  ('v230', audit.resource.data(False)),
                                  ('cycle', audit.cycle_capture.data(False))):
                if not files[name].exists():
                    legacy.passive.trace(files[name], payload)
    finally:
        legacy.ReleaseAudit = original
    write(check, result)
    print('V236_PATH_CHECK '+json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect'))
    for name in ('out-root', 'archive-root', 'v216-root', 'sealed-protocol', 'v235-summary'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    from preregister_freqduet_v234_passive_cycle_preflight import PAIR_CHECKS, registration, pair_passed
    spec = json.loads(a.sealed_protocol.read_text())
    assert spec == registration('6396ae6baa')
    parent = json.loads(a.v235_summary.read_text())
    assert parent['status'] == 'v235_synthetic_capture_tests_passed' and parent['source'] == '6c3397ff7f'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v236_passive_cycle_preflight.py')
    assert suite.countTestCases() == 5
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    write(a.out_root/'preflight.json', dict(source=a.source, tests=tested.testsRun,
        new_simulator_starts=0, scope='Synthetic slot gates and serialized V235 acceptance only.'))
    if a.command == 'tests':
        return
    auth = dict(protocol_source='6396ae6baa', observer_source='6c3397ff7f', runner_source=a.source,
        authorization='User approved c0 V107 then upper-only after ALL V234 gates;failed starts consume slots.',
        slots=spec['reserved_preflight']['slots'], max_started_paths=2, max_workers=1,
        extra_reference_paths=0, training_updates=0, new_futures=0,
        shadow_actions_executed=0, main_policy_changed=False)
    path = a.out_root/'execution_authorization.json'
    if path.exists():
        assert json.loads(path.read_text()) == auth
    else:
        write(path, auth)
    rows, start = [], time.monotonic()
    for arm in ('v107', 'upper'):
        row = collect(a.out_root, a.archive_root, a.v216_root, arm, PAIR_CHECKS)
        rows.append(row)
        summary = dict(status='v236_'+arm+('_passed' if row['passed'] else '_stopped'),
            source=a.source, observer_source='6c3397ff7f', protocol_source='6396ae6baa',
            paths=rows, started_paths=len(list((a.out_root/'started').glob('*.json'))),
            checked_paths=len(rows), max_started_paths=2, tests=tested.testsRun,
            training_updates=0, new_futures=0, shadow_actions_executed=0,
            main_policy_changed=False)
        if arm == 'upper' and row['passed']:
            assert pair_passed(rows)
            summary['status'] = 'v236_passive_pair_passed'
        write(a.out_root/'summary/status.json', summary)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=2,
            unit='path', phase=summary['status'], elapsed_seconds=time.monotonic()-start)), flush=True)
        if not row['passed']:
            raise SystemExit(1)
    print('V236_PASSIVE_PAIR_COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
