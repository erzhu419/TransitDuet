#!/usr/bin/env python3
"""Explicitly approved continuation after read-only native recovery; no native retry."""

import argparse
import json
from pathlib import Path
import sys
import traceback
import unittest

import run_freqduet_v258_atomic_smoke as driver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('tests','continue'))
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    a = p.parse_args()
    assert a.out_root.name == driver.COLLECTION
    native_gate = driver.read(a.out_root/'readonly_rechecks/22bd7999c5/native.json')
    native = driver.accepted(a.out_root,'native',a.source,native_gate)
    assert native['archived_stored_prefix_exact'] and native['fixed_authority_exact']
    assert native['simulator_starts'] == native['future_starts'] == 0
    assert native['plan']['physical'] == ['native','forced_native','b0','b2','b3','b4','b5','b6']
    assert len(list((a.out_root/'started').glob('*.json'))) == 1
    driver.interfaces()
    if a.command == 'tests':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
        tested = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('test_v258_atomic_smoke.RecoveredSlots'))
        assert tested.wasSuccessful()
        target = a.out_root/'continuation_preflight'/a.source/'summary.json'
        assert not target.exists()
        driver.write(target,dict(source=a.source,status='v258_recovered_continuation_tests_passed',
            tests=tested.testsRun,native_recheck_source='22bd7999c5',simulator_starts=0,future_starts=0))
        print('V258_CONTINUATION_TESTS_PASSED',flush=True)
        return
    ready = driver.read(a.out_root/'continuation_preflight'/a.source/'summary.json')
    assert ready['source'] == a.source and ready['status'] == 'v258_recovered_continuation_tests_passed'
    auth = dict(source=a.source,native_artifact_source='140949bcf8',native_recheck_source='22bd7999c5',
        authorization='User explicitly approved remaining seven ONLY after read-only native recheck passes.',
        native_retry_authorized=False,original_failed_start_debit=1,max_smoke_starts=8,
        remaining=['forced_native','b0','b2','b3','b4','b5','b6'],automatic_retry=False,
        failures_consume_slots=True,formal_authorized=False,training_authorized=False,
        main_policy_changed=False,v238_changed=False)
    target = a.out_root/'continuation_authorization.json'
    assert not target.exists(), 'continuation cannot restart automatically'
    driver.write(target,auth)
    for branch in auth['remaining']:
        driver.collect(a,branch,native_gate)
    legacy,joint,_ = driver.interfaces()
    paths = {b:driver.load_record(a.out_root/'records'/(driver.slot(b)+'.pkl.gz'))
             for b in native['plan']['physical']}
    try:
        gate = driver.atomic.validate_smoke(paths,joint.pilot.joint.assert_same_state,legacy.pilot.event.alignment)
    except Exception as exc:
        traceback.print_exc()
        gate = dict(passed=False,failure=type(exc).__name__+': '+str(exc))
    driver.write(a.out_root/'summary/engineering_gate.json',gate)
    driver.update_status(a.out_root,a.source,gate,native_gate)
    if not gate['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
