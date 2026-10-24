#!/usr/bin/env python3
"""Read-only recheck of the completed V258 native; cannot start a simulator."""

import argparse
import json
from pathlib import Path
import sys
import unittest

import run_freqduet_v258_atomic_smoke as driver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--v256-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    a = p.parse_args()
    assert a.collection.name == driver.COLLECTION
    original = driver.read(driver.check_file(a.collection,'native'))
    assert original['source'] == '140949bcf8' and not original['passed']
    assert original['failure'] == 'AssertionError: checkpoint value differs: V258_archived_authority.scenario_tape_id'
    starts = sorted(str(p) for p in (a.collection/'started').glob('*.json'))
    assert len(starts) == 1
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_v258_atomic_smoke.Authority')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    legacy,joint,_ = driver.interfaces()
    data = driver.load_record(a.collection/'records/c0_smoke_f0_native.pkl.gz')
    assert data['source'] == '140949bcf8' and data['complete'] and data['branch'] == 'native'
    result = dict(source=a.source,data_source='140949bcf8',original_task='t211462',
        original_check_preserved=True,original_failed_start_debit=1,
        tests=tested.testsRun,simulator_starts=0,future_starts=0,formal_starts=0,
        training_updates=0,main_policy_changed=False,v238_changed=False)
    result.update(driver.path_check(data,None,joint))
    result.update(driver.archived_prefix(data,a.v256_root,legacy,joint.pilot.joint.assert_same_state))
    assert starts == sorted(str(p) for p in (a.collection/'started').glob('*.json'))
    assert original == driver.read(driver.check_file(a.collection,'native'))
    result.update(status='v258_completed_native_readonly_recheck_passed',
        remaining_paths_started=0,continuation_authorized=False,
        next_gate='Keep original failure/debit; explicit continuation decision before remaining seven paths.')
    output = a.collection/'readonly_rechecks'/a.source/'native.json'
    assert not output.exists()
    driver.write(output,result)
    print('V258_NATIVE_READONLY_RECHECK '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
