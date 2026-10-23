#!/usr/bin/env python3
"""Server-only synthetic observer tests; no collection or archive-scan command."""

import argparse
import json
from pathlib import Path
import unittest

import freqduet_v230_resource_capture as resource
import freqduet_v235_passive_cycle_capture as capture
from preregister_freqduet_v234_passive_cycle_preflight import registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--sealed-protocol', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads(args.sealed_protocol.read_text()) == registration('6396ae6baa')
    assert 'source_7fdf968a5f' in capture.frozen.__file__, 'use frozen V232 directly'
    assert 'source_eec7a20c71' in resource.__file__, 'test frozen V230 hooks directly'
    output = args.out_root/'summary/status.json'
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved['source'] == args.source and saved['status'] == 'v235_synthetic_capture_tests_passed'
        print('V235_EXISTING_COMPLETED_TEST_RECEIPT_REUSED', flush=True)
        return
    tests = Path(__file__).resolve().parents[1]/'tests'
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests), pattern=pattern)
        for pattern in ('test_v234_passive_cycle_preflight.py', 'test_v235_passive_cycle_capture.py'))
    assert suite.countTestCases() == 16
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    summary = dict(status='v235_synthetic_capture_tests_passed', source=args.source,
        frozen_protocol_source='6396ae6baa', frozen_cycle_source='7fdf968a5f',
        frozen_resource_source='eec7a20c71', tests=result.testsRun,
        synthetic_original_payloads_exact=True, archive_scans=0, passive_paths=0,
        intervention_paths=0, training_updates=0, model_fits=0, shadow_actions_executed=0,
        new_futures=0, collection_authorized=False, main_policy_changed=False,
        empirical_new_path_equivalence_verified=False,
        reserved_slots_unspent=['c0_v107', 'c0_upper'],
        next_step='Separate execution authorization for V234 fixed pair;no path started by these tests.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print('V235_SYNTHETIC_TESTS_ONLY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
