#!/usr/bin/env python3
"""Run synthetic observer checks only; never load a runner or real archive."""

import argparse
import json
from pathlib import Path
import sys
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v247-root'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    from preregister_freqduet_v247_prepare_timing import registration
    assert json.loads((a.v247_root/'protocol.json').read_text()) == registration('e74d381e62')
    parent = json.loads((a.v247_root/'summary/status.json').read_text())
    assert parent['status'] == 'v247_prepare_mapping_and_minimal_budget_sealed'
    assert parent['source'] == 'e74d381e62' and not parent['collection_authorized']
    tests_dir = Path(__file__).resolve().parents[1]/'tests'
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests_dir), pattern=pattern)
        for pattern in ('test_v247_prepare_timing.py', 'test_v248_prepare_timing_capture.py'))
    assert suite.countTestCases() > 15
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert 'env.bus' not in sys.modules, 'synthetic checks imported the real simulator'
    status = dict(status='v248_independent_prepare_observer_synthetic_passed', source=a.source,
        protocol_source='e74d381e62', tests_passed=result.testsRun,
        synthetic_callback_payload_equality=True, real_path_equivalence_verified=False,
        collector_implemented=True, deployed_to_main=False, selector_defined=False,
        archive_scans=0, passive_paths=0, intervention_paths=0, training_updates=0,
        model_fits=0, shadow_actions_executed=0, reserved_preflight_paths=2,
        preflight_paths_consumed=0, collection_authorized=False,
        main_policy_changed=False, v238_changed=False,
        next_step='Separate approval for c0/V107 passive equality preflight;upper only after all gates pass.')
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print('V248_SYNTHETIC_ONLY_PASSED '+json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
