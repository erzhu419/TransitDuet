#!/usr/bin/env python3
"""Synthetic-only passive request integration tests; no collection command."""

import argparse
import json
from pathlib import Path
import sys
import unittest

import freqduet_v235_passive_cycle_capture as cycle
import preregister_freqduet_v238_persistent_requests as supply
import preregister_freqduet_v240_paired_coordination as protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sealed-protocol', 'parent-summary', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert json.loads(a.sealed_protocol.read_text()) == protocol.registration('e6788e6109')
    parent = json.loads(a.parent_summary.read_text())
    assert parent['status'] == 'v240_paired_protocol_sealed_no_replay' and parent['source'] == 'e6788e6109'
    assert not parent['replay_authorized'] and not parent['main_policy_changed']
    assert 'source_6c3397ff7f' in cycle.__file__
    assert 'source_7fdf968a5f' in cycle.frozen.__file__
    assert 'source_b4b1d00203' in supply.__file__
    assert 'source_e6788e6109' in protocol.__file__
    output = a.out_root/'summary/status.json'
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved['source'] == a.source and saved['status'] == 'v241_synthetic_passive_requests_passed'
        print('V241_COMPLETED_TEST_RECEIPT_REUSED', flush=True)
        return
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v241_passive_requests.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    assert not {'runner_v3', 'env.sim', 'env.bus'} & set(sys.modules), 'tests imported the real simulator'
    summary = dict(status='v241_synthetic_passive_requests_passed', source=a.source,
        protocol_source='e6788e6109', rule_source='b4b1d00203', tests=tested.testsRun,
        passive_integration_implemented=True, native_action_callback_inherited=True,
        synthetic_original_four_payloads_exact=True, synthetic_H0_aliases_native=True,
        synthetic_rng_and_native_calls_unchanged=True, archive_scans=0,
        passive_paths=0, smoke_paths=0, formal_paths=0, future_starts=0,
        training_updates=0, model_fits=0, shadow_actions_executed=0,
        runtime_connected=False, empirical_equivalence_verified=False, main_policy_changed=False,
        reserved_starts_consumed=0, reserved_starts_remaining=104,
        reserved_engine_mapping=dict(v107='v107', upper_only='upper'),
        collection_authorized=False,
        next_step='Separate authorization for V240 passive_c0_v107 first,then '
            'passive_c0_upper_only only after full equality. No smoke/formal launch admitted by these tests.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print('V241_SYNTHETIC_ONLY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
