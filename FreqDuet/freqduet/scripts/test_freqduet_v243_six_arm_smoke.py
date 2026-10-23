#!/usr/bin/env python3
"""Synthetic V240 adapter acceptance only; no smoke or rollout CLI."""

import argparse
import json
from pathlib import Path
import unittest

import freqduet_v243_six_arm_smoke as adapter
import preregister_freqduet_v240_paired_coordination as protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'sealed-protocol', 'v242-summary'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    assert json.loads(a.sealed_protocol.read_text()) == protocol.registration('e6788e6109')
    parent = json.loads(a.v242_summary.read_text())
    assert parent['status'] == 'v242_passive_pair_passed' and parent['source'] == 'c6710744f9'
    assert parent['started_paths'] == 2 and parent['reserved_starts_remaining'] == 102
    assert all(row['passed'] and row['original_payloads_exact'] and row['request_occurrences_complete']
               for row in parent['paths'])
    assert not parent['smoke_authorized'] and not parent['formal_authorized']
    output = a.out_root/'summary/status.json'
    if output.exists():
        prior = json.loads(output.read_text())
        assert prior['source'] == a.source and prior['status'] == 'v243_synthetic_six_arm_adapter_passed'
        print('V243_COMPLETED_TEST_RECEIPT_REUSED', flush=True)
        return
    # Building a class resolves the real frozen callback composition,not a runner or model.
    runtime_class = adapter.runtime_audit()
    assert runtime_class.__name__ == 'SixArmAudit'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v243_six_arm_smoke.py')
    assert suite.countTestCases() >= 10
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    summary = dict(status='v243_synthetic_six_arm_adapter_passed', source=a.source,
        protocol_source='e6788e6109', rule_source='b4b1d00203', passive_source='fcb5efdcf0',
        native_source='2b0115af7b', future_source='c45b2d0a93', tests=tested.testsRun,
        arms=list(adapter.ARM_RULES), runtime_adapter_class_built=True,
        synthetic_exact_common_prefix=True, synthetic_single_post_sample_fork=True,
        synthetic_current_window_and_H_preserved=True, synthetic_event_keyed_coupling=True,
        synthetic_frozen_v220_materialization=True, synthetic_issued_and_executed_chain=True,
        synthetic_all_three_upper_pairs_and_four_lower_arms_required=True,
        raw_archive_reads=0, new_passive_paths=0, smoke_paths=0, formal_paths=0,
        new_real_futures=0, training_updates=0, model_fits=0,
        real_shadow_actions_executed=0, reserved_starts_consumed=2, reserved_starts_remaining=102,
        smoke_authorized=False, formal_authorized=False, main_policy_changed=False,
        empirical_six_arm_smoke_verified=False,
        next_step='Separate authorization for the six frozen c0 smoke slots;no formal launch or retry admitted.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print('V243_SYNTHETIC_ONLY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
