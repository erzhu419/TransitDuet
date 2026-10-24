#!/usr/bin/env python3
"""Synthetic V254 adapter admission only; no real-path entrypoint."""

import argparse
import json
from pathlib import Path
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--v254-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    parent = json.loads((a.v254_root/'summary.json').read_text())
    assert parent['status'] == 'v254_manifest_reserved_no_replay' and parent['source'] == 'eaba2513da'
    assert parent['freshness']['decision'] == 'fresh' and not parent['launch_authorized']
    spec = json.loads((a.v254_root/'manifest.json').read_text())
    assert spec['preflight']['slots'] == ['passive_c0_v107','noop_c0_v107']
    assert not spec['launch_authorized']
    a.out_root.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v255_passive_pulse.py')
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = dict(status='v255_passive_noop_synthetic_'+('passed' if result.wasSuccessful() else 'failed'),
        source=a.source, parent_source='eaba2513da', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        collector_implemented=True, runtime_connected=False, selector_defined=False,
        simulator_starts=0, future_starts=0, training_updates=0, model_fits=0,
        raw_payloads_opened=0, launch_authorized=False, main_policy_changed=False, v238_changed=False,
        next_gate='Separate approval for original-future c0/V107 observer-only passive slot; '
            'exact six-payload and complete-record gates before native no-op. No smoke or S/E admission.')
    (a.out_root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V255_SYNTHETIC '+json.dumps(summary), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
