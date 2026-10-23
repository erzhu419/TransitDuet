#!/usr/bin/env python3
"""Execute the approved V224 passive first pair, then six only after exact gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import run_freqduet_v221_release_smoke as legacy
from freqduet_v224_prepare_capture import PrepareCapture, validate_saved_capture
from preregister_freqduet_v224_prepare_handoff import registration, require_legacy_equal


OriginalAudit = legacy.ReleaseAudit


class RecordingAudit(OriginalAudit):
    def __init__(self, seed, arm):
        super().__init__(seed, arm)
        assert arm in ('v107', 'upper'), 'V224 cannot execute lower interventions'
        self.handoff = PrepareCapture(self)

    def attach(self, runner):
        super().attach(runner)
        self.handoff.attach(runner, legacy)

    def __call__(self, policy, states, features, theta):
        self.handoff.bind_policy(policy)
        self.handoff.calls['lower_rule'] += 1
        value = super().__call__(policy, states, features, theta)
        if self.early is not None:
            self.handoff.bind_policy(self.early)
        self.handoff.query(self.queries[-1])
        return value

    def detach(self):
        self.handoff.detach()
        super().detach()


def collect(root, archive, v216, item):
    ci, arm = item
    slot = f'c{ci}_{arm}'
    summary = root/'checks'/f'{slot}.json'
    if summary.exists():
        return json.loads(summary.read_text())
    output, sidecar = root/'records'/f'{slot}.json.gz', root/'sidecars'/f'{slot}.json.gz'
    if output.exists() and sidecar.exists():
        data = legacy.prior.read(sidecar)
    else:
        assert not (root/'progress'/f'{slot}.json').exists(), 'started slot cannot automatically replay'
        data = None
    audits = []
    def factory(seed, name):
        audit = RecordingAudit(seed, name)
        audits.append(audit)
        return audit
    legacy.ReleaseAudit = factory
    try:
        if data is None:
            legacy.run_item(root, v216, item)
            assert len(audits) == 1
            data = dict(audits[0].handoff.data(), complete=True)
            legacy.passive.trace(sidecar, data)
    except Exception as exc:
        if audits:
            legacy.passive.trace(sidecar, dict(audits[0].handoff.data(), complete=False))
        result = dict(context=ci, arm=arm, legacy_exact=None, recording_complete=None,
                      passed=False, failure=type(exc).__name__+': '+str(exc))
        legacy.passive.write(summary, result)
        print('V224_PATH_FAILED', slot, json.dumps(result), flush=True)
        return result
    finally:
        legacy.ReleaseAudit = OriginalAudit
    recorded = legacy.prior.read(output)
    reference = legacy.prior.read(archive/'records'/f'{slot}.json.gz')
    result = dict(context=ci, arm=arm, legacy_exact=False, recording_complete=False, passed=False)
    try:
        require_legacy_equal(reference, recorded)
        result.update(legacy_exact=True, **validate_saved_capture(data, recorded))
        result['passed'] = True
    except AssertionError as exc:
        result['failure'] = str(exc)
    legacy.passive.write(summary, result)
    print('V224_PATH_CHECK', slot, json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--v216-root', type=Path, required=True)
    p.add_argument('--sealed-protocol', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.sealed_protocol.read_text())
    assert spec == json.loads(json.dumps(registration('282f0d884b'))), 'sealed V224 changed'
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                             pattern='test_v224_*.py')
    assert suite.countTestCases() > 0
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if a.command == 'tests':
        print('V224_COLLECTOR_TESTS_COMPLETE', flush=True)
        return
    legacy.pilot.joint.optimized_runtime(a.out_root)
    auth = dict(protocol_source='282f0d884b', implementation=os.environ['FREQDUET_HANDOFF_COMMIT'],
        authorization='User approved first c0 pair,then six conditional on exact equality AND complete records.',
        max_started_paths=8, new_training=0, shadow_execution=False, main_policy_changed=False)
    manifest = a.out_root/'execution_authorization.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == auth
    else:
        legacy.passive.write(manifest, auth)
    items = [(ci, arm) for ci in range(4) for arm in ('v107', 'upper')]
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        for label, stage in (('first_pair', items[:2]), ('remaining_six', items[2:])):
            futures = [pool.submit(collect, a.out_root, a.archive_root, a.v216_root, item) for item in stage]
            results.extend(f.result() for f in futures)
            passed = all(r['passed'] for r in results)
            status = dict(status='v224_'+label+('_passed' if passed else '_stopped'), paths=results,
                completed_paths=len(results), started_paths=len(list((a.out_root/'progress').glob('*.json'))),
                max_paths=8, training_updates=0, new_futures=0, shadow_actions_executed=0,
                main_policy_changed=False)
            legacy.passive.write(a.out_root/'summary/status.json', status)
            print('V224_STAGE_GATE', label, passed, flush=True)
            if not passed:
                return
    print('V224_PASSIVE_EQUIVALENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
