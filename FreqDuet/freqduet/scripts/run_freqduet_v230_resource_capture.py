#!/usr/bin/env python3
"""Execute the authorized V230 passive pair, then six only after exact equality."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
import traceback
import unittest

from freqduet_v230_resource_capture import ResourceCapture, dry_archive_prefix, validate
from preregister_freqduet_v224_prepare_handoff import require_legacy_equal
from preregister_freqduet_v230_resource_capture import first_pair_passed, registration


def frozen():
    passive = importlib.import_module('run_freqduet_v224_prepare_handoff')
    assert 'source_17043d5c03' in passive.__file__, 'use frozen V224,not a mutable collector'
    assert 'protocol_v6_v221_release_smoke_2b0115af7b/source/' in passive.legacy.__file__
    return passive


def collect(root, archive, v216, item):
    ci, arm = item
    slot = f'c{ci}_{arm}'
    check = root/'checks'/f'{slot}.json'
    if check.exists():
        return json.loads(check.read_text())
    old = frozen()
    legacy, original = old.legacy, old.legacy.ReleaseAudit
    output, handoff, extension = (root/folder/f'{slot}.json.gz'
                                  for folder in ('records', 'sidecars', 'resource_sidecars'))
    audits = []
    class RecordingAudit(old.RecordingAudit):
        def __init__(self, seed, name):
            super().__init__(seed, name)
            self.resource = ResourceCapture(self)

        def attach(self, runner):
            super().attach(runner)
            self.resource.attach()

        def __call__(self, policy, states, features, theta):
            value = super().__call__(policy, states, features, theta)
            self.resource.query(self.queries[-1])
            return value

        def detach(self):
            self.resource.detach()
            super().detach()

    def factory(seed, name):
        audit = RecordingAudit(seed, name)
        audits.append(audit)
        return audit
    try:
        if not all(p.exists() for p in (output, handoff, extension)):
            assert not (root/'progress'/f'{slot}.json').exists(), 'started slot cannot be replayed'
            legacy.ReleaseAudit = factory
            legacy.run_item(root, v216, item)
            assert len(audits) == 1
            legacy.passive.trace(handoff, dict(audits[0].handoff.data(), complete=True))
            legacy.passive.trace(extension, audits[0].resource.data(True))
            audits.clear()
        recorded = legacy.prior.read(output)
        require_legacy_equal(legacy.prior.read(archive/'records'/f'{slot}.json.gz'), recorded)
        sidecar = legacy.prior.read(handoff)
        require_legacy_equal(legacy.prior.read(archive/'sidecars'/f'{slot}.json.gz'), sidecar)
        result = dict(slot=slot, context=ci, arm=arm, legacy_exact=True, v224_sidecar_exact=True,
                      **validate(legacy.prior.read(extension), sidecar, recorded))
        result['passed'] = all(result[k] is True for k in registration('fields')['first_pair_gate'])
    except Exception as exc:
        if audits:
            if not handoff.exists():
                legacy.passive.trace(handoff, dict(audits[0].handoff.data(), complete=False))
            if not extension.exists():
                legacy.passive.trace(extension, audits[0].resource.data(False))
        result = dict(slot=slot, context=ci, arm=arm, passed=False,
                      failure=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
    finally:
        legacy.ReleaseAudit = original
    legacy.passive.write(check, result)
    print('V230_PATH_CHECK '+json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('tests', 'collect'))
    for name in ('out-root', 'archive-root', 'v216-root', 'sealed-protocol'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.sealed_protocol.read_text())
    assert spec == json.loads(json.dumps(registration('cf86ee990a'))), 'sealed V230 changed'
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern=pattern)
        for pattern in ('test_v230_resource_capture.py', 'test_v230_resource_extension.py'))
    assert suite.countTestCases() > 10
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    old = frozen()
    dry = {arm: dry_archive_prefix(old.legacy.prior.read(a.archive_root/'records'/f'c0_{arm}.json.gz'))
           for arm in ('v107', 'upper')}
    old.legacy.passive.write(a.out_root/'preflight.json', dict(tests=tests.testsRun,
        archive_interface_checks=dry, started_paths=0, source=os.environ['FREQDUET_RESOURCE_COMMIT']))
    if a.command == 'tests':
        print('V230_EXTENSION_PREFLIGHT_PASSED', flush=True)
        return
    authorization = dict(protocol_source='cf86ee990a', implementation=os.environ['FREQDUET_RESOURCE_COMMIT'],
        authorization='User explicitly approved passive c0 V107/upper,then six conditional on exact gates.',
        max_started_paths=8, max_workers=2, extra_reference_paths=0, shadow_actions_executed=0,
        training_updates=0, main_policy_changed=False)
    auth = a.out_root/'execution_authorization.json'
    if auth.exists():
        assert json.loads(auth.read_text()) == authorization
    else:
        old.legacy.passive.write(auth, authorization)
    items = [(s['context'], s['arm']) for s in spec['reserved_passive']['slots']]
    results, start = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        for label, stage in (('first_pair', items[:2]), ('remaining_six', items[2:])):
            if label == 'remaining_six':
                assert first_pair_passed(results), 'c0 pair did not pass exact engineering gates'
            jobs = [pool.submit(collect, a.out_root, a.archive_root, a.v216_root, item) for item in stage]
            for job in as_completed(jobs):
                results.append(job.result())
                results.sort(key=lambda r: (r['context'], r['arm'] != 'v107'))
                status = dict(status='v230_'+label+'_in_progress', paths=results,
                    source=authorization['implementation'], tests=tests.testsRun,
                    started_paths=len(list((a.out_root/'progress').glob('*.json'))), max_started_paths=8,
                    checked_paths=len(results), all_checked_passed=all(r['passed'] for r in results),
                    training_updates=0, new_futures=0, shadow_actions_executed=0, main_policy_changed=False)
                old.legacy.passive.write(a.out_root/'summary/status.json', status)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(results), total=8, unit='path',
                    phase=label, elapsed_seconds=time.monotonic()-start)), flush=True)
            passed = first_pair_passed(results) if label == 'first_pair' else all(r['passed'] for r in results)
            status['status'] = 'v230_'+label+('_passed' if passed else '_stopped')
            old.legacy.passive.write(a.out_root/'summary/status.json', status)
            print('V230_STAGE_GATE '+label+' '+str(passed), flush=True)
            if not passed:
                raise SystemExit(1)
    print('V230_PASSIVE_RESOURCE_EQUIVALENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
