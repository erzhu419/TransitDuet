"""Collect the eight reserved V414 days without changing the accepted observer."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load('v414_parity', HERE.parent / 'running_parity01/running_shadow_parity.py')
run, read, write = parity.run, parity.read, parity.write
slot_audit = parity.prior.readout.confirmation.slot_audit
PLAN = parity.PLAN
DAYS = PLAN['confirmation_days']
PHASE = 'running_shadow_confirmation'


def check_release(root):
    gate = read(root / 'running_shadow_parity/summary/results.json')
    assert gate['passed'] and gate['physical_starts'] == 1 and gate['paired_events'] == 5200
    assert gate['exact_public_slots'] and gate['exact_baseline_components'] and gate['exact_endpoints']
    assert gate['day'] == PLAN['engineering_day'] == run.PARITY and not gate['confirmation_started']
    assert len(set(DAYS)) == len(DAYS) == PLAN['proposed_confirmation_paths'] == 8
    assert not PLAN['automatic_retry'] and not set(DAYS) & {run.PARITY, *run.PLAN['confirmation_days']}
    assert PLAN['source_arm'] == run.CAP and PLAN['regime'] == run.REGIME


def validate_sidecar(slots, shadow):
    indexed = {(r['trip'], r['lead_s']): r for r in slots}
    assert len(indexed) == len(slots)
    seen, counts = set(), Counter()
    for sample in shadow:
        for pair in sample['slots']:
            key = tuple(pair)
            assert key not in seen
            seen.add(key)
            slot = indexed[key]
            assert sample['query_s'] == slot['query_s']
            assert len(sample['scenarios']) == len(slot['scenarios'])
            for old, new in zip(slot['scenarios'], sample['scenarios']):
                baseline = {r['bus']: r['trip'] for r in new['baseline']}
                assert {r['bus']: r['return_s'] for r in new['baseline'] if r['supported']} == old['returns']
                if not new['failure']:
                    assert baseline == {r['bus']: r['trip'] for r in new['candidate']}
                    assert len(baseline) == len(new['candidate'])
                for row in new['candidate']:
                    if row['supported']:
                        assert row['query_s'] == sample['query_s']
                        assert row['return_s'] == new['returns'][row['bus']]
                        assert set(row['components']) == set(parity.prior.FIELDS)
                        assert all(v >= 0 for v in row['components'].values())
                        assert sum(row['components'].values()) == row['return_s'] - row['query_s']
        for result in sample['scenarios']:
            counts.update(result['link_calls'])
            counts['shadow_scenarios'] += 1
            counts['shadow_failed'] += bool(result['failure'])
    assert seen == set(indexed)
    return dict(counts)


def checks(root):
    check_release(root)
    assert not (root / PHASE).exists()
    fixtures = load('v414_release_fixture', HERE.parent / 'source02/slot_checks.py')
    env, bridge, _ = fixtures.fixture()
    observer = sys.modules['observer']
    baseline = observer.PublicSlots(PLAN['lead_s'])
    captured = parity.ShadowSlots(observer, PLAN['lead_s'])
    baseline.before_step(env, bridge)
    captured.before_step(env, bridge)
    assert captured.records == baseline.records
    counts = validate_sidecar(captured.records, captured.shadow)
    assert counts['shadow_scenarios'] == 1 and not counts['shadow_failed']
    write(root / 'running_shadow_release_checks/summary/results.json', dict(passed=True,
        physical_starts=0, training_updates=0, confirmation_days=DAYS, counts=counts))


def case(root, day):
    check_release(root)
    gate = read(root / 'running_shadow_release_checks/summary/results.json')
    assert gate['passed'] and gate['physical_starts'] == 0 and gate['confirmation_days'] == DAYS
    assert day in DAYS
    captured = parity.ShadowSlots(sys.modules['observer'], PLAN['lead_s'])
    original = run.PublicSlots

    def observer(leads):
        assert leads == PLAN['lead_s']
        return captured

    target = root / PHASE / str(day)
    target.mkdir(parents=True, exist_ok=False)
    write(target / 'started.json', dict(day=day, physical_starts=1, automatic_retry=False))
    run.PublicSlots = observer
    try:
        result, trace, slots = run.collect(target, day)
    finally:
        run.PublicSlots = original
    assert result['completed'] and result['physical_account']['passed']
    assert result['physical_starts'] == 1 and not result['training_updates'] and not result['main_policy_changed']
    assert len(trace['uniforms']) == len(trace['queries']) == 5200
    audit = slot_audit(trace, slots)
    counts = validate_sidecar(slots, captured.shadow)
    write(target / 'cases' / run.REGIME / str(day) / run.CAP / 'raw/running_shadow.json', captured.shadow)
    write(target / 'summary/results.json', dict(completed=True, day=day, physical_starts=1,
        training_updates=0, main_policy_changed=False, automatic_retry=False,
        paired_events=5200, slots=audit, shadow=counts, endpoints=result['endpoints'], wall_s=result['wall_s']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('checks', 'case'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--day', type=int)
    args = parser.parse_args()
    if args.phase == 'checks':
        checks(args.root)
    else:
        case(args.root, args.day)
