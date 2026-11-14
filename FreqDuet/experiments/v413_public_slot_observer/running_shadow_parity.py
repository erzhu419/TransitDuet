"""One passive V414 engineering path against the accepted V413 archive."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_component_entry',
    HERE.parent / 'component_entry01/component_entry.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
run, read, write = prior.run, prior.read, prior.write
SOURCE = HERE.parent / 'running_shadow01'
sys.path.insert(0, str(SOURCE))
from running_shadow import ShadowSlots

PLAN = read(SOURCE / 'running_shadow_protocol.json')


def inspect_sidecar(slots, shadow, reference):
    indexed = {(r['trip'], r['lead_s']): r for r in slots}
    assert len(indexed) == len(slots)
    seen, counts = set(), Counter()
    assert len(shadow) == len(reference)
    for sample, old in zip(shadow, reference):
        assert sample['query_s'] == old['query_s'] and sample['slots'] == old['slots']
        assert bool(sample['scenarios']) == old['available']
        for key in map(tuple, sample['slots']):
            assert key not in seen and indexed[key]['query_s'] == sample['query_s']
            seen.add(key)
            assert len(indexed[key]['scenarios']) == len(sample['scenarios'])
        counts['query_ticks'] += 1
        counts['slot_rows'] += len(sample['slots'])
        for result, original in zip(sample['scenarios'], old['scenarios']):
            prior.same(original, result['baseline'], 'baseline_components')
            counts['shadow_scenarios'] += 1
            counts.update(result['link_calls'])
            if result['failure']:
                counts['shadow_failed'] += 1
                continue
            counts['shadow_completed'] += 1
            before = {r['bus']: r['trip'] for r in result['baseline']}
            after = {r['bus']: r['trip'] for r in result['candidate']}
            assert before == after and len(after) == len(result['candidate'])
            assert set(result['returns']) == {str(r['bus']) for r in result['candidate'] if r['supported']}
            for row in result['candidate']:
                if not row['supported']:
                    continue
                assert row['query_s'] == sample['query_s']
                assert row['return_s'] == result['returns'][str(row['bus'])]
                assert set(row['components']) == set(prior.FIELDS)
                assert all(v >= 0 for v in row['components'].values())
                assert sum(row['components'].values()) == row['return_s'] - row['query_s']
                counts['shadow_vehicle_records'] += 1
    assert seen == set(indexed)
    assert counts['shadow_completed'] and counts['hourly'] and counts['shadow_vehicle_records']
    return dict(counts)


def parity(root):
    gate = read(root / 'running_shadow_checks/summary/results.json')
    parent = read(root / 'component_parity/summary/results.json')
    day = PLAN['engineering_day']
    assert gate['passed'] and gate['physical_starts'] == 0 and not gate['confirmation_started']
    assert gate['engineering_day'] == parent['day'] == day == run.PARITY
    assert parent['passed'] and parent['exact_endpoints'] and parent['exact_public_slots']
    assert PLAN['proposed_old_day_paths'] == 1 and not PLAN['automatic_retry']
    assert PLAN['source_arm'] == run.CAP and PLAN['regime'] == run.REGIME
    assert PLAN['lead_s'] == run.PLAN['leads_s']
    target = root / 'running_shadow_parity'
    case = Path('cases') / run.REGIME / str(day) / run.CAP
    reference = root / 'component_parity' / case
    original_slots = read(reference / 'raw/public_slots.json')
    original_components = read(reference / 'raw/predicted_components.json')
    original_receipt = read(reference / 'summary/results.json')
    assert (reference / 'raw/path.pkl.gz').is_file()
    observer_module = sys.modules['observer']
    captured = ShadowSlots(observer_module, PLAN['lead_s'])
    original = run.PublicSlots

    def observer(leads):
        assert leads == PLAN['lead_s']
        return captured

    target.mkdir(exist_ok=False)
    write(target / 'started.json', dict(day=day, physical_starts=1, automatic_retry=False))
    run.PublicSlots = observer
    try:
        result, trace, slots = run.collect(target, day)
    finally:
        run.PublicSlots = original

    write(target / case / 'raw/running_shadow.json', captured.shadow)
    archive = run.old.frozen.helper(run.old.frozen.ASSETS, 'gate_default').raw(reference / 'raw/path.pkl.gz')
    assert set(trace) == set(archive) == set(prior.PLAN['exact_streams'])
    for name in prior.PLAN['exact_streams']:
        a, b = archive[name], trace[name]
        if name == 'graph_queries':
            a, b = ([{k: v for k, v in row.items() if k != 'total_s'} for row in rows] for rows in (a, b))
        prior.same(a, b, name)
    for name in prior.PLAN['exact_receipt_fields']:
        prior.same(original_receipt[name], result[name], name)
    prior.same(original_slots, read(target / case / 'raw/public_slots.json'), 'public_slots')
    counts = inspect_sidecar(slots, read(target / case / 'raw/running_shadow.json'), original_components)
    assert len(trace['uniforms']) == len(trace['queries']) == parent['paired_events'] == 5200
    write(target / 'summary/results.json', dict(passed=True, day=day, physical_starts=1,
        exact_streams=prior.PLAN['exact_streams'], exact_endpoints=True, exact_public_slots=True,
        exact_baseline_components=True, paired_events=5200, counts=counts,
        training_updates=0, main_policy_changed=False, confirmation_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parity(parser.parse_args().root)
