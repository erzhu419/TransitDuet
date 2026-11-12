"""Verify the paid path and collect only the eight reserved passive days."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_frozen_run', HERE.parent / 'source02/run.py')
frozen_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen_run)
PLAN = frozen_run.PLAN
read, write = frozen_run.read, frozen_run.write


def slot_audit(trace, slots):
    dispatches = {r['trip']: r for r in trace['dispatches']}
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    public = trace['public_clocks']
    assert len(dispatches) == len(returns) == len(public) == 262
    assert all(public[t] == r['public_s'] for t, r in dispatches.items())
    expected = {(trip, lead): due-lead for trip, due in public.items()
                for lead in PLAN['leads_s'] if due >= lead}
    actual = {(r['trip'], r['lead_s']): r['query_s'] for r in slots}
    assert actual == expected and len(actual) == len(slots)
    stats = {str(lead): Counter() for lead in PLAN['leads_s']}
    unsupported = {str(lead): Counter() for lead in PLAN['leads_s']}
    for r in slots:
        trip, clock, lead = r['trip'], r['query_s'], str(r['lead_s'])
        assert r['public_s'] == public[trip] and r['direction'] == dispatches[trip]['direction']
        expected_earlier = {t for t, row in dispatches.items() if row['direction'] == r['direction']
            and (public[t], t) < (public[trip], trip) and row['actual_s'] >= clock}
        assert set(r['earlier']) == expected_earlier
        assert len(r['ready']) == len(set(r['ready'])) and not set(r['ready']) & set(r['cohort'])
        for bus, active_trip in r['cohort'].items():
            row = dispatches[active_trip]
            assert row['bus'] == bus and row['direction'] != r['direction']
            assert row['actual_s'] < clock <= returns[bus, active_trip]
        s = stats[lead]
        s.update(snapshots=1, already_ready=int(bool(r['ready'])), earlier_queued=int(bool(r['earlier'])),
            model_available=int(bool(r['scenarios'])), no_model=int(not r['scenarios']))
        if r['unavailable']:
            unsupported[lead][r['unavailable']] += 1
        for model in r['scenarios']:
            name = f'scenario_{model["scenario"]}'
            s[name] += 1
            if not model['failure']:
                s[name+'_success'] += 1
                if r['cohort'] and r['cohort'].keys() <= model['returns'].keys():
                    s[name+'_complete_inbound'] += 1
    skipped = {str(h): sum(t < h for t in public.values()) for h in PLAN['leads_s']}
    return dict(leads=stats, unavailable=unsupported, before_service_start=skipped,
                schedule_exact=True, cohort_labels_complete=True)


def verify(root):
    parity = read(root / 'parity/summary/results.json')
    assert parity['passed'] and parity['physical_starts'] == 1 and parity['paired_events'] == 5200
    assert parity['exact_endpoints'] and parity['day'] == PLAN['engineering_day']
    assert not parity['confirmation_started'] and not (root / 'confirmation').exists()
    frozen = frozen_run.old.frozen
    archive = frozen.helper(frozen.ASSETS, 'gate_default')
    physical = frozen.helper(frozen.ASSETS, 'physical')
    case = f'cases/{PLAN["regime"]}/{PLAN["engineering_day"]}/{PLAN["arm"]}'
    original = archive.raw(frozen_run.V412_ROOT / case / 'raw/path.pkl.gz')
    current = archive.raw(root / 'parity' / case / 'raw/path.pkl.gz')
    assert set(original) == set(current) == {*parity['exact_streams'], 'uniforms', 'graph_queries'}
    for name in original:
        a, b = original[name], current[name]
        if name == 'graph_queries':
            a, b = ([{k: v for k, v in row.items() if k != 'total_s'} for row in values]
                    for values in (a, b))
        for left, right in ((a, b), (b, a)):
            diff = physical.first_diff(left, right, name)
            assert diff is None, diff
    assert len(current['uniforms']) == len(current['queries']) == len(current['graph_queries']) == 5200
    slots = read(root / 'parity' / case / 'raw/public_slots.json')
    for r in slots:
        r['cohort'] = {int(k): v for k, v in r['cohort'].items()}
        for m in r['scenarios']:
            m['returns'] = {int(k): v for k, v in m['returns'].items()}
    audit = slot_audit(current, slots)
    assert {h: s['snapshots'] for h, s in audit['leads'].items()} == parity['snapshot_counts']
    assert {h: s['model_available'] for h, s in audit['leads'].items()} == parity['supported_counts']
    days = PLAN['confirmation_days']
    for project in HERE.parent.parent.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in days:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
            assert not list(project.glob(f'run*/confirmation/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, days)) & frozen_run.old.base.declared_days(read(manifest)), manifest
    write(root / 'verify/summary/results.json', dict(passed=True, physical_starts=0,
        day=parity['day'], exact_streams=sorted(original), excluded_runtime_fields=['graph_queries.total_s'],
        paired_events=5200, exact_endpoints=True, slots=audit, confirmation_days=days))


def case(root, day):
    gate = read(root / 'verify/summary/results.json')
    days = PLAN['confirmation_days']
    assert gate['passed'] and gate['confirmation_days'] == days
    assert len(set(days)) == len(days) == PLAN['planned_confirmation_paths'] == 8 and day in days
    target = root / 'confirmation' / str(day)
    target.mkdir(parents=True, exist_ok=False)
    result, trace, slots = frozen_run.collect(target, day)
    assert result['completed'] and result['physical_account']['passed']
    assert result['physical_starts'] == 1 and not result['main_policy_changed'] and not result['training_updates']
    audit = slot_audit(trace, slots)
    write(target / 'summary/results.json', dict(completed=True, day=day, arm=PLAN['arm'],
        physical_starts=1, training_updates=0, main_policy_changed=False,
        automatic_retry=False, slots=audit, endpoints=result['endpoints'], wall_s=result['wall_s']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('verify', 'case'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--day', type=int)
    a = p.parse_args()
    if a.phase == 'verify':
        verify(a.root)
    else:
        case(a.root, a.day)
