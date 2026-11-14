"""Bounded full-day spacing comparison using the frozen V415 entrypoint."""

import argparse
import itertools
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
V415 = WORK / 'FreqDuet-v415-running-closed-loop'
sys.path[:0] = [str(HERE.parent / 'source01'), str(V415 / 'source01')]
from spacing import guarded_controller
import study415 as prior

read, write, frozen = prior.read, prior.write, prior.frozen
PLAN = read(HERE / 'traffic_protocol.json')
BASE, HOURLY, SPACING = PLAN['arms']
DAYS = PLAN['development_days']
OLD = PLAN['engineering_day']
DISPATCH = 'dispatch_cv_squared'
READINESS, DELAY, JOURNEY, ARRIVAL = prior.PRIMARY


def preflight(root):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and gate['physical_starts'] == 0
    assert (BASE, HOURLY) == (prior.BASE, prior.HOURLY)
    assert OLD in prior.DAYS and OLD != 2550004
    assert len(DAYS) == len(set(DAYS)) == PLAN['max_concurrent_days'] == 4
    assert PLAN['physical_start_cap'] == 1 + len(DAYS) * len(PLAN['arms'])
    assert not PLAN['automatic_retry'] and not (root / 'engineering').exists()
    assert not (root / 'cases').exists()
    reference = read(V415 / f'run01/pairs/{OLD}/summary/results.json')
    assert reference['physical_starts'] == 2 and reference['paired_events'] == 5200
    old_case = V415 / 'run01/cases' / prior.REGIME / str(OLD) / HOURLY
    assert (old_case / 'raw/path.pkl.gz').is_file()
    assert read(old_case / 'summary/results.json')['completed']
    capped = prior.old.base.Capped
    assert capped.select_plan.__globals__['choose'] is prior.frozen_v406.choose
    assert issubclass(guarded_controller(capped, prior.frozen_v406), capped)
    reserved, scanned = set(map(str, DAYS)), 0
    for project in WORK.glob('FreqDuet-*'):
        if project == HERE.parent or not project.is_dir():
            continue
        for day in DAYS:
            for pattern in ('run*/cases/*/{}', 'run*/confirmation/{}',
                            'run*/running_shadow_confirmation/{}', 'run*/pairs/{}'):
                assert not next(project.glob(pattern.format(day)), None), (project, day)
        for manifest in project.glob('source*/*protocol.json'):
            assert not reserved & prior.old.base.declared_days(read(manifest)), manifest
            scanned += 1
    for path in (root, root / 'engineering'):
        frozen.prepare(path)
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'traffic_preflight/summary/results.json', dict(passed=True, physical_starts=0,
          engineering_day=OLD, days=DAYS, arms=PLAN['arms'], manifests=scanned))


def run_arm(root, day, arm):
    capped = prior.old.base.Capped
    cls = guarded_controller(capped, prior.frozen_v406) if arm == SPACING else capped
    with patch.object(prior.old.base, 'Capped', cls):
        return prior.collect(root, day, arm, None if arm == BASE else prior.hourly)


def engineering(root):
    gate = read(root / 'traffic_preflight/summary/results.json')
    assert gate['passed'] and gate['engineering_day'] == OLD
    case = V415 / 'run01/cases' / prior.REGIME / str(OLD) / HOURLY
    archive = frozen.helper(frozen.ASSETS, 'gate_default').raw(case / 'raw/path.pkl.gz')
    reference = read(case / 'summary/results.json')
    ref_pair = read(V415 / f'run01/pairs/{OLD}/summary/results.json')['arms'][HOURLY]
    result, trace, counts, _ = run_arm(root / 'engineering', OLD, HOURLY)
    compare = frozen.helper(frozen.ASSETS, 'physical').first_diff
    assert set(archive) == set(trace) and counts == ref_pair['predictor']
    for name in archive:
        a, b = archive[name], trace[name]
        if name == 'graph_queries':
            a, b = ([{k: v for k,v in r.items() if k != 'total_s'} for r in rows] for rows in (a, b))
        for left, right in ((a,b), (b,a)):
            assert (diff := compare(left, right, name)) is None, diff
    for name in ('endpoints', 'execution', 'action_chain', 'contract'):
        assert reference[name] == result[name], name
    write(root / 'engineering/summary/results.json', dict(passed=True, physical_starts=1,
          day=OLD, arm=HOURLY, streams=sorted(trace), paired_events=5200,
          exact_endpoints=True, predictor=counts, wall_s=result['wall_s']))


def launch_gaps(trace):
    previous, gaps, excesses = {}, [], []
    for row in trace['dispatches']:
        direction = row['direction']
        if direction in previous:
            old = previous[direction]
            gap = row['actual_s']-old['actual_s']
            scheduled_gap = row['public_s']-old['public_s']
            assert gap >= 0 and scheduled_gap == 360
            gaps.append(gap)
            excesses.append(max(0, gap-scheduled_gap))
        previous[direction] = row
    assert len(gaps) == 260
    return dict(long_gap_count=sum(x > 0 for x in excesses), max_dispatch_gap_s=max(gaps),
                mean_long_gap_excess_s=sum(excesses)/len(excesses))


def day_case(root, day):
    engineering_gate = read(root / 'engineering/summary/results.json')
    assert day in DAYS and engineering_gate['passed'] and engineering_gate['exact_endpoints']
    physical = frozen.helper(frozen.ASSETS, 'physical')
    dose = frozen.helper(frozen.ASSETS, 'v324').actual_dose
    rows, doses, base_trace, hourly_trace = {}, {}, None, None
    for arm in PLAN['arms']:
        result, trace, counts, bins = run_arm(root, day, arm)
        detail, op = result['passenger_detail'], result['operating_detail']
        coordination = result['graph']['coordination']
        rows[arm] = dict(metrics=dict(result['endpoints'], **launch_gaps(trace),
                journey_p95_s=detail['journey_p50_p90_p95_s'][2], unboarded=detail['unserved'],
                incomplete=detail['incomplete']), generated=detail['generated'], predictor=counts,
                spacing={k:v for k,v in coordination.items() if k.startswith('spacing_')},
                bins=bins, wall_s=result['wall_s'], callback_p95_s=op['callback_p50_p95_max_s'][1])
        if arm == BASE:
            base_trace = trace
            continue
        assert rows[arm]['generated'] == rows[BASE]['generated']
        crn = physical.crn_pair(base_trace['uniforms'], trace['uniforms'])
        assert crn['exact'] and crn['common_events'] == 5200 and crn['left_only'] == crn['right_only'] == 0
        assert base_trace['public_clocks'] == trace['public_clocks']
        doses[f'{arm}_minus_{BASE}'] = dose(trace, base_trace)
        if arm == HOURLY:
            hourly_trace = trace
        else:
            doses[f'{arm}_minus_{HOURLY}'] = dose(trace, hourly_trace)
    assert all(not d['scheduled_release_changed_trips'] for d in doses.values())
    write(root / 'triples' / str(day) / 'summary/results.json', dict(day=day, paired_events=5200,
          physical_starts=3, training_updates=0, main_policy_changed=False, arms=rows, action_dose=doses))


def aggregate(root):
    rows = {d: read(root / 'triples' / str(d) / 'summary/results.json') for d in DAYS}
    assert all(r['day'] == d and r['paired_events'] == 5200 and r['physical_starts'] == 3
               and set(r['arms']) == set(PLAN['arms']) for d,r in rows.items())
    starts = read(root / 'engineering/summary/results.json')['physical_starts']
    starts += sum(1 for _ in root.glob('cases/slow/*/*/started.json'))
    assert starts == PLAN['physical_start_cap']
    np = prior.run.np
    draws = np.asarray(list(itertools.product(range(len(DAYS)), repeat=len(DAYS))))
    effects = {}
    for a,b in ((SPACING, HOURLY), (SPACING, BASE), (HOURLY, BASE)):
        by_metric = {}
        for metric in rows[DAYS[0]]['arms'][BASE]['metrics']:
            d = np.asarray([r['arms'][a]['metrics'][metric]-r['arms'][b]['metrics'][metric] for r in rows.values()])
            assert np.isfinite(d).all(), metric
            by_metric[metric] = dict(mean=float(d.mean()), by_day=dict(zip(DAYS, d.tolist())),
                days_improved=int((d < 0).sum()), ci95=np.quantile(d[draws].mean(1), [.025,.975]).tolist())
        effects[f'{a}_minus_{b}'] = by_metric
    hourly = effects[f'{SPACING}_minus_{HOURLY}']
    pooled = effects[f'{SPACING}_minus_{BASE}']
    checks = dict(dispatch_cv2_lower=hourly[DISPATCH]['mean'] < 0 and hourly[DISPATCH]['days_improved'] >= 3,
                  long_gap_excess_not_higher=hourly['mean_long_gap_excess_s']['mean'] <= 0,
                  readiness_lower_than_pooled=pooled[READINESS]['mean'] < 0,
                  public_delay_not_higher_than_pooled=pooled[DELAY]['mean'] <= 0,
                  journey_not_higher_than_either=max(hourly[JOURNEY]['mean'],pooled[JOURNEY]['mean']) <= 0,
                  arrival_cv2_not_higher_than_either=max(hourly[ARRIVAL]['mean'],pooled[ARRIVAL]['mean']) <= 0)
    write(root / 'aggregate/summary/results.json', dict(complete=True, development_screen_passed=all(checks.values()),
          physical_starts=starts, training_updates=0, main_policy_changed=False,
          checks=checks, effects=effects, by_day=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'engineering', 'day_case', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--day', type=int)
    a = parser.parse_args()
    globals()[a.phase](a.root, *(() if a.day is None else (a.day,)))
