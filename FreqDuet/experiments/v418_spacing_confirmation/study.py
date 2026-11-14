"""Use the frozen V416 and V417 arms on a separate demand panel."""

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
V417 = WORK / 'FreqDuet-v417-pooled-spacing'
spec = importlib.util.spec_from_file_location('v417', V417 / 'source01/study.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
v416, prior = pilot.v416, pilot.prior
read, write, frozen = pilot.read, pilot.write, pilot.frozen
PLAN = read(HERE / 'protocol.json')
DAYS = PLAN['confirmation_days']
ARMS = (pilot.BASE, pilot.ARM, pilot.SPACING)


def preflight(root):
    source = read(V417 / 'run01/aggregate/summary/results.json')
    assert source['complete'] and source['physical_starts'] == 4 and source['reused_paths'] == 12
    assert not source['v416_gate'] and not source['main_policy_changed']
    assert PLAN['arms'] == list(ARMS) and PLAN['candidate'] == pilot.SPACING
    assert PLAN['regime'] == prior.REGIME and PLAN['road_profile_shift_mps'] == prior.PLAN['road_profile_shift_mps']
    assert PLAN['demand_scale'] == prior.PLAN['regimes'][prior.REGIME]['demand_scale']
    assert len(set(DAYS)) == len(DAYS) == PLAN['max_concurrent_days'] == 16
    assert len(DAYS) * len(ARMS) == PLAN['physical_start_cap'] == 48
    assert not PLAN['automatic_retry'] and not (root / 'cases').exists()
    assert set(DAYS).isdisjoint((*pilot.DAYS, *prior.DAYS, v416.OLD))
    capped = prior.old.base.Capped
    assert capped.select_plan.__globals__['choose'] is prior.frozen_v406.choose
    assert Path(v416.guarded_controller.__code__.co_filename) == pilot.V416 / 'source01/spacing.py'
    reserved, inspected = set(map(str, DAYS)), 0
    for project in WORK.glob('FreqDuet-*'):
        if project == HERE.parent or not project.is_dir():
            continue
        for d in DAYS:
            for pattern in ('run*/cases/*/{}', 'run*/days/{}', 'run*/triples/{}', 'run*/pairs/{}',
                            'run*/confirmation/{}', 'run*/running_shadow_confirmation/{}'):
                assert not next(project.glob(pattern.format(d)), None), (project, d)
        for manifest in project.glob('source*/*protocol.json'):
            assert not reserved & prior.old.base.declared_days(read(manifest)), manifest
            inspected += 1
    frozen.prepare(root)
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(passed=True, physical_starts=0,
          reused_frozen_arms=True, days=DAYS, arms=ARMS, manifests=inspected))


def case(root, day):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and gate['days'] == DAYS and gate['arms'] == list(ARMS) and day in DAYS
    physical = frozen.helper(frozen.ASSETS, 'physical')
    action_dose = frozen.helper(frozen.ASSETS, 'v324').actual_dose
    rows, doses, base, pooled_guard = {}, {}, None, None
    tape = None
    for arm in ARMS:
        result, trace, predictor, bins = (pilot.collect(root, day) if arm == pilot.ARM
                                            else v416.run_arm(root, day, arm))
        detail, op = result['passenger_detail'], result['operating_detail']
        coordination = result['graph']['coordination']
        rows[arm] = dict(metrics=dict(result['endpoints'], **v416.launch_gaps(trace),
            journey_p95_s=detail['journey_p50_p90_p95_s'][2], unboarded=detail['unserved'],
            incomplete=detail['incomplete']), generated=detail['generated'], bins=bins, predictor=predictor,
            spacing={k: v for k,v in coordination.items() if k.startswith('spacing_')},
            wall_s=result['wall_s'], callback_p95_s=op['callback_p50_p95_max_s'][1])
        assert bool(predictor.get('evaluations')) == (arm == pilot.SPACING)
        if arm == pilot.BASE:
            base = trace
            tape = result['contract']['scenario_tape_id']
            continue
        assert result['contract']['scenario_tape_id'] == tape
        assert rows[arm]['generated'] == rows[pilot.BASE]['generated']
        assert base['public_clocks'] == trace['public_clocks']
        crn = physical.crn_pair(base['uniforms'], trace['uniforms'])
        assert crn['exact'] and crn['common_events'] == 5200 and crn['left_only'] == crn['right_only'] == 0
        doses[f'{arm}_minus_{pilot.BASE}'] = action_dose(trace, base)
        if arm == pilot.ARM:
            pooled_guard = trace
        else:
            doses[f'{arm}_minus_{pilot.ARM}'] = action_dose(trace, pooled_guard)
    assert all(not d['scheduled_release_changed_trips'] for d in doses.values())
    write(root / 'days' / str(day) / 'summary/results.json', dict(day=day, arms=rows, scenario_tape_id=tape,
          paired_events=5200, physical_starts=3, training_updates=0, main_policy_changed=False, action_dose=doses))


def aggregate(root):
    rows = {d: read(root / 'days' / str(d) / 'summary/results.json') for d in DAYS}
    assert all(v['day'] == d and set(v['arms']) == set(ARMS) and v['paired_events'] == 5200
               and v['physical_starts'] == 3 and not v['training_updates'] and not v['main_policy_changed']
               for d,v in rows.items())
    assert len({v['scenario_tape_id'] for v in rows.values()}) == len(DAYS)
    starts = list((root / 'cases' / prior.REGIME).glob('*/*/started.json'))
    assert {(int(p.parent.parent.name), p.parent.name) for p in starts} == {(d,a) for d in DAYS for a in ARMS}
    assert len(starts) == PLAN['physical_start_cap']
    np, config = prior.run.np, PLAN['bootstrap']
    draws = np.random.default_rng(config['seed']).integers(len(DAYS), size=(config['draws'], len(DAYS)))
    effects = {}
    for a,b in ((pilot.SPACING, pilot.BASE), (pilot.SPACING, pilot.ARM), (pilot.ARM, pilot.BASE)):
        metrics = effects[f'{a}_minus_{b}'] = {}
        for key in rows[DAYS[0]]['arms'][a]['metrics']:
            delta = np.asarray([r['arms'][a]['metrics'][key]-r['arms'][b]['metrics'][key] for r in rows.values()])
            assert np.isfinite(delta).all(), key
            metrics[key] = dict(mean=float(delta.mean()), ci95=np.quantile(delta[draws].mean(1), [.025,.975]).tolist(),
                by_day=dict(zip(DAYS, delta.tolist())), days_lower=int((delta < 0).sum()))
    gates = {}
    primary = PLAN['readiness_superiority']
    for b in (pilot.BASE, pilot.ARM):
        metrics = effects[f'{pilot.SPACING}_minus_{b}']
        gates[b] = {k: metrics[k]['ci95'][1] <= v for k,v in PLAN['nonharm_upper_limits'].items()}
        gates[b][primary] = metrics[primary]['ci95'][1] < 0
    write(root / 'aggregate/summary/results.json', dict(complete=True,
          passed=all(all(g.values()) for g in gates.values()), checks=gates, independent_days=DAYS,
          physical_starts=len(starts), training_updates=0, main_policy_changed=False, effects=effects, by_day=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'case', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--day', type=int)
    args = parser.parse_args()
    if args.phase == 'case':
        case(args.root, args.day)
    else:
        globals()[args.phase](args.root)
