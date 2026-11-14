"""Run only the missing pooled-plus-guard arm; read the other three archives."""

import argparse
import importlib.util
import itertools
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
V416 = HERE.parent.parent / 'FreqDuet-v416-dispatch-spacing'
ARCHIVE = V416 / 'run01'
spec = importlib.util.spec_from_file_location('frozen_v416_traffic', V416 / 'traffic01/traffic.py')
v416 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v416)
prior, read, write, frozen = v416.prior, v416.read, v416.write, v416.frozen
PLAN = read(HERE / 'protocol.json')
DAYS, ARM = PLAN['development_days'], PLAN['new_arm']
BASE, HOURLY, SPACING = v416.BASE, v416.HOURLY, v416.SPACING


def reference_path(day, arm):
    return ARCHIVE / 'cases' / prior.REGIME / str(day) / arm


def collect(root, day):
    capped = prior.old.base.Capped
    guarded = v416.guarded_controller(capped, prior.frozen_v406)
    with patch.object(prior.old.base, 'Capped', guarded):
        return prior.collect(root, day, ARM)


def preflight(root):
    source = read(ARCHIVE / 'aggregate/summary/results.json')
    assert source['complete'] and not source['development_screen_passed'] and source['physical_starts'] == 13
    assert DAYS == v416.DAYS and len(set(DAYS)) == len(DAYS) == PLAN['physical_start_cap'] == 4
    assert not PLAN['automatic_retry'] and ARM not in v416.PLAN['arms']
    assert not (root / 'cases').exists()
    for day in DAYS:
        row = read(ARCHIVE / 'triples' / str(day) / 'summary/results.json')
        assert row == source['by_day'][str(day)] and row['paired_events'] == 5200
        assert row['physical_starts'] == 3 and set(row['arms']) == {BASE, HOURLY, SPACING}
        for arm in row['arms']:
            case = reference_path(day, arm)
            saved = read(case / 'summary/results.json')
            assert saved['completed'] and saved['physical_account']['passed']
            assert saved['arm'] == arm and saved['day'] == day
            assert saved['passenger_detail']['generated'] == row['arms'][arm]['generated']
            assert all(row['arms'][arm]['metrics'][k] == v for k,v in saved['endpoints'].items())
            if arm != HOURLY:
                assert (case / 'raw/path.pkl.gz').is_file()

    capped = prior.old.base.Capped
    assert capped.select_plan.__globals__['choose'] is prior.frozen_v406.choose
    assert Path(prior.collect.__code__.co_filename) == v416.V415 / 'source01/study415.py'
    with patch.object(prior, 'collect', side_effect=lambda root, day, arm, clock_factory=None:
                      (prior.old.base.Capped, day, arm, clock_factory)):
        cls, day, arm, clock = collect(root, DAYS[0])
        assert cls is not capped and issubclass(cls, capped) and prior.old.base.Capped is capped
        assert day == DAYS[0] and arm == ARM and clock is None
    frozen.prepare(root)
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(passed=True, physical_starts=0,
          days=DAYS, arm=ARM, reused_paths=12, pooled_binding=True, guarded_selector=True))


def case(root, day):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and gate['days'] == DAYS and gate['arm'] == ARM and day in DAYS
    paid = read(ARCHIVE / 'triples' / str(day) / 'summary/results.json')['arms']
    result, trace, counts, bins = collect(root, day)
    detail, op = result['passenger_detail'], result['operating_detail']
    assert not counts and all(detail['generated'] == r['generated'] for r in paid.values())
    raw = frozen.helper(frozen.ASSETS, 'gate_default').raw
    physical = frozen.helper(frozen.ASSETS, 'physical')
    dose = frozen.helper(frozen.ASSETS, 'v324').actual_dose
    actions = {}
    for arm in (BASE, SPACING):
        archive = raw(reference_path(day, arm) / 'raw/path.pkl.gz')
        crn = physical.crn_pair(archive['uniforms'], trace['uniforms'])
        assert crn['exact'] and crn['common_events'] == 5200 and crn['left_only'] == crn['right_only'] == 0
        assert archive['public_clocks'] == trace['public_clocks']
        actions[arm] = dose(trace, archive)
        assert not actions[arm]['scheduled_release_changed_trips']
    coordination = result['graph']['coordination']
    row = dict(metrics=dict(result['endpoints'], **v416.launch_gaps(trace),
               journey_p95_s=detail['journey_p50_p90_p95_s'][2], unboarded=detail['unserved'],
               incomplete=detail['incomplete']), generated=detail['generated'], predictor=counts,
               spacing={k:v for k,v in coordination.items() if k.startswith('spacing_')},
               bins=bins, wall_s=result['wall_s'], callback_p95_s=op['callback_p50_p95_max_s'][1])
    assert set(row['metrics']) == set(paid[BASE]['metrics'])
    write(root / 'days' / str(day) / 'summary/results.json', dict(day=day, arm=ARM, physical_starts=1,
          paired_events=5200, training_updates=0, main_policy_changed=False, result=row, action_dose=actions))


def aggregate(root):
    parent = read(ARCHIVE / 'aggregate/summary/results.json')
    saved = {d: read(root / 'days' / str(d) / 'summary/results.json') for d in DAYS}
    assert all(r['day'] == d and r['arm'] == ARM and r['physical_starts'] == 1 and r['paired_events'] == 5200
               and not r['training_updates'] and not r['main_policy_changed'] for d,r in saved.items())
    assert sum(1 for _ in root.glob(f'cases/{prior.REGIME}/*/*/started.json')) == PLAN['physical_start_cap']
    arms = {d: dict(parent['by_day'][str(d)]['arms'], **{ARM: r['result']}) for d,r in saved.items()}
    pairs = dict(guard_pooled=(ARM, BASE), guard_hourly=(SPACING, HOURLY),
                 hourly_unguarded=(HOURLY, BASE), hourly_guarded=(SPACING, ARM))
    np = prior.run.np
    draws = np.asarray(list(itertools.product(range(len(DAYS)), repeat=len(DAYS))))

    def summary(values):
        assert np.isfinite(values).all()
        return dict(mean=float(values.mean()), by_day=dict(zip(DAYS, values.tolist())),
                    days_lower=int((values < 0).sum()),
                    ci95=np.quantile(values[draws].mean(1), [.025,.975]).tolist())

    effects = {k: {} for k in (*pairs, 'interaction')}
    for metric in arms[DAYS[0]][BASE]['metrics']:
        delta = {k: np.asarray([r[a]['metrics'][metric]-r[b]['metrics'][metric] for r in arms.values()])
                 for k,(a,b) in pairs.items()}
        delta['interaction'] = delta['guard_hourly']-delta['guard_pooled']
        for label, values in delta.items():
            effects[label][metric] = summary(values)
    write(root / 'aggregate/summary/results.json', dict(complete=True, physical_starts=4, reused_paths=12,
          training_updates=0, main_policy_changed=False, v416_gate=parent['development_screen_passed'],
          contrasts=pairs, effects=effects, new_arm_by_day=saved))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'case', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--day', type=int)
    args = parser.parse_args()
    globals()[args.phase](args.root, *(() if args.day is None else (args.day,)))
