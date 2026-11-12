"""Execute and score the frozen heavy-demand confirmation on the servers."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
FROZEN = WORKSPACE / 'FreqDuet-v407-service-continuation/source03'
sys.path.insert(0, str(FROZEN))
import run as frozen
import numpy as np

PLAN = frozen.study.read(HERE / 'protocol.json')
REGIME = 'heavy'
DAYS = PLAN['days'][REGIME]
ARMS = PLAN['arms']
BASES = PLAN['reference_weights']
CANDIDATE = 'v406_single'
PRIMARY = ('journey_s_per_passenger', 'cv_squared', 'readiness_s_per_trip',
           'public_delay_s_per_day')


def declared_days(doc):
    values = []
    for key in ('days', 'confirmation_days', 'development_days', 'engineering_day'):
        group = doc.get(key, [])
        if isinstance(group, dict):
            for days in group.values():
                if isinstance(days, list):
                    values.extend(days)
        elif isinstance(group, list):
            values.extend(group)
        elif isinstance(group, (int, str)):
            values.append(group)
    return set(map(str, values))


def preflight(root):
    assert len(DAYS) == len(set(DAYS)) == PLAN['max_concurrent_days']
    assert len(DAYS) * len(ARMS) == PLAN['physical_start_cap']
    assert set(ARMS) == {CANDIDATE, *BASES}
    assert sum(BASES.values()) == 1
    assert PLAN['regimes'][REGIME] == frozen.PLAN['regimes'][REGIME]
    assert set(ARMS) <= set(frozen.ARMS)
    assert Path(sys.modules[frozen.FrozenController.__module__].__file__) == FROZEN / 'frozen_v406.py'
    assert Path(frozen.study.__file__).parent == frozen.RUNTIME / 'python'
    for project in WORKSPACE.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for d in DAYS:
            assert not list(project.glob(f'run*/cases/*/{d}')), (project, d)
        for manifest in project.glob('source*/protocol.json'):
            seen = declared_days(json.loads(manifest.read_text()))
            assert not seen.intersection(map(str, DAYS)), manifest

    frozen.prepare(root)
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    frozen.study.write(root / 'preflight/summary/results.json', dict(
        passed=True, physical_starts=0, days=DAYS, arms=ARMS, source=str(FROZEN)))


def day_pair(root, day):
    gate = frozen.study.read(root / 'preflight/summary/results.json')
    assert gate['passed'] and gate['days'] == DAYS and gate['arms'] == ARMS
    assert day in DAYS
    frozen.study.HERE = HERE
    frozen.study.make_audit = frozen.audit
    args = argparse.Namespace(root=root, assets=frozen.ASSETS)
    physical = frozen.helper(frozen.ASSETS, 'physical')
    arms, uniforms, generated = {}, None, None
    for arm in ARMS:
        result, trace, _ = frozen.study.collect(args, REGIME, day, arm)
        assert result['physical_account']['passed'] and len(trace['uniforms']) == 5200
        assert len(trace['queries']) == len(trace['graph_queries']) == 5200
        detail = result['passenger_detail']
        if uniforms is None:
            uniforms, generated = trace['uniforms'], detail['generated']
        assert detail['generated'] == generated
        assert physical.crn_pair(uniforms, trace['uniforms'])['exact']
        operating = result['operating_detail']
        assert operating['early_departures'] == 0
        arms[arm] = dict(metrics=dict(result['endpoints'], unboarded=detail['unserved'],
            incomplete=detail['incomplete'], journey_p95_s=detail['journey_p50_p90_p95_s'][2]),
            graph=result['graph'], wall_s=result['wall_s'],
            callback_p95_s=operating['callback_p50_p95_max_s'][1])
    frozen.study.write(root / 'paired' / str(day) / 'summary/results.json',
        dict(day=day, generated=generated, arms=arms, paired_events=5200))


def aggregate(root):
    rows, missing = {}, []
    for day in DAYS:
        path = root / 'paired' / str(day) / 'summary/results.json'
        if not path.is_file():
            missing.append(day)
            continue
        row = frozen.study.read(path)
        assert row['day'] == day and set(row['arms']) == set(ARMS)
        rows[day] = row
    starts = sum(1 for _ in (root / 'cases' / REGIME).glob('*/*/started.json'))
    assert starts <= PLAN['physical_start_cap']
    report = dict(protocol_id=PLAN['protocol_id'], complete=not missing, passed=False,
        missing_days=missing, physical_starts=starts, reference_weights=BASES,
        training_updates=0, main_policy_changed=False, by_day=rows)
    if not missing:
        assert starts == PLAN['physical_start_cap']
        cfg = PLAN['bootstrap']
        rng = np.random.default_rng(cfg['seed'])
        draws = rng.integers(0, len(DAYS), size=(cfg['draws'], len(DAYS)))
        effects = {}
        for name, weights in [('frozen_mixture', BASES),
                              *((arm, {arm: 1.}) for arm in BASES)]:
            effect = {}
            for metric in rows[DAYS[0]]['arms'][CANDIDATE]['metrics']:
                values = np.asarray([
                    row['arms'][CANDIDATE]['metrics'][metric] -
                    sum(w * row['arms'][arm]['metrics'][metric] for arm, w in weights.items())
                    for row in rows.values()])
                effect[metric] = dict(mean=float(values.mean()),
                    ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
                    by_day=dict(zip(DAYS, values.tolist())))
            effects[name] = effect
        primary = effects['frozen_mixture']
        gates = {metric: primary[metric]['ci95'][1] <= 0 for metric in PRIMARY}
        gates[PRIMARY[0]] = primary[PRIMARY[0]]['ci95'][1] < 0
        report.update(effects=effects, gates=gates, passed=all(gates.values()))
    frozen.study.write(root / 'aggregate/summary/results.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'day', 'aggregate'))
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--day', type=int)
    a = p.parse_args()
    if a.phase == 'day':
        day_pair(a.root, a.day)
    else:
        globals()[a.phase](a.root)
