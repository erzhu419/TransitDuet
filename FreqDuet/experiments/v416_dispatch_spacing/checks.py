"""Zero-traffic exercise of the real capped selector with controlled forecasts."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spacing import backlog_edges, excess_ok, guarded_controller


def check_selector(study):
    native, capped = study.frozen_v406, study.old.base.Capped
    assert capped.select_plan.__globals__['choose'] is native.choose
    controller = guarded_controller(capped, native)
    now, key = 720, (0, 10, 20)
    def plan(trip, release, direction=False):
        return NS(trip=trip, public_s=release, release_s=release, target_s=360,
                  direction=direction, min_gap_s=None)

    plans = [plan(2, 360), plan(3, 720), plan(4, 1080)]
    snapshot = NS(now_s=now, current_bus=0, native_action_s=0, plans=plans,
                  vehicles=[NS(bus=0, trip=10, direction=True, index=0, legal=(0,5,10,15))],
                  routes={True: [NS(station=20)]})
    reference = {2: 1200, 3: 1320, 4: 1530}
    fast = {2: 750, 3: 1320, 4: 1450}
    safe = {2: 1000, 3: 1310, 4: 1460}
    clocks = {0: reference, 5: fast, 10: safe}
    scores = {0: 100., 5: 70., 10: 90.}
    public = {p.trip: p.public_s for p in plans}

    def row(times, cost=100.):
        delay = {t: clock-public[t] for t,clock in times.items()}
        return dict(coverage=True, failure=False, cost=cost, cv2=.1, missing_commands=(),
                    launches={t: (s, t) for t,s in times.items()},
                    public_delays=delay, readiness_delays=delay)

    base = NS(snapshot=snapshot, initial_last_dispatch_s={False: 0, True: 0},
              evaluate=lambda commands=None: row(clocks[(commands or {}).get(key, 0)],
                                                 scores[(commands or {}).get(key, 0)]))
    blocks = backlog_edges(base)
    assert blocks == [(0, [(None, 2, 360), (2, 3, 360), (3, 4, 360)])]
    assert native.admissible(row(fast, 70.), row(reference), public)
    assert not excess_ok(row(fast), row(reference), blocks)
    assert excess_ok(row(safe), row(reference), blocks)
    assert not excess_ok(row({2: 1201, 3: 1320, 4: 1530}), row(reference), blocks)
    assert not excess_ok(row({2: 1200, 3: 1320, 4: 1700}), row(reference), blocks)
    assert excess_ok(row({2: 1000, 3: 1310}), row({2: 1200, 3: 1320}), blocks)

    def select(cls, models):
        from collections import Counter
        bridge = object.__new__(cls)
        bridge.models, bridge.pending, bridge.activity = models, None, Counter()
        bridge.manager = NS(method='single', due=False)
        return bridge.select_plan(models[0]), dict(bridge.activity)

    # First scenario permits 5 s; the second is the one that must veto it.
    other = NS(**vars(base))
    other.evaluate = lambda commands=None: row(
        (reference if not commands else safe), scores[(commands or {}).get(key, 0)])
    original, _ = select(capped, [other, base])
    picked, counts = select(controller, [other, base])
    assert original['action_s'] == 5 and picked['action_s'] == 10
    assert original['evaluations'] == picked['evaluations'] == 6
    assert counts['spacing_active_queries'] == counts['spacing_queries_with_rejection'] == 1
    assert counts['spacing_rejected_scenarios'] == 1
    assert select(capped, [other, base])[0] == original

    snapshot.now_s = now-1
    assert not backlog_edges(base)
    assert select(controller, [other, base]) == select(capped, [other, base])
    snapshot.plans = [*plans, plan(5, 360, True)]
    assert not backlog_edges(base)  # Two due trips in total, but in opposite directions.
    snapshot.now_s = now
    snapshot.plans += [plan(6, 720, True)]
    assert len(backlog_edges(base)) == 2
    return ['causal direction-specific backlog and both boundaries',
            'next-plan edge omitted only when absent from the reference continuation',
            'late-delay improvement can be vetoed by wider dispatch gaps',
            'each demand scenario can veto the same candidate',
            'inactive selector exact; active selector adds no model evaluations']


def preflight(root):
    v415 = HERE.parent.parent / 'FreqDuet-v415-running-closed-loop'
    audit = v415 / 'run01/divergence/summary/results.json'
    spec = importlib.util.spec_from_file_location('v415_study', v415/'source01/study415.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    evidence = study.read(audit)
    assert evidence['day'] == 2550004 and evidence['complete'] and evidence['physical_starts'] == 0
    result = dict(passed=True, physical_starts=0, training_updates=0, main_policy_changed=False,
                  checks=[*check_selector(study), *study.check_graph()])
    study.write(root/'preflight/summary/results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    preflight(parser.parse_args().root)
