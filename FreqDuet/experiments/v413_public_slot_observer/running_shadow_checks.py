"""Exercise the new shadow predictor without starting traffic or opening labels."""

import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load('v413_frozen_run', HERE.parent / 'source02/run.py')
sys.path[:0] = [str(HERE), str(HERE.parent / 'components02')]
from dataclasses import replace
from running_shadow import HOUR_S, MIN_VISITS, RunningByHour, ShadowSlots, Unsupported, detached_kernel
from observer import slot_snapshot
from freqduet_fleet_observation import capture_fleet
from freqduet_native_continuation import inputs
from frozen_v406 import partial_forecasts, ReferenceCommands, ReferenceGraph
from coupled_model import _BudgetContinuation
from plan_model import observe_plans


def graph_fixture(env, bridge):
    snapshot = slot_snapshot(env, bridge, capture_fleet(env))
    bases, censored = partial_forecasts(snapshot, bridge.prefix.strict(env.current_time), bridge.prefix.flow_events, {})
    assert not censored
    plans, last = observe_plans(env)
    return ReferenceGraph(bases[0], inputs(env, bridge.prefix, bridge.evidence), plans, last, {})


def checks():
    fixtures = load('v413_slot_checks', HERE.parent / 'source02/slot_checks.py')
    env, bridge, _ = fixtures.fixture()
    key, now = (False, 20), env.current_time
    seed = bridge.prefix.visits[0]
    early = [replace(seed, arrival_s=t+25, run_s=25, completed_s=t+30, available_s=t+25)
             for t in (3590, 3580, 3570)]
    later = [replace(seed, arrival_s=t+50, run_s=50, completed_s=t+55, available_s=t+50)
             for t in (3600, 3610, 3620)]
    cutoff = [replace(later[0], run_s=55, completed_s=now), replace(later[0], run_s=55, available_s=now)]
    clock = RunningByHour({key:early + later + cutoff}, {key:dict(run_s=16.1)}, now)
    assert [clock.seconds(key,t) for t in (3599,3600,7200)] == [25,50,17]
    assert RunningByHour({key:later[:2]}, {key:dict(run_s=16.1)}, now).seconds(key,3600) == 17

    simulate = detached_kernel()
    model = graph_fixture(env, bridge)

    class Pooled:
        def seconds(self, key, entry_s):
            return model.forecasts[key]['run_s']  # Fixture running values are integral.

    def detached(clock):
        submitted = ReferenceCommands({}, {})
        return simulate(model, submitted, False, _BudgetContinuation(model, submitted), clock)

    baseline = model.evaluate()
    equivalent = detached(Pooled())
    assert {k:v for k,v in baseline.items() if k != 'cv2'} == {k:v for k,v in equivalent.items() if k != 'cv2'}

    class Probe(Pooled):
        def __init__(self):
            self.calls = []

        def seconds(self, key, entry_s):
            self.calls.append((key,entry_s))
            return super().seconds(key, entry_s) + 10

    probe = Probe()
    changed = detached(probe)
    assert changed['returns'][2] == baseline['returns'][2]  # Already on its final link.
    assert ((False,20), now+5) in probe.calls  # Already-issued command stays locked.
    first = model.snapshot.plans[0]
    assert ((first.direction,model.snapshot.routes[first.direction][0].station),
            changed['launches'][first.trip][0]) in probe.calls
    assert changed['returns'][1] == baseline['returns'][1]+10

    module = sys.modules['observer']
    original = module.PublicSlots([300,0])
    shadow = ShadowSlots(module, [300,0])
    graph = module.ReferenceGraph
    original.before_step(env,bridge)
    shadow.before_step(env,bridge)
    assert original.records == shadow.records and len(shadow.shadow) == 1 and module.ReferenceGraph is graph
    record = shadow.shadow[0]['scenarios'][0]
    assert record['link_calls']['hourly'] > 0 and not record['failure']
    assert {r['bus'] for r in record['baseline']} == {r['bus'] for r in record['candidate']} == {0,1,2}

    def unsupported(*args):
        raise Unsupported('shadow only')

    failed = ShadowSlots(module, [300,0])
    failed.simulate = unsupported
    failed.before_step(env,bridge)
    assert failed.records == original.records and failed.shadow[0]['scenarios'][0]['failure'] == 'shadow only'
    bridge.evidence.packets.clear()
    shadow.before_step(env,bridge)
    assert not shadow.shadow[-1]['scenarios'] and module.ReferenceGraph is graph
    return ['entry-hour boundary, strict completion and pooled fallback',
            'exact pooled kernel equivalence', 'unchanged partial link and issued command',
            'simulated launch and departure clocks used for full links',
            'unchanged public-slot records and restored hook after abstention',
            'candidate abstention does not suppress baseline forecast']


def preflight(root, parent):
    plan = run.read(HERE / 'running_shadow_protocol.json')
    assert plan['source_arm'] == run.CAP and plan['regime'] == run.REGIME
    assert plan['engineering_day'] == run.PARITY and plan['lead_s'] == run.PLAN['leads_s']
    assert plan['hour_seconds'] == HOUR_S and plan['min_completed_visits'] == MIN_VISITS
    gate = run.read(parent/'component_parity/summary/results.json')
    assert gate['passed'] and gate['exact_public_slots'] and gate['exact_endpoints']
    days = plan['confirmation_days']
    assert len(days) == len(set(days)) == plan['proposed_confirmation_paths'] == 8
    assert not plan['physical_starts_released'] and not plan['automatic_retry']
    assert not set(days) & {run.PARITY, *run.PLAN['confirmation_days']}
    examined = 0
    for project in HERE.parent.parent.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in days:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
            assert not list(project.glob(f'run*/confirmation/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, days)) & run.old.base.declared_days(run.read(manifest)), manifest
            examined += 1
    results = checks()
    run.write(root/'running_shadow_checks/summary/results.json', dict(passed=True, physical_starts=0,
        training_updates=0, main_policy_changed=False, checks=results, metadata_manifests=examined,
        engineering_day=plan['engineering_day'], confirmation_days=days, confirmation_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    preflight(args.root, args.parent)
