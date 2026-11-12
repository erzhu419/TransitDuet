"""Bind a passive recorder to V412's unchanged slow capped path."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
V412 = HERE.parent.parent / 'FreqDuet-v412-running-speed-transfer'
sys.path[:0] = [str(HERE), str(V412 / 'source02')]
from study412 import CAP, SHIFT, old, np, read, shifted_profile, write
from observer import PublicSlots

PLAN = read(HERE / 'protocol.json')
REGIME, PARITY = PLAN['regime'], PLAN['engineering_day']
V412_ROOT = V412 / 'run01'
GATE = read(V412_ROOT / 'preflight/summary/results.json')


def preflight(root):
    from slot_checks import checks
    assert GATE['passed'] and PLAN['arm'] == CAP and REGIME == 'slow'
    days = PLAN['confirmation_days']
    assert len(days) == len(set(days)) == PLAN['planned_confirmation_paths'] == 8
    assert PARITY in GATE['days'][REGIME] and PARITY not in days
    assert PLAN['planned_parity_paths'] == 1 and PLAN['leads_s'] == [300, 0]
    prior = V412_ROOT / 'cases' / REGIME / str(PARITY) / CAP
    assert (prior / 'raw/path.pkl.gz').is_file() and read(prior / 'summary/results.json')['completed']
    examined = 0
    for project in HERE.parent.parent.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in days:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, days)) & old.base.declared_days(read(manifest)), manifest
            examined += 1
    old.frozen.prepare(root)
    bound = old.frozen.study.bind(old.frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    names = checks()
    write(root / 'preflight/summary/results.json', dict(passed=True, checks=names,
        physical_starts=0, training_updates=0, metadata_manifests=examined,
        parity_day=PARITY, confirmation_days=days, confirmation_started=False))


def collect(root, day):
    frozen = old.frozen
    frozen.prepare(root)
    original = old.make_audit
    observer = PublicSlots(PLAN['leads_s'])

    def make_audit(runtime, credit, arm, regime):
        gate = GATE['frozen_gate']
        base = original(runtime, credit, arm, regime, day, gate['counts'], gate['tilt'])

        class SlotAudit(type(base)):
            def attach(self, runner):
                super().attach(runner)
                frame = self.env.routes_set
                self.expected = frame.iloc[:, 5:].to_numpy(dtype=float) + SHIFT[REGIME]
                self.env.routes_set = shifted_profile(frame, SHIFT[REGIME])
                self.step_with_native_audit = self.env.step
                self.first_tick = self.last_tick = None

                def step(*args, **kwargs):
                    self.last_tick = self.env.current_time
                    if self.first_tick is None:
                        self.first_tick = self.last_tick
                    observer.before_step(self.env, self.bridge)
                    return self.step_with_native_audit(*args, **kwargs)

                self.env.step = step

            def detach(self):
                self.env.step = self.step_with_native_audit
                super().detach()

            def finish(self, runner, records, holds):
                super().finish(runner, records, holds)
                np.testing.assert_array_equal(
                    [r.speed_history.to_numpy(dtype=float) for r in self.env.routes], self.expected)
                assert self.env._demand_scale == 1.2 and self.contract['physical_vehicle_count'] == 12
                self.contract['applied_road_profile_shift_mps'] = SHIFT[REGIME]
                observer.finish(self.first_tick, self.last_tick)
                assert sum(r['lead_s'] == 0 for r in observer.records) == len(self.dispatches) == 262

        return SlotAudit('joint', credit)

    frozen.study.HERE = V412 / 'source02'
    frozen.study.make_audit = make_audit
    result, trace, out = frozen.study.collect(NS(root=root, assets=frozen.ASSETS), REGIME, day, CAP)
    write(out / 'raw/public_slots.json', observer.records)
    return result, trace, observer.records


def parity(root):
    assert read(root / 'preflight/summary/results.json')['passed']
    assert not (root / 'parity').exists()
    result, trace, slots = collect(root / 'parity', PARITY)
    prior = V412_ROOT / 'cases' / REGIME / str(PARITY) / CAP
    reference = read(prior / 'summary/results.json')
    archive = old.frozen.helper(old.frozen.ASSETS, 'gate_default').raw(prior / 'raw/path.pkl.gz')
    physical = old.frozen.helper(old.frozen.ASSETS, 'physical')
    names = ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'returns',
             'resource_predictions', 'slow_events', 'publish_ticks', 'public_clocks')
    for name in names:
        diff = physical.first_diff(archive[name], trace[name], name)
        assert diff is None, diff
    assert physical.crn_pair(archive['uniforms'], trace['uniforms'])['exact']
    for name in ('endpoints', 'execution', 'action_chain', 'contract'):
        diff = physical.first_diff(reference[name], result[name], name)
        assert diff is None, diff
    assert any(r['scenarios'] and not r['ready'] for r in slots)
    write(root / 'parity/summary/results.json', dict(passed=True, physical_starts=1,
        day=PARITY, exact_streams=list(names), exact_endpoints=True, paired_events=5200,
        snapshot_counts={h:sum(r['lead_s'] == h for r in slots) for h in PLAN['leads_s']},
        supported_counts={h:sum(r['lead_s'] == h and bool(r['scenarios']) for r in slots)
                          for h in PLAN['leads_s']}, confirmation_started=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'parity'))
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    globals()[a.phase](a.root)
