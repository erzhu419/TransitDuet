"""One extra passive parity path; reuse the checked V413 recorder and runtime."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_readout', HERE.parent/'readout01/readout.py')
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
run, read, write = readout.run, readout.read, readout.write
sys.path.insert(0,str(HERE.parent/'components02'))
from components import ComponentSlots, FIELDS

PLAN = read(HERE/'component_parity_protocol.json')


class PublicComponentSlots:
    def __init__(self, leads):
        self.tap = ComponentSlots(sys.modules['observer'], leads)
        self.records = self.tap.base.records

    def before_step(self, env, bridge):
        self.tap.before_step(env,bridge)

    def finish(self, start, end):
        self.tap.base.finish(start,end)


def same(a, b, label):
    physical = run.old.frozen.helper(run.old.frozen.ASSETS,'physical')
    for left,right in ((a,b),(b,a)):
        difference = physical.first_diff(left,right,label)
        assert difference is None, difference


def component_audit(slots, recorded):
    indexed = {(s['trip'],s['lead_s']):s for s in slots}
    seen, counts = set(), Counter()
    for sample in recorded:
        for key in sample['slots']:
            key = tuple(key)
            assert key not in seen
            seen.add(key)
            slot = indexed[key]
            assert sample['query_s'] == slot['query_s']
            assert sample['available'] == bool(slot['scenarios'])
            counts['slots'] += 1
            if not sample['available']:
                counts['unsupported_slots'] += 1
                continue
            assert len(sample['scenarios']) == len(slot['scenarios'])
            for scenario, rows in zip(slot['scenarios'],sample['scenarios']):
                counts['slot_scenarios'] += 1
                buses = [r['bus'] for r in rows]
                assert len(buses) == len(set(buses))
                assert set(scenario['returns']) <= set(buses)
                for row in rows:
                    if not row['supported']:
                        assert row['bus'] not in scenario['returns']
                        continue
                    assert row['return_s'] == scenario['returns'][row['bus']]
                    assert row['query_s'] == slot['query_s']
                    assert set(row['components']) == set(FIELDS)
                    assert all(v >= 0 for v in row['components'].values())
                    assert sum(row['components'].values()) == row['return_s']-row['query_s']
                    counts['vehicle_scenario_records'] += 1
    assert seen == set(indexed)
    return dict(counts)


def checks(root):
    gate = read(root/'component_audit/summary/results.json')
    assert gate['complete'] and gate['physical_starts'] == 0
    assert PLAN['day'] == run.PARITY and PLAN['physical_start_cap'] == 1
    assert not PLAN['automatic_retry'] and not PLAN['new_days']
    assert not (root/'component_parity').exists()
    prior = root/PLAN['reference']/'cases'/run.REGIME/str(run.PARITY)/run.CAP
    for name in ('raw/path.pkl.gz','raw/public_slots.json','summary/results.json'):
        assert (prior/name).is_file(), name
    spec = importlib.util.spec_from_file_location('slot_fixture', HERE.parent/'source02/slot_checks.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    env,bridge,_ = fixture.fixture()
    original_class = run.PublicSlots
    expected = original_class([300,0])
    actual = PublicComponentSlots([300,0])
    now = env.current_time
    for second in range(301):
        env.current_time = now+second
        if second:
            bridge.credit.stream.pending_commands.clear()
        expected.before_step(env,bridge)
        actual.before_step(env,bridge)
    expected.finish(now,now+300)
    actual.finish(now,now+300)
    same(expected.records,actual.records,'public_slots')
    counts = component_audit(actual.records,actual.tap.records)
    assert counts['slots'] and counts['vehicle_scenario_records']
    assert run.PublicSlots is original_class
    write(root/'component_checks/summary/results.json',dict(passed=True,physical_starts=0,
        day=PLAN['day'],physical_start_cap=1,counts=counts,
        checks=['complete legacy observer interface','same public slots across clock progression',
                'component associations and first-return closure','frozen reference and budget present']))


def parity(root):
    gate = read(root/'component_checks/summary/results.json')
    assert gate['passed'] and gate['day'] == PLAN['day']
    target = root/'component_parity'
    target.mkdir(exist_ok=False)
    # This launch allowance is consumed even if simulation construction fails.
    write(target/'started.json',dict(day=PLAN['day'],physical_starts=1,automatic_retry=False))
    original, recorders = run.PublicSlots, []

    def observer(leads):
        value = PublicComponentSlots(leads)
        recorders.append(value)
        return value

    run.PublicSlots = observer
    try:
        result,trace,slots = run.collect(target,PLAN['day'])
    finally:
        run.PublicSlots = original
    assert len(recorders) == 1
    recorded = recorders[0].tap.records
    case = Path('cases')/run.REGIME/str(PLAN['day'])/run.CAP
    write(target/case/'raw/predicted_components.json',recorded)
    prior = root/PLAN['reference']/case
    archive = run.old.frozen.helper(run.old.frozen.ASSETS,'gate_default').raw(prior/'raw/path.pkl.gz')
    assert set(trace) == set(archive) == set(PLAN['exact_streams'])
    for name in PLAN['exact_streams']:
        a,b = archive[name],trace[name]
        if name == 'graph_queries':
            a,b = ([{k:v for k,v in row.items() if k != 'total_s'} for row in values] for values in (a,b))
        same(a,b,name)
    receipt = read(prior/'summary/results.json')
    for field in PLAN['exact_receipt_fields']:
        same(receipt[field],result[field],field)
    same(read(prior/'raw/public_slots.json'),read(target/case/'raw/public_slots.json'),'public_slots')
    counts = component_audit(slots,recorded)
    assert len(trace['uniforms']) == len(trace['queries']) == 5200
    write(target/'summary/results.json',dict(passed=True,physical_starts=1,day=PLAN['day'],
        exact_streams=PLAN['exact_streams'],exact_public_slots=True,exact_endpoints=True,
        paired_events=5200,counts=counts,training_updates=0,main_policy_changed=False,
        snapshot_counts={lead:sum(s['lead_s']==lead for s in slots) for lead in run.PLAN['leads_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('checks','parity'))
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    globals()[args.phase](args.root)
