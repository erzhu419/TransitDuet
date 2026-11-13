"""Zero-traffic component tests and actual-component inventory for the paid 81 cases."""

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_readout', HERE.parent / 'readout01/readout.py')
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
run = readout.run
PLAN, read, write = run.PLAN, run.read, run.write
sys.path.insert(0,str(HERE))
from components import ComponentSlots, FIELDS, actual_components, model_components


def checks():
    row = dict(returns={0:46}, nodes=[(15,'prepare',0), (20,'departure',0),
        (30,'arrival',0), (35,'prepare',0), (36,'departure',0), (46,'arrival',0)],
        actions={(0,5,1):(15,5), (0,5,2):(35,0)})
    value = NS(now_s=12, vehicles=[NS(bus=0,trip=5,phase='service')])
    model = model_components(value, row)[0]
    assert model['components'] == dict(travel_s=20, service_prepare_s=8, hold_residence_s=6)
    trace = dict(dispatches=[dict(bus=0,trip=5,actual_s=0)],
        returns=[dict(bus_id=0,trip_id=5,physical_time_s=46)],
        arrivals=[dict(trip_id=5,station_id=1,time_s=10),dict(trip_id=5,station_id=2,time_s=30)],
        departures=[dict(trip_id=5,station_id=1,time_s=20),dict(trip_id=5,station_id=2,time_s=36)],
        queries=[dict(bus=0,trip=5,station=1,time_s=15,executed_request_s=5),
                 dict(bus=0,trip=5,station=2,time_s=35,executed_request_s=0)])
    assert actual_components(trace,0,5,12) == model
    value.now_s, value.vehicles[0].phase = 17, 'hold'
    held = dict(row, nodes=row['nodes'][1:], actions={(0,5,2):(35,0)})
    assert model_components(value,held)[0] == actual_components(trace,0,5,17)
    assert model_components(value,held)[0]['components']['hold_residence_s'] == 4

    spec = importlib.util.spec_from_file_location('v413_slot_checks', HERE.parent / 'source02/slot_checks.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    env, bridge, _ = fixtures.fixture()
    observer = sys.modules['observer']
    original_graph = observer.ReferenceGraph
    original = observer.PublicSlots([300,0])
    original.before_step(env,bridge)
    recorded = ComponentSlots(observer,[300,0])
    recorded.before_step(env,bridge)
    physical = run.old.frozen.helper(run.old.frozen.ASSETS,'physical')
    assert physical.first_diff(original.records,recorded.base.records) is None
    assert physical.first_diff(recorded.base.records,original.records) is None
    assert observer.ReferenceGraph is original_graph
    models = recorded.base.records[0]['scenarios']
    for model, rows in zip(models, recorded.records[0]['scenarios']):
        assert all(r['supported'] and r['return_s'] == model['returns'][r['bus']] for r in rows)
        assert all(sum(r['components'].values()) == r['return_s']-r['query_s'] for r in rows)
    bridge.evidence.packets.clear()
    recorded.before_step(env,bridge)
    assert not recorded.records[-1]['available'] and observer.ReferenceGraph is original_graph
    return ['query clipping across service and committed holding', 'zero command retains physical tick',
            'unchanged frozen public-slot predictions', 'component sums reproduce every first return',
            'passive hook restored after unsupported observation']


def audit(root):
    tested = checks()
    details = read(root / 'error_audit/details.json')
    frozen = read(root / 'error_audit/summary/results.json')
    assert frozen['complete']
    archive = run.old.frozen.helper(run.old.frozen.ASSETS,'gate_default')
    totals, by_day, records = defaultdict(Counter), {}, {}
    for day in PLAN['confirmation_days']:
        folder = root/'confirmation'/str(day)/'cases'/PLAN['regime']/str(day)/PLAN['arm']
        trace = archive.raw(folder/'raw/path.pkl.gz')
        daily, actuals = defaultdict(Counter), []
        for error in details[str(day)]:
            bus, trip, now = error['actual_bus'], error['actual_predecessor'], error['query_s']
            result = actual_components(trace,bus,trip,now)
            kind = error['kind']
            daily[kind]['n'] += 1
            if result['supported']:
                assert result['return_s'] == error['same_cycle']['actual_return_s']
                daily[kind]['supported'] += 1
                daily[kind].update(result['components'])
                daily[kind]['future_requested_h_s'] += result['future_requested_h_s']
                daily[kind]['remaining_s'] += result['return_s']-now
            actuals.append(dict(result, target_trip=error['trip'], bus=bus, trip=trip, kind=kind))
        by_day[day], records[day] = daily, actuals
        for kind, counts in daily.items():
            totals[kind].update(counts)
    assert sum(r['n'] for r in totals.values()) == 81
    for kind, counts in totals.items():
        assert counts['n'] == frozen['totals'][kind]['n']
        assert sum(counts.get(k,0) for k in FIELDS) == counts.get('remaining_s',0)
    write(root/'component_audit/actual_components.json',records)
    write(root/'component_audit/summary/results.json',dict(complete=True, physical_starts=0,
        training_updates=0, main_policy_changed=False, synthetic_checks=tested,
        totals=totals, by_day=by_day, predicted_components_available_in_archive=False,
        observer_ready_for_physical_parity=False,
        fields='travel; service plus PREPARE/command latency; holding residence including zero ticks',
        scope='Actual remaining time on matched cycles; no attribution of forecast error to a component'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
