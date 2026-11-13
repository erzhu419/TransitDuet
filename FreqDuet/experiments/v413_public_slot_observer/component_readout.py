"""Read only the accepted old-day component sidecar; no traffic or fitting."""

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_readout',HERE.parent/'readout01/readout.py')
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
run, read, write = readout.run, readout.read, readout.write
sys.path.insert(0,str(HERE.parent/'components02'))
from components import FIELDS, actual_components


def compare(predicted, actual):
    differences = {key:predicted['components'][key]-actual['components'][key] for key in FIELDS}
    error = predicted['return_s']-actual['return_s']
    assert sum(differences.values()) == error
    return dict(n=1, return_error_s=error, absolute_return_error_s=abs(error), **differences,
        future_requested_h_error_s=predicted['future_requested_h_s']-actual['future_requested_h_s'])


def checks():
    a = dict(return_s=150, components=dict(travel_s=70,service_prepare_s=20,hold_residence_s=10),
             future_requested_h_s=10)
    b = dict(return_s=170, components=dict(travel_s=100,service_prepare_s=15,hold_residence_s=5),
             future_requested_h_s=5)
    result = compare(a,b)
    assert result['return_error_s'] == -20 and result['travel_s'] == -30
    assert result['service_prepare_s'] == result['hold_residence_s'] == 5


def review(root):
    checks()
    gate = read(root/'component_parity/summary/results.json')
    assert gate['passed'] and gate['exact_endpoints'] and gate['exact_public_slots']
    assert gate['physical_starts'] == 1 and gate['day'] == run.PARITY
    folder = root/'component_parity/cases'/run.REGIME/str(run.PARITY)/run.CAP
    trace = run.old.frozen.helper(run.old.frozen.ASSETS,'gate_default').raw(folder/'raw/path.pkl.gz')
    slots = read(folder/'raw/public_slots.json')
    recorded = read(folder/'raw/predicted_components.json')
    indexed = {}
    for row in recorded:
        for key in row['slots']:
            assert tuple(key) not in indexed
            indexed[tuple(key)] = row
    assert len(indexed) == len(slots) == gate['counts']['slots']
    launches = {r['trip']:r for r in trace['dispatches']}
    previous, predecessors = {}, {}
    for launch in sorted(launches.values(),key=lambda r:(r['actual_s'],r['trip'])):
        predecessors[launch['trip']] = previous.get(launch['bus'])
        previous[launch['bus']] = launch['trip']
    groups, cache, examples = {}, {}, []

    def actual(bus,trip,query):
        key = bus,trip,query
        if key not in cache:
            cache[key] = actual_components(trace,bus,trip,query)
        return cache[key]

    for slot in slots:
        trip,lead,query = slot['trip'],slot['lead_s'],slot['query_s']
        sidecar = indexed[trip,lead]
        if not sidecar['available']:
            continue
        for scenario, components in zip(slot['scenarios'],sidecar['scenarios']):
            scenario_id = scenario['scenario']
            group = groups.setdefault(f'{lead}/scenario_{scenario_id}',dict(coverage=Counter(),by_status=defaultdict(Counter)))
            coverage = group['coverage']
            coverage['slots'] += 1
            if scenario['failure']:
                coverage['model_failure'] += 1
                continue
            predicted = scenario['launches'].get(str(trip))
            if predicted is None:
                coverage['missing_launch'] += 1
                continue
            predicted_launch,predicted_bus = predicted
            launch = launches[trip]
            late,real_late = predicted_launch>slot['public_s'],launch['actual_s']>slot['public_s']
            status = ('correct_late' if real_late else 'correct_on_time') if late==real_late else ('false_alarm' if late else 'miss')
            coverage[status] += 1
            by_bus = {c['bus']:c for c in components if c['supported']}
            if lead == 300 and scenario_id == 0 and trip in (225,227,236,249,260):
                for bus, component in by_bus.items():
                    if bus != 10 and not (trip==225 and component['trip']==214):
                        continue
                    measured = actual(bus,component['trip'],query)
                    if measured['supported']:
                        examples.append(dict(public_trip=trip, query_s=query, bus=bus, active_trip=component['trip'],
                            predicted_assignment=predicted_bus,actual_assignment=launch['bus'],
                            predicted=component,actual=measured,errors=compare(component,measured)))
            if predicted_bus != launch['bus']:
                coverage['assignment_changed'] += 1
                continue
            component = by_bus.get(predicted_bus)
            if component is None or component['trip'] != predecessors[trip]:
                coverage['not_current_cycle'] += 1
                continue
            if any(t<predicted_launch and bus==predicted_bus for other,(t,bus) in scenario['launches'].items()
                   if int(other)!=trip):
                coverage['model_reused_vehicle'] += 1
                continue
            measured = actual(predicted_bus,component['trip'],query)
            if not measured['supported']:
                coverage['actual_component_unsupported'] += 1
                continue
            coverage['matched'] += 1
            difference = compare(component,measured)
            difference['launch_error_s'] = predicted_launch-launch['actual_s']
            group['by_status'][status].update(difference)
    write(root/'component_readout/summary/results.json',dict(complete=True, physical_starts=0,
        day=gate['day'], training_updates=0, main_policy_changed=False, groups=groups,
        focal_chain=examples, matched_vehicle_queries=len(cache),
        scope='One consumed engineering day. Component differences include reference-continuation vs actual-control differences.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    review(parser.parse_args().root)
