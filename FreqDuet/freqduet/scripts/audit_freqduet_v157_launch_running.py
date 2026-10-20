#!/usr/bin/env python3
"""Read the seven consumed V156 launch queries; leave every forecast frozen."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v156_clearance_support as audit


class RecordedRunning(audit.candidate.RunningLaw):
    def __init__(self, cells, held, prefix):
        super().__init__(cells, held, prefix)
        self.calls = {}

    def estimate(self, direction, station, now, entry, native):
        value = super().estimate(direction, station, now, entry, native)
        self.calls[direction, station] = dict(query_time_s=now, entry_time_s=entry,
            native_mean_s=native['mean'], selected=value,
            prior_day_moments=self.prior.get((direction, station, int(entry//3600)), []))
        return value


def case(args, target, running, services, geometry):
    seed, day, mode = [target[k] for k in ('seed', 'day', 'mode')]
    path, objects = audit.frozen.prepare(args.archive_root, seed, day, mode, geometry)
    service = audit.frozen.frozen.candidate.ServiceLaw(services['actors'][str(seed)],
        path['arrivals'], path['ready'], path['cutoff'])
    phases = audit.frozen.frozen.candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
    prefix = audit.frozen.prior.RunningPrefix(path, geometry)
    cells = running['actors'][str(seed)]
    histories = dict(full=cells, single=audit.retain_days(cells, [min(running['days'])]),
        none=audit.retain_days(cells, []))
    rows, ledgers = {a: [] for a in audit.ARMS}, []
    for k, row in path['trace'].items():
        now = row['prediction_time_s']
        label, truth = audit.stats.target_label(row, path['ready'])
        if label != 'resolved' or now < path['cutoff']:
            continue
        trip = audit.stats.follower(row)[0]
        if path['dispatch'][trip]['actual_s'] != now:
            continue
        p, status, _ = audit.candidate.frozen.position(path, row, objects[-1], geometry, objects, service, True)
        if status != 'observed_same_clock_launch':
            continue
        reference, _, _ = audit.accounting.structured_packet(path, row, geometry, objects, phases, service, True)
        before = sum(reference.values())
        actual = audit.parts.actual_parts(path, row, geometry[bool(row['direction'])])
        audit.close(sum(actual.values()), truth)
        native_legs = audit.frozen.prior.legs(path, row, p, prefix, objects[0])
        audit.close(sum(v['truth_s'] for v in native_legs), actual['running'])
        audit.close(sum(v['predicted_s'] for v in native_legs), reference['running'])
        points, components, laws = {}, {}, {}
        for arm, history in histories.items():
            law = RecordedRunning(history, day, prefix)
            after, detail = audit.candidate.forecast(path, row, geometry, objects, phases, service, law)
            if arm == 'none':
                audit.close(before, after)
            packet = audit.shadow_packet(reference, before, after, detail['running_s'])
            components[arm] = dict(predicted=audit.accounting.coarse(packet),
                errors=audit.accounting.component_errors(dict(v149=reference, joint=packet), actual))
            points[arm], laws[arm] = after, law
            record = audit.full_record((*audit.stats.follower(row), *k[1:]), before, after, truth)
            b, a = reference['running']-actual['running'], detail['running_s']-actual['running']
            record.update(reference_running_bias_s=b, shadow_running_bias_s=a,
                reference_running_mae_s=abs(b), shadow_running_mae_s=abs(a), delta_running_mae_s=abs(a)-abs(b))
            if detail['mean_prior_days'] is not None:
                record['mean_prior_days'] = detail['mean_prior_days']
            rows[arm].append(record)
        legs = []
        for i, leg in enumerate(native_legs, p['origin']+1):
            station = p['route']['ids'][i]
            shadows = {}
            for arm, law in laws.items():
                call = law.calls.get((p['direction'], station))
                prediction = leg['predicted_s'] if call is None else call['selected']['mean']
                shadows[arm] = dict(predicted_s=prediction, bias_s=prediction-leg['truth_s'],
                    partial_leg_retained=call is None, completed_prefix_and_prior=call)
            legs.append(dict(**leg, arms=shadows))
        for arm in audit.ARMS:
            audit.close(sum(v['arms'][arm]['predicted_s'] for v in legs), components[arm]['predicted']['running'])
        ledgers.append(dict(seed=seed, day=day, mode=mode, query=list(k), time_s=now,
            follower=list(audit.stats.follower(row)), reference_s=before, truth_s=truth,
            points_s=points, true_components=audit.accounting.coarse(actual),
            components=components, legs=legs))
    for arm in audit.ARMS:
        reproduced = audit.stats.describe(rows[arm])
        expected = target['arms'][arm]
        if reproduced['queries'] != expected['queries']:
            raise AssertionError('consumed launch-query support changed')
        for field, value in expected['metrics'].items():
            audit.close(reproduced['metrics'][field], value)
    return ledgers


def run(args):
    parent = audit.read(args.tradeoffs_root/'summary/status.json')
    if parent['status'] != 'v156_saved_cell_tradeoffs_complete' or parent['execution_source'] != '830be90017':
        raise AssertionError('requires the completed V156 paired readout')
    running, services = audit.read(args.v154_root/'model.json'), audit.read(args.v151_root/'model.json')
    if services['execution_source'] != '1b783ea8f3' or running['status'] != 'sealed_development_moments':
        raise AssertionError('frozen priors differ')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v157_launch_running.py', '-v'], check=True)
    geometry = audit.frozen.native.model.load_geometry(args.runtime)
    ledgers = [r for t in parent['clearance_launch_cells'] for r in case(args, t, running, services, geometry)]
    audit.write(args.out_root/'summary/status.json', dict(status='v157_saved_launch_accounting_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], queries=len(ledgers),
        saved_path_count=len(parent['clearance_launch_cells']), ledgers=ledgers,
        original_v156_endpoints_reproduced=True, new_rollouts=0, calibration_refit=False,
        main_policy_changed=False, v151_model_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], controller_admitted=False,
        limitations='Seven outcome-selected consumed launch queries, not independent confirmation '
            'or population inference. Only original full/single/none points are reconstructed. '
            'All prior-day moments are shown at each frozen arm entry clock, not scored as new '
            'candidate arms. True running durations and future entry hours are diagnostic labels, '
            'never forecast inputs. No best-day selection, weight change or gate reinterpretation.'))
    print('V157_SAVED_LAUNCH_ACCOUNTING_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v151-root', 'v154-root', 'tradeoffs-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
