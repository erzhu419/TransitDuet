#!/usr/bin/env python3
"""Test the corrected mapping, then validate existing V274 records without replay."""

import argparse
from pathlib import Path
import sys
import time
import unittest

from freqduet_v274_passive_flow import compare_legacy, validate_flow
from run_freqduet_protocol_v6_checkpoint_trajectory import assert_same_state as same
from run_freqduet_v274_passive_flow import load_trace
from seal_freqduet_v274_flow_equivalence import REFERENCE, read, write


def run(a):
    start = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'),pattern=p) for p in
        ('test_v274_passive_flow.py','test_v273_flow_contract.py','test_v271_native_event_graph.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun > 0 and result.wasSuccessful()
    assert 'runner_v3' not in sys.modules and 'run_freqduet_v267_bilateral_factorial' not in sys.modules
    write(a.out_root/'synthetic.json',dict(source=a.source,tests=result.testsRun,status='passed',new_traffic_paths=0))
    root = a.results_root/'protocol_v6_v274_flow_equivalence'
    old = a.results_root/REFERENCE
    actual = read(root/'paths/flow_passive_c0.json')
    previous = read(old/'paths/noop_fix.json')
    failure = read(root/'failures/flow_passive_c0.json')
    assert actual['source'] == failure['source'] == 'bed2255dca'
    assert previous['source'] == '40eddb8fb6'
    assert failure['status'] == 'failed_slot_consumed'
    assert {p.stem for p in (root/'started').glob('*.json')} == {'flow_passive_c0'}
    native = load_trace(root/'traces/flow_passive_c0.pkl.gz')
    graph = load_trace(root/'traces/graph_queries_flow_passive_c0.pkl.gz')
    data = load_trace(root/'traces/flow_flow_passive_c0.pkl.gz')
    compare_legacy(native,actual,graph,load_trace(old/'traces/noop_fix.pkl.gz'),previous,
                   load_trace(old/'traces/graph_queries_noop_fix.pkl.gz'),same)
    flow = validate_flow(data,native,graph,same)
    same((flow['arrivals'],flow['headway_arrivals'],flow['terminal_arrivals'],flow['queries'],
          flow['cases']['first_arrivals'],flow['completed_visits']), (5502,5240,262,5200,42,5460),
         'V274_corrected_full_record_counts')
    assert len(native['holds']) == len(native['queries']) == 5200
    execution = actual['execution']
    assert execution['commanded_s'] == execution['executed_s'] == 38930
    assert execution['physical_vehicles'] == 12 and execution['done_reason'] == 'service_cleared'
    metrics = ('passenger_person_s','journey_min','cv_squared','readiness_delay_s',
               'public_launch_delay_total_s','holding_s_per_trip','trips_unlaunched','trips_incomplete')
    write(a.out_root/'results.json',dict(source=a.source,execution_source='bed2255dca',
        status='corrected_mapping_archived_checks_passed',tests=result.testsRun,
        original_task='t211740',original_error=failure['error'],
        original_verdict_preserved=failure['status'],original_exit_code=1,
        old_actions_timing_endpoints_and_records_exact=True,corrected_record_validation_passed=True,
        flow=flow,endpoint_deltas={k:actual['results'][k]-previous['results'][k] for k in metrics},
        limitations=['Two terminal first-return clocks lack independent old individual labels.',
                     'Old native/flow/whole-callback latency arrays were not persisted; no reconstruction is claimed.'],
        physical_slots_consumed=1,physical_start_cap=1,new_traffic_paths=0,training_updates=0,
        main_policy_changed=False,controller_admitted=False,complete_original_protocol_admission=False,
        wall_s=time.perf_counter()-start,
        scope='Supplemental read-only record validation after a mapping correction; not a replay or rewritten original verdict.'))
    print('V274_CORRECTED_MAPPING_ARCHIVE_CHECKS_COMPLETE',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
