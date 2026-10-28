#!/usr/bin/env python3
"""V277 server-only synthetic checks; no traffic path or training entry point."""

import argparse
import json
from pathlib import Path
import time
import unittest

import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, choose_current
from check_freqduet_v276_indexed import stress_state


def run(a):
    tests = Path(__file__).resolve().parents[1]/'tests'
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests),pattern=name,
        top_level_dir=str(tests)) for name in
        ('test_v277_arrival_metric.py','test_v277_matrix.py','test_v276_indexed_continuation.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun >= 27
    reference = stress_state()
    model = ArrivalMetricGraph(reference,reference.observed)
    baseline = model.evaluate()
    assert baseline['coverage'] and not baseline['failure']
    measured = {}
    for name,solver in (('aligned_beam',flow.choose),('aligned_current',choose_current)):
        start = time.perf_counter()
        value = solver(model)
        measured[name] = dict(wall_s=time.perf_counter()-start,evaluations=value['evaluations'],
            action_s=value['action_s'], objective=value['selected']['cost'],
            cv2=value['selected']['cv2'], reference_cv2=value['reference']['cv2'],
            feasible=flow.graph.feasible(value['fallback'],value['reference']))
    assert measured['aligned_current']['evaluations'] <= 7
    assert measured['aligned_current']['objective'] <= measured['aligned_beam']['objective']+1e-9
    result = dict(source=a.source,status='synthetic_checks_passed',tests=result.testsRun,
        synthetic_same_state=measured,traffic_paths=0,training_updates=0,
        main_policy_changed=False,real_matrix_started=False,
        scope='Model comparison, not V276 output equivalence or full-day traffic benefit.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('V277_SYNTHETIC_CHECKS_COMPLETE',result['tests'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',required=True)
    run(parser.parse_args())
