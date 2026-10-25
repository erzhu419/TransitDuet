#!/usr/bin/env python3
"""Synthetic observation/recourse contract checks; no traffic data or rollout."""

import argparse
import json
from pathlib import Path
import unittest

import numpy as np

from freqduet_apc_flow_evidence import APCArrival, completed_flow
import freqduet_event_graph_design as design


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    tests = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern=pattern)
        for pattern in ('test_v273_flow_contract.py','test_v271_native_event_graph.py'))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun > 0 and result.wasSuccessful()
    rows = [APCArrival(i,i,i,False,1,float(i*60),float(i*60),float(i*60+1),
                      0.,0.,float(b),50.) for i,b in enumerate((10,50,40))]
    evidence = completed_flow(rows,122.)
    spec = design.fixture('alighting')
    planned = np.array((20,10,0,20,20,20))
    first_only = spec['baseline'].copy()
    first_only[0] = planned[0]
    good, full, ref = design.admissible(planned,spec,matched_amount=True)
    bad, partial, _ = design.admissible(first_only,spec,matched_amount=False)
    summary = dict(source=a.source, status='flow_evidence_and_recourse_contract_checks_passed',
        tests=result.testsRun, example_flow=evidence,
        recourse_counterexample=dict(full_plan_feasible=bool(good[0]), first_only_feasible=bool(bad[0]),
            full_plan_cv2_delta=float(full['model_cv_squared'][0]-ref['model_cv_squared'][0]),
            first_only_cv2_delta=float(partial['model_cv_squared'][0]-ref['model_cv_squared'][0])),
        traffic_paths=0, real_data_reads=0, model_fits=0, interval_fits=0,
        training_updates=0, online_action_selector_implemented=False, main_policy_changed=False,
        scope='Constructed observation identities and loss-of-recourse counterexamples, not an explanation of V272 or traffic benefit.')
    path = a.out_root/'summary/status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('V273_FLOW_CONTRACT_CHECKS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
