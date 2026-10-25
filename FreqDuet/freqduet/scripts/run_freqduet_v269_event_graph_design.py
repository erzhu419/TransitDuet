#!/usr/bin/env python3
"""Server-only mathematical mechanism checks; zero traffic rollouts or fits."""

import argparse
import json
from pathlib import Path
import unittest

import freqduet_event_graph_design as graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / 'tests'),
                                               pattern='test_v269_event_graph_design.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.testsRun or not result.wasSuccessful():
        raise AssertionError('V269 structural mechanism checks failed; no traffic phase is released')
    values = [graph.certificate(graph.fixture(case)) for case in graph.CASES]
    summary = dict(source=a.source, status='v269_event_graph_model_certificate_complete', tests=result.testsRun,
        cases=values, traffic_rollouts=0, new_model_fits=0, policy_parameter_search=False,
        real_controller_implemented=False, field_forecast_validated=False,
        main_policy_changed=False, confirmation_started=False,
        interpretation='Constructed event-graph mechanisms only, not traffic effectiveness or a causal deployed policy. '
            'Equal future H is a model comparison, not an online right to spend accumulated H. '
            'The exact six-control enumeration is a small reference solver, not a full-fleet runtime optimizer.')
    path = a.out_root / 'summary/status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('V269_EVENT_GRAPH_DESIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
