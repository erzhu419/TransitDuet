#!/usr/bin/env python3
"""Synthetic-only server admission. No simulator import or physical path entry."""

import argparse
import json
from pathlib import Path
import sys
import time
import unittest

import freqduet_causal_event_graph as graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_v270_causal_event_graph import fixture


def protocol():
    return dict(stage='v270_complete_day_development_pilot', release='sealed_not_started',
        arms=['fixed360_bilateral025', 'event_graph_apc', 'event_graph_homogeneous_cost'],
        demand_days=[2071001, 2071002, 2071003, 2071004],
        actor_seed=12011, independent_runtime_seeds=[27011001, 27011002],
        paired_unit='demand day x runtime seed; common exogenous streams, not shared policy outcomes',
        engineering_preflight_cap=2, paired_full_days=8, matrix_paths=24,
        total_physical_start_cap=26, max_concurrent=8, automatic_retry=False,
        engineering_order=['fixed360 native replay', 'event graph observational no-op'],
        admission='exact native action/timing/endpoint equivalence, causal query/APC fields and prefix clocks; '
                  'verify common exogenous streams are indexed independently of policy event counts',
        fleet=12, action_bins_s=graph.ACTIONS.tolist(), upper='unchanged exact360 public schedule',
        forecasts='strict completed current-day APC/AVL visits per direction/stop; minimum 3; '
                  'last 3 visits must be uncensored for boarded/gap rate; all completed alight/incoming and mean run; '
                  'no future queues or destinations',
        horizon='query + longest route upper bound (ceil mean run +104 nonterminal service, '
                'native query/transition ticks, up to45 legal holding) '
                '+360*fleet+1; launches with public release <= query+360*fleet; fixed across candidates',
        solver=dict(beam_width=graph.BEAM_WIDTH, variables=graph.MAX_VARIABLES,
                    evaluations=graph.MAX_EVALUATIONS, events=graph.MAX_EVENTS,
                    optimality_claim=False, fallback='retain native on unsupported prefix or no feasible strict gain'),
        operational_obligations='commands already issued and current service are fixed; first current legal command only; '
                                'future plan revisable at every query; FIFO native return+1 dispatch, no terminal holding',
        model_constraints='CV2, first return and next launch per active vehicle, and each included public launch delay no worse than '
                          'same-query bilateral forecast; equal comparison boundaries, no H-as-slack',
        statistics='8 day/runtime pairs, paired means and 95% crossed day/runtime bootstrap; '
                   'single actor is a development anchor, not an actor-generalization claim',
        endpoints=['once-only waiting+onboard person-seconds', 'actual full-day arrival CV2',
                   'vehicle readiness delay', 'total public dispatch delay including control waits',
                   'unlaunched/incomplete trips', 'actual holding vehicle/person-seconds'],
        continue_gate='APC arm passenger upper95<0 and CV2/readiness/public-delay upper95<=0 vs unchanged baseline; '
                      'no extra unfinished service; repeat effect direction in both runtime-seed halves; '
                      'APC-vs-homogeneous must improve the same tradeoff, otherwise no allocation-information claim',
        failure_decision='close this frozen pilot if joint gate fails; no forecast/window/weight tuning on its outcomes',
        independent_confirmation='new demand days and remaining actors require a separate release after development gain; '
                                 'only then matched-budget frequency ablations',
        physical_execution_authorized_now=False, main_policy_changed=False, training_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / 'tests'),
                                               pattern='test_v270_causal_event_graph.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.testsRun or not result.wasSuccessful():
        raise AssertionError('synthetic admission failed; traffic protocol remains closed')
    snapshot, visits = fixture()
    g = graph.assemble(snapshot, visits)
    records = {}
    for name, homogeneous in (('apc', False), ('homogeneous', True)):
        start = time.perf_counter()
        selected = graph.solve(g, homogeneous)
        records[name] = dict(action_s=selected['action_s'], evaluations=selected['evaluations'],
            status=selected['status'], wall_s=time.perf_counter()-start,
            model_deltas={k: selected['selected'][k]-selected['reference'][k]
                          for k in ('cost', 'waiting', 'onboard', 'cv2')},
            feasible=graph.feasible(selected['selected'], selected['reference']))
    # Detect whether a full-fleet query is computationally usable before wiring a simulator.
    routes = {d: tuple(graph.Stop(100*int(d)+i, control=i < 22, terminal=i == 22) for i in range(23))
              for d in (False, True)}
    visits = [graph.Visit(d, stop.station, snapshot.now_s-gap, snapshot.now_s-gap+4,
                         snapshot.now_s-gap+5, 20., 40., 0., 40. if i else 0., 360.)
              for d, stops in routes.items() for i, stop in enumerate(stops) for gap in (1080, 720, 360)]
    fleet = tuple(graph.Vehicle(i, i, bool(i % 2), 0 if i == 0 else i*2 % 22,
        'prepare' if i == 0 else 'travel', 40., arrival_s=snapshot.now_s-4 if i == 0 else None,
        boarded=0. if i == 0 else None) for i in range(12))
    plans = tuple(graph.Plan(100+2*i+int(d), d, snapshot.now_s+360*(i+1))
                  for i in range(12) for d in (False, True))
    large = graph.Snapshot(snapshot.now_s, snapshot.now_s, routes, fleet, plans, 0, 30)
    start = time.perf_counter()
    timed = graph.request(large, visits)
    records['full_fleet_query'] = dict(vehicles=12, stops_per_direction=23,
        wall_s=time.perf_counter()-start, evaluations=timed['evaluations'], status=timed['status'])
    summary = dict(source=a.source, status='v270_causal_graph_synthetic_checks_passed',
        tests=result.testsRun, examples=records, traffic_paths=0, training_updates=0,
        native_simulator_adapter_admitted=False, main_policy_changed=False,
        interpretation='Causal model and bounded solver checks only; no full-day policy benefit established.')
    for name, value in (('summary/status.json', summary), ('protocol.json', protocol())):
        path = a.out_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('V270_CAUSAL_EVENT_GRAPH_SYNTHETIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
