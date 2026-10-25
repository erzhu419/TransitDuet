"""Causal inputs, physical invariants and bounded joint search, not traffic efficacy."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import freqduet_causal_event_graph as graph
import freqduet_event_graph_design as v269


def fixture():
    q = 2000.
    routes = {d: (graph.Stop(10 * int(d)), graph.Stop(10 * int(d) + 1),
                   graph.Stop(10 * int(d) + 2, control=False, terminal=True)) for d in (False, True)}
    visits = []
    for d, stops in routes.items():
        for i, stop in enumerate(stops):
            for gap in (1080, 720, 360):
                visits.append(graph.Visit(d, stop.station, q - gap, q - gap + 4, q - gap + 5,
                    100., 40., 0., 40. if i else 0., 360.))
    vehicles = (graph.Vehicle(0, 0, False, 0, 'prepare', 40., arrival_s=q-4, boarded=0.),
                graph.Vehicle(1, -1, True, 0, 'terminal', 0., ready_s=q-100))
    plans = (graph.Plan(10, True, q+100), graph.Plan(11, False, q+720), graph.Plan(12, True, q+720))
    return graph.Snapshot(q, q, routes, vehicles, plans, 0, 30), visits


class CausalGraphTests(unittest.TestCase):
    def setUp(self):
        self.snapshot, self.visits = fixture()

    def test_strict_completion_and_availability(self):
        original = graph.assemble(self.snapshot, self.visits)
        q = self.snapshot.now_s
        extra = [replace(self.visits[0], arrival_s=q-1, completed_s=q, available_s=q, run_s=99999),
                 replace(self.visits[0], arrival_s=q-5, completed_s=q-1, available_s=q, run_s=99999)]
        candidate = graph.assemble(self.snapshot, self.visits + extra)
        self.assertEqual(original.forecasts, candidate.forecasts)
        self.assertEqual(original.horizon_s, candidate.horizon_s)
        self.assertEqual(original.evaluate()['cost'], candidate.evaluate()['cost'])

    def test_invalid_clocks_and_stale_query(self):
        with self.assertRaises(ValueError):
            graph.assemble(self.snapshot, self.visits + [replace(self.visits[0], available_s=0)])
        result = graph.request(replace(self.snapshot, sampled_s=self.snapshot.now_s-1), self.visits)
        self.assertEqual((result['status'], result['action_s']), ('unsupported_prefix', 30))

    def test_apc_can_be_available_before_the_visit_finishes(self):
        rows = [replace(r, available_s=r.arrival_s) for r in self.visits]
        self.assertEqual(graph.assemble(self.snapshot, rows).forecasts,
                         graph.assemble(self.snapshot, self.visits).forecasts)

    def test_support_and_capacity_censor_abstain(self):
        for rows in (self.visits[1:], [replace(r, left_behind=1) for r in self.visits]):
            result = graph.request(self.snapshot, rows)
            self.assertEqual((result['action_s'], result['evaluations']), (30, 0))

    def test_old_censor_does_not_disable_the_rest_of_day(self):
        old = replace(self.visits[0], arrival_s=100., completed_s=104., available_s=105., left_behind=10.)
        g = graph.assemble(self.snapshot, [old, *self.visits])
        self.assertEqual(g.forecasts[False, 0]['demand_visits'], 3)

    def test_terminal_has_no_holding_authority(self):
        bad = dict(self.snapshot.routes)
        bad[False] = (*bad[False][:-1], graph.Stop(2, control=True, terminal=True))
        with self.assertRaises(ValueError):
            graph.assemble(replace(self.snapshot, routes=bad), self.visits)

    def test_missing_current_arrival_or_board_count_abstains(self):
        for own in (replace(self.snapshot.vehicles[0], arrival_s=None),
                    replace(self.snapshot.vehicles[0], boarded=None)):
            result = graph.request(replace(self.snapshot, vehicles=(own, self.snapshot.vehicles[1])), self.visits)
            self.assertEqual(result['status'], 'unsupported_prefix')

    def test_arrival_service_departure_and_transfer(self):
        g = graph.assemble(self.snapshot, self.visits)
        base = g.evaluate({(0, 0, 0): 30, (0, 0, 1): 0})
        moved = g.evaluate({(0, 0, 0): 0, (0, 0, 1): 30})
        self.assertTrue(base['coverage'] and moved['coverage'])
        self.assertEqual(base['returns'], moved['returns'])
        self.assertEqual(base['launches'], moved['launches'])
        self.assertAlmostEqual(moved['onboard'] - base['onboard'], -1160.)
        self.assertEqual(base['waiting'], moved['waiting'])
        # Equal H and earlier alighting do not imply no-harm to the NEXT headway.
        self.assertGreater(moved['cv2'], base['cv2'])
        self.assertFalse(graph.feasible(moved, base))

    def test_homogeneous_control_changes_no_physical_forecast(self):
        g = graph.assemble(self.snapshot, self.visits)
        plan = {(0, 0, 0): 0, (0, 0, 1): 30}
        physical, blind = g.evaluate(plan), g.evaluate(plan, homogeneous=True)
        for key in ('arrivals', 'boardings', 'returns', 'launches', 'actions', 'cv2', 'waiting'):
            self.assertEqual(physical[key], blind[key])
        self.assertNotEqual(physical['onboard'], blind['onboard'])

    def test_committed_hold_is_irreversible(self):
        own = self.snapshot.vehicles[0]
        other = graph.Vehicle(1, 2, True, 0, 'hold', 10., remaining_s=20.,
                              arrival_s=self.snapshot.now_s-4, boarded=0.)
        g = graph.assemble(replace(self.snapshot, vehicles=(own, other)), self.visits)
        illegal = g.evaluate({(1, 2, 10): 0})
        self.assertIn((1, 2, 10), illegal['missing_commands'])
        departures = [t for t, kind, bus in illegal['nodes'] if kind == 'departure' and bus == 1]
        self.assertEqual(departures[0], self.snapshot.now_s+19)
        self.assertNotIn((1, 2, 10), illegal['legal'])

    def test_executed_h_is_not_spendable_slack(self):
        first = graph.assemble(self.snapshot, self.visits).evaluate()
        fleet = tuple(replace(v, executed_h_s=1e8) for v in self.snapshot.vehicles)
        second = graph.assemble(replace(self.snapshot, vehicles=fleet), self.visits).evaluate()
        for field in ('actions', 'returns', 'launches', 'cost'):
            self.assertEqual(first[field], second[field])

    def test_fifo_return_plus_one_and_no_added_stock(self):
        fleet = (*self.snapshot.vehicles, graph.Vehicle(2, -1, True, 0, 'terminal', 0., ready_s=1950))
        q = self.snapshot.now_s
        plans = (graph.Plan(10, True, q), graph.Plan(13, True, q), *self.snapshot.plans[1:])
        result = graph.assemble(replace(self.snapshot, vehicles=fleet, plans=plans), self.visits).evaluate()
        self.assertEqual(result['launches'][10], (q, 1))
        self.assertEqual(result['launches'][13], (q, 2))
        self.assertTrue(all(bus in (0, 1, 2) for _, bus in result['launches'].values()))
        returned = result['returns'][0]
        next_launch = min(t for t, bus in result['launches'].values() if bus == 0)
        self.assertGreaterEqual(next_launch, returned + 1)
        # A return observed during this tick cannot supply an earlier dispatch phase.
        later = replace(fleet[1], ready_s=q)
        result = graph.assemble(replace(self.snapshot, vehicles=(fleet[0], later), plans=plans[:1]), self.visits).evaluate()
        self.assertEqual(result['launches'][10], (q+1, 1))

    def test_boarding_cutoff_capacity_and_left_behind(self):
        visits = [replace(r, incoming_load=0., alighted=0., boarded=30.) if r.station in (0, 1) else r
                  for r in self.visits]
        own = replace(self.snapshot.vehicles[0], boarded=10.)
        g = graph.assemble(replace(self.snapshot, vehicles=(own, self.snapshot.vehicles[1])), visits)
        short, long = g.evaluate({(0, 0, 0): 0}), g.evaluate({(0, 0, 0): 45})
        # Already boarded riders are not left waiting in the forecast queue.
        self.assertAlmostEqual(short['initial_queue'][False, 0], 20.)
        for result in (short, long):
            self.assertTrue(all(row[5] <= 50. and row[6] >= 0. for row in result['boardings']))
            self.assertFalse(any(row[1:4] == (0, 0, 0) for row in result['boardings']))
        at_b = [r for r in long['boardings'] if r[1:4] == (0, 0, 1)][0]
        self.assertGreater(at_b[6], 0.)  # Future capacity pressure remains in the queue.
        # Independent closed form verifies the lazy aggregate waiting integral.
        span = g.horizon_s - self.snapshot.now_s
        rate = sum(f['rate'] for f in g.forecasts.values())
        expected = sum(long['initial_queue'].values()) * span + rate * span**2 / 2
        expected -= sum(r[4] * (g.horizon_s-r[0]) for r in long['boardings'])
        self.assertAlmostEqual(long['waiting'], expected, places=6)
        self.assertAlmostEqual(long['endpoint_queue'], sum(long['initial_queue'].values()) + rate*span
                               - sum(r[4] for r in long['boardings']), places=7)

    def test_arrivals_reorder_instead_of_using_hindsight_neighbors(self):
        v = graph.Vehicle(2, 3, False, 0, 'hold', 10., remaining_s=30.,
                          arrival_s=self.snapshot.now_s-5, boarded=0.)
        g = graph.assemble(replace(self.snapshot, vehicles=(*self.snapshot.vehicles, v)), self.visits)
        a = g.evaluate({(0, 0, 0): 45})
        b = g.evaluate({(0, 0, 0): 0})
        self.assertEqual([r[1] for r in a['boardings'] if r[2] in (0, 3) and r[3] == 1], [2, 0])
        self.assertEqual([r[1] for r in b['boardings'] if r[2] in (0, 3) and r[3] == 1], [0, 2])
        for result in (a, b):
            gaps = np.array([x for times in result['arrivals'].values() for x in np.diff(sorted(times))])
            self.assertAlmostEqual(result['cv2'], np.var(gaps) / np.mean(gaps)**2)

    def test_solver_masks_budget_and_first_action(self):
        own = replace(self.snapshot.vehicles[0], legal=(0, 30))
        g = graph.assemble(replace(self.snapshot, vehicles=(own, self.snapshot.vehicles[1])), self.visits)
        result = graph.solve(g)
        self.assertIn(result['action_s'], own.legal)
        self.assertLessEqual(result['evaluations'], graph.MAX_EVALUATIONS)
        self.assertFalse(result['optimality_claim'])
        self.assertTrue(graph.feasible(result['selected'], result['reference']))
        self.assertLessEqual(result['selected']['cost'], result['reference']['cost'])
        first_departure = next(t for t, kind, bus in result['selected']['nodes'] if kind == 'departure' and bus == 0)
        self.assertEqual(first_departure, self.snapshot.now_s + max(1, result['action_s']))
        with patch.object(graph, 'MAX_EVALUATIONS', 1):
            bounded = graph.solve(g)
        self.assertEqual((bounded['action_s'], bounded['evaluations']), (30, 1))

    def test_bounded_solver_finds_v269_known_feasible_joint_transfer(self):
        # The old finite mathematical certificate is a solver regression only;
        # its prescribed future prepares are never inputs to the causal assembler.
        spec = v269.fixture('alighting')
        keys = [(i, i, station) for station in (0, 1) for i in range(3)]

        class Reference:
            snapshot = SimpleNamespace(current_bus=0, native_action_s=30,
                vehicles=tuple(graph.Vehicle(i, i, False, 0, 'prepare', 40.) for i in range(3)),
                routes={False: (graph.Stop(0), graph.Stop(1))})

            def evaluate(self, commands=None, homogeneous=False):
                self_check.assertFalse(homogeneous)
                commands = commands or {}
                h = np.array([commands.get(k, spec['baseline'][j]) for j, k in enumerate(keys)])
                values = v269.model(h, spec)
                times = [*spec['prepare'], *(values['arrival_a'][0] + spec['service'])]
                return dict(cost=float(values['model_passenger_variable_person_s'][0]),
                    cv2=float(values['model_cv_squared'][0]),
                    returns=dict(enumerate(values['ready'][0])), next_launch=dict(enumerate(values['launch'][0])),
                    delays=dict(enumerate(values['launch'][0] - spec['published'])),
                    actions={k: (times[j], int(h[j])) for j, k in enumerate(keys)},
                    legal={k: tuple(graph.ACTIONS) for k in keys}, missing_commands=set(),
                    coverage=True, failure=None)

        self_check = self
        result = graph.solve(Reference())
        self.assertEqual(result['status'], 'model_improvement')
        self.assertTrue(graph.feasible(result['selected'], result['reference']))
        self.assertLess(result['selected']['cost'], result['reference']['cost'])
        self.assertNotEqual(result['action_s'], 30)
        self.assertLessEqual(result['evaluations'], graph.MAX_EVALUATIONS)

    def test_online_proposals_are_not_executed_commands(self):
        planner = graph.OnlinePlanner()
        result = planner.choose(self.snapshot, self.visits)
        current = (0, 0, 0)
        self.assertNotIn(current, planner.proposals)
        self.assertEqual(result['action_s'], result['commands'].get(current, 30))
        stale = replace(self.snapshot, sampled_s=self.snapshot.now_s-1)
        self.assertEqual(planner.choose(stale, self.visits)['action_s'], 30)
        self.assertEqual(planner.proposals, {})

    def test_event_budget_retains_native(self):
        g = graph.assemble(self.snapshot, self.visits)
        with patch.object(graph, 'MAX_EVENTS', 1):
            result = graph.solve(g)
        self.assertEqual((result['status'], result['action_s']), ('retain_native', 30))
        self.assertEqual(result['reference']['failure'], 'event_limit')

    def test_zero_holding_and_service_query_ticks_are_not_instantaneous(self):
        g = graph.assemble(self.snapshot, self.visits)
        result = g.evaluate({(0, 0, 0): 0, (0, 0, 1): 0})
        q = self.snapshot.now_s
        own = [(t, kind) for t, kind, bus in result['nodes'] if bus == 0]
        self.assertEqual(own[:5], [(q, 'prepare'), (q+1, 'departure'), (q+101, 'arrival'),
                                   (q+146, 'prepare'), (q+147, 'departure')])


if __name__ == '__main__':
    unittest.main()
