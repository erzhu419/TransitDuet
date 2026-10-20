from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v150_service_support as run


class SupportTests(unittest.TestCase):
    def test_existing_nine_sample_convention_is_descriptive(self):
        self.assertEqual([run.support_group(n) for n in (0, 1, 8, 9, 20)],
            ['none', 'one_to_eight', 'one_to_eight', 'nine_plus', 'nine_plus'])

    def test_budgets_preserve_global_episode_weights(self):
        def row(episode, group, base, phase):
            p = {g: [0., 0.] for g in (*run.SUPPORT, 'retained_structured')}
            p[group] = [base, phase]
            return dict(episode=[episode], group=group, base=base, phase=phase, stop_parts=p)
        rows = [row(1, 'none', 10., 10.), row(1, 'one_to_eight', 6., 2.),
            row(2, 'nine_plus', -4., -2.)]
        budgets = run.budgets(rows)
        self.assertEqual(sum(budgets['query/'+g]['metrics']['phase_allocated_bias_s']
            for g in run.QUERY_GROUPS), 2.)
        self.assertEqual(sum(budgets['query/'+g]['metrics']['fraction'] for g in run.QUERY_GROUPS), 1.)
        self.assertEqual(sum(budgets['stop/'+g]['metrics']['base_allocated_bias_s']
            for g in (*run.SUPPORT, 'retained_structured')), 2.)

    def test_empty_conditional_group_is_not_fabricated_observation(self):
        self.assertEqual(run.parts.prior.describe([]), dict(queries=0, episodes=0, metrics={}))

    def test_future_phase_samples_do_not_become_available_early(self):
        phases = run.frozen.candidate.PhaseServices(
            {(1, 1, True): dict(time_s=101.)}, {(1, 1, True): dict(ready_time_s=110.)}, 100.)
        self.assertIsNone(phases.estimate(True, 1, 110., 120.))
        self.assertEqual(phases.estimate(True, 1, 111., 120.)['count'], 1)

    def test_per_stop_reconstruction_keeps_unsupported_mean(self):
        route = dict(ids=[0, 1, 2], positions=[0., 100., 200.])
        history = run.frozen.candidate.model.CompletedMeans([
            ((ch, True, stop), 20., value) for stop in (1, 2)
            for ch, value in (('running', 10.), ('service', 20.), ('holding', 3.), ('ticks', 1.))])
        phases = run.frozen.candidate.PhaseServices({(1, 2, True): dict(time_s=101.)},
            {(1, 2, True): dict(ready_time_s=107.)}, 100.)
        p = dict(route=route, origin=0, target=2, progress=0., now=108., initial_s=0., direction=True)
        stops, gap = run.stop_predictions(p, history, phases)
        self.assertEqual([s['phase_n'] for s in stops], [0, 1])
        self.assertEqual([s['selected_s'] for s in stops], [20., 6.])
        self.assertEqual(gap, 50.)

    def test_same_time_launch_timestamp_is_not_observation_proof(self):
        route = dict(ids=[0, 1], positions=[0., 100.], index={0: 0, 1: 1})
        row = dict(prediction_time_s=20., station_id=1, direction=True, online={},
            evidence=dict(predicted_follower_trip_id=7, spatial_gap_m=100.))
        index = SimpleNamespace(motion=None, inspect=lambda *args: dict(
            origin=None, same_clock_events=['departure'], reason='no_completed_departure'))
        history = run.frozen.candidate.model.CompletedMeans([
            ((ch, True, 1), 19., 5.) for ch in ('running', 'service')])
        p = dict(dispatch={7: dict(actual_s=20.)})
        tag, data = run.boundary(p, row, dict(status='no_strict_physical_segment'),
            {7: index}, history, None, {True: route})
        self.assertEqual(tag, 'launch_at_query')
        self.assertEqual(data['all_required_component_means_available'], 1.)
        self.assertEqual(data['native_launch_observation_logged'], 0.)
        row['online']['launch_observed_s'] = 20.
        _, data = run.boundary(p, row, dict(status='no_strict_physical_segment'),
            {7: index}, history, None, {True: route})
        self.assertEqual(data['native_launch_observation_logged'], 1.)

    def test_service_survivor_failure_is_not_always_low_sample_count(self):
        current = (7, 1, True)
        arrival = dict(trip_id=7, station_id=1, direction=True, time_s=1.)
        index = SimpleNamespace(motion=SimpleNamespace(at=[1.], arrivals=[arrival]),
            inspect=lambda *args: dict(same_clock_events=['ready'], reason='pending_stop'))
        service = run.parts.frozen.ServiceResiduals([(True, 1, float(t), 4.) for t in range(2, 11)])
        p = dict(dispatch={7: dict(actual_s=0.)}, ready={current: dict(ready_time_s=20.)},
            trace={current: dict(policy_time_s=20., query_load=100)})
        row = dict(prediction_time_s=20., station_id=2, direction=True, online={},
            evidence=dict(predicted_follower_trip_id=7, spatial_gap_m=100.))
        route = dict(ids=[0, 1, 2], positions=[0., 100., 200.], index={0: 0, 1: 1, 2: 2})
        tag, data = run.boundary(p, row, dict(status='no_completed_service_survivor'),
            {7: index}, None, service, {True: route})
        self.assertEqual(tag, 'survivor_depleted_pool_ge9')
        self.assertEqual(data['pool_n'], 9.)
        self.assertEqual(data['retrospective_current_service_remaining_s'], 0.)
        self.assertEqual(data['current_follower_query_load_strictly_observed'], 0.)


if __name__ == '__main__':
    unittest.main()
