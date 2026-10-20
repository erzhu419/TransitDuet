from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
