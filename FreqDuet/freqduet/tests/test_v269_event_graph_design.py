import unittest

import numpy as np

import freqduet_event_graph_design as graph


class EventGraphDesign(unittest.TestCase):
    def test_same_amount_relocation_preserves_downstream_clock_and_service(self):
        spec = graph.fixture('alighting')
        relocated = np.array((20, 10, 0, 20, 20, 20))
        allowed, selected, ref = graph.admissible(relocated, spec, matched_amount=True)
        self.assertTrue(allowed[0])
        for field in ('holding_by_vehicle', 'arrival_b', 'ready', 'launch', 'cv_numerator', 'cv_denominator'):
            np.testing.assert_array_equal(selected[field], ref[field])
        self.assertEqual(selected['model_passenger_variable_person_s'][0] - ref['model_passenger_variable_person_s'][0], -1200.)
        self.assertEqual(selected['model_waiting_person_s'][0], ref['model_waiting_person_s'][0])

    def test_joint_native_plan_has_a_strict_gain_without_less_holding(self):
        value = graph.certificate(graph.fixture('alighting'))
        self.assertTrue(value['strict_improvement'])
        self.assertEqual(value['per_vehicle_holding_before'], value['per_vehicle_holding_after'])
        self.assertLess(value['model_effects']['model_passenger_variable_person_s'], 0.)
        self.assertLessEqual(value['model_effects']['model_cv_squared'], 0.)
        self.assertEqual(value['universal_offset_joint_improvements'], 0)

    def test_no_cheap_position_no_artificial_passenger_gain(self):
        spec = graph.fixture('through')
        value = graph.certificate(spec)
        self.assertFalse(value['strict_improvement'])
        self.assertEqual(value['selected_actions'], value['baseline_actions'])

    def test_later_stop_at_cap_cannot_borrow_extra_authority(self):
        value = graph.certificate(graph.fixture('saturated'))
        self.assertFalse(value['strict_improvement'])
        self.assertTrue(np.isin(value['selected_actions'], graph.ACTIONS).all())

    def test_committed_actions_are_not_recovered_as_slack(self):
        spec = graph.fixture('committed')
        value = graph.certificate(spec)
        self.assertFalse(value['strict_improvement'])
        allowed, _, _ = graph.admissible([20, 10, 0, 20, 20, 20], spec, matched_amount=True)
        self.assertFalse(allowed[0])

    def test_dispatch_is_exact_max_not_a_free_lateness_variable(self):
        spec = graph.fixture('alighting')
        row = graph.model(spec['baseline'], spec)
        np.testing.assert_array_equal(row['launch'][0], [344, 714, 1084])
        np.testing.assert_array_equal(row['launch'], np.maximum(spec['published'], row['ready'] + 1))

    def test_non_native_action_is_not_silently_rounded(self):
        with self.assertRaises(AssertionError):
            graph.model([25, 10, 0, 20, 20, 20], graph.fixture('alighting'))


if __name__ == '__main__':
    unittest.main()
