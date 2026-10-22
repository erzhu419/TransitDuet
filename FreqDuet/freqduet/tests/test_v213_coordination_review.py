import unittest

import numpy as np

import audit_freqduet_v213_coordination as v
import audit_freqduet_v213_dispatch_binding as binding


def prepared(plan, actual, final):
    change = dict(trip_id=2, before=dict(planned_s=100.), after=dict(planned_s=plan))
    call = dict(trip_id=1, time_s=50., fresh_commands=[dict(coefficient=[0.])], changes=[change])
    return dict(data=dict(result=dict(upper=dict(trip=1, query_s=50.)), trace=dict(upper_callbacks=[call])),
        launches={2: dict(actual_s=actual)}, latest_plan={2: final})


class CoordinationReviewTests(unittest.TestCase):
    def test_vehicle_availability_follows_assigned_physical_vehicle(self):
        data = dict(trace=dict(actual_launches=[dict(trip=1, bus=4, actual_s=0.),
            dict(trip=2, bus=4, actual_s=151.)], vehicle_returns={'1': dict(return_s=150.)}, upper_callbacks=[]))
        result = binding.frame(data)
        self.assertEqual(result['availability'][2], dict(previous_trip=1, return_s=150.))
        self.assertIsNone(result['availability'][1]['return_s'])

    def test_plan_persists_but_execution_can_be_identical(self):
        counts, sums = v.plan_realization(prepared(100., 150., 100.), prepared(110., 150., 110.))
        self.assertEqual(counts['anchor_plan_changed_trips'], 1)
        self.assertEqual(counts['affected_launch_changed'], 0)
        self.assertEqual(counts['plan_difference_persisted_but_actual_unchanged'], 1)
        self.assertEqual(sums['absolute_anchor_plan_delta_s'], 10.)

    def test_later_plan_erasure_is_not_execution_blockage(self):
        counts, _ = v.plan_realization(prepared(100., 150., 100.), prepared(110., 150., 100.))
        self.assertEqual(counts['plan_difference_erased_before_launch'], 1)
        self.assertEqual(counts['plan_difference_persisted_but_actual_unchanged'], 0)

    def test_plan_realization_sign_and_missing_final_support(self):
        base, candidate = prepared(100., 100., 100.), prepared(110., 110., 110.)
        candidate['latest_plan'] = {}
        counts, _ = v.plan_realization(base, candidate)
        self.assertEqual(counts['affected_launch_same_sign'], 1)
        self.assertEqual(counts['latest_plan_unobserved'], 1)

    def test_ready_gain_and_completed_departure_gain_can_disagree(self):
        context = dict(evidence_valid=True, forward_headway_s=100., target_headway_s=360.)
        self.assertGreater(v.ready_marginal(context, 500., 0., 20.), 0.)
        neighbours = tuple(dict(time_s=t) for t in (0., 500., 600.))
        self.assertLess(v.departure_marginal(neighbours, 20., 360.), 0.)

    def test_unsupported_removal_or_ready_overtake_keeps_missing(self):
        context = dict(evidence_valid=True, forward_headway_s=100., target_headway_s=360.)
        self.assertIsNone(v.ready_marginal(context, None, 0., 5.))
        self.assertIsNone(v.ready_marginal(context, 3., 0., 5.))
        self.assertIsNone(v.departure_marginal(tuple(dict(time_s=t) for t in (0., 3., 100.)), 5., 360.))
        self.assertEqual(v.sign(None), 'unresolved')

    def test_cv_decomposition_includes_covariance_and_mean_normalization(self):
        def events(holds):
            return [dict(trip_id=i, station_id=2, direction=True, release=100.*i,
                realized_holding=float(h), other_elapsed=float(i*i), time_s=100.*i+h+i*i)
                for i, h in enumerate(holds)]
        b, c = v.moments(events([0, 5, 0, 10, 5])), v.moments(events([0, 0, 10, 5, 15]))
        parts = v.cv_parts(b, c)
        self.assertAlmostEqual(sum(parts.values()), c['cv2']-b['cv2'], places=12)
        self.assertNotEqual(parts['normalization'], 0.)
        self.assertNotEqual(parts['holding_other'], 0.)

    def test_descriptive_intervals_retain_contexts_and_future_zeroes(self):
        result = v.intervals([0.]*64)
        self.assertEqual(result['context_means'], [0.]*4)
        self.assertEqual(result['conditional_ci95'], [0., 0.])
        self.assertEqual(result['crossed_ci95'], [0., 0.])
        result = v.intervals(np.repeat([1., 2., 3., 4.], 16))
        self.assertEqual(result['mean'], 2.5)
        self.assertEqual(result['context_means'], [1., 2., 3., 4.])


if __name__ == '__main__':
    unittest.main()
