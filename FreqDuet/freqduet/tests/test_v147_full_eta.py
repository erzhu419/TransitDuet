from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v147_full_eta import (
    target_label, symmetric_identity, holding_parts, describe, inference, successors)


def row(trip=2, bus=22, now=100., prediction=120., forward=40., executed=10.):
    return dict(trip_id=1, station_id=5, direction=False, prediction_time_s=now,
        evidence=dict(predicted_follower_trip_id=trip, predicted_follower_bus_id=bus),
        context=dict(evidence_valid=True, follower_evidence_valid=True,
            follower_departure_gap_s=prediction, forward_headway_s=forward),
        online=dict(gap_s=prediction, component_used=True),
        feedback_error_s=(prediction-forward)/2., executed_s=executed)


def labels(a=160., b=190.):
    return {(2, 5, False): dict(bus_id=22, ready_time_s=a),
        (3, 5, False): dict(bus_id=33, ready_time_s=b)}


class FullEtaTests(unittest.TestCase):
    def test_target_is_the_predicted_trip_and_bus_not_the_next_vehicle(self):
        status, truth = target_label(row(), labels())
        self.assertEqual((status, truth), ('resolved', 60.))
        self.assertEqual(target_label(row(bus=23), labels())[0], 'same_trip_bus_mismatch')
        self.assertEqual(target_label(row(), {})[0], 'missing_same_trip_ready')
        self.assertIsNone(target_label(row(now=170.), labels())[1])

    def test_invalid_evidence_is_not_rescued_by_future_labels(self):
        r = row()
        r['context']['follower_evidence_valid'] = False
        self.assertEqual(target_label(r, labels()), ('no_valid_follower_evidence', None))

    def test_symmetric_identity_split_recovers_truth_change(self):
        b, c = row(), row(3, 33, 110.)
        physical, identity = symmetric_identity(b, c, labels(), labels(180., 200.))
        self.assertEqual((physical, identity), (5., 25.))
        self.assertEqual(physical+identity, (200.-110.)-(160.-100.))

    def test_stable_identity_has_zero_replacement(self):
        self.assertEqual(symmetric_identity(row(), row(now=110.), labels(), labels(180., 200.)), (10., 0.))

    def test_missing_cross_identity_anchor_is_not_filled(self):
        r = labels()
        del r[3, 5, False]
        self.assertIsNone(symmetric_identity(row(), row(3, 33, 110.), r, labels(180., 200.)))

    def test_holding_split_separates_prediction_and_quantization(self):
        b, c = row(), row(3, 33, 110., prediction=132., forward=44.)
        result = holding_parts(b, c, 60., 90., 5., 25.)
        self.assertEqual(result['holding_delta_s'], 0.)
        self.assertEqual(result['quantization_or_execution_s'], -1.)
        self.assertEqual(result['forecast_error_s'], -2.25)
        self.assertAlmostEqual(sum(result[k] for k in ('physical_ready_s', 'identity_s',
            'forecast_error_s', 'forward_gap_s', 'quantization_or_execution_s')), 0.)

    def test_holding_split_keeps_clipping(self):
        b = row(prediction=500., executed=45.)
        c = row(3, 33, 110., prediction=132., forward=44., executed=10.)
        result = holding_parts(b, c, 60., 90., 5., 25.)
        self.assertEqual(result['holding_delta_s'], -35.)
        self.assertEqual(result['quantization_or_execution_s'], -1.)

    def test_episode_weighting_precedes_path_weighting(self):
        d = describe([dict(episode=[1], error=10.), dict(episode=[1], error=20.),
            dict(episode=[2], error=100.)])
        self.assertEqual((d['queries'], d['episodes']), (3, 2))
        self.assertEqual(d['metrics']['error'], 57.5)

    def test_incomplete_group_has_no_subset_bootstrap(self):
        cells = [dict(seed=1, day=10, stats={'g': {'metrics': {'error': 1.}}}),
            dict(seed=1, day=20, stats={})]
        result = inference(cells, lambda c: c['stats'], 'g', 'error')
        self.assertEqual(result['contributing_cells'], 1)
        self.assertIsNone(result['ci95'])

    def test_departure_successor_uses_recorded_event_order_per_stop(self):
        events = [dict(trip_id=t, station_id=s, direction=False) for t, s in ((1, 5), (4, 6), (2, 5), (3, 5))]
        self.assertEqual(successors(events), {(1, 5, False): (2, 5, False), (2, 5, False): (3, 5, False)})


if __name__ == '__main__':
    unittest.main()
