from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v161_joint_intervals as protocol
from extract_freqduet_v159_joint_residuals import outcome_day_ids, day_ids


def parent():
    # Synthetic identifiers test the split contract without accessing residuals.
    cal, confirm = list(range(101, 133)), list(range(201, 233))
    return dict(status='v160_metadata_and_preregistration_complete', execution_source='14d930ee25',
        new_days_collected=0, interval_fit=False, registration=dict(
            status='preregistered_not_collected', collector_submitted=False,
            calibration_days=cal, confirmation_days=confirm, actors=list(range(12)),
            modes=['old', 'handoff'], checkpoint_ep=199,
            sampler_seeds={str(d): 1600000000+i for i, d in enumerate(cal+confirm)},
            service_point_source='1b783ea8f3', running_shadow_source='5be4cea534',
            v146_extensions_s=[3., 0.], window_s=3600., tails=[.1, .1],
            freshness=dict(overlap_days=[], overlap_samplers=[])))


def readouts():
    return [dict(mode=m, scope=s, complete_actor_day_grid=True, finite_unresolved_n=0,
        missing_reference_n=0, coverage_lower95=.82, below_upper95=.09, above_upper95=.09,
        finite_fraction_lower95=.85, score_delta_upper95=-1.)
        for m in protocol.MODES for s in protocol.SCOPES]


class JointProtocolTests(unittest.TestCase):
    def test_parent_is_not_modified_and_reservation_remains_unconsumed(self):
        source = parent()
        before = deepcopy(source)
        plan = protocol.registration(source)
        self.assertEqual(source, before)
        self.assertEqual(len(day_ids(plan) & set(range(101, 133))), 32)
        self.assertEqual(outcome_day_ids(dict(status='complete', protocol=plan)), set())
        self.assertFalse(plan['execution']['collector_submitted'])
        self.assertFalse(plan['execution']['interval_fit'])

    def test_consumed_or_overlapping_cohorts_block_seal(self):
        source = parent()
        source['new_days_collected'] = 1
        with self.assertRaisesRegex(AssertionError, 'not consumed'):
            protocol.registration(source)
        source = parent()
        source['registration']['confirmation_days'][0] = 101
        with self.assertRaisesRegex(AssertionError, 'disjoint'):
            protocol.registration(source)

    def test_frozen_parameters_and_matched_samplers_cannot_drift(self):
        for field, changed in (('tails', [.05, .05]), ('window_s', 7200.),
                ('v146_extensions_s', [0., 0.]), ('service_point_source', 'candidate')):
            source = parent()
            source['registration'][field] = changed
            with self.assertRaisesRegex(AssertionError, 'frozen forecast'):
                protocol.registration(source)
        source = parent()
        source['registration']['sampler_seeds']['101'] += 1
        with self.assertRaisesRegex(AssertionError, 'sampler'):
            protocol.registration(source)

    def test_ranks_count_days_not_queries_and_keep_fixed_tails(self):
        self.assertIsNone(protocol.rank_levels(18))
        result = protocol.rank_levels(19)
        self.assertEqual((result['lower_rank'], result['upper_rank']), (2, 18))
        result = protocol.rank_levels(32)
        self.assertEqual((result['lower_rank'], result['upper_rank']), (3, 30))

    def test_joint_contract_and_coarse_full_eta_reference(self):
        plan = protocol.registration(parent())
        self.assertEqual(plan['target']['channels'], ['running', 'service', 'residence'])
        self.assertEqual(plan['comparator']['key'], ['actor', 'mode', 'query_phase'])
        self.assertIn('predicted_target_phase', plan['conditioning']['candidate_key'])
        self.assertIn('no confirmation-day residual updates', plan['estimator']['update'])
        self.assertIn('late policy load', plan['conditioning']['excluded'])

    def test_complete_joint_pass_never_admits_a_controller(self):
        result = protocol.gate(readouts())
        self.assertEqual(result['decision'], 'pass')
        self.assertFalse(result['controller_admitted'])

    def test_any_mode_scope_gate_can_fail_the_joint_claim(self):
        for field, value, name in (('coverage_lower95', .799, 'coverage'),
                ('below_upper95', .101, 'below'), ('above_upper95', .101, 'above'),
                ('finite_fraction_lower95', .799, 'support'), ('score_delta_upper95', .001, 'score')):
            rows = readouts()
            rows[-1][field] = value
            result = protocol.gate(rows)
            self.assertEqual(result['decision'], 'not_admitted')
            self.assertIn('handoff/hour13/'+name, result['failures'])

    def test_score_improvement_is_strict_only_in_registered_scopes(self):
        rows = readouts()
        for row in rows:
            row['score_delta_upper95'] = 0.
        result = protocol.gate(rows)
        self.assertEqual(set(result['failures']), {m+'/'+s+'/score' for m in protocol.MODES
            for s in ('all', 'clearance')})

    def test_missing_or_unresolved_evidence_is_inconclusive(self):
        self.assertEqual(protocol.gate(readouts()[:-1])['decision'], 'inconclusive')
        self.assertEqual(protocol.gate(readouts()+readouts()[:1])['decision'], 'inconclusive')
        for field, value in (('coverage_lower95', None), ('score_delta_upper95', float('nan')),
                ('finite_unresolved_n', 1), ('missing_reference_n', 1), ('complete_actor_day_grid', False)):
            rows = readouts()
            rows[0][field] = value
            self.assertEqual(protocol.gate(rows)['decision'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
