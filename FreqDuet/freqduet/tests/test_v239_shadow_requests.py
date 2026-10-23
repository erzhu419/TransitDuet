import copy
from pathlib import Path
import tempfile
import unittest

from test_v238_persistent_requests import snapshot
from analyze_freqduet_v239_shadow_requests import (
    ARMS, JOIN, label_rows, prepare_requests, read_evaluation, supply_groups, tally, write)


def fixtures():
    state, row, meta = snapshot()
    join = dict(zip(JOIN, (3, 5, 2, 0)))
    row.update(join)
    q = dict(join, view=state.query(row, meta),
             identity=dict(bus_id=2, trip_id=1, direction=False, station_id=3))
    extension = dict(complete=True, query_extensions=[row])
    capture = dict(complete=True, queries=[q])
    label = dict(key=[1, 3, False, 0], bus_id=2, query_s=10, H_prefix_s=45,
        holding_s=20, own_obligation=11, next_clock='no_next_launch',
        fixed=dict(kind='benefit'), next_label_matches=None)
    return extension, capture, dict(decisions=[label])


class ShadowAuditTest(unittest.TestCase):
    def test_requests_retain_physical_state_and_memoryless_is_native(self):
        extension, capture, _ = fixtures()
        before = copy.deepcopy((extension, capture))
        r = prepare_requests(extension, capture)[0]
        self.assertEqual((r['executed_holding_s'], r['native_s'], r['shadow_s']), (45, 20, 10))
        self.assertEqual(r['memoryless_bin'], r['native_bin'])
        self.assertEqual((extension, capture), before)

    def test_wrong_occurrence_or_repeated_key_stops_audit(self):
        extension, capture, _ = fixtures()
        bad = copy.deepcopy(capture)
        bad['queries'][0]['query_occurrence'] = 1
        with self.assertRaises(AssertionError):
            prepare_requests(extension, bad)
        extension['query_extensions'] *= 2
        capture['queries'] *= 2
        with self.assertRaises(AssertionError):
            prepare_requests(extension, capture)

    def test_tally_distinguishes_requested_dose_and_local_compensation_loss(self):
        extension, capture, _ = fixtures()
        s = tally(prepare_requests(extension, capture))
        self.assertEqual((s['changed'], s['requested_reduction_s'], s['memoryless_changed']), (1, 10, 0))
        self.assertEqual(s['local_R_loss_vs_native_on_changed_s2'], 2600)
        self.assertAlmostEqual(s['retained_local_gain_fraction_on_changed'], 3000/5600)
        self.assertIsNone(tally([])['retained_local_gain_fraction_on_changed'])

    def test_all_abstentions_remain_in_denominator(self):
        extension, capture, _ = fixtures()
        extension['query_extensions'][0]['shadow_frontier']['status'] = 'missing_follower'
        rows = prepare_requests(extension, capture)
        s = tally(rows)
        self.assertEqual((s['queries'], s['supported'], s['changed'], s['requested_reduction_s']), (1, 0, 0, 0))
        self.assertEqual(rows[0]['reason'], 'missing_follower')

    def test_labels_never_rewrite_requests_and_mismatched_identity_stops(self):
        extension, capture, labels = fixtures()
        requests = prepare_requests(extension, capture)
        before = copy.deepcopy(requests)
        joined = label_rows(requests, labels)
        self.assertEqual(joined[0]['association_match_label'], 'unavailable')
        labels['decisions'][0].update(next_clock='arbitrary_future', next_label_matches=False)
        self.assertEqual(label_rows(requests, labels)[0]['shadow_bin'], requests[0]['shadow_bin'])
        self.assertEqual(requests, before)
        labels['decisions'][0]['bus_id'] = 999
        with self.assertRaises(AssertionError):
            label_rows(requests, labels)

    def test_missing_supply_labels_remain_null_and_cycles_count_once(self):
        extension, capture, _ = fixtures()
        requests = prepare_requests(extension, capture)
        cycles = [dict(bus_id=2, trip_id=1, next_trip=None, holding_s=20,
                       next_public_delay_s=None, clock=dict(kind='no_next_launch')),
                  dict(bus_id=8, trip_id=2, next_trip=3, holding_s=0,
                       next_public_delay_s=0, clock=dict(kind='postreturn_wait'))]
        result = supply_groups(requests, cycles)
        self.assertIsNone(result['no_next_launch']['original_next_public_delay_s'])
        self.assertEqual(result['no_next_launch']['requested_reduction_s'], 10)
        self.assertEqual(result['postreturn_wait']['original_next_public_delay_s'], 0)
        self.assertEqual(sum(r['cycles'] for r in result.values()), 2)

    def test_evaluation_cannot_open_before_both_request_files_are_sealed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'labels.json', dict(synthetic=True))
            write(root/'requests_sealed.json', dict(arms=list(ARMS), evaluation_labels_read=0))
            with self.assertRaises(AssertionError):
                read_evaluation(root, root/'labels.json')
            for arm in ARMS:
                write(root/'requests'/f'c0_{arm}.json.gz', [])
            self.assertEqual(read_evaluation(root, root/'labels.json'), dict(synthetic=True))


if __name__ == '__main__':
    unittest.main()
