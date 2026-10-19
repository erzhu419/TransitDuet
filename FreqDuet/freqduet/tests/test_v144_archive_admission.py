import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v144_archive_admission import metadata_days, case_id, trace_schema, inspect_bundle, artifact_bases


class ArchiveAdmissionTests(unittest.TestCase):
    def test_days_are_metadata_not_numeric_metrics(self):
        self.assertEqual(metadata_days(dict(demand_days=[2351001],
            paths=[dict(day=2351002, values=dict(day=999))], effects=dict(day=888))), {2351001, 2351002})

    def test_day_used_by_other_actor_stays_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_bundle(Path(d), '12071_2351015_old', {2351015: {'old/summary/status.json'}})
            self.assertEqual(r['use_status'], 'prior_aggregate_or_diagnostic')
            self.assertFalse(r['fields_complete'])

    def test_absent_reference_does_not_certify_unused(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_bundle(Path(d), '12071_9999999_old', {})
            self.assertEqual(r['use_status'], 'unconsumed_not_established')

    def test_schema_reads_past_the_first_row(self):
        good = dict(trip_id=1, station_id=2, direction=0, policy_time_s=10, chosen_s=0, executed_s=0,
            evidence=dict(ready_time_s=9, bus_id=1, follower_source='journey',
                predicted_follower_trip_id=2, predicted_follower_bus_id=2),
            online=dict(gap_s=30), original_prediction=dict(status='not_causally_moving'))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'trace.jsonl.gz'
            with gzip.open(p, 'wt') as f:
                f.write(json.dumps(good)+'\n'+json.dumps(dict(good, original_prediction={}))+'\n')
            self.assertEqual(trace_schema(p)['missing'], ['original_prediction.status'])

    def test_natural_case_identity_does_not_merge_modes(self):
        self.assertNotEqual(case_id('12071_2351015_old'), case_id('12071_2351015_handoff'))
        self.assertIsNone(case_id('p15'))

    def test_recorder_layout_does_not_traverse_counterfactual_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for p in ('records', 'smoke_runs/traces', 'cells/c15_f22_a10/cells/traces', 'source/traces'):
                (root/p).mkdir(parents=True)
            self.assertEqual(set(artifact_bases(root)), {root, root/'smoke_runs'})

    def test_v137_reuses_exact_v135_terminal_instead_of_requiring_a_copy(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)/'protocol_v6_v137_passive_ready_exposure'
            name = '12071_2351015_old'
            receipt = base/'paths'/(name+'.json')
            receipt.parent.mkdir(parents=True)
            terminal = base.parent/'protocol_v6_v135_handoff_confirmation'/'terminal'/(name+'.json')
            terminal.parent.mkdir(parents=True)
            terminal.write_text('{}')
            for exact in (True, False):
                receipt.write_text(json.dumps(dict(terminal_path_exact=exact)))
                r = inspect_bundle(base, name, {})
                self.assertEqual('terminal' not in r['missing'], exact)


if __name__ == '__main__':
    unittest.main()
