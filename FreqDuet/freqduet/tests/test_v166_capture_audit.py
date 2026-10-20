import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v166_capture as audit


class FreshReadbackTests(unittest.TestCase):
    def receipts(self):
        receipt = dict(seed=1, day=2951001, mode='old', sampler_seed=1650000000,
            execution_source=audit.CAPTURE_SOURCE, main_policy_changed=False,
            actor_updates=0, critic_updates=0, dual_updates=0)
        cell = dict(receipt, protocol_source=audit.PROTOCOL_SOURCE,
            all_queries_recorded_before_target_labels=True, interval_fit=False)
        return receipt, cell

    def test_assigned_source_sampler_and_zero_updates(self):
        values = self.receipts()
        audit.receipts(*values, 1, 2951001, 'old', 1650000000)
        for field, value in (('execution_source', 'older'), ('sampler_seed', 42),
                ('actor_updates', 1), ('critic_updates', 1), ('dual_updates', 1)):
            broken = deepcopy(values)
            broken[0][field] = value
            with self.assertRaises(AssertionError):
                audit.receipts(*broken, 1, 2951001, 'old', 1650000000)
        values[1]['all_queries_recorded_before_target_labels'] = False
        with self.assertRaisesRegex(AssertionError, 'workflow'):
            audit.receipts(*values, 1, 2951001, 'old', 1650000000)

    def shard(self, start=0):
        return dict(status='v166_calibration_shard_complete', execution_source=audit.CAPTURE_SOURCE,
            protocol_source=audit.PROTOCOL_SOURCE, confirmation_paths=0, interval_fit=False,
            main_policy_changed=False, start=start, end=start+8, calibration_paths=16)

    def test_shards_cannot_omit_duplicate_or_relabel_paths(self):
        good = [self.shard(), self.shard(8)]
        audit.shard_coverage(good, 16)
        for broken in (good[:1], [self.shard(), self.shard()], [self.shard(), self.shard(16)]):
            with self.assertRaisesRegex(AssertionError, 'gaps or overlap'):
                audit.shard_coverage(broken, 16)
        for field, value in (('execution_source', 'older'), ('confirmation_paths', 1), ('interval_fit', True)):
            broken = deepcopy(good)
            broken[0][field] = value
            with self.assertRaisesRegex(AssertionError, 'source/freeze'):
                audit.shard_coverage(broken, 16)

    def live(self):
        evidence = dict(bus_id=1, ready_time_s=10., predicted_follower_bus_id=4, predicted_follower_trip_id=2)
        ready = dict(evidence, trip_id=1, station_id=3, direction=True)
        raw = dict(query_key=[1, 3, True], query_time_s=10., ready_callback_order=0, evidence=deepcopy(evidence),
            events_seen=dict(ready=1, last_arrival_s=9., last_departure_s=None),
            query_load=None, query_load_available=False, own_progress_m=100., own_speed_mps=5.,
            follower=dict(bus_id=4, trip_id=2, progress_m=90., speed_mps=4., state='TRAVEL'))
        return raw, ready

    def test_all_ready_clocks_identity_and_no_late_load(self):
        values = self.live()
        audit.live_record(*values, 0)
        for field, value in (('query_time_s', 11.), ('ready_callback_order', 1), ('query_load', 20)):
            broken = deepcopy(values)
            broken[0][field] = value
            with self.assertRaises(AssertionError):
                audit.live_record(*broken, 0)
        broken = deepcopy(values)
        broken[0]['events_seen']['last_departure_s'] = 11.
        with self.assertRaisesRegex(AssertionError, 'future event'):
            audit.live_record(*broken, 0)
        broken = deepcopy(values)
        broken[0]['follower']['trip_id'] = 9
        with self.assertRaisesRegex(AssertionError, 'physical follower'):
            audit.live_record(*broken, 0)

    def test_explicitly_absent_follower_is_not_a_missing_snapshot(self):
        raw, ready = self.live()
        raw['follower'] = None
        ready['predicted_follower_bus_id'] = raw['evidence']['predicted_follower_bus_id'] = None
        ready['predicted_follower_trip_id'] = raw['evidence']['predicted_follower_trip_id'] = None
        audit.live_record(raw, ready, 0)

    def test_matrix_rejects_missing_artifacts_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(capture_root=Path(directory))
            sealed = dict(reservation=dict(actors=[1], calibration_days=list(range(8))))
            with self.assertRaisesRegex(AssertionError, 'matrix incomplete'):
                audit.matrix(args, sealed)

    def test_no_replay_fitting_or_collection_calls_in_readback(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)}
        self.assertTrue(called.isdisjoint({'rollout', 'fit', 'save_queries', 'attach_labels', 'collect', 'train'}))


if __name__ == '__main__':
    unittest.main()
