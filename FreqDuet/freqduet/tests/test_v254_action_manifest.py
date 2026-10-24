import json
from pathlib import Path
import tempfile
import unittest

from seal_freqduet_v254_action_manifest import (
    ROOT_BASE, PAYLOADS, audit_namespace, contexts, future_manifest, manifest,
    preflight_interface, runtime_stream_names, seed_hits, stream_offsets)


STREAMS = ['fleet','upper_residual_selector','headway_value_selector','terminal_value_selector',
           'fixed_expert_selector','tpc_mixture','reachability_replay']


class ActionManifestTests(unittest.TestCase):
    def test_existing_four_contexts_are_not_called_new_days(self):
        rows = contexts()
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['actor'] for r in rows}, {12037, 933083})
        self.assertEqual({r['demand_day'] for r in rows}, {2151001,2151002})
        self.assertEqual(rows[0]['prefix_sampler'], 402112137)
        self.assertTrue(all(r['prefix_status'] == 'consumed_development_not_new_day' for r in rows))

    def test_banks_and_derived_streams_are_disjoint_and_uint32(self):
        rows = future_manifest(STREAMS)
        self.assertEqual(len(rows), 65)
        all_seeds = [s for r in rows for s in r['streams'].values()]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(len(all_seeds), 65*13)
        self.assertLess(max(all_seeds), 2**32)
        for c in range(4):
            evaluation = [r for r in rows if r['context'] == c and r['phase'] == 'evaluation']
            self.assertEqual([r['half'] for r in evaluation], ['A']*4+['B']*4)

    def test_derived_collision_detected_even_without_equal_root(self):
        target = ROOT_BASE+50000000
        hits = seed_hits(dict(smoke_seed=ROOT_BASE), {target}, set(stream_offsets(STREAMS).values()))
        self.assertEqual(hits, [dict(recorded_seed=ROOT_BASE, derived_offset=50000000, reserved_seed=target)])
        self.assertTrue(seed_hits('--future-seed '+str(ROOT_BASE), {ROOT_BASE}, {0}))

    def test_reference_and_noop_use_original_future_not_new_bank(self):
        p = preflight_interface()
        self.assertEqual(len(PAYLOADS), 6)
        self.assertEqual(p['slots'], ['passive_c0_v107','noop_c0_v107'])
        self.assertIn('No future switch', p['expected_boundary'])
        self.assertIn('original action object', p['no_op'])

    def test_caps_count_physical_starts_not_alias_labels(self):
        spec = manifest('test', STREAMS)
        self.assertEqual(spec['physical_start_caps'], dict(passive=2, smoke=8, selection=224, evaluation=224, total=458))
        self.assertEqual(len(spec['reserved_slots']), 2+65*8+1)
        self.assertEqual(sum(s['kind'] == 'explicit_engineering_native_alias' for s in spec['reserved_slots']), 1)
        self.assertTrue(all(s['skip_if_native_alias'] for s in spec['reserved_slots'] if s['kind'] == 'fixed_bin_label'))
        self.assertFalse(spec['launch_authorized'] or spec['collector_implemented'] or spec['selector_defined'])

    def test_metadata_and_scheduler_conflicts_block_but_raw_content_is_not_read(self):
        spec = manifest('test', STREAMS)
        snapshot = dict(tasks=[], source_files=['queue.json','queue_archive.jsonl'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excluded = root/'current'
            excluded.mkdir()
            (excluded/'manifest.json').write_text(json.dumps(spec))
            raw = root/'old'/'records'
            raw.mkdir(parents=True)
            (raw/'c0.json.gz').write_bytes(b'not a readable raw payload')
            metadata = root/'old'/'manifest.json'
            metadata.write_text(json.dumps(dict(future_seed=123)))
            audit = audit_namespace([root], excluded, spec, snapshot)
            self.assertEqual(audit['decision'], 'fresh')
            self.assertEqual(audit['artifact_names_checked'], 1)
            metadata.write_text(json.dumps(dict(future_seed=ROOT_BASE)))
            self.assertEqual(audit_namespace([root], excluded, spec, snapshot)['decision'], 'blocked')
            metadata.write_text('{}')
            snapshot['tasks'] = [dict(id='old', cmd='--future '+str(ROOT_BASE))]
            self.assertEqual(audit_namespace([root], excluded, spec, snapshot)['decision'], 'blocked')

    def test_bad_metadata_is_reported_not_silently_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'protocol.json').write_text('{unfinished')
            result = audit_namespace([root], root/'excluded', manifest('test', STREAMS),
                dict(tasks=[], source_files=['queue.json']))
            self.assertEqual(result['decision'], 'blocked')
            self.assertEqual(len(result['errors']), 1)

    def test_runtime_names_are_parsed_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runner.py'
            path.write_text("raise RuntimeError('must not import')\nclass Runner:\n"
                "    def _runtime_numpy_streams(self):\n        return {'fleet': self.rng, 'other': self.other}\n")
            self.assertEqual(runtime_stream_names(path), ['fleet','other'])


if __name__ == '__main__':
    unittest.main()
