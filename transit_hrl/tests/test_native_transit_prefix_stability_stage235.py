import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_prefix_stability as diagnostic
from scripts import run_native_transit_prefix_stability_stage235 as spec
from scripts.submit_native_transit_prefix_stability_stage235_scheduleurm import task_specification
from tests import test_native_transit_cost_strength_stage234 as fixtures

strength, joint = diagnostic.strength, diagnostic.joint


def curves_for(groups, indices=(0, 1, 2, 3)):
    first = {root: min(g for g in groups if g[0] == root) for root in {g[0] for g in groups}}
    curves, ledger = {a: [] for a in joint.ANCHORS}, []
    halves = (list(indices[:2]), list(indices[2:]))
    for group in sorted(groups):
        for direction in (0, 1):
            ledger.append({'held_group': list(group), 'direction': direction, 'training_draws': halves[direction],
                'validation_draws': halves[1-direction], 'held_states': 6,
                'fits': {a: {'scales': [2., 3.]} for a in joint.ANCHORS}})
            for a in joint.ANCHORS:
                curve = {}
                for k in strength.CANDIDATES:
                    gain = 0. if k == 'zero' else (-2. if group == first[group[0]] else .02) if k == '1@0.25' else -.001
                    risk = int(k == '1@0.25' and group == first[group[0]] and direction == 0)
                    curve[k] = {'error_sum': .01 if k == '1@0.25' else 1., 'error_cases': 30,
                        'changed_count': 0 if k == 'zero' else 1,
                        'by_root': {str(group[0]): {'cases': 6, 'cost_sum': gain,
                            'wait_sum_min': gain, 'unsafe_delta_count': risk}}}
                curves[a].append(curve)
    return curves, ledger


def source_for(train, held, partition, direction):
    groups = lambda rows: sorted({(r['root'], r['scene_seed']) for r in rows})
    a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
    curves, ledger = curves_for(groups(train), a)
    scopes = {}
    for scope in spec.SCOPES:
        rows = train if scope.startswith('train') else held
        def record(row):
            raw = {'selected': 0, 'cost': -.01, 'wait': -.02, 'unsafe': False}
            new = {'selected': 1, 'cost': -.03 if scope.startswith('train') else 0., 'wait': -.025, 'unsafe': False}
            return {**row, 'decisions': {'raw': raw, **{f'cost_strength_{a}_{m}': dict(new)
                for a in joint.ANCHORS for m in diagnostic.FIELDS}}}
        scopes[scope] = [record(r) for r in rows]
    return {'partition': partition, 'direction': direction, 'training_draws': list(a), 'evaluation_draws': list(b),
        'training_groups': list(map(list, groups(train))), 'held_groups': list(map(list, groups(held))),
        'selection': {a: diagnostic.select(curves[a], ledger, a) for a in joint.ANCHORS}, 'scopes': scopes}, curves, ledger


class PrefixStabilityTest(unittest.TestCase):
    def test_delete_both_directions_and_distinguish_selection_from_eligibility(self):
        groups = [(397, 0), (397, 1), (401, 0), (401, 1)]
        curves, ledger = curves_for(groups)
        full = diagnostic.select(curves['raw'], ledger, 'raw')
        self.assertEqual((full['mean_level'], full['safe_level']), ('1@0.25', '1@0.5'))
        d = diagnostic.diagnose(curves['raw'], ledger, 'raw', full)
        self.assertEqual(len(d['deletions']), 4)
        self.assertTrue(all(x['removed_blocks'] == 2 for x in d['deletions']))
        dominant = d['deletions'][0]['modes']['mean_selected']
        self.assertTrue(dominant['changed'])
        self.assertFalse(dominant['fixed_eligible'])
        self.assertGreater(dominant['fixed_curve']['by_root']['397']['mean_delta_cost'], 0.)
        nondominant = d['deletions'][1]['modes']['mean_selected']
        self.assertFalse(nondominant['changed'])
        self.assertTrue(nondominant['fixed_eligible'])
        self.assertEqual(d['prefixes'][0]['modes']['mean_selected']['cases'], 12)
        self.assertEqual(d['future_halves'][0]['validation_draws'], [2, 3])
        self.assertFalse(d['future_halves'][0]['modes']['safe_selected']['changed'])
        self.assertTrue(d['future_halves'][1]['modes']['safe_selected']['changed'])

    def test_signed_concentration_is_source_local_and_hold_is_not_improvement(self):
        curves, ledger = curves_for([(397, 0), (397, 1), (401, 0), (401, 1)])
        full = diagnostic.select(curves['raw'], ledger, 'raw')
        d = diagnostic.diagnose(curves['raw'], ledger, 'raw', full)
        c = d['concentration']['mean_selected']['397']['cost']
        self.assertEqual(c['prefixes'], 2)
        self.assertEqual((c['improving_prefixes'], c['worsening_prefixes']), (1, 1))
        self.assertEqual(c['top1_benefit_share'], 1.)
        self.assertAlmostEqual(c['net_sum'], -3.96)
        for curve in curves['raw']:
            for k, stats in curve.items():
                if k != 'zero':
                    for x in stats['by_root'].values(): x['cost_sum'] = 1.
        full = diagnostic.select(curves['raw'], ledger, 'raw')
        d = diagnostic.diagnose(curves['raw'], ledger, 'raw', full)
        self.assertIsNone(d['full']['mean_selected']['candidate'])
        self.assertTrue(all(x['modes']['mean_selected']['fixed_eligible'] is None for x in d['deletions']))
        self.assertIsNone(d['concentration']['mean_selected']['397']['cost']['top1_benefit_share'])
        bad = copy.deepcopy(full)
        bad['mean_level'] = '1@0.25'
        with self.assertRaisesRegex(ValueError, 'Stage234 inner selection changed'):
            diagnostic.diagnose(curves['raw'], ledger, 'raw', bad)

    def test_physical_weighting_and_unsafe_sign(self):
        p = diagnostic.weighted([{'cases': 6, 'mean_delta_cost': 1., 'mean_delta_wait_min': 2., 'mean_delta_unsafe': 0.},
            {'cases': 12, 'mean_delta_cost': -1., 'mean_delta_wait_min': -2., 'mean_delta_unsafe': .5}])
        self.assertAlmostEqual(p['mean_delta_cost'], -1/3)
        self.assertAlmostEqual(p['mean_delta_unsafe'], 1/3)
        rows = [{'decisions': {'raw': {'selected': 0, 'cost': 2., 'wait': 3., 'unsafe': True},
            'chosen': {'selected': 1, 'cost': 1., 'wait': 2., 'unsafe': False}}}]
        self.assertEqual(diagnostic.outer_stats(rows, 'chosen'), {'cases': 1, 'mean_delta_cost': -1.,
            'mean_delta_wait_min': -1., 'mean_delta_unsafe': -1., 'switch_count': 1})

    def test_clean_short_inner_replay_matches_original_selection(self):
        bank = fixtures.old.old.old.fixtures.native_fixture.bank_fixture()[0]['training_data']
        groups = sorted({(r['root'], r['scene_seed']) for r in bank})[:3]
        rows = [r for r in bank if (r['root'], r['scene_seed']) in groups]
        panels = fixtures.old.old.old.fixtures.old.panels_for(rows)
        batch = joint.binding.learned.neural.batches(rows)
        reused_data = []
        with patch.object(joint.binding.learned.neural, 'STEPS', 3):
            for _, training, held in joint.binding.policy.prefix_folds(rows):
                rs, hs = [rows[i] for i in training], [rows[i] for i in held]
                for a in ((0, 1), (2, 3)):
                    ref, _ = joint.fit_reference(rs, panels, a, batch=tuple(x[training] for x in batch))
                    reused_data.append(joint.causal_data(hs, ref, batch=tuple(x[held] for x in batch)))
            reused = {'originals': np.concatenate([d['originals'] for d in reused_data]),
                'roots': np.concatenate([d['roots'] for d in reused_data]), 'costs': np.concatenate([d['heads'][..., 0] for d in reused_data]),
                'state_conditioned_pair': np.concatenate([d['raw_logits'] for d in reused_data])}
            old, old_ledger, old_repro, _ = strength.collect(rows, panels, (0, 1, 2, 3), reused)
            curves, ledger, repro = joint.collect_paths(rows, panels, (0, 1, 2, 3), reused, path_fitter=strength.fit_path)
        self.assertEqual(ledger, old_ledger)
        self.assertEqual(repro, old_repro)
        for kind in joint.ANCHORS:
            d = diagnostic.diagnose(curves[kind], ledger, kind, old[kind])
            self.assertEqual(len(d['prefixes']), 3)

    def test_outer_labels_cannot_change_inner_diagnostics(self):
        rows, registered = fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        train, held = spec.training.cohort.split_rows(views['mixed_b'], 0)
        source, curves, ledger = source_for(train, held, 'first_last', 0)
        changed = copy.deepcopy(source)
        for row in changed['scopes']['held_other']:
            for name, decision in row['decisions'].items():
                if name != 'raw': decision['cost'] += 10.
        with patch.object(joint, 'collect_paths', return_value=(curves, ledger, {'max_abs_difference': 0., 'changed_choices': 0})):
            before, after = [spec.fit_cell(train, {}, s, {}) for s in (source, changed)]
        self.assertEqual(before['anchors'], after['anchors'])
        self.assertNotEqual(before['outer'], after['outer'])

    def test_complete_cohort_summary_and_missing_prefix_rejection(self):
        rows, registered = fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        results = {}
        for key in spec.KEYS:
            train, held = spec.training.cohort.split_rows(views[key[0]], key[1])
            source, curves, ledger = source_for(train, held, key[2], key[3])
            with patch.object(joint, 'collect_paths', return_value=(curves, ledger, {'max_abs_difference': 0., 'changed_choices': 0})):
                cell = spec.fit_cell(train, {}, source, {})
            results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key), 'cell': cell}
        report = spec.summarize(results, views)
        self.assertEqual(report['selection_reproduction'], {'anchor_selections': 64, 'differences': 0})
        self.assertEqual(report['selector_deletion_comparisons'], 1920)
        for v in report['views'].values():
            for a in joint.ANCHORS:
                for m in diagnostic.FIELDS:
                    x = v['anchors'][a][m]
                    self.assertEqual(x['deletion_checks'], 80)
                    self.assertEqual(x['future_half_checks'], 16)
                    self.assertEqual(sum(p['cases'] for p in x['inner_by_root'].values()), 960)
                    self.assertEqual(sum(p['cases'] for p in x['outer_scopes']['held_other'].values()), 480)
                    self.assertEqual(sum(p['switch_count'] for p in x['outer_scopes']['held_other'].values()), 480)
                    self.assertEqual(len(x['inner_by_cell']), 8)
                    self.assertEqual(len(x['outer_scopes_by_cell']), 8)
                    self.assertTrue(all(len(h) == 2 for h in x['future_halves_by_cell']))
        results[spec.KEYS[0]]['cell']['blocks'].pop()
        with self.assertRaisesRegex(ValueError, 'Dropped validation prefix'): spec.summarize(results, views)

    def test_protocol_and_scheduler_are_frozen_unbound_and_cpu_only(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_native_episodes'], 0)
        self.assertEqual(c['budget']['new_outer_prediction_calls'], 0)
        with patch.object(spec.previous, 'contract', wraps=spec.previous.contract) as parent:
            spec.contract()
            self.assertEqual(parent.call_count, 1)
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification('unit_prefix_stability', *key)
            self.assertEqual((task['cpu'], task['ram_mb']), (1, 3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', task['result_dir'])


if __name__ == '__main__': unittest.main()
