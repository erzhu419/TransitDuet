import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_objective_oof as learning
from scripts import run_native_transit_objective_oof_stage239 as spec
from scripts.submit_native_transit_objective_oof_stage239_scheduleurm import task_specification
from tests.test_native_transit_refit_matrix_stage237 import small_fixture


def candidate_curve(cost, wait, risk=0.):
    def one(c, w, r): return {'by_root': {'397': {'cases': 6, 'delta_cost': c, 'delta_wait_min': w,
        'delta_unsafe': r, 'scaled_cost': c, 'scaled_wait': w}}, 'score': max(c, w), 'changed_count': 1}
    return {'pooled': one(cost, wait, risk), 'halves': [one(cost, wait, risk), one(cost, wait, risk)]}


class ObjectiveOOFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(learning.joint.binding.learned.neural, 'STEPS', 3):
            args = small_fixture()
            source, cached = learning.base.previous.fit_matrix(*args)
        train, held, panels, reference, paths, old, _, oof = args
        cls.args = train, held, panels, reference, paths, source, cached, oof, old['selection']

    def test_half_safe_rejects_future_and_risk_regressions_joint_selection(self):
        curves = {o: {c: candidate_curve(0., 0.) for c in learning.strength.CANDIDATES} for o in learning.OBJECTIVES}
        curves['rank_regret']['1@1'] = candidate_curve(-2., -1.)
        curves['rank_regret']['1@1']['halves'][0] = candidate_curve(-3., 1.)['pooled']
        curves['wait_rank']['3@0.5'] = candidate_curve(-.5, -.5)
        curves['wait_rank']['8@1'] = candidate_curve(-3., -3., 1.)
        selected = learning.choose(curves)
        self.assertEqual(selected['joint']['pooled_safe']['objective'], 'rank_regret')
        self.assertEqual(selected['joint']['half_safe']['candidate'], '3@0.5')
        self.assertIsNone(selected['rank_regret']['half_safe']['candidate'])
        self.assertEqual(selected['wait_rank']['half_safe']['eligible_count'], 1)

    def test_no_reference_refits_cached_OOF_reproduced_and_all_objectives_covered(self):
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('No raw training')):
            result, cache = learning.fit_cell(*self.args)
        self.assertEqual(len(result['blocks']), 6)
        self.assertEqual(len(cache), 7)
        self.assertEqual(result['reproduction']['oof_choice_comparisons'], 36)
        self.assertEqual(result['reproduction']['legacy_changed_choices'], 0)
        for block in result['blocks']:
            self.assertEqual(set(block['training_draws']) & set(block['validation_draws']), set())
            for kind in learning.joint.ANCHORS:
                self.assertEqual(len(block['fits'][kind]), 12)
                self.assertEqual(set(block['curves'][kind]), set(learning.OBJECTIVES))
        for scope in learning.SCOPES:
            self.assertEqual(set(result['scopes'][scope]['all']['variants']), set(learning.VARIANTS))
            self.assertNotIn('cost_mse', result['scopes'][scope]['all']['variants']['raw'])

    def test_outer_and_other_future_truth_cannot_change_selection_or_weights(self):
        first, models = learning.fit_cell(*self.args)
        train, held, panels, reference, paths, source, cached, oof, frozen = self.args
        changed = copy.deepcopy(panels)
        for r in train+held:
            values = changed[learning.joint.binding.learned.state_key(r)]['values']
            if r in held: values[:, 1:, (0, 3)] += 100.
            else: values[4:, 1:, (0, 3)] += 100.
        second, other = learning.fit_cell(train, held, changed, reference, paths, source, cached, oof, frozen)
        self.assertEqual(models, other)
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['blocks'], second['blocks'])
        self.assertNotEqual(first['scopes']['held_other'], second['scopes']['held_other'])

    def test_wait_labels_change_wait_rank_only_with_same_subspace_and_budget(self):
        train, _, panels, reference, paths = self.args[:5]
        d = learning.joint.causal_data(train, reference)
        v = learning.joint.physical_values(train, panels, (0, 1, 2, 3))
        first, fits, _ = learning.fit_paths(d, v, paths['raw'])
        changed = v.copy()
        changed[..., 1] += np.arange(6)[None, :]*paths['raw']['scales'][1]*2
        second, _, _ = learning.fit_paths(d, changed, paths['raw'])
        for o in learning.base.OBJECTIVES: self.assertEqual(first[o], second[o])
        self.assertNotEqual(first['wait_rank'], second['wait_rank'])
        mask = np.arange(6)[None, :] != d['raw_choices'][:, None]
        x = learning.joint.relative(d['x'][..., :198], d['raw_choices'])[mask]
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        for f in fits:
            self.assertLessEqual(f['norm'], f['radius']+1e-12)
            self.assertLessEqual(f['optimality'], 1e-8)
            w = np.asarray(first[f['objective']]['weights'][f['level']][0])
            k = f['retained_rank']
            np.testing.assert_allclose(w, vt[:k].T@(vt[:k]@w), atol=1e-12)

    def test_protocol_and_unbound_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['objective_solves'], 16128)
        for key in (spec.KEYS[0], (None, None, None, None)):
            t = task_specification('unit_objective_oof', *key)
            self.assertEqual((t['cpu'], t['ram_mb']), (1, 3072))
            self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', t['result_dir'])

    def test_complete_report_and_wrong_future_or_missing_objective_rejected(self):
        train = [{'root': 397, 'scene_seed': g} for g in range(10) for _ in range(6)]
        groups = [(397, g) for g in range(10)]
        curves = {o: {c: candidate_curve(0., 0.) for c in learning.strength.CANDIDATES} for o in learning.OBJECTIVES}
        fit = [{'objective': o, 'level': l, 'success': True, 'parameter_count': int(l), 'retained_rank': int(l),
            'optimality': 0., 'norm': 1., 'radius': 1., 'loss_before': 2., 'loss_after': 1.}
            for o in learning.OBJECTIVES[1:] for l in learning.strength.lowrank.LEVELS[1:]]
        fits = {k: fit for k in learning.joint.ANCHORS}
        results = {}
        for key in spec.KEYS:
            a = spec.PARTITIONS[key[2]][key[3]]
            stats = {'cases': 60, 'variants': {v: {'delta_cost_own': 0.} for v in learning.VARIANTS}}
            blocks = [{'held_group': list(g), 'half': h, 'training_states': 54, 'held_states': 6,
                'training_draws': list(a[2*h:2*h+2]), 'validation_draws': list(a[2*(1-h):2*(1-h)+2]), 'fits': fits,
                'curves': {k: {o: dict.fromkeys(learning.strength.CANDIDATES) for o in learning.OBJECTIVES} for k in learning.joint.ANCHORS}}
                for g in groups for h in (0, 1)]
            cell = {'blocks': blocks, 'training_draws': list(a), 'evaluation_draws': list(spec.PARTITIONS[key[2]][1-key[3]]),
                'full_fits': fits, 'curves': {k: curves for k in learning.joint.ANCHORS},
                'selection': {k: learning.choose(curves) for k in learning.joint.ANCHORS},
                'scopes': {s: {'all': stats, 'by_root': {'397': stats}} for s in learning.SCOPES},
                'reproduction': {'oof_choice_comparisons': 120, 'legacy_choice_comparisons': 6, 'oof_changed_choices': 0,
                    'legacy_changed_choices': 0, 'oof_max_abs_difference': 0., 'subspace_max_abs_difference': 0., 'legacy_max_abs_difference': 0.}}
            results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key), 'cell': cell}
        with patch.object(spec.training.cohort, 'split_rows', return_value=(train, train)):
            report = spec.summarize(results, dict.fromkeys(spec.VIEWS))
            self.assertEqual(report['reproduction']['oof_choice_comparisons'], 3840)
            self.assertEqual(report['mixed_pooled']['held_other']['all']['cases'], 960)
            self.assertEqual(report['admission']['joint_raw_half_safe']['admitted_cells'], 0)
            cell = results[spec.KEYS[0]]['cell']
            ds = cell['blocks'][0]['validation_draws']
            cell['blocks'][0]['validation_draws'] = cell['blocks'][0]['training_draws']
            with self.assertRaisesRegex(ValueError, 'Changed clean OOF labels'): spec.summarize(results, dict.fromkeys(spec.VIEWS))
            cell['blocks'][0]['validation_draws'] = ds
            cell['blocks'][0]['curves']['raw'].pop('wait_rank')
            with self.assertRaisesRegex(ValueError, 'Dropped OOF objective'): spec.summarize(results, dict.fromkeys(spec.VIEWS))


if __name__ == '__main__': unittest.main()
