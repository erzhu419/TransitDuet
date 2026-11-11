from copy import deepcopy
import json
import unittest

import numpy as np

from freq_hrl.experiments.plan_validity_boundary_information_fit import frozen_predictions
from freq_hrl.experiments.plan_validity_boundary_information_readout import evaluation_data, summarize
from scripts import analyze_plan_validity_boundary_information_readout as merge
from test_plan_validity_boundary_information_readout import fixture


def panels():
    cell, design, rows, models = fixture()
    data = evaluation_data(rows, cell, design, cell["readout_prefix_seeds"])
    scores = frozen_predictions(data, models)
    subsets = []
    for mask in (data["groups"] < 204, data["groups"] >= 204):
        part = {k: data[k][mask] for k in ("tail", "groups", "windows")}
        subsets.append(summarize(part, {k: p[mask] for k, p in scores.items()}))
    return subsets, summarize(data, scores)


class BoundaryInformationMergeTest(unittest.TestCase):
    def test_public_moment_merge_equals_direct_all_prefix_readout(self):
        parts, direct = panels()
        combined = merge.merge_root(parts)
        self.assertEqual((combined["prefixes"], combined["states"], combined["windows"], combined["E_pairs"]), (8, 24, 96, 768))
        for method, mse in direct["MSE_to_E_mean"].items():
            self.assertAlmostEqual(combined["MSE_to_E_mean"][method], mse)
        for name, target in direct["comparisons"].items():
            result = combined["comparisons"][name]
            for field in ('MSE_delta', 'conditional_suffix_MC_SE', 'prefix_delta_SD_descriptive'):
                self.assertAlmostEqual(result[field], target[field])
            self.assertEqual(result['prefix_reports'], target['prefix_reports'])
            for half in ('suffix_halves', 'window_halves'):
                for j in range(2):
                    for field in ('MSE_delta', 'conditional_suffix_MC_SE'):
                        self.assertAlmostEqual(result[half][j][field], target[half][j][field])

    def test_audit_rejects_missing_shards_fits_and_wrong_prefix_moments(self):
        design = json.loads((merge.ROOT / 'protocols/plan_validity_boundary_information_2026-11-11.json').read_text())
        release = json.loads((merge.ROOT / 'protocols/plan_validity_boundary_information_readout_2026-11-11.json').read_text())
        parts, _ = panels()
        records = []
        for shard in release['shards']:
            c = design['cells'][shard['cell_index']]
            panel = deepcopy(parts[shard['shard_index']])
            for comparison in panel['comparisons'].values():
                for row, seed in zip(comparison['prefix_reports'], shard['prefix_seeds']):
                    row['seed'] = seed
            records.append({'run_name': design['run_name'], 'task': c['task'], 'optimizer_seed': c['optimizer_seed'],
                'shard_index': shard['shard_index'], 'status': 'complete', 'stage': 'boundary_information_readout_complete',
                'sampled_prefix_seeds': shard['prefix_seeds'], 'completed_states': 12, 'completed_pairs': 384,
                'new_environment_steps': 926400, 'new_fits': 0, 'all_models_frozen': True, 'evaluation_prefixes_accessed': True,
                'readout': panel, 'measured_cost': {'counts': {'environment_steps': 926400, 'frozen_model_window_predictions': 288},
                    'cpu_seconds': 1., 'wall_seconds': 2.}})
        report = merge.aggregate_records(design, release, records)
        self.assertEqual((report['evaluation_environment_steps'], report['completed_states'], report['E_pairs']), (7411200, 96, 3072))
        with self.assertRaisesRegex(ValueError, 'eight'):
            merge.aggregate_records(design, release, records[:-1])
        changed = deepcopy(records)
        changed[-1]['measured_cost']['counts']['ridge_fit_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'ledger'):
            merge.aggregate_records(design, release, changed)
        changed = deepcopy(records)
        changed[-1]['readout']['comparisons']['wide_minus_dense']['prefix_reports'][0]['seed'] = -1
        with self.assertRaisesRegex(ValueError, 'prefix moments'):
            merge.aggregate_records(design, release, changed)


if __name__ == '__main__':
    unittest.main()
