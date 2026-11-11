from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments import plan_validity_boundary_information_readout as readout
from freq_hrl.experiments import plan_validity_boundary_parent_scale as parent
from freq_hrl.experiments import plan_validity_boundary_tail as old
from freq_hrl.experiments.plan_validity_boundary_information import draw_seed
from freq_hrl.experiments.plan_validity_boundary_information_fit import frozen_predictions
from scripts import run_plan_validity_boundary_information_readout as worker
from scripts import submit_plan_validity_boundary_information_readout_scheduleurm as submitter
from test_plan_validity_boundary_information_fit import banks


def fixture():
    cell, design, base, _, _ = banks()
    cell.update(task="fixture", readout_prefix_seeds=list(range(200, 208)))
    rows = deepcopy(base)
    for row in rows:
        row["seed"] += 200
        for j, win in enumerate(row["windows"]):
            win.update(window_index=j, window_seed=draw_seed(17, row["seed"], row["opportunity_step"], j, 11401),
                A_replicas=[{"suffix_index": k, "suffix_seed": draw_seed(17, row["seed"], row["opportunity_step"], j, k, 11402),
                    "W": 1., "tail": k * .02, "G": 1 + k * .02} for k in range(20, 28)])
            del win["suffix_draws"]
    data = parent.parent_dataset(base, horizon=1200, period=50)
    _, model = parent.fit_parent(data, np.ones(96, dtype=bool), total_df=3)
    _, controls = old.fit_views(data, np.ones(96, dtype=bool), total_df=3)
    models = {"base": model, "dense": deepcopy(model), "wide": deepcopy(model),
        "constant": controls["constant"], "causal_t": controls["causal_t"]}
    models["wide"]["weights"] = (np.asarray(model["weights"]) * .7).tolist()
    return cell, design, rows, models


class BoundaryInformationReadoutTest(unittest.TestCase):
    def test_E_provenance_roster_and_predictions_do_not_use_E_labels(self):
        cell, design, rows, models = fixture()
        data = readout.evaluation_data(rows, cell, design, cell["readout_prefix_seeds"])
        self.assertEqual(data["tail"].shape, (96, 8))
        saved = deepcopy(models)
        predictions = frozen_predictions(data, models)
        changed = deepcopy(rows)
        for row in changed:
            for window in row["windows"]:
                for replica in window["A_replicas"]:
                    replica["tail"] += 1000
                    replica["G"] += 1000
        other = frozen_predictions(readout.evaluation_data(changed, cell, design, cell["readout_prefix_seeds"]), models)
        self.assertEqual(models, saved)
        for key in predictions:
            np.testing.assert_array_equal(predictions[key], other[key])
        bad = deepcopy(rows)
        bad[0]["windows"][0]["A_replicas"][0]["suffix_index"] = 4
        with self.assertRaises(ValueError):
            readout.evaluation_data(bad, cell, design, cell["readout_prefix_seeds"])
        with self.assertRaises(ValueError):
            readout.evaluation_data(rows + rows[:1], cell, design, cell["readout_prefix_seeds"])

    def test_shared_label_loss_cancellation_MC_and_both_half_partitions(self):
        cell, design, rows, models = fixture()
        data = readout.evaluation_data(rows, cell, design, cell["readout_prefix_seeds"])
        predictions = frozen_predictions(data, models)
        report = readout.summarize(data, predictions)
        p, q, y = predictions["wide"], predictions["base"], data["tail"]
        result = report["comparisons"]["wide_minus_base"]
        self.assertAlmostEqual(result["MSE_delta"], np.mean((p - y.mean(axis=1)) ** 2 - (q - y.mean(axis=1)) ** 2))
        var = 4 * (p - q) ** 2 * y.var(axis=1, ddof=1) / 8
        self.assertAlmostEqual(result["conditional_suffix_MC_SE"], np.sqrt(var.sum()) / len(p))
        self.assertAlmostEqual(result["MSE_delta"], np.mean([r["MSE_delta"] for r in result["prefix_reports"]]))
        for name in ("suffix_halves", "window_halves"):
            self.assertAlmostEqual(result["MSE_delta"], np.mean([r["MSE_delta"] for r in result[name]]))
        np.testing.assert_array_equal(readout.paired_loss(p, p, y)["conditional_suffix_MC_SE"], 0.)

    def test_release_uses_only_reserved_E_and_exact_full_replay_budget(self):
        root = worker.ROOT
        design = json.loads((root / "protocols/plan_validity_boundary_information_2026-11-11.json").read_text())
        result = submitter.prepare(design, Path('design.json'), Path('fit.json'), Path('release.json'))
        self.assertEqual((len(result["shards"]), result["released_environment_steps"]), (8, 7411200))
        self.assertTrue(all(s["environment_steps"] == 926400 and s["new_pairs"] == 384 for s in result["shards"]))
        for i, cell in enumerate(design["cells"]):
            self.assertEqual([s for shard in result["shards"] if shard["cell_index"] == i for s in shard["prefix_seeds"]], cell["readout_prefix_seeds"])
        for job in result["jobs"]:
            self.assertEqual((job["cpu"], job["vram"], job["ram_mb"], job["require_node"]), (1, 0, 2048, None))
            self.assertTrue(job["allow_cpu_training"] and job["protect_from_auto_eviction"])
        bad = deepcopy(design)
        bad["cells"][0]["readout_prefix_seeds"][0] = bad["cells"][0]["development_prefix_seeds"][0]
        with self.assertRaises(ValueError):
            submitter.prepare(bad, Path('design.json'), Path('fit.json'), Path('release.json'))

    def test_global_freeze_checks_all_four_roots_before_E_access(self):
        cell, design, _, models = fixture()
        design["cells"] = [{**cell, "optimizer_seed": 17 + i} for i in range(4)]
        rosters = {"dense": {"window_keys": [[s, t, j] for s in range(8) for t in (50, 105, 150) for j in range(4)],
            "label_counts": [9] * 12 + [8] * 84},
            "wide": {"window_keys": [[s, t, j] for s in range(32) for t in (50, 105, 150) for j in range(4)], "label_counts": [2] * 384}}
        bundle = {"run_name": design["run_name"], "evaluation_prefixes_accessed": False,
            "cells": [{"task": c["task"], "optimizer_seed": c["optimizer_seed"], "models": models, "training_rosters": rosters} for c in design["cells"]]}
        report = {"run_name": design["run_name"], "status": "complete", "stage": "boundary_information_fit_complete",
            "all_models_frozen": True, "new_ridge_fits": 8, "new_environment_steps": 0, "evaluation_prefixes_accessed": False,
            "measured_cost": {"counts": {"ridge_fit_calls": 8}}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for part in ('public', 'private'):
                (root / 'fit' / part).mkdir(parents=True)
            (root / 'fit.json').write_text(json.dumps({"output_directory": "fit"}))
            (root / 'fit/public/result.json').write_text(json.dumps(report))
            target = root / 'fit/private/frozen_models.json'
            target.write_text(json.dumps(bundle))
            with patch.object(worker, 'ROOT', root):
                self.assertEqual(len(worker.load_global_freeze(design, {"fit_manifest": "fit.json"})), 4)
                del bundle["cells"][-1]["models"]["wide"]
                target.write_text(json.dumps(bundle))
                with self.assertRaises(ValueError):
                    worker.load_global_freeze(design, {"fit_manifest": "fit.json"})

    def test_worker_global_gate_before_native_work_and_exact_no_fit_budget(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        cell, design, rows, models = fixture()
        design.update(cells=[cell], source_hardware_profile={"fixture": True})
        prefixes = cell["readout_prefix_seeds"][:4]
        lookup = {(r["seed"], r["opportunity_step"]): r for r in rows}
        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'design.json').write_text(json.dumps(design))
                shard = {"cell_index": 0, "shard_index": 0, "prefix_seeds": prefixes, "output_directory": "out",
                    "environment_steps": 926400, "new_pairs": 384}
                manifest = root / 'release.json'
                manifest.write_text(json.dumps({"design_manifest": "design.json", "shards": [shard]}))

                def observe(*_, seed, **kwargs):
                    count_work('environment_steps', 1200)
                    return {"seed": seed, "check_steps": [50, 105, 150]}

                def acquire(*args, **kwargs):
                    count_work('environment_steps', 76800)
                    return deepcopy(args[2])

                gate = patch.object(worker, 'load_global_freeze', side_effect=ValueError('fixture invalid final root')) if fail else patch.object(
                    worker, 'load_global_freeze', return_value={(cell['task'], cell['optimizer_seed']): models})
                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), gate, \
                    patch.object(worker, 'load_frozen_C0', return_value=(None, None, SimpleNamespace(), None)), \
                    patch.object(worker, 'observe', side_effect=observe) as observation, \
                    patch.object(worker, 'fresh_reference', side_effect=lambda o, step, **kw: lookup[o['seed'], step]), \
                    patch.object(worker, 'acquire_windows', side_effect=acquire):
                    args = ['--manifest', str(manifest), '--shard-index', '0']
                    self.assertEqual(worker.main(args), int(fail))
                    self.assertEqual(observation.call_count, 0 if fail else 4)
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                report = json.loads((root / 'out/public/result.json').read_text())
                self.assertEqual(report['new_environment_steps'], 0 if fail else 926400)
                self.assertEqual(report['measured_cost']['counts'].get('ridge_fit_calls', 0), 0)
                if not fail:
                    self.assertEqual(report['measured_cost']['counts']['frozen_model_window_predictions'], 288)


if __name__ == '__main__':
    unittest.main()
