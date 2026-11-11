from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_information_fit as fit
from freq_hrl.experiments import plan_validity_boundary_parent_scale as parent
from freq_hrl.experiments import plan_validity_boundary_tail as old
from freq_hrl.experiments.plan_validity_boundary_information import dense_extra_windows, draw_seed
from scripts import run_plan_validity_boundary_information_fit as worker
from test_plan_validity_boundary_tail import bank


def banks():
    root = 17
    design = json.loads((worker.ROOT / "protocols/plan_validity_boundary_information_2026-11-11.json").read_text())
    cell = {"optimizer_seed": root, "development_prefix_seeds": list(range(8)), "wide_training_prefix_seeds": list(range(100, 124)),
        "dense_extra_windows": dense_extra_windows(list(range(8)))}
    base = bank()
    for r in base:
        for j, w in enumerate(r["windows"]):
            w.update(window_seed=draw_seed(root, r["seed"], r["opportunity_step"], j, 11401), window_costs=[1., 2.])
            for k, d in enumerate(w["suffix_draws"]):
                d.update(suffix_seed=draw_seed(root, r["seed"], r["opportunity_step"], j, k, 11402), W=1., G=1. + d["tail"])
    extra = {(k["seed"], k["state_index"], k["window_index"]) for k in cell["dense_extra_windows"]}

    def new_row(original, seed, dense):
        r = deepcopy(original)
        r["seed"] = seed
        state = [50, 105, 150].index(r["opportunity_step"])
        for j, w in enumerate(r["windows"]):
            indices = [4, 5, 6, 7, 8, 9] + ([10] if (seed, state, j) in extra else []) if dense else [4, 5]
            w.update(window_index=j, window_seed=draw_seed(root, seed, r["opportunity_step"], j, 11401),
                A_replicas=[{"suffix_index": k, "suffix_seed": draw_seed(root, seed, r["opportunity_step"], j, k, 11402),
                    "W": 1., "tail": float(k), "G": float(k + 1)} for k in indices])
            del w["suffix_draws"]
        return r

    dense = [new_row(r, r["seed"], True) for r in base]
    wide = [new_row(base[j], seed, False) for seed in cell["wide_training_prefix_seeds"] for j in range(3)]
    return cell, design, base, dense, wide


class BoundaryInformationFitTest(unittest.TestCase):
    def test_all_A_means_use_eight_or_nine_labels_not_only_the_first_two(self):
        cell, design, base, dense, wide = banks()
        d = fit.assemble(cell, design, base, dense, "dense")
        w = fit.assemble(cell, design, base, wide, "wide")
        self.assertEqual((len(d["prefixes"]), len(d["training_mean"]), sum(d["label_counts"])), (8, 96, 780))
        self.assertEqual((d["label_counts"].count(8), d["label_counts"].count(9)), (84, 12))
        self.assertEqual((len(w["prefixes"]), len(w["training_mean"]), sum(w["label_counts"])), (32, 384, 768))
        for i, (seed, step, j) in enumerate(d["window_keys"]):
            original = next(r for r in base if r["seed"] == seed and r["opportunity_step"] == step)
            added = next(r for r in dense if r["seed"] == seed and r["opportunity_step"] == step)
            labels = [r["tail"] for r in original["windows"][j]["suffix_draws"][:2]] + [r["tail"] for r in added["windows"][j]["A_replicas"]]
            self.assertAlmostEqual(d["training_mean"][i], np.mean(labels))
        self.assertEqual(d["now"].shape, (96, 120))
        self.assertEqual(w["now"].shape, (384, 120))

    def test_B_labels_cannot_change_assembled_targets_and_bad_indices_are_rejected(self):
        cell, design, base, dense, _ = banks()
        first = fit.assemble(cell, design, base, dense, "dense")
        changed = deepcopy(base)
        for row in changed:
            for win in row["windows"]:
                for r in win["suffix_draws"][2:]:
                    r["tail"] += 100000
                    r["G"] += 100000
        other = fit.assemble(cell, design, changed, dense, "dense")
        np.testing.assert_array_equal(first["training_mean"], other["training_mean"])
        bad = deepcopy(dense)
        bad[0]["windows"][0]["A_replicas"][0]["suffix_index"] = 2
        with self.assertRaisesRegex(ValueError, "wrong A indices"):
            fit.assemble(cell, design, base, bad, "dense")
        with self.assertRaisesRegex(ValueError, "roster"):
            fit.assemble(cell, design, base, dense + dense[:1], "dense")

    def test_explicit_target_solver_preserves_old_two_A_fit_exactly(self):
        data = parent.parent_dataset(bank(), horizon=1200, period=50)
        mask = data["groups"] != 0
        p, m = parent.fit_parent(data, mask, total_df=3)
        q, n = parent.fit_parent_targets(data["now"], data["wait_one_check"], data["tail"][:, :2].mean(axis=1), mask, total_df=3)
        self.assertEqual(m, n)
        np.testing.assert_array_equal(p, q)

    def test_public_audit_requires_every_matching_closed_shard_and_zero_fits(self):
        root = worker.ROOT
        design = json.loads((root / "protocols/plan_validity_boundary_information_2026-11-11.json").read_text())
        release = json.loads((root / "protocols/plan_validity_boundary_information_sample_2026-11-11.json").read_text())
        records = []
        for s in release["shards"]:
            c = design["cells"][s["cell_index"]]
            records.append({"run_name": design["run_name"], "task": c["task"], "optimizer_seed": c["optimizer_seed"],
                "method": s["method"], "shard_index": s["shard_index"], "status": "complete", "stage": "boundary_information_sample_complete",
                "sampled_prefix_seeds": s["prefix_seeds"], "completed_states": 12, "completed_pairs": s["new_pairs"],
                "new_environment_steps": s["environment_steps"], "fits": 0, "evaluation_prefixes_accessed": False,
                "measured_cost": {"cpu_seconds": 1., "counts": {"environment_steps": s["environment_steps"]}}})
        costs = fit.audit_public(design, release, records)
        self.assertEqual(sum(c["environment_steps"] for c in costs.values()), 11289600)
        self.assertEqual(sum(c["new_pairs"] for c in costs.values()), 4656)
        with self.assertRaises(ValueError):
            fit.audit_public(design, release, records[:-1])
        changed = deepcopy(records)
        changed[0]["measured_cost"]["counts"]["ridge_fit_calls"] = 1
        with self.assertRaisesRegex(ValueError, "fitting"):
            fit.audit_public(design, release, changed)

    def test_worker_audits_before_fits_and_freezes_all_models_before_readout(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        cell, design, base, dense, wide = banks()
        data = {"dense": fit.assemble(cell, design, base, dense, "dense"), "wide": fit.assemble(cell, design, base, wide, "wide")}
        original = parent.parent_dataset(bank(), horizon=1200, period=50)
        _, base_model = parent.fit_parent(original, np.ones(96, dtype=bool), total_df=3)
        _, old_models = old.fit_views(original, np.ones(96, dtype=bool), total_df=3)
        controls = {"base": base_model, "constant": old_models["constant"], "causal_t": old_models["causal_t"]}
        prepared = [{"cell": {"task": "fixture", "optimizer_seed": i}, "banks": data, "controls": controls} for i in range(4)]
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "design.json").write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"fixture": True}}))
                (root / "sample.json").write_text('{}')
                path = root / "fit.json"
                path.write_text(json.dumps({"design_manifest": "design.json", "sample_manifest": "sample.json", "output_directory": "out",
                    "total_effective_df": 3, "ridge_fit_calls": 8}))

                def readout(*_):
                    bundle = json.loads((root / "out/private/frozen_models.json").read_text())
                    self.assertEqual(len(bundle["cells"]), 4)
                    self.assertTrue(all(set(c["models"]) == {"base", "constant", "causal_t", "dense", "wide"} for c in bundle["cells"]))
                    return {"scope": "training_only"}

                loader = patch.object(worker, "load_training_inputs", side_effect=ValueError("fixture invalid last shard")) if fail else patch.object(worker, "load_training_inputs", return_value=({}, prepared))
                with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), loader, \
                     patch.object(worker, "training_readout", side_effect=readout):
                    args = ["--manifest", str(path)]
                    self.assertEqual(worker.main(args), int(fail))
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                report = json.loads((root / "out/public/result.json").read_text())
                self.assertEqual(report["measured_cost"]["counts"].get("ridge_fit_calls", 0), 0 if fail else 8)
                self.assertEqual(report["new_environment_steps"], 0)
                self.assertEqual((root / "out/private/frozen_models.json").exists(), not fail)


if __name__ == "__main__":
    unittest.main()
