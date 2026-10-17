from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_abstention_confirmation_stage292 as spec
from scripts import submit_native_transit_abstention_confirmation_stage292_scheduleurm as submit


def fixture():
    rows = [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p, "original_choice": 0}
        for k in spec.cache.keys("train") for p in spec.cache.PROBES]
    raw = np.zeros((16, 360), int)
    proposal = np.broadcast_to([int(r["probe_index"] == spec.cache.PROBES[0]) for r in rows], raw.shape).copy()
    keep = (proposal != raw)&np.asarray([r["replica"] < 2 for r in rows])[None]
    old = np.where(np.asarray([r["replica"] == 4 for r in rows])[None]&(proposal != raw), 2, raw)
    choices = np.stack((raw, old, proposal, np.where(keep, proposal, raw), raw, proposal), axis=1)
    probability, matching = spec.prior.thin_probabilities(spec.prior.row_cube(rows), proposal != raw, keep)
    ledger = {"choices": choices.tolist(), "arms": list(spec.prior.ARMS), "query": [0, 1, 2],
        "matched_thin_probabilities": probability.tolist(), "matching": matching}
    legacy = {"choices": np.stack((raw, old, proposal, proposal, proposal, proposal), axis=1).tolist()}
    return rows, ledger, legacy


class AbstentionConfirmationTest(unittest.TestCase):
    def test_seal_exact_actions_weights_and_only_disagreement_union(self):
        rows, ledger, legacy = fixture()
        sealed = spec.seal(ledger, legacy, rows)
        np.testing.assert_array_equal(sealed["choices"], np.asarray(ledger["choices"])[:, [0, 1, 3, 4]])
        self.assertEqual(sealed["matched_thin_probabilities"], ledger["matched_thin_probabilities"])
        self.assertEqual(sealed["states"][0]["candidates"], [0, 1])
        self.assertEqual(sealed["states"][1]["candidates"], [])
        self.assertEqual(sealed["states"][24]["candidates"], [0, 1, 2])
        self.assertEqual(len(sealed["pilot_indices"]), 10)
        ledger["matched_thin_probabilities"][0][0] = .5
        with self.assertRaisesRegex(ValueError, "matching changed"):
            spec.seal(ledger, legacy, rows)

    def test_seed_namespace_and_train_only_zero_fit_freeze(self):
        rows, ledger, legacy = fixture()
        new = [spec.cache.future_seed(k, p, d) for k in spec.cache.keys("train")
            for p in spec.cache.PROBES for d in (*spec.DRAWS, *spec.PILOT_DRAWS)]
        old = {spec.cache.future_seed(k, p, d) for k in spec.cache.keys("train") for p in spec.cache.PROBES for d in range(24)}
        self.assertEqual(len(new), len(set(new)))
        self.assertFalse(set(new)&old)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for module in (spec, spec.prior, spec.prior.source):
                stack.enter_context(patch.object(module, "ROOT", Path(temp)))
            spec.write_json(spec.prior.directory(spec.SOURCE_RUN)/"report/result.json",
                {"contract": spec.prior.contract(), "stage290_control_replay_exact": True})
            spec.write_json(spec.prior.directory(spec.SOURCE_RUN)/"report_raw/primary.json", ledger)
            spec.write_json(spec.prior.source.directory(spec.prior.SOURCE_RUN)/"report_raw/primary.json", legacy)

            def load(run, role):
                self.assertEqual((run, role), (spec.CACHE_RUN, "train"))
                return rows

            stack.enter_context(patch.object(spec.cache, "load_rows", side_effect=load))
            stack.enter_context(patch.object(spec.cache, "load_panels", side_effect=AssertionError("No labels for freeze")))
            stack.enter_context(patch.object(spec.cache, "fit", side_effect=AssertionError("No fit")))
            stack.enter_context(patch.object(np, "load", side_effect=AssertionError("No models")))
            spec.freeze("unit292")
            plan = spec.read(spec.directory("unit292")/"freeze/result.json")
            self.assertEqual(plan["collection_tasks"], 60*16)
            self.assertEqual(plan["state_futures"], 60*64)
            self.assertEqual(plan["native_episodes"], 64*190)
            self.assertEqual(plan["pilot_native_episodes"], 60)
            self.assertEqual(plan["held_prefixes_read"], 0)

    def test_thinning_expectation_frozen_bootstrap_and_shared_profile_MC(self):
        panels = np.zeros((3, 64, 6, 4))
        panels[:, :, 1, 0] = np.asarray([-2, 2, 2])[:, None]+(np.arange(64)-31.5)[None]/64
        panels[:, :, 1, 1] = panels[:, :, 1, 0]
        panels[:, :, 1, 3] = -.5
        raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
        witness = np.array([[1, 0, 0]])
        sealed = {"choices": np.stack((raw, raw, witness, raw), axis=1).tolist(),
            "proposal": proposal.tolist(), "matched_thin_probabilities": [[1/3]*3]}
        result = spec.summarize(sealed, panels, np.arange(3).reshape(1, 3, 1))
        expected = np.mean([panels[list(c), :, 1, 0].sum()/3/64 for c in combinations(range(3), 1)])
        c = result["comparisons"]["matched_thin_vs_raw"]
        self.assertAlmostEqual(c["mean"]["cost"], expected)
        ix = np.random.default_rng(292).choice(3, (5000, 3), replace=True)
        points = panels[:, :, 1, 0].mean(axis=1)/3
        np.testing.assert_allclose(c["prefix_CI95"]["cost"], np.quantile(points[ix].mean(axis=1), [.025, .975]))
        expected_se = np.sqrt(np.var(panels[:, :, 1, 0]/3, axis=1, ddof=1).sum()/64)/3
        self.assertAlmostEqual(c["conditional_MC_se"]["cost"], expected_se)
        repeated = {k: np.repeat(np.asarray(v), 16, axis=0).tolist() for k, v in sealed.items()}
        duplicate = spec.summarize(repeated, panels, np.arange(3).reshape(1, 3, 1))
        for interval in ("mean", "conditional_MC_se"):
            np.testing.assert_allclose(list(c[interval].values()), list(duplicate["comparisons"]["matched_thin_vs_raw"][interval].values()), atol=1e-15)

    def test_joint_requires_future_and_prefix_uncertainty_plus_halves(self):
        comparisons = {name: {interval: {f: [-2., -1.] for f in spec.FIELDS}
            for interval in ("prefix_CI95", "conditional_MC_CI95")}
            for name in ("zero_abstain_vs_raw", "zero_abstain_vs_matched_thin")}
        result = {"comparisons": comparisons, "halves": {h: {"zero_abstain": {"cost": -1, "wait_min": -1}}
            for h in ("first32", "last32")}}
        self.assertTrue(spec.flags(result)["joint_retention_supported"])
        comparisons["zero_abstain_vs_raw"]["conditional_MC_CI95"]["cost"] = [-2, 1]
        self.assertFalse(spec.flags(result)["joint_retention_supported"])
        self.assertTrue(spec.flags(result)["cost_information_supported"])
        comparisons["zero_abstain_vs_raw"]["conditional_MC_CI95"]["cost"] = [-2, -1]
        result["halves"]["last32"]["zero_abstain"]["wait_min"] = 1
        self.assertFalse(spec.flags(result)["joint_retention_supported"])

    def test_native_pilot_restores_archived_rng_and_shared_futures(self):
        rows, ledger, legacy = fixture()
        sealed = spec.seal(ledger, legacy, rows)
        state = sealed["states"][0]
        native = spec.cache.native
        calls = []

        def episode(*args, **kwargs):
            candidate = kwargs.get("candidate")
            seed = kwargs["plan_class"].keywords["continuation_seed"]
            self.assertEqual(kwargs["reference"].prefix, "archived_prefix")
            calls.append((candidate, seed))
            return {"supply": [], "probe": {"selected": candidate}}, SimpleNamespace(prefix="archived_prefix", snapshot={"source_rng": 1},
                env=SimpleNamespace(scenario_tape=SimpleNamespace(seed=seed, _streams={"used": True})))

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(spec, "ROOT", Path(temp)))
            stack.enter_context(patch.object(spec.cache, "ROOT", Path(temp)))
            path = spec.cache.prefix_path(spec.CACHE_RUN, tuple(state["key"]))
            path.mkdir(parents=True)
            with (path/"prefix_refs.pkl").open("wb") as f:
                pickle.dump({"prefix": "archived_prefix", "snapshots": {state["probe"]: {"archived_rng": 1}}}, f)
            spec.write_json(spec.directory("unit292")/"sealed_raw/primary.json", sealed)
            spec.write_json(spec.directory("unit292")/"freeze/result.json",
                {"contract": spec.contract(), "software_qualified": True, "pilot_indices": [0]})
            stack.enter_context(patch.object(native, "native_inputs", return_value=({}, "frozen_checkpoint")))
            restore = stack.enter_context(patch.object(native, "restore_caller_torch_rng"))
            stack.enter_context(patch.object(native.source.shared, "episode", side_effect=episode))
            stack.enter_context(patch.object(native.source, "qualify_observer"))
            stack.enter_context(patch.object(native.source.shared, "qualify"))
            stack.enter_context(patch.object(native.source.shared, "policy_trace", return_value=[]))
            validate = stack.enter_context(patch.object(native.source, "validate_continuations"))
            stack.enter_context(patch.object(native, "paired_values", return_value=[0., 0., 0., 0.]))
            spec.collect("unit292", "pilot", 0)
            receipt = spec.read(spec.receipt_path("unit292", "pilot", 0))
            self.assertEqual(receipt["native_episodes"], 6)
            self.assertEqual(receipt["exact_identities"], 2)
            self.assertEqual(restore.call_count, 6)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(validate.call_args.kwargs["candidates"], [0, 1])
            self.assertEqual([c for c, _ in calls], [None, 0, 1]*2)
            self.assertEqual({s for _, s in calls}, {spec.cache.future_seed(tuple(state["key"]), state["probe"], d) for d in spec.PILOT_DRAWS})
            receipt["outcomes_used_for_selection"] = True
            spec.write_json(spec.receipt_path("unit292", "pilot", 0), receipt)
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                spec.pilot_report("unit292")

    def test_complete_scalar_report_roundtrip_and_missing_chunk_rejection(self):
        rows, ledger, legacy = fixture()
        sealed = spec.seal(ledger, legacy, rows)
        active = [s["state_index"] for s in sealed["states"] if s["candidates"]]
        original_read = spec.read
        missing = False

        def read(path):
            if "receipts" in path.parts:
                if missing:
                    raise FileNotFoundError(path)
                i = int(path.parts[-3][5:])
                chunk = int(path.parts[-2][5:])
                return {"contract": spec.contract(), "software_qualified": True,
                    "draws": list(spec.DRAWS[4*chunk:4*chunk+4]),
                    "native_episodes": 4*(1+len(sealed["states"][i]["candidates"]))}
            if "samples" in path.parts:
                i = int(path.parts[-3][5:])
                chunk = int(path.parts[-2][5:])
                state = sealed["states"][i]
                return [{"draw": d, "future_seed": spec.cache.future_seed(tuple(state["key"]), state["probe"], d),
                    "values": {str(a): ([0., 0., 0., 0.] if a != 1 else [-3., -2., -1., -.5]
                        if state["key"][3] < 2 else [4., 5., -1., -.5]) for a in state["candidates"]}}
                    for d in spec.DRAWS[4*chunk:4*chunk+4]]
            return original_read(path)

        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)):
            spec.write_json(spec.directory("unit292")/"sealed_raw/primary.json", sealed)
            spec.write_json(spec.directory("unit292")/"freeze/result.json",
                {"active_indices": active, "collection_tasks": len(active)*16, "state_futures": len(active)*64})

            def load(run, role):
                self.assertEqual((run, role), (spec.CACHE_RUN, "train"))
                return rows

            with patch.object(spec, "read", side_effect=read), patch.object(spec.cache, "load_rows", side_effect=load), \
                    patch.object(np, "load", side_effect=AssertionError("No model inference")):
                spec.report("unit292")
                result = original_read(spec.directory("unit292")/"report/result.json")
                self.assertTrue(result["joint_retention_supported"])
                self.assertTrue(result["cost_information_supported"])
                self.assertEqual(result["execution"]["native_episodes"], 64*190)
                self.assertEqual(result["held_prefixes_read"], 0)
                missing = True
                with self.assertRaises(FileNotFoundError):
                    spec.report("unit292")

    def test_scheduler_metadata_no_pin_small_receipts_and_dependency_gate(self):
        task = submit.task_specification("unit292", "collect", 0, 1)
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 1536, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn(str(spec.directory("unit292")), task["stage_input_paths"])
        self.assertTrue(task["result_dir"].endswith("receipts/state0/chunk1"))
        records = [{"id": "t1", "status": "running", "exit_code": None, **submit.task_specification("unit292", "pilot-report")}]
        plan = {"contract": spec.contract(), "software_qualified": True, "active_indices": [0], "pilot_indices": [0]}
        with patch.dict(sys.modules, {"scheduler": SimpleNamespace(load_state=lambda: {"tasks": records})}), \
                patch.object(sys, "argv", ["submit", "--run-name", "unit292", "--phase", "collect"]), \
                patch.object(submit, "inventory", return_value=records), patch.object(spec, "read", return_value=plan), \
                patch.object(submit, "execute_bulk") as execute:
            with self.assertRaises(SystemExit):
                submit.main()
            execute.assert_not_called()
            records[0].update(status="done", exit_code=0)
            submit.main()
            self.assertEqual(len(execute.call_args.args[0]), 16)


if __name__ == "__main__":
    unittest.main()
