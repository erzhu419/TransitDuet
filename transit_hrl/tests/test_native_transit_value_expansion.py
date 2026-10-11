import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_native_transit_value_expansion_stage187 as spec
from scripts.analyze_native_transit_value_expansion_stage187 import summarize
from scripts.submit_native_transit_value_expansion_stage187_scheduleurm import task_specification
from scripts import submit_native_transit_value_expansion_stage187_scheduleurm as submit
from tests.test_native_transit_timetable_context import fixture as context_fixture


def fixture(*, preflight=True):
    old, rows, sources = context_fixture()
    result = {}
    templates = {(r["root"], r["scenario"], r["probe_index"]): r for r in rows}
    for root, scenario, replica, probe in spec.cells(preflight=preflight):
        row = copy.deepcopy(templates[root, scenario, probe])
        row["scene_seed"] = spec.scene_seed(root, scenario, replica)
        row["context"]["cost_before"] = .4
        row["timetable_context"] = copy.deepcopy(next(c["timetable_context"] for c in
            old[root, scenario, 1]["contexts"] if c["index"] == probe))
        reference = sources[root, scenario, 1]
        baseline = {k: reference[k] for k in spec.OUTCOME_KEYS if k in reference}
        baseline.update(scene_seed=row["scene_seed"], peak_fleet=14, passengers_generated=1000,
            passengers_unserved=0, trips_completed=262, ep_steps=61380, window_ledger_qualified=True)
        branches = []
        for candidate in range(6):
            identity = candidate == row["original_choice"]
            branch = {**baseline, "candidate": candidate, "identity_control_reproduced": identity}
            if not identity:
                branch.update(peak_fleet=15, service_cost_restricted=.4,
                    restricted_wait_horizon_min=-.1, ep_reward=2.)
            branches.append(branch)
        baseline_probe = {**{k: row[k] for k in ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")},
            "greedy": row["original_choice"]}
        label_cell = {"seed": root, "scenario": scenario, "probe_index": probe, "observer": {**baseline, "probe": baseline_probe},
            "branches": [{**b, "probe": {"selected": b["candidate"]}} for b in branches], "training_context": row["context"]}
        row["labels"] = spec.labels.records(label_cell)["labels"]
        for label, branch in zip(row["labels"], branches):
            label["future_fleet_cost"] = max(branch["peak_fleet"]-12, 0)**2/12 - 4/12
        trace = spec.shared.policy_trace(reference["supply"])
        result[root, scenario, replica, probe] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "seed": root, "scenario": scenario, "replica": replica, "scene_seed": row["scene_seed"], "probe_index": probe,
            "software_qualified": True, "training_updates": 0, "native_steps": 7*61380,
            "frozen_weights_qualified_episodes": 7, "window_ledger_qualified_episodes": 7,
            "matched_prefix_state_rng_branches": 6, "exact_identity_reproductions": 1,
            "continuation_qualified": True, "observer": baseline, "branches": branches,
            "policy_trace": trace, "training_record": row}
    return result, old


class ValueExpansionTest(unittest.TestCase):
    def test_native_clock_is_not_the_lower_transition_count(self):
        baseline = {k: 0. for k in spec.shared.METRICS}
        baseline.update(N_fleet=12, simulation_end_time_s=61380, trips_completed=262, ep_steps=4970,
            window_ledger_qualified=True, schedule={"upper_decisions": 44, "scheduled_trips": 262})
        with patch.object(spec.shared, "validate_schedule"):
            spec.qualify_observer(baseline)
            baseline["simulation_end_time_s"] -= 1
            with self.assertRaises(ValueError): spec.qualify_observer(baseline)

    def test_fixed_seeds_and_budget_leave_reserved_confirmation_unopened(self):
        reserved = {900000000+r*3000+j*10+i for r in spec.ROOTS for j in range(5) for i in range(4)}
        spec.validate_scene_roles(reserved)
        self.assertEqual(len(spec.cells()), 480)
        self.assertEqual(len(spec.cells(preflight=True)), 10)
        self.assertEqual(7*61380*len(spec.cells()), 206236800)
        self.assertTrue(set(spec.cells(preflight=True)).issubset(spec.cells()))
        reserved.add(spec.scene_seed(397, "low_noise", 0))
        with self.assertRaises(ValueError): spec.validate_scene_roles(reserved)

    def test_physical_targets_identity_floor_and_signed_units(self):
        cells, _ = fixture()
        cell = cells[397, "low_noise", 0, 8]
        row = spec.qualify_cell(cell)
        target = row["labels"][0]
        self.assertAlmostEqual(target["physical_return_delta"], -40.)
        self.assertAlmostEqual(target["fleet_cost_delta"], 5/12)
        self.assertAlmostEqual(target["future_fleet_cost"], 5/12)
        self.assertAlmostEqual(target["nonfleet_cost_delta"], .4-5/12)
        self.assertAlmostEqual(target["wait_delta_min"], -.1)
        for failure in ("seed", "budget", "rng", "continuation", "order", "identity", "units", "floor", "curve"):
            broken = copy.deepcopy(cell)
            if failure == "seed": broken["scene_seed"] -= 1
            elif failure == "budget": broken["native_steps"] -= 1
            elif failure == "rng": broken["matched_prefix_state_rng_branches"] = 5
            elif failure == "continuation": broken["continuation_qualified"] = False
            elif failure == "order": broken["branches"].reverse()
            elif failure == "identity": broken["branches"][row["original_choice"]]["ep_reward"] += 1
            elif failure == "units": broken["training_record"]["labels"][0]["physical_return_delta"] *= -1
            elif failure == "floor": broken["training_record"]["labels"][0]["future_fleet_cost"] -= 1
            else: broken["training_record"]["timetable_context"]["candidate_planned_s"][0][-1] += 1
            with self.subTest(failure=failure), self.assertRaises(ValueError): spec.qualify_cell(broken)

    def test_scene_grouping_does_not_count_correlated_callbacks_as_independent(self):
        cells, _ = fixture(preflight=False)
        result = summarize(cells)
        self.assertEqual((result["independent_scene_groups"], result["training_states"], result["paired_labels"]), (80, 480, 2880))
        self.assertEqual((result["new_native_episodes"], result["exact_identity_reproductions"]), (3360, 480))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        broken = copy.deepcopy(cells)
        broken[397, "low_noise", 0, 9]["policy_trace"][0]["selected"] = 3
        with self.assertRaises(ValueError): summarize(broken)
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells)

    def test_preflight_requires_all_regimes_and_both_lower_roots(self):
        cells, _ = fixture()
        result = summarize(cells, preflight=True)
        self.assertEqual((result["independent_scene_groups"], result["training_states"], result["new_native_episodes"]), (10, 10, 70))
        with self.assertRaises(ValueError): summarize(cells)

    def test_scheduler_unique_unbound_tasks_only_sync_compact_result_directories(self):
        tasks = [task_specification("unit_expansion", *k) for k in spec.cells()]
        self.assertEqual(len({t["signature"] for t in tasks}), 480)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 480)
        for t in tasks:
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertEqual([Path(p).name for p in t["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
            self.assertNotIn("_raw", t["result_dir"])

    def test_remaining_release_counts_logical_signatures_not_failed_retries(self):
        tasks = [task_specification("unit_release", *k) for k in spec.cells(preflight=True)]
        registered = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract()}
        with patch.object(submit, "inventory", return_value=tasks+copy.deepcopy(tasks)), \
                patch.object(Path, "read_text", return_value=json.dumps(registered)), \
                patch.object(submit, "load_summary", return_value={"software_qualified": True}) as load, \
                patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as execute, \
                patch.object(sys, "argv", ["submit", "--run-name", "unit_release", "--phase", "remaining"]):
            submit.main()
        self.assertEqual(len(execute.call_args.args[0]), 470)
        self.assertTrue(load.call_args.kwargs["preflight"])

    def test_worker_runs_identity_first_and_keeps_full_rollouts_on_server(self):
        cells, old = fixture()
        cell = cells[397, "low_noise", 0, 8]
        row = cell["training_record"]
        context = next(c for c in old[397, "low_noise", 1]["contexts"] if c["index"] == 8)
        baseline = {**cell["observer"], "probe": {**{k: row[k] for k in
            ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")}, "greedy": row["original_choice"]},
            "supply": {"decisions": [{**d, "candidates": [{"score": abs(i-d["greedy"])} for i in range(6)]}
                for d in cell["policy_trace"]]}}
        snapshot = {"buses": [{"on_route": True, "direction": b["direction"], "passengers": [None]*b["onboard_count"]}
                for b in context["active_buses"]],
            "queues": [[s, d, [None]*n] for s, d, n in context["queue_by_station"]],
            "routes": [[i, speed] for i, speed in enumerate(context["route_speed_mps"])], "cost_before": row["context"]["cost_before"]}
        baseline["supply"]["decisions"][8]["historical_peak"] = 14
        calls = []
        def episode(*args, candidate=None, **kwargs):
            calls.append(candidate)
            result = copy.deepcopy(baseline)
            if candidate is not None:
                result.update(cell["branches"][candidate], source_control_reproduced=candidate == row["original_choice"])
                result["probe"]["selected"] = candidate
            return result, SimpleNamespace(contexts=[context], snapshot=snapshot)
        with patch.object(spec.shared, "episode", side_effect=episode), patch.object(spec.shared, "qualify"), \
                patch.object(spec, "qualify_observer"), patch.object(spec, "validate_continuations"), \
                patch.object(spec.shared.source.incumbent.source.distributional, "validate_model"), \
                patch.object(spec.native, "load_source", return_value=(None, Path("server_ckpt"))), \
                patch.object(spec, "raw_directory", return_value=Path("server_probe_raw")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                contextlib.redirect_stdout(io.StringIO()):
            result = spec.collect(397, "low_noise", 0, 8, {}, Path("local_probe/result.json"))
        self.assertEqual(calls, [None, 2, 0, 1, 3, 4, 5])
        self.assertEqual(str(write.call_args.args[0]), "server_probe_raw/paired_rollouts.json")
        self.assertTrue(all("probe" not in b and "window_panels" not in b for b in result["branches"]))
        self.assertNotIn("supply", result["observer"])
        self.assertEqual(result["training_record"]["timetable_context"], context["timetable_context"])


if __name__ == "__main__":
    unittest.main()
