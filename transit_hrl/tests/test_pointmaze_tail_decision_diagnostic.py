from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_tail_decision_diagnostic as diagnostic
from scripts import pointmaze_tail_decision_diagnostic_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


def case(seed, window, tails):
    return {"seed": seed, "check_step": 50, "window_advantage": window,
            "tail_advantages": np.asarray(tails, dtype=float),
            "predictions": {m: 0. for m in diagnostic.FROZEN_MODELS}}


class TailDecisionDiagnosticTest(unittest.TestCase):
    def test_split_is_exhaustive_disjoint_and_fixed(self):
        a, b = diagnostic.split_indices(64)
        self.assertEqual(a, list(range(32)))
        self.assertEqual(b, list(range(32, 64)))
        with self.assertRaisesRegex(ValueError, "equal"):
            diagnostic.split_indices(5)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            diagnostic.score_direction([case(1, 0., [1., 2., 3., 4.])], [0, 1], [1, 2])

    def test_oracle_can_lose_on_disjoint_scoring_futures(self):
        cases = [case(1, 0., [2., 2., -1., -1.])]
        primary = diagnostic.score_direction(cases, [0, 1], [2, 3])
        reverse = diagnostic.score_direction(cases, [2, 3], [0, 1])
        self.assertTrue(primary["rows"][0]["now_choices"]["oracle_tail"])
        self.assertEqual(primary["summary"]["comparisons"]["oracle_tail"]["mean_ise_benefit_vs_short_window"], -1.)
        self.assertEqual(reverse["summary"]["comparisons"]["oracle_tail"]["mean_ise_benefit_vs_short_window"], 0.)
        self.assertGreater(cases[0]["tail_advantages"].mean(), 0.)

    def test_scoring_labels_do_not_choose_primary_actions(self):
        cases = [case(1, 0., [2., 2., -1., -1.])]
        before = diagnostic.score_direction(cases, [0, 1], [2, 3])
        cases[0]["tail_advantages"][2:] = -100.
        after = diagnostic.score_direction(cases, [0, 1], [2, 3])
        self.assertEqual(before["rows"][0]["now_choices"], after["rows"][0]["now_choices"])
        self.assertEqual(after["rows"][0]["scoring_total_advantage_mean"], -100.)
        cases[0]["tail_advantages"][:2] = -2.
        changed = diagnostic.score_direction(cases, [0, 1], [2, 3])
        self.assertFalse(changed["rows"][0]["now_choices"]["oracle_tail"])
        self.assertEqual(changed["rows"][0]["scoring_total_advantage_mean"], -100.)

    def test_benefit_sign_weighting_standard_error_and_ties(self):
        cases = [case(1, -1., [2., 2., 0., 4.]), case(2, 1., [-2., -2., -4., 0.])]
        result = diagnostic.score_direction(cases, [0, 1], [2, 3])["summary"]
        oracle = result["comparisons"]["oracle_tail"]
        self.assertEqual((result["short_window_now_count"], oracle["now_count"], oracle["switches_vs_short_window"]), (1, 1, 2))
        self.assertEqual(oracle["mean_ise_benefit_vs_short_window"], 1.)
        self.assertAlmostEqual(oracle["conditional_mc_standard_error"], np.sqrt(2))
        tied = diagnostic.score_direction([case(1, 0., [0., 0., 1., 1.])], [0, 1], [2, 3])
        self.assertFalse(tied["rows"][0]["short_window_now"])
        self.assertFalse(any(tied["rows"][0]["now_choices"].values()))
        self.assertEqual(tied["summary"]["comparisons"]["oracle_tail"]["conditional_mc_standard_error"], 0.)

    def preflight_inputs(self):
        options = cell_options(208001, preflight=True)
        args = SimpleNamespace(optimizer_seed=208001,
                               branch_fit_seeds=list(options["branch_fit"]),
                               trigger_eval_seeds=list(options["trigger_eval"]),
                               **spec.input_results(208001, preflight=True),
                               **spec.sampling_options(preflight=True))
        cells = [json.loads(getattr(args, k + "_result").read_text())["cells"][0]
                 for k in ("endpoint", "fresh", "prediction")]
        return args, cells

    def test_cached_preflight_has_no_fits_and_keeps_all_states(self):
        args, _ = self.preflight_inputs()
        result = diagnostic.run_cell(args)
        for key in ("critic_fits", "controller_training_iterations", "additional_primitive_steps",
                    "policy_updates", "evaluation_paths_used"):
            self.assertEqual(result[key], 0)
        for direction in result["directions"].values():
            self.assertEqual(len(direction["rows"]), 2)
            self.assertEqual(sorted(direction["selection_indices"] + direction["scoring_indices"]), list(range(4)))
        first, second = (result["directions"][key]["rows"] for key in ("first_to_second", "second_to_first"))
        for a, b in zip(first, second):
            for method in diagnostic.FROZEN_MODELS:
                self.assertEqual(a["now_choices"][method], b["now_choices"][method])

    def test_join_rejects_changed_credit_identity_and_missing_state(self):
        args, cells = self.preflight_inputs()
        options = dict(fit_seeds=args.branch_fit_seeds, opportunities_per_path=1, replicates=4)
        bad = deepcopy(cells)
        selected = bad[1]["rows"][0]
        row = next(r for r in bad[0]["branch_fit_rows"] if (r["seed"], r["check_step"]) == (selected["seed"], selected["check_step"]))
        row["window_ise_advantage"] += 1.
        with self.assertRaisesRegex(ValueError, "credit identity"):
            diagnostic.join_cases(*bad, **options)
        bad = deepcopy(cells)
        bad[2]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "exact frozen opportunities"):
            diagnostic.join_cases(*bad, **options)

    def test_scheduler_stages_only_three_caches_on_dynamic_cpu_pool(self):
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_tail_decision", root, preflight=preflight, protocol_spec=spec)
                self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 1536, None))
                inputs = spec.input_results(root, preflight=preflight)
                self.assertEqual(len(inputs), 3)
                for source in inputs.values():
                    self.assertIn(str(source.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
