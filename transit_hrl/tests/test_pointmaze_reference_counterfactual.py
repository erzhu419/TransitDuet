import copy
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_reference_counterfactual as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def query(start):
    return {"scenario_seed": 123001, "start": start, "prefix_noise_seed": 123002,
        "suffix_noise_seeds": {"A": 123002, "B": 123003}}


def test_zero_intervention_matches_exact_stage121_forecast_source(source_data):
    models, predictor, cal, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    q = query(50)
    with patch.object(experiment.joint.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.joint.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        row, audit = experiment.intervention_episode(trainer, args=args, query=q, panel="A",
            action_delta=np.zeros(8), period=50, predictor=predictor, envelope=cal["50"]["envelope"])
        _, reference, old_audit = experiment.joint.native_episode(trainer, args=args,
            seed=q["scenario_seed"], noise_seed=q["prefix_noise_seed"], arm="forecast", period=50,
            predictor=predictor, envelope=cal["50"]["envelope"], collect=False)
    np.testing.assert_allclose(row["episode_return"], reference["episode_return"], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(audit["measurements"], old_audit["measurements"])
    np.testing.assert_array_equal(audit["innovations"], old_audit["innovations"])
    assert row["reference_correction_peak"] == 0.
    assert audit["state"].shape == (392,)


def test_paired_native_worker_has_common_prefix_noise_and_exact_suffix_credit(source_data):
    models, predictor, cal, args, teachers = source_data
    snapshot = copy.deepcopy(models["50"].state_dict())
    job = (experiment.joint.weights(models["50"]), teachers["50"], query(50), 50,
        predictor, cal["50"]["envelope"])
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), \
            patch.object(experiment.joint.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.joint.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        result = experiment.worker_query(job)
    assert result["native_episodes"] == 38 and result["native_steps"] == 3800
    assert result["prefix_suffix_noise_and_authority_checks"] == result["policy_freeze"] == "passed"
    assert result["max_innovation_error"] < 3e-5
    assert result["max_reference_peak"] <= .05 + 1e-8
    assert set(result["gradients"]) == {"A", "B"}
    assert set(result["crossfit"]) == {"A_to_B", "B_to_A"}
    experiment.joint.source.native.curves.support.assert_frozen(models["50"], snapshot)


def test_finite_difference_and_causal_state_pullback(source_data):
    models, _, _, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    state = np.linspace(0., .25, 392).astype(np.float32)
    native = np.arange(1., 9.)
    panels = {p: {"zero": {"suffix_return": 100.}} for p in ("A", "B")}
    for p in panels:
        for i in range(8):
            for sign, direction in (("plus", 1), ("minus", -1)):
                panels[p][f"axis{i}_{sign}"] = {"suffix_return": 100. + direction * experiment.spec.EPSILON * native[i]}
    gradients = experiment.gradients(panels)
    np.testing.assert_allclose(gradients["A"], native, atol=1e-10, rtol=0)
    snapshot = experiment.joint.weights(trainer)
    report = experiment.pullback(trainer, [{"state": state, "gradients": gradients,
        "crossfit": {"A_to_B": {"plus_zero": .1, "plus_minus": .2},
                     "B_to_A": {"plus_zero": .3, "plus_minus": .4}}}])
    np.testing.assert_allclose(report["native_gradient_cosine"], 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(report["actor_pullback_cosine"], 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(report["crossfit_plus_zero"], .2, atol=1e-12, rtol=0)
    torch.testing.assert_close(experiment.joint.weights(trainer), snapshot, atol=0, rtol=0)


def test_reduced_run_records_actual_native_budget_without_adopting_policy(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    with patch.object(experiment.spec, "arguments", return_value=args), \
            patch.object(experiment.spec, "STARTS", (0,)), patch.object(experiment.spec, "SCENARIOS_PER_START", 1), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(experiment.joint.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.joint.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        result = experiment.run(410011, tmp_path / "result.json")
        assert result["cost"] == experiment.spec.budget()
    assert result["cost"]["native_episodes"] == 76 and result["cost"]["native_steps"] == 7600
    assert result["cost"]["optimizer_steps"] == result["cost"]["checkpoint_writes"] == result["cost"]["native_trace_writes"] == 0
    assert result["teacher_upper_lower_and_source_frozen"] == "passed"
    assert (tmp_path / "completion" / "ready.json").is_file()
    assert all("state" not in q for g in result["groups"].values() for q in g["queries"])


def test_frozen_rosters_cover_both_periods_and_do_not_reuse_stage121_paths():
    spec = experiment.spec
    assert spec.budget()["native_episodes"] == 608
    assert spec.budget()["native_steps"] == 729600
    assert spec.ROOTS == (410011, 410023) and spec.EPSILON == .005
    seen = set()
    for root in spec.ROOTS:
        roles = spec.source.seed_roles(root, preflight=False)
        old = {s for round_ in roles["training_rounds"] for r in round_ for s in (r["scenario_seed"], *r["noise_seeds"])} | set(roles["native_evaluation"])
        queries = spec.queries(root)
        assert [q["start"] for q in queries] == [0, 0, 300, 300, 600, 600, 900, 900]
        seeds = {s for q in queries for s in (q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values())}
        assert len(seeds) == 32 and not seeds.intersection(old | seen)
        assert all(q["start"] % p == 0 for q in queries for p in spec.PERIODS)
        seen.update(seeds)
