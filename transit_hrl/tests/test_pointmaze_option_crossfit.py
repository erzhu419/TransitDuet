from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_option_crossfit as experiment
from scripts import pointmaze_option_crossfit_stage144_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_mean_option_query import job_for
from test_pointmaze_option_conditioning import cached_rows
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def test_split_excludes_both_noise_panels_of_whole_scene():
    rows = [{"row": {"seed": s}, "panel": p} for s in (11, 12, 13) for p in spec.PANELS]
    for seed in (11, 12, 13):
        training, held = experiment.scene_split(rows, seed)
        assert len(training) == 4 and len(held) == 2
        assert all(r["row"]["seed"] != seed for r in training)
        assert all(r["row"]["seed"] == seed for r in held)
        assert {r["panel"] for r in held} == set(spec.PANELS)


def test_held_out_labels_only_change_diagnostic_not_fitted_actor(source_data):
    models, predictor, calibrations, args, teachers = source_data
    roles = spec.seed_roles(spec.ROOTS[0])["replayed_training"][:3]
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        _, rows = cached_rows(source_data, 50, roles)
    training, held = experiment.scene_split(rows, roles[0]["scenario_seed"])
    trainer = warm_trainer(source_data)
    initial = job_for(source_data, 50)[2]
    experiment.joint.load_weights(trainer, initial)
    uppers, _, _, _ = experiment.previous.fit_candidates(trainer, training)
    frozen = {v: {n: t.clone() for n, t in w.items()} for v, w in uppers.items()}
    records, calls = experiment.held_out_geometry(trainer, uppers, held)
    multiplied = [{**r, "gradients": {p: 4*g for p, g in r["gradients"].items()}} for r in held]
    scaled, scaled_calls = experiment.held_out_geometry(trainer, uppers, multiplied)
    assert calls == scaled_calls == 18
    for a, b in zip(records, scaled):
        assert a["held_out_mean_step_RMS"] == b["held_out_mean_step_RMS"]
        assert a["raw_to_compact_mean_step_cosine"] == b["raw_to_compact_mean_step_cosine"]
        for variant in spec.VARIANTS:
            np.testing.assert_allclose(b["predicted_return_increment"][variant],
                4*a["predicted_return_increment"][variant], atol=1e-12, rtol=0)
    torch.testing.assert_close(uppers, frozen, atol=0, rtol=0)
    torch.testing.assert_close(experiment.joint.weights(trainer), initial, atol=0, rtol=0)
    assert not trainer.upper_actor_optimizer.state and not trainer.lower_actor_optimizer.state


def test_budget_is_cached_crossfit_only():
    b = spec.budget()
    assert b["native_episodes"] == 288 and b["native_steps"] == 345600
    assert b["replay_episodes"] == b["credit_checks"] == 32
    assert b["evaluation_episodes"] == 256 and b["empirical_fisher_solves"] == 32
    assert b["held_out_geometry_forward_batches"] == 288
    assert b["collection_episodes"] == b["counterfactual_episodes"] == 0
    for root in spec.ROOTS:
        roles = spec.seed_roles(root)
        assert roles["replayed_training"] == spec.source.seed_roles(root)["replayed_training"]
        assert roles["held_out_scene_order"] == [r["scenario_seed"] for r in roles["replayed_training"]]


def test_full_crossfit_runner_excludes_scenes_and_never_queries(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    conditioning_path, local_path, wide_path, warm_path = [tmp_path/n/"result.json"
        for n in ("conditioning", "local", "wide", "warm")]
    warm_spec = spec.source.source.source.source.warm_source
    with patch.object(spec, "arguments", return_value=args), \
            patch.object(spec.source.source.source.source, "SCENARIOS", 3), \
            patch.object(spec, "source_result", return_value=conditioning_path), \
            patch.object(spec.source, "source_result", side_effect=lambda r, wide=False: wide_path if wide else local_path), \
            patch.object(spec.source, "warm_result", return_value=warm_path), \
            patch.object(experiment.warm.spec, "source_result", return_value=warm_path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r, p, **kw: teachers[str(p)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        caches = {name: {"status": "complete", "protocol": s.PROTOCOL, "root": root, "cost": s.budget(),
            "seed_roles": s.seed_roles(root), "contract": s.contract(), "groups": {}}
            for name, s in (("conditioning", spec.source), ("local", spec.source.source), ("wide", spec.source.source.source))}
        caches["conditioning"]["inherited_Stage142_cost"] = {"retained": True}
        warm = {"status": "complete", "protocol": warm_spec.PROTOCOL, "root": root, "cost": warm_spec.budget(), "groups": {}}
        for period in spec.PERIODS:
            with patch.object(experiment.joint.source.native, "_WORKER", (models[str(period)], args)):
                paths, _ = cached_rows(source_data, period, spec.seed_roles(root)["replayed_training"])
            caches["wide"]["groups"][str(period)] = {"mean_query_paths": [p["wide"] for p in paths]}
            caches["local"]["groups"][str(period)] = {"local_query_paths": [p["local"] for p in paths]}
            warm["groups"][str(period)] = {"selection": {"method": "refresh"},
                "training_native_return_fits": {"refresh": {"pooled": {"scale": .4}}}}
            ckpt = warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol": warm_spec.PROTOCOL, "root": root, "period": period, "method": "refresh",
                "fit": {"scale": .4}, "weights": experiment.joint.weights(warm_trainer(source_data))["upper_actor"]}, ckpt)
        for path, payload in ((conditioning_path, caches["conditioning"]), (local_path, caches["local"]),
                              (wide_path, caches["wide"]), (warm_path, warm)):
            experiment.write_json(path, payload)
        original_fit = experiment.previous.fit_candidates
        fitted_rosters = []
        def fit_without_test(trainer, rows):
            seeds = [r["row"]["seed"] for r in rows]
            assert len(seeds) == 4 and len(set(seeds)) == 2
            fitted_rosters.append(set(seeds))
            return original_fit(trainer, rows)
        with patch.object(experiment.previous, "fit_candidates", side_effect=fit_without_test), \
                patch.object(experiment.query, "worker_mean_query", side_effect=AssertionError("new query")):
            result = experiment.run(root, tmp_path/"output"/"result.json")
        expected_budget = spec.budget()
    assert result["cost"] == expected_budget and result["cost"]["native_episodes"] == 108
    assert result["cost"]["native_steps"] == 10800
    assert result["cost"]["counterfactual_episodes"] == result["cost"]["collection_episodes"] == 0
    assert result["cost"]["held_out_geometry_forward_batches"] == 108
    assert result["inherited_source_cost"]["inherited_Stage142_cost"] == {"retained": True}
    assert len(fitted_rosters) == 6
    for group in result["groups"].values():
        assert group["whole_scene_exclusion"] == group["frozen_deployment_and_noise_pairing"] == "passed"
        assert len(group["folds"]) == 3
        for fold in group["folds"]:
            assert fold["held_out_scenario_seed"] not in fold["training_scenario_seeds"]
            assert len(fold["held_out_geometry"]) == 2
            for effect in fold["native_control"]["effects"].values(): assert len(effect["paired_differences"]) == 2
        for effect in group["held_out_control"]["effects"].values(): assert len(effect["paired_differences"]) == 6
    assert not list((tmp_path/"output").rglob("*.pt")) and not list((tmp_path/"output").rglob("*.npz"))
    assert (tmp_path/"output"/"completion"/"ready.json").is_file()
