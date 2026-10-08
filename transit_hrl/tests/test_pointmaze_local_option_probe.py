from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_local_option_probe as experiment
from scripts import pointmaze_local_option_probe_stage142_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_mean_option_query import job_for
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def test_local_probe_reduces_cubic_bias_without_changing_policy_std():
    center, innovation = .2, np.array([[1.]])
    targets = {}
    for method, scale in spec.PROBE_SCALES.items():
        probe_std = spec.STD*scale
        targets[method] = experiment.previous.paired_query_gradient(
            [(center+probe_std)**3], [(center-probe_std)**3], innovation, probe_std).item()
    derivative = 3*center**2
    np.testing.assert_allclose(targets["local"]-derivative, (targets["wide"]-derivative)/100, atol=1e-14, rtol=0)
    assert spec.STD == .05 and spec.PROBE_SCALES == {"wide": 1., "local": .1}


def test_probe_only_change_keeps_mean_path_noise_and_replays_wide_labels(source_data):
    models, predictor, calibrations, args, teachers = source_data
    job = job_for(source_data, 50)
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        wide = experiment.previous.worker_mean_query(job)
        local = experiment.previous.worker_mean_query(job, probe_scale=.1)
        replay = experiment.worker_replay(job+(wide["path"],))
    np.testing.assert_array_equal(wide["batch"].state, local["batch"].state)
    np.testing.assert_array_equal(wide["batch"].action, local["batch"].action)
    np.testing.assert_array_equal(wide["gradient"], replay["gradient"])
    assert wide["path"]["mean_return"] == local["path"]["mean_return"] == replay["row"]["episode_return"]
    assert local["cost"] == wide["cost"]
    assert not np.array_equal(local["gradient"], wide["gradient"])
def test_budget_and_rosters_separate_new_queries_from_prior_wide_cost():
    b = spec.budget()
    assert b["native_episodes"] == 1920 and b["native_steps"] == 2304000
    assert b["replay_episodes"] == b["collection_episodes"] == 32
    assert b["counterfactual_episodes"] == 1152 and b["mean_query_label_pairs"] == 576
    assert b["evaluation_episodes"] == 704 and b["evaluation_alias_assignments"] == 64
    assert b["score_gradient_batches"] == b["mean_score_forward_batches"] == 0
    assert b["empirical_fisher_solves"] == 4 and b["credit_checks"] == 1216
    seen = set()
    for root in spec.ROOTS:
        roles, old = spec.seed_roles(root), spec.source.seed_roles(root)
        assert roles["replayed_training"] == old["replayed_training"]
        assert not set(roles["evaluation"]).intersection(set(old["evaluation"]) | seen)
        seen.update(roles["evaluation"])


def test_native_run_exact_replay_fixed_steps_and_compact_artifacts(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    source_path, warm_path = tmp_path/"source"/"result.json", tmp_path/"warm"/"result.json"
    warm_spec = spec.source.source.warm_source
    with patch.object(spec, "arguments", return_value=args), patch.object(spec.source, "arguments", return_value=args), \
            patch.object(spec.source.source, "SCENARIOS", 2), patch.object(spec, "EVALUATION_EPISODES", 2), \
            patch.object(spec.source, "EVALUATION_EPISODES", 2), patch.object(spec, "source_result", return_value=source_path), \
            patch.object(spec, "warm_result", return_value=warm_path), \
            patch.object(experiment.warm.spec, "source_result", return_value=warm_path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r, p, **kw: teachers[str(p)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        cached = {"status": "complete", "protocol": spec.source.PROTOCOL, "root": root, "cost": spec.source.budget(),
            "seed_roles": spec.source.seed_roles(root), "contract": spec.source.contract(), "groups": {},
            "inherited_Stage140_cost": {"retained": True}}
        warm = {"status": "complete", "protocol": warm_spec.PROTOCOL, "root": root, "cost": warm_spec.budget(), "groups": {}}
        for period in spec.PERIODS:
            roles = spec.seed_roles(root)["replayed_training"]
            with patch.object(experiment.joint.source.native, "_WORKER", (models[str(period)], args)):
                rows = [experiment.previous.worker_mean_query(job_for(source_data, period, p, r)) for r in roles for p in spec.PANELS]
            trainer = warm_trainer(source_data)
            experiment.joint.load_weights(trainer, job_for(source_data, period)[2])
            _, fit, _ = experiment.previous.mean_candidates(trainer, rows)
            cached["groups"][str(period)] = {"mean_query_paths": [r["path"] for r in rows], "learning": {"mean_query": fit}}
            warm["groups"][str(period)] = {"selection": {"method": "refresh"},
                "training_native_return_fits": {"refresh": {"pooled": {"scale": .4}}}}
            checkpoint = warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol": warm_spec.PROTOCOL, "root": root, "period": period, "method": "refresh",
                "fit": {"scale": .4}, "weights": experiment.joint.weights(warm_trainer(source_data))["upper_actor"]}, checkpoint)
        experiment.write_json(source_path, cached); experiment.write_json(warm_path, warm)
        result = experiment.run(root, tmp_path/"output"/"result.json")
        assert result["cost"] == spec.budget() and result["cost"]["native_episodes"] == 84
        assert result["cost"]["native_steps"] == 8400 and result["cost"]["mean_query_label_pairs"] == 12
        assert result["inherited_Stage141_cost"] == cached["cost"]
        assert result["inherited_earlier_source_cost"]["inherited_Stage140_cost"] == {"retained": True}
        assert (tmp_path/"output"/"completion"/"ready.json").is_file()
        assert not list((tmp_path/"output").rglob("*.pt")) and not list((tmp_path/"output").rglob("*.npz"))
        for g in result["groups"].values():
            assert g["wide_label_and_policy_replay"] == g["shared_mean_path_and_probe_innovations"] == "passed"
            for method in spec.METHODS:
                for rms in g["learning"][method]["mean_step_RMS"].values():
                    np.testing.assert_allclose(rms, spec.MEAN_STEP_RMS, atol=1e-8, rtol=0)
            assert g["sampled_minus_mean_return"]["source_forecast"]["paired_differences"] == [0., 0.]
            assert np.isfinite(g["query_gradient_wide_to_local_cosine"])
