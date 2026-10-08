from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_option_conditioning as experiment
from scripts import pointmaze_option_conditioning_stage143_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_mean_option_query import job_for
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def cached_rows(source_data, period, roles):
    paths, rows = [], []
    for role in roles:
        for panel in spec.PANELS:
            job = job_for(source_data, period, panel, role)
            queries = {p: experiment.query.worker_mean_query(job, probe_scale=spec.source.PROBE_SCALES[p]) for p in spec.PROBES}
            label = {p: row["path"] for p, row in queries.items()}
            row = experiment.worker_replay(job+(label,))
            for probe in spec.PROBES:
                np.testing.assert_array_equal(row["gradients"][probe], queries[probe]["gradient"])
            paths.append(label); rows.append(row)
    return paths, rows


def test_one_replay_reconstructs_both_probe_gradients_with_correct_denominators(source_data):
    models, predictor, calibrations, args, teachers = source_data
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        paths, rows = cached_rows(source_data, 50, spec.seed_roles(spec.ROOTS[0])["replayed_training"][:1])
    assert len(rows) == 2
    assert all(r["row"]["upper_sample"] is False for r in rows)
    assert not np.array_equal(rows[0]["gradients"]["wide"], rows[0]["gradients"]["local"])
    for row, path in zip(rows, paths):
        assert row["row"]["episode_return"] == path["wide"]["mean_return"] == path["local"]["mean_return"]


def test_raw_reconstruction_compact_update_subspace_and_equal_functional_steps(source_data):
    models, predictor, calibrations, args, teachers = source_data
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        _, rows = cached_rows(source_data, 50, spec.seed_roles(spec.ROOTS[0])["replayed_training"][:2])
    trainer = warm_trainer(source_data)
    initial = job_for(source_data, 50)[2]
    experiment.joint.load_weights(trainer, initial)
    actors, fits, similarities, cost = experiment.fit_candidates(trainer, rows)
    assert cost["empirical_fisher_solves"] == 2 and cost["upper_candidate_weight_steps"] == 8
    assert cost["policy_geometry_forward_batches"] == 9
    projection = experiment.causal_summary_projection()
    for probe in spec.PROBES:
        old_rows = [{"batch": r["batch"], "gradient": r["gradients"][probe]} for r in rows]
        old, _, _ = experiment.query.mean_candidates(trainer, old_rows)
        for sign in ("plus", "minus"):
            torch.testing.assert_close(actors[probe+"_raw_"+sign], old[sign], atol=1e-8, rtol=1e-5)
            for rep in spec.REPRESENTATIONS:
                np.testing.assert_allclose(fits[probe+"_"+rep]["mean_step_RMS"][sign], spec.MEAN_STEP_RMS, atol=1e-8, rtol=0)
                torch.testing.assert_close(actors[probe+"_"+rep+"_"+sign]["log_std"], initial["upper_actor"]["log_std"], atol=0, rtol=0)
            delta = (actors[probe+"_compact_"+sign]["net.0.weight"].double()-initial["upper_actor"]["net.0.weight"].double()).numpy()
            projected = delta@projection.T@np.linalg.solve(projection@projection.T, projection)
            np.testing.assert_allclose(delta, projected, atol=1e-7, rtol=0)
        assert fits[probe+"_raw"]["geometry"]["state_dimensions"] == 392
        assert fits[probe+"_compact"]["geometry"]["state_dimensions"] == 26
        assert np.isfinite(similarities[probe])
    torch.testing.assert_close(experiment.joint.weights(trainer), initial, atol=0, rtol=0)
    assert not trainer.upper_actor_optimizer.state and not trainer.upper_value_optimizer.state


def test_budget_has_no_new_queries_and_fresh_evaluation_roster():
    b = spec.budget()
    assert b["native_episodes"] == 1248 and b["native_steps"] == 1497600
    assert b["replay_episodes"] == b["credit_checks"] == 32
    assert b["collection_episodes"] == b["counterfactual_episodes"] == 0
    assert b["evaluation_episodes"] == 1216 and b["evaluation_alias_assignments"] == 64
    assert b["empirical_fisher_solves"] == 4 and b["upper_candidate_weight_steps"] == 16
    assert b["policy_geometry_forward_batches"] == 18
    seen = set()
    for root in spec.ROOTS:
        roles, old = spec.seed_roles(root), spec.source.seed_roles(root)
        assert roles["replayed_training"] == old["replayed_training"]
        assert not set(roles["evaluation"]).intersection(set(old["evaluation"]) | seen)
        seen.update(roles["evaluation"])


def test_native_runner_has_zero_query_cost_and_preserves_label_lineage(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    local_path, wide_path, warm_path = [tmp_path/n/"result.json" for n in ("local", "wide", "warm")]
    warm_spec = spec.source.source.source.warm_source
    with patch.object(spec, "arguments", return_value=args), patch.object(spec, "EVALUATION_EPISODES", 2), \
            patch.object(spec.source.source.source, "SCENARIOS", 2), \
            patch.object(spec, "source_result", side_effect=lambda r, wide=False: wide_path if wide else local_path), \
            patch.object(spec, "warm_result", return_value=warm_path), \
            patch.object(experiment.warm.spec, "source_result", return_value=warm_path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r, p, **kw: teachers[str(p)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        caches = {p: {"status": "complete", "protocol": protocol.PROTOCOL, "root": root, "cost": protocol.budget(),
            "seed_roles": protocol.seed_roles(root), "contract": protocol.contract(), "groups": {}}
            for p, protocol in (("local", spec.source), ("wide", spec.source.source))}
        caches["local"]["inherited_earlier_source_cost"] = {"retained": True}
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
        for path, payload in ((local_path, caches["local"]), (wide_path, caches["wide"]), (warm_path, warm)):
            experiment.write_json(path, payload)
        result = experiment.run(root, tmp_path/"output"/"result.json")
        assert result["cost"] == spec.budget() and result["cost"]["native_episodes"] == 84
        assert result["cost"]["native_steps"] == 8400 and result["cost"]["counterfactual_episodes"] == 0
        assert result["inherited_Stage142_cost"] == caches["local"]["cost"]
        assert result["inherited_Stage141_cost"] == caches["wide"]["cost"]
        assert result["inherited_earlier_source_cost"] == {"retained": True}
        assert not list((tmp_path/"output").rglob("*.pt")) and not list((tmp_path/"output").rglob("*.npz"))
        assert (tmp_path/"output"/"completion"/"ready.json").is_file()
        for group in result["groups"].values():
            assert group["both_probe_label_replays"] == group["frozen_deployment_and_noise_pairing"] == "passed"
            assert group["sampled_minus_mean_return"]["source_forecast"]["paired_differences"] == [0., 0.]
