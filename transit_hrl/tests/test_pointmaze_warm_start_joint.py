import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_warm_start_joint as experiment
from scripts import pointmaze_warm_start_joint_stage136_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def warm_trainer(source_data):
    models, predictor, calibrations, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    with torch.no_grad():
        trainer.upper_actor.net[0].bias.copy_(torch.linspace(-.1, .1, 8))
    return trainer


def native_patch():
    return patch.object(experiment.joint.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task())


def bounds_patch():
    return patch.object(experiment.joint.source.native.joint, "pointmaze_goal_bounds",
        return_value=(-2*np.ones(2), 2*np.ones(2)))


def test_interventions_replay_same_batch_and_joint_equals_separate_updates(source_data):
    models, predictor, calibrations, args, teachers = source_data
    initial = experiment.joint.weights(warm_trainer(source_data))
    job = (experiment.joint.weights(models["50"]), teachers["50"], {"warm_start": initial},
        136001, [136002, 136003, 136004, 136005], 50, predictor, calibrations["50"]["envelope"], True)
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        outputs = experiment.worker_group(job)
    batches = [b for _, b, _ in outputs if b is not None]
    assert len(batches) == 4 and all(b.upper.size == 2 and b.lower.size == 100 for b in batches)
    assert all(row["upper_sample"] == (v == "sampled") for v, _, row in outputs)
    states = {}
    for method in spec.METHODS:
        trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
        experiment.joint.load_weights(trainer, initial)
        reports = experiment.update_intervention(trainer, batches, root=spec.ROOTS[0], period=50, method=method)
        assert set(reports) == ({"upper", "lower"} if method == "joint" else {method.split("_")[0]})
        for level, report in reports.items():
            inactive = "lower" if level == "upper" else "upper"
            assert all(report["parameter_delta_rms"][level+s] > 0 for s in ("_actor", "_value"))
            assert all(report["parameter_delta_rms"][inactive+s] == 0 for s in ("_actor", "_value"))
            assert report["old_logp_replay_max_error"] < spec.ppo.LOGP_REPLAY_TOLERANCE
        torch.testing.assert_close(trainer.lower_actor.teacher.state_dict(), teachers["50"], atol=0, rtol=0)
        torch.testing.assert_close(trainer.upper_actor.log_std, initial["upper_actor"]["log_std"], atol=0, rtol=0)
        states[method] = experiment.joint.weights(trainer)
    for level in ("upper", "lower"):
        for suffix in ("_actor", "_value"):
            torch.testing.assert_close(states["joint"][level+suffix], states[level+"_only"][level+suffix], atol=0, rtol=0)
            inactive = "lower_only" if level == "upper" else "upper_only"
            torch.testing.assert_close(states[inactive][level+suffix], initial[level+suffix], atol=0, rtol=0)


def test_rosters_budget_and_interaction_are_not_confirmation_statistics():
    assert spec.ROOTS == (410037, 410049)
    assert spec.budget()["native_episodes"] == 480
    assert spec.budget()["native_steps"] == 576000
    assert spec.budget()["upper_actor_optimizer_steps"] == 8
    assert spec.budget()["lower_actor_optimizer_steps"] == 152
    assert spec.budget()["ppo_update_calls"] == 8
    assert spec.budget()["checkpoint_writes"] == spec.budget()["native_trace_writes"] == 0
    seen = set()
    for root in spec.ROOTS:
        roles = spec.seed_roles(root)
        seeds = [s for r in roles["training"] for s in (r["scenario_seed"], *r["noise_seeds"])] + roles["evaluation"]
        assert len(set(seeds)) == len(seeds) and not seen.intersection(seeds)
        seen.update(seeds)
        inherited = spec.source.training_roles(root) + spec.source.validation_roles(root) + spec.source.label_roles(root)
        old = {s for r in inherited for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        assert not old.intersection(seeds)
    values = {"source_forecast": 1., "warm_start": 3., "upper_only": 4., "lower_only": 2., "joint": 3.5, "joint_blinded": 1.5}
    metrics = ("reference_correction_rms", "reference_correction_peak", "learned_residual_rms",
        "learned_residual_peak", "plan_delta_rms", "upper_mean_rms")
    rows = [{v: {"episode_return": x, **dict.fromkeys(metrics, 0.)} for v, x in values.items()}]
    report = experiment.evaluation_summary(rows)
    assert report["effects"]["joint_interaction"]["mean"] == .5
    assert report["effects"]["joint_minus_warm_start"]["mean"] == .5
    assert all("ci" not in e for e in report["effects"].values())


def test_reduced_runner_counts_real_updates_and_does_not_write_checkpoints(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    cached = {"status": "complete", "protocol": spec.source.PROTOCOL, "root": root,
        "cost": spec.source.budget(), "inherited_source_cost": spec.source.source.budget(), "groups": {}}
    path = tmp_path / "source" / "result.json"
    for period in spec.PERIODS:
        initial = experiment.joint.weights(warm_trainer(source_data))["upper_actor"]
        cached["groups"][str(period)] = {"selection": {"method": "refresh"},
            "training_native_return_fits": {"refresh": {"pooled": {"scale": .4}}}}
        checkpoint = path.parent / "final_weights" / f"period_{period}_refresh_upper.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": spec.source.PROTOCOL, "root": root, "period": period,
            "method": "refresh", "fit": {"scale": .4}, "weights": initial}, checkpoint)
    experiment.write_json(path, cached)
    with patch.object(spec, "arguments", return_value=args), patch.object(spec, "SCENARIOS", 2), \
            patch.object(spec, "EVALUATION_EPISODES", 2), patch.object(spec, "source_result", return_value=path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r, p, **kw: teachers[str(p)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        result = experiment.run(root, tmp_path / "output" / "result.json")
        assert result["cost"] == spec.budget()
        assert result["cost"]["native_episodes"] == 48
        assert (tmp_path / "output" / "completion" / "ready.json").is_file()
        assert not list((tmp_path / "output").rglob("*.pt"))
        for group in result["groups"].values():
            assert group["layer_parameter_isolation"] == "passed"
            assert group["diagnostic"]["upper"]["episodes"] == 4
            assert set(group["effects"]) == {f"{a}_minus_{b}" for a, b in spec.CONTRASTS} | {"joint_interaction"}
        saved = torch.load(checkpoint, weights_only=False)
        saved["fit"]["scale"] = .8
        torch.save(saved, checkpoint)
        with pytest.raises(ValueError, match="source fit"):
            experiment.load_selected_upper(cached, root, period)
