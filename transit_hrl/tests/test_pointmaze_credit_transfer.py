import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_credit_transfer as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


def test_projection_retains_current_state_and_causal_stream_moments_but_is_noninvertible():
    projection = experiment.causal_summary_projection()
    assert projection.shape == (26, 392) and np.linalg.matrix_rank(projection) == 26
    state = np.zeros(392)
    state[:6], state[-2:] = np.arange(6), [.0, .75]
    trend = np.arange(1., 7.) * .01
    history = np.arange(6.) + np.arange(64)[:, None] * trend
    state[6:390] = history.ravel()
    value = projection @ state
    np.testing.assert_array_equal(value[:8], np.r_[state[:6], state[-2:]])
    np.testing.assert_allclose(value[8:14], history[-1], atol=1e-14)
    np.testing.assert_allclose(value[14:20], history.mean(0), atol=1e-14)
    np.testing.assert_allclose(value[20:26], trend, atol=1e-14)
    variation = np.zeros(392)
    variation[6:390:6][:3] = [1., -2., 1.]
    np.testing.assert_allclose(projection @ variation, 0., atol=1e-16)
    assert np.linalg.norm(variation) > 0


def test_compact_candidates_stay_in_summary_subspace_and_match_native_fisher(source_data):
    models, _, _, args, teachers = source_data
    actor = experiment.joint.make_trainer(models["50"], teachers["50"], args).upper_actor
    before = copy.deepcopy(actor.state_dict())
    rng = np.random.default_rng(128)
    labels = [{"state": rng.normal(size=392).astype(np.float32), "gradient": rng.normal(size=8).tolist(),
        "query": {"scenario_seed": 1 + i // 2, "panel": ("A", "B")[i % 2]}} for i in range(8)]
    candidates, learning, work = experiment.learn_directions(actor, labels)
    projection = experiment.causal_summary_projection()
    for sign, weights in candidates["compact"].items():
        weight = weights["net.0.weight"].numpy()
        np.testing.assert_allclose(weight @ np.linalg.pinv(projection) @ projection, weight, atol=2e-9, rtol=0)
    for method in candidates:
        for sign, weights in candidates[method].items():
            torch.testing.assert_close(weights["log_std"], before["log_std"], atol=0, rtol=0)
            np.testing.assert_allclose(learning[method]["radius"]["exact_kl"][sign], experiment.spec.FISHER_RADIUS, rtol=2e-6)
    assert learning["raw"]["geometry"]["state_dimensions"] == 392
    assert learning["compact"]["geometry"]["state_dimensions"] == 26
    assert work == {"empirical_fisher_solves": 2, "fisher_jvp_batches": 2, "exact_kl_forward_batches": 4}
    torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)


def test_reduced_runner_holds_out_entire_scene_and_fits_only_new_training(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    before = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    cache = tmp_path / "source" / "result.json"
    spec = experiment.spec
    seen_training, original_fit = [], experiment.source.source.fit_method
    def record_fit(rows, method):
        seen_training.extend(r["zero"]["seed"] for r in rows)
        return original_fit(rows, method)
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(spec.source, "LABEL_SCENARIOS", 2))
        stack.enter_context(patch.object(spec.source, "arguments", return_value=args))
        stack.enter_context(patch.object(spec, "TRAINING_SCENARIOS", 1))
        stack.enter_context(patch.object(spec, "EVALUATION_EPISODES", 1))
        stack.enter_context(patch.object(spec, "source_result", return_value=cache))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(experiment.source.source, "fit_method", side_effect=record_fit))
        groups = {}
        for period in spec.PERIODS:
            trainer = experiment.joint.make_trainer(models[str(period)], teachers[str(period)], args)
            labels = []
            for q in spec.source.queries(410011, period):
                query = {"scenario_seed": q["scenario_seed"], "start": q["start"], "prefix_noise_seed": q["noise_seed"],
                    "suffix_noise_seeds": {"A": q["noise_seed"]}}
                row, _ = experiment.source.credit.intervention_episode(trainer, args=args, query=query, panel="A",
                    action_delta=np.zeros(8), period=period, predictor=predictor, envelope=cal[str(period)]["envelope"])
                labels.append({"query": q, "gradient": np.arange(1., 9.).tolist(),
                    "zero_episode_return": row["episode_return"], "zero_suffix_return": row["suffix_return"]})
            groups[str(period)] = {"native_labels": labels}
        experiment.write_json(cache, {"status": "complete", "protocol": spec.source.PROTOCOL, "root": 410011,
            "cost": spec.source.budget(), "groups": groups})
        result = experiment.run(410011, tmp_path / "candidate" / "result.json")
        assert result["cost"] == spec.budget()
        assert set(seen_training) == {r["scenario_seed"] for r in spec.training_roles(410011)}
    assert result["cost"]["native_episodes"] == 88 and result["cost"]["native_steps"] == 8800
    assert len(list((tmp_path / "candidate").rglob("*.pt"))) == 4 and not list(tmp_path.rglob("*.npz"))
    for p, group in result["groups"].items():
        for fold in group["leave_scene_out"]:
            assert fold["held_out_scenario"] not in fold["training_scenarios"]
            assert len(fold["training_scenarios"]) == 1
            for method in fold["methods"].values():
                held = method["held_out_prediction"]["trajectory_derivatives"]
                assert {r["scenario_seed"] for r in held} == {fold["held_out_scenario"]}
                assert {r["panel"] for r in held} == {"A", "B"}
                assert method["training"]["geometry"]["training_rows"] == (4 if p == "50" else 2)
        assert group["effects"]["compact_minus_source_forecast"] == group["effects"]["compact_minus_compact_blinded"]
        assert group["lower_and_critics_frozen"] == "passed"
        experiment.joint.source.native.curves.support.assert_frozen(models[p], before[p])


def test_fixed_rosters_budgets_and_no_reuse_of_calibration_or_evaluation():
    spec = experiment.spec
    assert spec.budget()["native_episodes"] == 1248 and spec.budget()["native_steps"] == 1497600
    assert spec.budget()["leave_scene_out_fits"] == 16 and spec.budget()["training_state_replays"] == 288
    assert spec.MINIMUM_GAIN == .5 and spec.FISHER_RADIUS == spec.source.FISHER_RADIUS
    seen = set()
    for root in spec.ROOTS:
        old = {s for r in spec.source.label_roles(root) + spec.source.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(spec.source.evaluation_seeds(root))
        training = {s for r in spec.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        evaluation = set(spec.evaluation_seeds(root))
        assert len(training) == 48 and len(evaluation) == 32
        assert not training & evaluation and not (training | evaluation) & (old | seen)
        seen.update(training | evaluation)
