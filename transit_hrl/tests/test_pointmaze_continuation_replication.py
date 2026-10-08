import copy
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from freq_hrl.experiments import pointmaze_compact_continuation as kernel
from freq_hrl.experiments import pointmaze_continuation_replication as experiment
from scripts import pointmaze_third_update_stage132_spec as third_step
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


@pytest.mark.parametrize("spec", (experiment.spec, third_step))
def test_shared_kernel_uses_source_protocol_method_baseline_and_training_roles(source_data, tmp_path, spec):
    models, predictor, cal, original_args, teachers = source_data
    args = copy.copy(original_args)
    root = args.optimizer_seed = spec.ROOTS[0]
    cache = tmp_path/"source"/"result.json"
    groups = {}
    for period in spec.PERIODS:
        actor = kernel.joint.make_trainer(models[str(period)], teachers[str(period)], args).upper_actor
        with torch.no_grad():
            actor.net[0].bias.copy_(torch.arange(1., 9.)*.001)
        fit = {"scale": 1.}
        path = cache.parent/"final_weights"/f"period_{period}_{spec.SOURCE_METHOD}_upper.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": spec.source.PROTOCOL, "root": root, "period": period,
            "method": spec.SOURCE_METHOD, "fit": fit, "weights": actor.state_dict()}, path)
        groups[str(period)] = {"training_native_return_fits": {spec.SOURCE_METHOD: {"pooled": fit}}}
    source_cost = spec.source.budget()
    kernel.write_json(cache, {"status": "complete", "protocol": spec.source.PROTOCOL,
        "root": root, "cost": source_cost, "groups": groups})
    seen, original_fit = [], kernel.fit_method
    def record_fit(rows, method):
        seen.extend(r["zero"]["seed"] for r in rows)
        assert all(r["zero"]["upper_mean_rms"] > 0. for r in rows)
        return original_fit(rows, method)
    with ExitStack() as stack:
        native_task(stack)
        for name in ("LABEL_SCENARIOS", "TRAINING_SCENARIOS"):
            stack.enter_context(patch.object(spec, name, 1))
        stack.enter_context(patch.object(spec, "EVALUATION_EPISODES", 2))
        stack.enter_context(patch.object(spec, "arguments", return_value=args))
        stack.enter_context(patch.object(spec, "source_result", return_value=cache))
        stack.enter_context(patch.object(kernel.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(kernel.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(kernel, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(kernel, "fit_method", side_effect=record_fit))
        result = kernel.run(root, tmp_path/"candidate"/"result.json", protocol_spec=spec)
        assert result["cost"] == spec.budget()
        assert set(seen) == {r["scenario_seed"] for r in spec.training_roles(root)}
        for p, g in result["groups"].items():
            assert g["evaluation_seeds"] == spec.evaluation_seeds(root)
            assert g["effects"]["refresh_minus_source_forecast"] == g["effects"]["refresh_minus_refresh_blinded"]
            assert g["mean_metrics"][spec.SOURCE_BASELINE]["upper_mean_rms"] > 0.
            assert f"refresh_minus_{spec.SOURCE_BASELINE}" in g["effects"]
    assert result["protocol"] == spec.PROTOCOL and result["source_protocol"] == spec.source.PROTOCOL
    assert result["inherited_source_cost"] == source_cost and result["kind"] == spec.EVIDENCE_ROLE
    assert result["cost"]["native_episodes"] == 162 and result["cost"]["native_steps"] == 16200


def synthetic_cells(gains, increments=.3, relabel50=-.01, relabel100=.2):
    spec, cells = experiment.spec, []
    for root, gain in zip(spec.ROOTS, gains):
        groups = {}
        for p in spec.PERIODS:
            effects = {f"refresh_minus_{b}": {"mean": x, "paired_differences": [x]*spec.EVALUATION_EPISODES}
                for b, x in (("source_forecast", gain), ("single", increments),
                    ("stale", relabel50 if p == 50 else relabel100), ("refresh_blinded", gain))}
            groups[str(p)] = {"effects": effects, "evaluation_seeds": spec.evaluation_seeds(root),
                "training_roles": spec.training_roles(root), "lower_and_critics_frozen": "passed"}
        cells.append({"status": "complete", "protocol": spec.PROTOCOL, "root": root, "cost": spec.budget(),
            "source_protocol": spec.source.PROTOCOL, "inherited_source_cost": spec.source.budget(), "groups": groups})
    return cells


def test_inference_keeps_material_increment_and_relabeling_claims_separate():
    summary = experiment.aggregate(synthetic_cells([1.]*6))
    assert summary["material_continuation_gate"] == "supported_both_periods"
    assert summary["universal_relabeling_gate"] == "not_closed"
    assert summary["period_relabeling_gate"] == {"50": False, "100": True}
    assert summary["statistics"]["n_independent"] == 6 and summary["statistics"]["bonferroni_family_size"] == 6
    assert experiment.aggregate(synthetic_cells([.3]*6))["material_continuation_gate"] == "not_closed"
    assert experiment.aggregate(synthetic_cells([1.]*6, increments=-.1))["material_continuation_gate"] == "not_closed"
    varied = experiment.aggregate(synthetic_cells([-1.]*3+[1.]*3))
    assert varied["endpoints"]["50/refresh_minus_source_forecast"]["ci"][1] > .6


def test_inference_rejects_wrong_roots_source_cost_and_pairing():
    cells = synthetic_cells([1.]*6)
    with pytest.raises(ValueError, match="six additional"):
        experiment.aggregate(cells[:-1])
    wrong = copy.deepcopy(cells)
    wrong[0]["inherited_source_cost"]["native_steps"] += 1
    with pytest.raises(ValueError, match="source or budget"):
        experiment.aggregate(wrong)
    wrong = copy.deepcopy(cells)
    wrong[0]["groups"]["50"]["evaluation_seeds"][0] += 1
    with pytest.raises(ValueError, match="paired roster"):
        experiment.aggregate(wrong)


def test_replication_freezes_recipe_and_uses_disjoint_fresh_scenes():
    spec = experiment.spec
    assert spec.METHODS == spec.recipe.METHODS and spec.VARIANTS == spec.recipe.VARIANTS
    assert spec.FISHER_RADIUS == spec.recipe.FISHER_RADIUS and spec.DAMPING == spec.recipe.DAMPING
    assert spec.EPSILON == kernel.spec.EPSILON and spec.ACTION_DIM == kernel.spec.ACTION_DIM
    assert not set(spec.ROOTS) & set(spec.recipe.ROOTS)
    assert spec.budget()["native_episodes"] == 6304 and spec.budget()["native_steps"] == 7564800
    assert spec.budget()["label_episodes"] == 4896 and spec.budget()["evaluation_episodes"] == 896
    seen = set()
    for root in spec.ROOTS:
        labels = {s for r in spec.label_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        train = {s for r in spec.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        evaluation = set(spec.evaluation_seeds(root))
        old = {s for r in spec.source.label_roles(root)+spec.source.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(spec.source.evaluation_seeds(root))
        assert not labels & train and not (labels|train) & evaluation
        assert not (labels|train|evaluation) & (seen|old)
        seen.update(labels|train|evaluation)


def test_third_step_retains_geometry_and_names_the_actual_source_and_budget():
    spec = third_step
    assert spec.SOURCE_METHOD == "refresh" and spec.SOURCE_BASELINE == "two_step"
    assert "single" not in spec.VARIANTS and spec.source.PROTOCOL == kernel.spec.PROTOCOL
    assert spec.FISHER_RADIUS == spec.source.FISHER_RADIUS and spec.EPSILON == spec.source.EPSILON
    assert spec.budget()["native_episodes"] == 5856 and spec.budget()["native_steps"] == 7027200
    for root in spec.ROOTS:
        old = {s for r in spec.source.label_roles(root)+spec.source.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(spec.source.evaluation_seeds(root))
        fresh = {s for r in spec.label_roles(root)+spec.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(spec.evaluation_seeds(root))
        assert not fresh & old
