import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from freq_hrl.experiments import pointmaze_compact_replication as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


def test_reduced_fresh_root_run_has_exact_budget_and_training_only_step_selection(source_data, tmp_path):
    models, predictor, cal, original_args, teachers = source_data
    args = copy.copy(original_args)
    args.optimizer_seed = 410037
    before = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    fit_seeds, original_fit = [], experiment.fit_method
    def record_fit(rows, method):
        fit_seeds.extend(r["zero"]["seed"] for r in rows)
        return original_fit(rows, method)
    with ExitStack() as stack:
        native_task(stack)
        for name in ("LABEL_SCENARIOS", "TRAINING_SCENARIOS", "EVALUATION_EPISODES"):
            stack.enter_context(patch.object(experiment.spec, name, 1))
        stack.enter_context(patch.object(experiment.spec, "arguments", return_value=args))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(experiment, "fit_method", side_effect=record_fit))
        result = experiment.run(410037, tmp_path / "result.json")
        assert result["cost"] == experiment.spec.budget()
        assert set(fit_seeds) == {r["scenario_seed"] for r in experiment.spec.training_roles(410037)}
    assert result["cost"]["native_episodes"] == 146 and result["cost"]["native_steps"] == 14600
    assert len(list(tmp_path.rglob("*.pt"))) == 4 and not list(tmp_path.rglob("*.npz"))
    for p, g in result["groups"].items():
        assert all("state" not in r and r["pairing_and_freeze"] == "passed" for r in g["native_labels"])
        assert g["effects"]["compact_minus_source_forecast"] == g["effects"]["compact_minus_compact_blinded"]
        assert g["learning"]["compact"]["geometry"]["state_dimensions"] == 26
        assert g["learning"]["raw"]["geometry"]["state_dimensions"] == 392
        assert g["lower_and_critics_frozen"] == "passed"
        experiment.joint.source.native.curves.support.assert_frozen(models[p], before[p])


def synthetic_cells(forecast_gains, raw_gain=.2):
    cells = []
    spec = experiment.spec
    for root, gain in zip(spec.ROOTS, forecast_gains):
        effects = {"compact_minus_source_forecast": {"mean": gain, "paired_differences": [gain] * spec.EVALUATION_EPISODES},
            "compact_minus_compact_blinded": {"mean": gain, "paired_differences": [gain] * spec.EVALUATION_EPISODES},
            "compact_minus_raw": {"mean": raw_gain, "paired_differences": [raw_gain] * spec.EVALUATION_EPISODES}}
        cells.append({"status": "complete", "protocol": spec.PROTOCOL, "root": root, "cost": spec.budget(),
            "groups": {str(p): {"effects": copy.deepcopy(effects), "evaluation_seeds": spec.evaluation_seeds(root),
                "training_roles": spec.training_roles(root), "lower_and_critics_frozen": "passed"} for p in spec.PERIODS}})
    return cells


def test_aggregation_clusters_by_root_and_keeps_positive_subthreshold_results():
    low = experiment.aggregate(synthetic_cells([.3] * 6))
    assert low["material_replication_gate"] == "not_closed"
    assert not any(low["period_material_replication_gate"].values())
    assert all(e["positive_gain_supported"] for e in low["endpoints"].values())
    assert experiment.aggregate(synthetic_cells([1.] * 6))["material_replication_gate"] == "supported_both_periods"
    varied = experiment.aggregate(synthetic_cells([-1.] * 3 + [1.] * 3))
    assert varied["statistics"]["n_independent"] == 6
    assert varied["statistics"]["bonferroni_family_size"] == 4
    for period in experiment.spec.PERIODS:
        endpoint = varied["endpoints"][f"{period}/compact_minus_source_forecast"]
        assert endpoint["root_means"] == [-1.] * 3 + [1.] * 3
        assert endpoint["ci"][1] - endpoint["ci"][0] > 1.
        assert endpoint["mean"] == 0.


def test_aggregation_rejects_missing_or_development_roots_and_changed_pair_rosters():
    cells = synthetic_cells([1.] * 6)
    with pytest.raises(ValueError, match="six additional roots"):
        experiment.aggregate(cells[:-1])
    wrong = copy.deepcopy(cells)
    wrong[0]["root"] = 410011
    with pytest.raises(ValueError, match="six additional roots"):
        experiment.aggregate(wrong)
    wrong = copy.deepcopy(cells)
    wrong[0]["groups"]["50"]["evaluation_seeds"][0] += 1
    with pytest.raises(ValueError, match="paired roster"):
        experiment.aggregate(wrong)


def test_frozen_recipe_budgets_and_disjoint_roots_and_seed_roles():
    spec = experiment.spec
    assert len(spec.ROOTS) == 6 and not set(spec.ROOTS) & set(spec.source.ROOTS)
    assert spec.budget()["native_episodes"] == 6176 and spec.budget()["native_steps"] == 7411200
    assert spec.budget()["label_episodes"] == 4896 and spec.budget()["evaluation_episodes"] == 768
    assert spec.EPSILON == experiment.source.source.spec.EPSILON and spec.ACTION_DIM == 8
    assert spec.FISHER_RADIUS == spec.source.FISHER_RADIUS and spec.DAMPING == spec.source.DAMPING
    assert spec.MINIMUM_GAIN == .5 and len(spec.ENDPOINTS) == 4
    seen = set()
    for root in spec.ROOTS:
        labels = {s for r in spec.label_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        train = {s for r in spec.training_roles(root) for s in (r["scenario_seed"], *r["noise_seeds"].values())}
        evaluation = set(spec.evaluation_seeds(root))
        assert len(labels) == 12 and len(train) == 48 and len(evaluation) == 64
        assert not labels & train and not (labels | train) & evaluation
        assert not (labels | train | evaluation) & seen
        for period in spec.PERIODS:
            assert len(spec.queries(root, period)) == (192 if period == 50 else 96)
        seen.update(labels | train | evaluation)
