"""Equal-shape PointMaze attribution of frequency routing inside HRL."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .pointmaze_multiscale_validation import _json_ready
from .pointmaze_routing_attribution import (
    POINTMAZE_ROUTING_METHODS,
    POINTMAZE_ROUTING_SCENARIOS,
    build_parser,
    build_pointmaze_routing_model,
    resolved_pointmaze_routing_protocol,
    train_pointmaze_routing_cell,
)


POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION = (
    "pointmaze_frequency_routing_stage4_v2"
)
POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH = (
    "pointmaze_frequency_routing_masked_attribution"
)
POINTMAZE_MASKED_ROUTING_METHODS = POINTMAZE_ROUTING_METHODS
POINTMAZE_MASKED_ROUTING_SCENARIOS = POINTMAZE_ROUTING_SCENARIOS
POINTMAZE_MASKED_ROUTING_REPRESENTATIONS = {
    "hrl_history": "history",
    "hrl_causal_filter": "filtered",
    "hrl_multiscale_all": "multiscale_all",
    "hrl_multiscale_routed": "multiscale_routed_masked",
    "hrl_multiscale_swapped": "multiscale_swapped_masked",
}
POINTMAZE_MASKED_ROUTING_CONTRACT = {
    "history": "raw_history_to_both_levels",
    "filtered": "causal_filtered_history_to_both_levels",
    "multiscale_all": "all_haar_bands_to_both_levels",
    "multiscale_routed_masked": (
        "fixed_shape_slow_mid_upper_and_mid_high_lower_masks"
    ),
    "multiscale_swapped_masked": (
        "fixed_shape_mid_high_upper_and_slow_mid_lower_masks"
    ),
}
POINTMAZE_MASKED_ROUTING_SHAPE_CONTRACT = (
    "identical_upper_lower_state_shapes_parameters_and_initialization_per_root"
)


def build_pointmaze_masked_routing_model(**kwargs):
    return build_pointmaze_routing_model(
        **kwargs,
        method_representations=POINTMAZE_MASKED_ROUTING_REPRESENTATIONS,
    )


def train_pointmaze_masked_routing_cell(
    **kwargs,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    payload, rows, model = train_pointmaze_routing_cell(
        **kwargs,
        method_representations=POINTMAZE_MASKED_ROUTING_REPRESENTATIONS,
        protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
        algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
        routing_contract=POINTMAZE_MASKED_ROUTING_CONTRACT,
        domain="pointmaze_frequency_routing_masked_attribution",
    )
    payload["routing_shape_contract"] = (
        POINTMAZE_MASKED_ROUTING_SHAPE_CONTRACT
    )
    return payload, rows, model


def resolved_pointmaze_masked_routing_protocol(**kwargs) -> dict[str, Any]:
    protocol = resolved_pointmaze_routing_protocol(
        **kwargs,
        method_representations=POINTMAZE_MASKED_ROUTING_REPRESENTATIONS,
        protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
        algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
    )
    protocol["routing_shape_contract"] = (
        POINTMAZE_MASKED_ROUTING_SHAPE_CONTRACT
    )
    return protocol


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    parser.description = (
        "Run equal-shape PointMaze Stage-4 V2 routing attribution."
    )
    args = parser.parse_args(None if argv is None else list(argv))
    protocol = resolved_pointmaze_masked_routing_protocol(
        methods=args.methods,
        scenarios=args.scenarios,
        env_id=args.env_id,
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=args.eval_seeds,
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        for scenario in args.scenarios:
            for method in args.methods:
                payload, _, _ = train_pointmaze_masked_routing_cell(
                    method=method,
                    scenario=scenario,
                    env_id=args.env_id,
                    train_seeds=args.train_seeds,
                    selection_seeds=args.selection_seeds,
                    eval_seeds=args.eval_seeds,
                    iterations=args.iterations,
                    horizon=args.horizon,
                    optimizer_seed=args.optimizer_seed,
                    upper_period_seconds=args.upper_period_seconds,
                    history_seconds=args.history_seconds,
                    fast_period_seconds=args.fast_period_seconds,
                    maximum_subgoal_delta=args.maximum_subgoal_delta,
                    reference_hidden_dim=args.reference_hidden_dim,
                    learning_rate=args.learning_rate,
                    checkpoint_evaluation_interval=(
                        args.checkpoint_evaluation_interval
                    ),
                )
                output["cells"].append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
