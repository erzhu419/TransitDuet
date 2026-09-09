#!/usr/bin/env python3
"""Strictly aggregate the frozen V30 expanded-observation discovery matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_freqduet_prefix_counterfactual import (
    PrefixMatrixContract,
    aggregate,
)
from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_EVAL_EPISODE,
    DISCOVERY_REPLAY_SEED,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPANDED_CONTEXT_COLUMNS,
    LABEL_PROTOCOL_VERSION,
    MATRIX_PROTOCOL_VERSION,
    OFFSETS_S,
    discovery_checkpoint_dir,
)


DISCOVERY_CONTRACT = PrefixMatrixContract(
    label_protocol_version=LABEL_PROTOCOL_VERSION,
    matrix_protocol_version=MATRIX_PROTOCOL_VERSION,
    config=CONFIG,
    train_seeds=tuple(DISCOVERY_TRAIN_SEEDS),
    eval_seeds=tuple(DISCOVERY_SCENARIO_SEEDS),
    decision_indices=tuple(DISCOVERY_DECISION_INDICES),
    offsets_s=tuple(OFFSETS_S),
    checkpoint_ep=CHECKPOINT_EP,
    eval_episode=DISCOVERY_EVAL_EPISODE,
    replay_seed=DISCOVERY_REPLAY_SEED,
    checkpoint_dir_for_seed=discovery_checkpoint_dir,
    required_context_columns=tuple(EXPANDED_CONTEXT_COLUMNS),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args()
    try:
        manifest = aggregate(
            args.jobs_root,
            args.out_dir,
            expected_commit=args.expected_source_commit,
            contract=DISCOVERY_CONTRACT,
        )
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "aggregate_invalid.json").write_text(json.dumps({
            "protocol_version": MATRIX_PROTOCOL_VERSION,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE V30 discovery aggregate status={manifest['status']} "
        f"jobs={manifest['jobs']} rows={manifest['rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
