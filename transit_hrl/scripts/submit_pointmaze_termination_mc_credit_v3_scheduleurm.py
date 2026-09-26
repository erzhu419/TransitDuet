#!/usr/bin/env python3
"""Submit the two-root full-return termination-credit development screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_termination_mc_credit_v3_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    LINUX_CPU_NODES,
    SCHEDULER,
    execute_bulk,
)
from scripts.submit_pointmaze_learned_termination_stage10_scheduleurm import (  # noqa: E402
    cell_relative_dir,
    task_specification as base_task_specification,
)
from scripts.submit_pointmaze_multiscale_stage3_scheduleurm import (  # noqa: E402
    _inventory_by_signature,
)


def task_signature(run_name: str, root: int) -> str:
    return f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.POLICY}/{root}"


def task_specification(
    run_name: str, root: int, *, preflight: bool
) -> dict[str, object]:
    spec.cell_options(root, preflight=preflight)
    task = base_task_specification(
        run_name, root, preflight=preflight,
        stochastic_repetitions=spec.STOCHASTIC_REPETITIONS,
        termination_gae_lambda=spec.GAE_LAMBDA,
    )
    task.update({
        "project": spec.EXPERIMENT_PROTOCOL,
        "description": f"Freq-HRL full-return termination credit root{root}",
        "signature": task_signature(run_name, root),
        "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/cell",
    })
    return task


def inventory(run_name: str) -> list[dict[str, object]]:
    process = subprocess.run(
        [
            sys.executable, str(SCHEDULER), "results", "--signature",
            f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/*",
            "--status", "queued", "launching", "running", "done", "failed",
            "cancelled", "--limit", "0", "--include-empty", "--no-log-scan",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(process.stdout[process.stdout.index("{"):])
    return list(payload.get("results", []))


def sync_results(run_name: str, *, preflight: bool, workers: int) -> None:
    tasks = _inventory_by_signature(inventory(run_name))
    expected = []
    for root in spec.roots(preflight=preflight):
        signature = task_signature(run_name, root)
        task = tasks.get(signature)
        if task is None or task.get("status") != "done" or not task.get("node"):
            raise SystemExit(f"MC-credit task is not done: {signature}")
        expected.append((signature, ROOT / cell_relative_dir(run_name, root), task))
    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    def sync_one(item: tuple[str, Path, dict[str, object]]) -> tuple[str, bool, str]:
        signature, path, task = item
        if (path / "result.json").is_file():
            return signature, True, "already present"
        ok, message = scheduler_runtime._sync_one_result({
            "node": task["node"],
            "result_dir": str(path),
            "local_result_dir": str(path),
        })
        return signature, bool(ok), str(message)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(sync_one, expected))
    failures = [item for item in results if not item[1]]
    if failures:
        raise SystemExit(f"compact MC-credit result sync failed: {failures}")
    for signature, path, _ in expected:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if (
            result.get("status") != "complete"
            or result.get("protocol", {}).get("protocol_version")
            != spec.EXPERIMENT_PROTOCOL
            or len(result.get("cells", [])) != 1
        ):
            raise SystemExit(f"invalid MC-credit result: {signature}")
    print(f"synced {len(expected)} compact MC-credit result JSON files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results", action="store_true")
    parser.add_argument("--sync-workers", type=int, default=2)
    args = parser.parse_args()
    if args.sync_results:
        sync_results(args.run_name, preflight=args.preflight, workers=args.sync_workers)
        return 0
    subprocess.run(
        ["git", "diff", "--exit-code", spec.ALGORITHM_REVISION, "--",
         "freq_hrl", "scripts/run_pointmaze_learned_termination_stage10.py"],
        cwd=ROOT,
        check=True,
    )
    if inventory(args.run_name):
        raise SystemExit("MC-credit run already registered")
    run_dir = ROOT / "results" / args.run_name
    if any(run_dir.glob("cells/**/result.json")):
        raise SystemExit("MC-credit run already contains completed cells")
    run_dir.mkdir(parents=True, exist_ok=True)
    roots = spec.roots(preflight=args.preflight)
    registration = {
        "protocol": spec.EXPERIMENT_PROTOCOL,
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "preflight": bool(args.preflight),
        "optimizer_roots": list(roots),
        "termination_gae_lambda": spec.GAE_LAMBDA,
        "stochastic_repetitions": spec.STOCHASTIC_REPETITIONS,
        "evidence_role": "post_failure_development_only",
        "options": {
            str(root): spec.cell_options(root, preflight=args.preflight)
            for root in roots
        },
        "scheduler": {
            "nodes": list(LINUX_CPU_NODES), "require_node": None,
            "cpu_per_cell": 1, "ram_mb_per_cell": 1536,
        },
        "artifacts": {"synced": ["result.json"], "checkpoints": "disabled"},
    }
    (run_dir / "preregistration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execute_bulk(
        [task_specification(args.run_name, root, preflight=args.preflight)
         for root in roots],
        dry_run=args.dry_run,
        intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}",
    )
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
