"""Submit, monitor, and read out Route A (Unprivileged Queries + Multi-Budget Pareto) via scheduleurm."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    execute_bulk,
)

sys.path.insert(0, str(SCHEDULER.parent))
import scheduler  # noqa: E402


REMOTE = Path("/home/zhengliang01/scheduleurm_work/TransitDuet_freqhrl_v3/transit_hrl")
PLANS = (
    (
        Path("protocols/plan_validity_stage9_unprivileged_queries_2026-11-18.json"),
        "scripts/run_stage9_unprivileged_queries.py",
    ),
    (
        Path("protocols/plan_validity_stage9_budget_pareto_2026-11-18.json"),
        "scripts/run_stage9_budget_pareto.py",
    ),
)


def tasks_with(signatures: set[str], state: dict | None = None) -> list[dict]:
    tasks = (state if state is not None else scheduler.load_state())["tasks"]
    items = tasks.values() if isinstance(tasks, dict) else tasks
    # The scheduler appends registrations; a manual retry can supersede an old task.
    return list({t["signature"]: t for t in items if t.get("signature") in signatures}.values())


def submit_jobs(jobs: list[dict], label: str) -> tuple[list[str], set[str]]:
    by_signature = {j["signature"]: j for j in jobs}
    signatures = set(by_signature)
    existing = {t["signature"] for t in tasks_with(signatures)}
    missing = [j for sig, j in by_signature.items() if sig not in existing]
    if missing:
        execute_bulk(missing, dry_run=False, intent_label=label)
    with scheduler.state_lock():
        state = scheduler.load_state()
        registered = {t["signature"]: t for t in tasks_with(signatures, state)}
        if registered.keys() != signatures:
            raise RuntimeError(f"scheduler registered {len(registered)}/{len(jobs)} jobs for {label}")
        tasks = [registered[sig] for sig in by_signature]
        pending = [task for task in tasks if task.get("status") == "queued"]
        for task in pending:
            task.update(allow_cpu_training=True, protect_from_auto_eviction=True)
        if pending:
            scheduler.save_state(state)
    ids = [t["id"] for t in tasks]
    print(f"Registered {label}: {ids}", flush=True)
    if pending:
        subprocess.run(
            [
                sys.executable,
                str(SCHEDULER),
                "dispatch",
                *(arg for task in pending for arg in ("--task-id", task["id"])),
                "--intent-label",
                label,
            ],
            check=True,
        )
    return ids, signatures


def remote_python(args: list[str], timeout: int = 120) -> str:
    shell = (
        "cd "
        + shlex.quote(str(REMOTE))
        + " && env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        + shlex.join([DEFAULT_LINUX_PYTHON, *args])
    )
    for attempt in range(3):
        rc, out, err = scheduler.run_on("node006", shell, timeout=timeout, check=False)
        if rc == 0:
            return out
        if rc != 255 or attempt == 2:
            raise RuntimeError(f"remote command failed (rc={rc}): {err[-2000:]}")
        time.sleep(15)
    raise RuntimeError("unreachable")


def wait_for_study(plan: dict, signatures: set[str], *, hours: float = 2.0) -> list[dict]:
    rel_paths = [f"{c['public']}/result.json" for c in plan["cells"]]
    probe = (
        "import json\n"
        "from pathlib import Path\n"
        f"paths = {rel_paths!r}\n"
        "rows = [json.loads(Path(p).read_text()) if Path(p).is_file() else None for p in paths]\n"
        "print(json.dumps({'statuses': [r['status'] if r else None for r in rows], "
        "'done': all(rows)}, separators=(',', ':')))\n"
    )
    cutoff = time.monotonic() + 3600.0 * hours
    previous = None
    next_queue_check = 0.0
    while time.monotonic() < cutoff:
        out = remote_python(["-c", probe], timeout=60)
        status_obj = json.loads(out)
        if status_obj["statuses"] != previous:
            print(plan["run_name"], "statuses:", status_obj["statuses"], flush=True)
            previous = status_obj["statuses"]
        if status_obj["done"]:
            if any(s != "complete" for s in status_obj["statuses"]):
                err_probe = (
                    "import json\nfrom pathlib import Path\n"
                    f"paths = {rel_paths!r}\n"
                    "print(json.dumps([json.loads(Path(p).read_text()) for p in paths]))\n"
                )
                raise RuntimeError(
                    f"{plan['run_name']} had failed cells: {remote_python(['-c', err_probe])}"
                )
            return status_obj["statuses"]
        if time.monotonic() >= next_queue_check:
            next_queue_check = time.monotonic() + 300.0
            state_tasks = tasks_with(signatures)
            if len(state_tasks) == len(rel_paths) and all(
                t.get("status") in {"done", "failed", "cancelled", "canceled"}
                for t in state_tasks
            ):
                # Re-check once after terminal state in case file just landed on NFS
                out2 = json.loads(remote_python(["-c", probe], timeout=60))
                if out2["done"] and all(s == "complete" for s in out2["statuses"]):
                    return out2["statuses"]
                raise RuntimeError(
                    f"{plan['run_name']} tasks terminated without complete reports: "
                    + str([(t["id"], t.get("status"), t.get("exit_code")) for t in state_tasks])
                )
        time.sleep(30)
    raise TimeoutError(f"{plan['run_name']} timed out after {hours}h")


def run_study(plan_rel: Path, runner_rel: str) -> None:
    plan = json.loads((ROOT / plan_rel).read_text())
    run_name = plan["run_name"]
    local_dir = ROOT / "results" / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    summary_file = local_dir / "summary.json"
    if summary_file.is_file():
        print(f"=== {run_name} already complete locally; skipping ===", flush=True)
        return

    ids, sigs = submit_jobs(plan["jobs"], run_name)
    release_file = local_dir / "release.json"
    if not release_file.exists():
        release_file.write_text(
            json.dumps(
                {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(plan_rel),
                    "task_ids": ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    statuses = wait_for_study(plan, sigs, hours=2.0)
    headline = remote_python([runner_rel, "readout"], timeout=180)
    print(f"=== {run_name} READOUT ===\n{headline}", flush=True)

    full_summary = remote_python(
        [
            "-c",
            f"from pathlib import Path; print(Path('results/{run_name}/summary.json').read_text(), end='')",
        ],
        timeout=180,
    )
    summary_file.write_text(full_summary, encoding="utf-8")
    (local_dir / "completion.json").write_text(
        json.dumps(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "task_ids": ids,
                "statuses": statuses,
                "summary_bytes": len(full_summary.encode("utf-8")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    for plan_rel, runner_rel in PLANS:
        run_study(plan_rel, runner_rel)
    print("ALL ROUTE A EXPERIMENTS COMPLETED SUCCESSFULLY.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
