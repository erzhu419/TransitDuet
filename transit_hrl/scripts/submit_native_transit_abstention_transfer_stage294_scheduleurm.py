#!/usr/bin/env python3
"""Schedule prospective frozen-rule transfer on the unpinned CPU pool."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_abstention_transfer_stage294 as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, phase, key=None, probe=None, bank=None, index=None, chunk=None):
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_abstention_transfer_stage294.py",
        "--run-name", run, "--phase", phase]
    suffix = phase
    if key is not None:
        command += ["--role", key[0], "--root", str(key[1]), "--scenario", key[2], "--replica", str(key[3])]
        suffix += "/"+"/".join(map(str, key))
    if probe is not None:
        command += ["--probe", str(probe), "--bank", bank]
        suffix += f"/probe{probe}/{bank}"
    if index is not None:
        command += ["--state-index", str(index), "--chunk", str(chunk)]
        suffix += f"/state{index}/chunk{chunk}"
    if phase in ("pilot", "capture", "menus", "evaluate"):
        output = spec.task_receipt_dir(run, phase, key, probe, bank, index, chunk)
    else:
        output = spec.directory(run)/phase
    # The result root itself is never staged: it contains server-only snapshots.
    return {"project": spec.EXPERIMENT_PROTOCOL, "description": "Freq-HRL new-prefix frozen abstention "+suffix,
        "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}",
        "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{phase}", "cwd": str(ROOT),
        "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        "cpu": 1, "ram_mb": 1536 if phase in ("pilot", "capture", "menus", "evaluate") else 3072,
        "vram": 0, "priority": "normal", "require_node": None,
        "allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "allow_cpu_training": True,
        "cpu_training_justification": "Frozen raw/247 controls and finite-MC rule on genuinely new prefixes; no fits or actor updates.",
        "reroute_on_node_down": True, "node_down_requeue_s": 300,
        "result_dir": str(output), "local_result_dir": str(output),
        "stage_excludes": ["results", "**/__pycache__"],
        "stage_input_paths": [str(ROOT/"scripts"), str(ROOT/"freq_hrl"), str(spec.directory(run)/"inputs")]}


def tasks(run, phase):
    if phase in ("pilot", "capture"):
        return [task_specification(run, phase, k) for k in spec.keys("pilot" if phase == "pilot" else "new")]
    if phase == "menus":
        return [task_specification(run, phase, k, p, b) for k in spec.keys() for p in spec.PROBES for b in ("proposal", "screen")]
    if phase == "evaluate":
        plan = spec.read(spec.directory(run)/"seal/result.json")
        if plan["contract"] != spec.contract() or not plan["software_qualified"]:
            raise ValueError("Changed frozen evaluation plan")
        return [task_specification(run, phase, index=i, chunk=c) for i in plan["active_indices"] for c in range(4)]
    return [task_specification(run, phase)]


def require_completed(existing, required):
    latest = {t["signature"]: t for t in existing}
    for task in required:
        prior = latest.get(task["signature"])
        if prior is None or prior["status"] != "done" or prior.get("exit_code") != 0:
            raise ValueError("Preceding phase is not completely done/exit0: "+task["signature"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=spec.PHASES, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(SCHEDULER.parent))
    import scheduler
    prefix = f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{a.run_name}/"
    existing = [t for t in scheduler.load_state()["tasks"] if str(t.get("signature", "")).startswith(prefix)]
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if a.phase == "freeze":
        if existing or registration.exists():
            p.error("Stage294 already registered")
        parent = spec.read(spec.directory("native_transit_abstention_attribution_stage293_20261018_r1")/"completion_receipt.json")
        if (parent["scheduler"]["status_counts"] != {"done": 1}
                or parent["scheduler"]["exit_code_counts"] != {"0": 1} or not parent["frozen_statistics_replayed"]):
            p.error("Complete Stage293 first")
        write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "require_node": None,
                "cpu_per_task": 1, "native_ram_mb": 1536, "offline_ram_mb": 3072}})
    else:
        if spec.read(registration)["contract"] != spec.contract():
            p.error("Changed prospective transfer registration")
        preceding = dict(zip(spec.PHASES[1:], spec.PHASES[:-1]))[a.phase]
        required = tasks(a.run_name, preceding)
        signatures = {t["signature"] for t in required}
        if signatures-{t["signature"] for t in existing}:
            existing = [t for t in scheduler.load_archive_tasks() if str(t.get("signature", "")).startswith(prefix)]+existing
        require_completed(existing, required)
        if a.phase in ("capture", "menus", "seal", "evaluate", "report") and not spec.read(spec.directory(a.run_name)/"pilot-report/result.json")["software_qualified"]:
            p.error("Native software preflight must pass first")
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in existing}&{t["signature"] for t in batch}:
        p.error("Phase already dispatched")
    execute_bulk(batch, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}:{a.phase}")


if __name__ == "__main__":
    main()
