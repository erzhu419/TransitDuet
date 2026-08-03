"""Source-level audit for Freq-HRL shared training-core boundaries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CORE_DIRS = (
    Path("transit_hrl/freq_hrl/core"),
    Path("transit_hrl/freq_hrl/encoders"),
    Path("transit_hrl/freq_hrl/rl"),
)

FORBIDDEN_CORE_IMPORT_PREFIXES = (
    "freq_hrl.domains",
    "freq_hrl.experiments",
    "freq_hrl.policies",
    "freq_transitduet",
    "FreqDuet",
)


@dataclass(frozen=True)
class AdapterEvidenceSpec:
    adapter: str
    path: Path
    required_symbol: str
    role: str


ADAPTER_EVIDENCE = (
    AdapterEvidenceSpec(
        adapter="trading_ppo",
        path=Path("transit_hrl/freq_hrl/experiments/trading/ppo_actor_critic.py"),
        required_symbol="train_frequency_separated_ppo",
        role="Trading Freq-HRL calls the asynchronous SMDP training loop.",
    ),
    AdapterEvidenceSpec(
        adapter="transit_surrogate_ppo",
        path=Path("transit_hrl/freq_hrl/experiments/transit/ppo_surrogate.py"),
        required_symbol="train_frequency_separated_ppo",
        role="Transit surrogate must migrate to the asynchronous SMDP loop.",
    ),
    AdapterEvidenceSpec(
        adapter="transit_native_replay_update",
        path=Path("transit_hrl/freq_hrl/experiments/transit/native_shared_ppo.py"),
        required_symbol="apply_smdp_updates",
        role="Native Transit must update separate upper and lower SMDP trajectories.",
    ),
    AdapterEvidenceSpec(
        adapter="transit_native_actor_core",
        path=Path("transit_hrl/freq_hrl/experiments/transit/native_shared_ppo.py"),
        required_symbol="FrequencySeparatedActorCriticPPO",
        role="Native Transit bridge must instantiate the v2 frequency-separated actor-critic.",
    ),
)


def _parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_files(source_root: Path, rel_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for rel_dir in rel_dirs:
        root = source_root / rel_dir
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def audit_core_import_boundaries(source_root: Path = Path(".")) -> dict[str, Any]:
    """Check that core/encoder/RL modules do not import domain code."""
    violations: list[dict[str, Any]] = []
    checked = 0
    for path in _python_files(source_root, CORE_DIRS):
        checked += 1
        tree = _parse_python(path)
        rel = path.relative_to(source_root)
        for module in _imported_modules(tree):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_CORE_IMPORT_PREFIXES):
                violations.append({
                    "path": str(rel),
                    "module": module,
                })
    return {
        "status": "supported" if not violations else "failed",
        "checked_files": int(checked),
        "violations": violations,
    }


def audit_adapter_shared_entries(source_root: Path = Path(".")) -> list[dict[str, Any]]:
    """Check that domain adapters call or instantiate the registered shared symbols."""
    rows: list[dict[str, Any]] = []
    for spec in ADAPTER_EVIDENCE:
        path = source_root / spec.path
        status = "missing"
        evidence = "file is missing"
        if path.exists():
            tree = _parse_python(path)
            calls = _called_names(tree)
            imports = _imported_names(tree)
            has_symbol = spec.required_symbol in calls and spec.required_symbol in imports
            status = "supported" if has_symbol else "failed"
            evidence = (
                f"`{spec.required_symbol}` is imported and called"
                if has_symbol else f"`{spec.required_symbol}` is not both imported and called"
            )
        rows.append({
            "adapter": spec.adapter,
            "status": status,
            "path": str(spec.path),
            "required_symbol": spec.required_symbol,
            "role": spec.role,
            "evidence": evidence,
        })
    return rows


def audit_shared_training_core(source_root: Path = Path(".")) -> dict[str, Any]:
    """Return the reviewer-facing shared-core source audit."""
    boundary = audit_core_import_boundaries(source_root)
    adapters = audit_adapter_shared_entries(source_root)
    adapter_status = all(row["status"] == "supported" for row in adapters)
    status = (
        "supported"
        if boundary["status"] == "supported" and adapter_status
        else "partial"
        if boundary["status"] == "supported" and any(row["status"] == "supported" for row in adapters)
        else "failed"
    )
    return {
        "status": status,
        "core_boundary": boundary,
        "adapter_evidence": adapters,
        "boundary_statement": (
            "Core/encoder/RL modules stay domain-agnostic; Quant and Transit "
            "adapters must collect separate upper/lower trajectories and delegate "
            "learning to FrequencySeparatedActorCriticPPO v2. Legacy joint-PPO "
            "entries do not satisfy this audit."
        ),
    }
