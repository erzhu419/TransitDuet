"""Deterministic seed-role utilities for confirmatory Freq-HRL experiments."""

from __future__ import annotations

from hashlib import blake2b, sha256
from pathlib import Path
import subprocess
from typing import Iterable


MAX_NUMPY_SEED = 2**32 - 1
SOURCE_MANIFEST_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".toml"})


def _source_manifest_digest(entries: Iterable[tuple[str, bytes]]) -> str:
    digest = sha256()
    count = 0
    for relative_text, content in entries:
        relative = str(relative_text).encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
        count += 1
    if count == 0:
        raise ValueError("source manifest contains no registered source files")
    return digest.hexdigest()


def source_manifest_sha256(source_root: Path) -> str:
    """Hash registered source/config paths and bytes without filesystem metadata."""

    root = Path(source_root).resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.suffix.lower() in SOURCE_MANIFEST_SUFFIXES
        and "__pycache__" not in path.parts
        and path.is_file()
    )
    if not files:
        raise ValueError(f"source manifest contains no registered source files: {root}")
    return _source_manifest_digest(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in files
    )


def git_source_manifest_sha256(
    repository_root: Path,
    source_relative_path: Path,
    *,
    revision: str,
) -> str:
    """Hash the registered source files stored in a Git commit."""

    repository = Path(repository_root).resolve()
    git_root = Path(subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()).resolve()
    source_absolute = (repository / source_relative_path).resolve()
    try:
        source_prefix = source_absolute.relative_to(git_root).as_posix()
    except ValueError as exc:
        raise ValueError("source root must be inside the Git worktree") from exc
    listing = subprocess.run(
        [
            "git", "-C", str(git_root), "ls-tree", "-r", "--name-only", "-z",
            str(revision), "--", source_prefix,
        ],
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted(
        item.decode("utf-8")
        for item in listing.split(b"\0")
        if item
        and Path(item.decode("utf-8")).suffix.lower() in SOURCE_MANIFEST_SUFFIXES
        and "__pycache__" not in Path(item.decode("utf-8")).parts
    )
    prefix = source_prefix + "/"
    entries = []
    for repository_path in paths:
        if not repository_path.startswith(prefix):
            continue
        content = subprocess.run(
            ["git", "-C", str(git_root), "show", f"{revision}:{repository_path}"],
            check=True,
            capture_output=True,
        ).stdout
        entries.append((repository_path[len(prefix):], content))
    return _source_manifest_digest(entries)


def registered_git_source_identity(
    repository_root: Path,
    source_relative_path: Path,
) -> tuple[str, str]:
    """Return HEAD identity only when staged source bytes equal that commit."""

    repository = Path(repository_root).resolve()
    source_relative = Path(source_relative_path)
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    working_manifest = source_manifest_sha256(repository / source_relative)
    committed_manifest = git_source_manifest_sha256(
        repository,
        source_relative,
        revision=revision,
    )
    if working_manifest != committed_manifest:
        raise RuntimeError(
            "working source manifest does not match the registered Git revision"
        )
    return revision, working_manifest


def current_freq_hrl_source_manifest_sha256() -> str:
    return source_manifest_sha256(Path(__file__).resolve().parents[1])


def is_hex_digest(value: object, *, length: int) -> bool:
    text = str(value).strip().lower()
    return len(text) == int(length) and all(
        character in "0123456789abcdef" for character in text
    )


def verify_current_freq_hrl_source_identity(
    *,
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
    require_verified: bool = False,
) -> dict[str, str]:
    """Verify that the executing package matches its registered source bytes."""

    revision = str(code_revision).strip().lower()
    expected_manifest = str(expected_source_manifest_sha256).strip().lower()
    if revision and not is_hex_digest(revision, length=40):
        raise ValueError("source identity requires a full Git revision")
    if expected_manifest and not is_hex_digest(expected_manifest, length=64):
        raise ValueError("source identity requires a source manifest SHA-256")
    actual_manifest = current_freq_hrl_source_manifest_sha256()
    if expected_manifest and expected_manifest != actual_manifest:
        raise RuntimeError(
            "staged source manifest mismatch: expected "
            f"{expected_manifest}, got {actual_manifest}"
        )
    status = (
        "verified"
        if is_hex_digest(revision, length=40)
        and expected_manifest == actual_manifest
        else "unregistered_local"
    )
    if require_verified and status != "verified":
        raise ValueError("confirmatory runs require verified source identity")
    return {
        "code_revision": revision,
        "source_manifest_sha256": actual_manifest,
        "source_identity_status": status,
    }


def derive_seed(namespace: str, *parts: object) -> int:
    """Derive a stable NumPy/PyTorch-compatible seed from structured inputs."""

    payload = "\x1f".join([str(namespace), *(str(part) for part in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int(int.from_bytes(digest, byteorder="big") % MAX_NUMPY_SEED)


def training_rollout_seed(
    training_replicate_seed: int,
    rollout_seed_root: int,
    iteration: int,
    *,
    domain: str,
) -> int:
    """Return one fresh, paired training path for a replicate and iteration."""

    return derive_seed(
        "freq_hrl_training_rollout_v2",
        str(domain),
        int(training_replicate_seed),
        int(rollout_seed_root),
        int(iteration),
    )


def validate_unique_seeds(values: Iterable[int], *, role: str) -> list[int]:
    seeds = [int(value) for value in values]
    if not seeds:
        raise ValueError(f"{role} must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{role} must contain unique seeds")
    return seeds


def validate_evaluation_seed_roles(
    validation_seeds: Iterable[int],
    heldout_test_seeds: Iterable[int],
) -> tuple[list[int], list[int]]:
    """Validate that checkpoint selection and final testing use disjoint paths."""

    validation = validate_unique_seeds(validation_seeds, role="validation_seeds")
    heldout = validate_unique_seeds(heldout_test_seeds, role="eval_seeds")
    overlap = sorted(set(validation) & set(heldout))
    if overlap:
        raise ValueError(
            "validation_seeds and eval_seeds must be disjoint; overlap="
            + ",".join(str(seed) for seed in overlap)
        )
    return validation, heldout
