"""Deterministic seed-role utilities for confirmatory Freq-HRL experiments."""

from __future__ import annotations

from hashlib import blake2b
from typing import Iterable


MAX_NUMPY_SEED = 2**32 - 1


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
