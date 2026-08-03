"""Named random streams for reproducible FreqDuet experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import random

import numpy as np
import torch


_SUPPORTED_MODES = {"global_legacy", "isolated_streams_v4"}


def _derive_seed(base_seed: int, namespace: str) -> int:
    payload = f"freqduet-v4:{int(base_seed)}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class RandomnessContract:
    """Derive stable, named component seeds from one experiment seed.

    ``global_legacy`` is retained only to reproduce historical configurations.
    ``isolated_streams_v4`` prevents an ablation in one component from changing
    another component's initialization, replay sampling, or exploration stream.
    """

    base_seed: int
    mode: str = "global_legacy"

    def __post_init__(self):
        mode = str(self.mode).strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                "randomness.mode must be global_legacy or isolated_streams_v4")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "base_seed", int(self.base_seed))

    @property
    def isolated(self) -> bool:
        return self.mode == "isolated_streams_v4"

    def seed(self, namespace: str) -> int:
        if not namespace:
            raise ValueError("random stream namespace must be non-empty")
        if not self.isolated:
            return int(self.base_seed)
        return _derive_seed(self.base_seed, str(namespace))

    def python(self, namespace: str):
        if not self.isolated:
            return random
        return random.Random(self.seed(namespace))

    def numpy(self, namespace: str):
        if not self.isolated:
            return np.random
        return np.random.RandomState(self.seed(namespace))

    def torch_generator(self, namespace: str, device="cpu"):
        if not self.isolated:
            return None
        generator = torch.Generator(device=torch.device(device).type)
        generator.manual_seed(self.seed(namespace))
        return generator

    @contextmanager
    def torch_initialization(self, namespace: str):
        """Seed module construction without consuming the global CPU stream."""
        if not self.isolated:
            yield
            return
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed(namespace))
            yield
        finally:
            torch.random.set_rng_state(state)

    def manifest(self, namespaces):
        names = sorted({str(name) for name in namespaces})
        seeds = {name: self.seed(name) for name in names}
        payload = {
            "contract": self.mode,
            "base_seed": self.base_seed,
            "component_seeds": seeds,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload
