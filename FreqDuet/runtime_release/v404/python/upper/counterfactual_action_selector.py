"""Pure-Python runtime scorer for counterfactual fixed-action selectors.

Offline scripts may use scikit-learn to fit a shallow decision tree from
matched counterfactual labels. The simulator runtime only needs this small
JSON-backed tree walker, so HPC training environments do not need sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


ACTION_SPECS = {
    "target_m20": {"delta_s": -20.0, "terminal_dispatch": False},
    "target0": {"delta_s": 0.0, "terminal_dispatch": False},
    "target_p20": {"delta_s": 20.0, "terminal_dispatch": False},
    "term45_m20": {"delta_s": -20.0, "terminal_dispatch": True},
    "term45_0": {"delta_s": 0.0, "terminal_dispatch": True},
    "term45_p20": {"delta_s": 20.0, "terminal_dispatch": True},
}


@dataclass(frozen=True)
class CounterfactualActionPrediction:
    method: str
    delta_s: float
    terminal_dispatch: bool
    confidence: float
    node_id: int


class CounterfactualActionTreeSelector:
    """Load and score an exported sklearn DecisionTreeClassifier artifact."""

    def __init__(self, payload: Mapping[str, object]):
        self.payload = dict(payload)
        self.feature_cols = [
            str(x) for x in self.payload.get("feature_cols", [])
        ]
        self.feature_medians = {
            str(k): float(v)
            for k, v in dict(self.payload.get("feature_medians", {})).items()
        }
        self.classes = [str(x) for x in self.payload.get("classes", [])]
        self.nodes = list(self.payload.get("nodes", []))
        if not self.feature_cols:
            raise ValueError("counterfactual action selector has no features")
        if not self.classes:
            raise ValueError("counterfactual action selector has no classes")
        if not self.nodes:
            raise ValueError("counterfactual action selector has no tree nodes")

    @classmethod
    def load(cls, path: str | Path) -> "CounterfactualActionTreeSelector":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def vectorize(self, values: Mapping[str, object]) -> np.ndarray:
        xs = []
        for name in self.feature_cols:
            raw = values.get(name, self.feature_medians.get(name, 0.0))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = self.feature_medians.get(name, 0.0)
            if not np.isfinite(value):
                value = self.feature_medians.get(name, 0.0)
            xs.append(value)
        return np.asarray(xs, dtype=np.float64)

    def predict(self, values: Mapping[str, object]) -> CounterfactualActionPrediction:
        x = self.vectorize(values)
        node_id = 0
        while True:
            node = self.nodes[int(node_id)]
            if bool(node.get("leaf", False)):
                probs = np.asarray(node.get("proba", []), dtype=np.float64)
                if probs.size != len(self.classes):
                    probs = np.ones(len(self.classes), dtype=np.float64)
                if probs.sum() <= 0.0:
                    probs = np.ones(len(self.classes), dtype=np.float64)
                probs = probs / probs.sum()
                class_idx = int(np.argmax(probs))
                method = self.classes[class_idx]
                spec = ACTION_SPECS.get(method, ACTION_SPECS["target0"])
                return CounterfactualActionPrediction(
                    method=method,
                    delta_s=float(spec["delta_s"]),
                    terminal_dispatch=bool(spec["terminal_dispatch"]),
                    confidence=float(probs[class_idx]),
                    node_id=int(node_id),
                )
            feature_idx = int(node["feature"])
            threshold = float(node["threshold"])
            if x[feature_idx] <= threshold:
                node_id = int(node["left"])
            else:
                node_id = int(node["right"])
