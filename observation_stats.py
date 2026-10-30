"""Transparent baseline arithmetic, without fitted ML or inferred fault semantics."""
from __future__ import annotations

import numpy as np


class MetricStatistics:
    uses_iforest = False

    def __init__(self):
        self.assets = {}
        self.n_trained = 0

    def fit(self, history: dict) -> None:
        for asset_id, row in history.items():
            values = np.asarray(row["X"], dtype=float)
            if values.ndim != 2 or len(values) < 20 or not np.isfinite(values).all():
                raise ValueError("Baseline requires at least 20 finite readings per asset.")
            median = np.median(values, axis=0)
            mad = np.median(np.abs(values - median), axis=0)
            self.assets[asset_id] = {
                "features": row["features"], "median": median,
                "scale": np.where(mad < 1e-6, 1.0, 1.4826 * mad),
            }
        self.n_trained = len(self.assets)

    def score(self, aid: str, metrics: dict) -> dict:
        state = self.assets[aid]
        values = np.array([metrics[name] for name in state["features"]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Readings must be finite.")
        z = (values - state["median"]) / state["scale"]
        top = int(np.argmax(np.abs(z)))
        maximum = float(abs(z[top]))
        return {
            "score": round(min(1.0, maximum / 6), 3), "is_anomaly": maximum >= 3,
            "zmax": round(maximum, 2), "top_metric": state["features"][top],
            "per_metric_z": {name: round(float(value), 2) for name, value in zip(state["features"], z)},
            "iforest": None,
        }


class CodeMembership:
    def __init__(self):
        self.vocab = []
        self.V = 0
        self.n_transitions = 0

    def fit(self, sequences: list[list[str]]) -> None:
        self.vocab = sorted({code for sequence in sequences for code in sequence})
        self.V = len(self.vocab)

    def scan(self, recent_codes: list[str]) -> dict:
        unknown = sorted(set(recent_codes[-8:]) - set(self.vocab))
        return {"surprise": 0.0, "is_anomaly": bool(unknown), "flagged": unknown,
                "rare_rate": sum(code not in self.vocab for code in recent_codes[-8:]) / max(1, len(recent_codes[-8:])),
                "transition": "", "method": "baseline code membership, no learned transitions"}
