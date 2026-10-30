"""Per-asset baseline arithmetic and explicit demo code membership."""
from __future__ import annotations

import numpy as np

from fleet_data import BENIGN_CODES, FAULT_CODES, FleetDataset, Reading


class BaselineDetectors:
    def __init__(self, data: FleetDataset):
        self.states = {}
        self.vocabulary = set(BENIGN_CODES) | FAULT_CODES
        self.last_codes = {}
        for asset in data.assets:
            rows = [row for row in data.history if row.asset_id == asset.id]
            features = sorted(rows[0].metrics)
            values = np.array([[row.metrics[key] for key in features] for row in rows])
            median = np.median(values, axis=0)
            mad = np.median(np.abs(values - median), axis=0)
            scale = 1.4826 * np.where(mad > 0, mad, np.maximum(np.abs(median) * .01, .01))
            mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), .01)
            covariance = np.cov(((values - mean) / std).T)
            covariance = np.atleast_2d(covariance) + np.eye(len(features)) * 1e-6
            inverse = np.linalg.pinv(covariance)
            centered = (values - mean) / std
            distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0))
            self.states[asset.id] = {
                "features": features, "median": median, "scale": scale, "mean": mean, "std": std,
                "inverse": inverse, "maha_limit": max(float(np.percentile(distances, 99)), 1),
                "rows": rows,
            }
            primary = [row.event_codes[0] if row.event_codes else "OK" for row in rows]
            self.vocabulary.update(primary)
            self.last_codes[asset.id] = primary[-1]

    def score(self, row: Reading) -> dict:
        state = self.states[row.asset_id]
        values = np.array([row.metrics[key] for key in state["features"]])
        z = (values - state["median"]) / state["scale"]
        centered = (values - state["mean"]) / state["std"]
        distance = float(np.sqrt(max(float(centered @ state["inverse"] @ centered), 0)))
        anomalous = bool(np.max(np.abs(z)) >= 3)
        score = min(float(np.max(np.abs(z))) / 6, 1)
        return {
            "is_anomaly": anomalous, "score": round(score, 3),
            "isolation_forest_outlier": False, "mahalanobis": round(distance, 3),
            "mahalanobis_limit": round(state["maha_limit"], 3),
            "mahalanobis_outlier": distance > state["maha_limit"],
            "signals": [{"metric": key, "latest": row.metrics[key], "unit": "",
                         "robust_z": round(float(value), 2), "anomalous": bool(abs(value) >= 3)}
                        for key, value in zip(state["features"], z)],
        }

    def scan_codes(self, row: Reading, recent: list[str]) -> dict:
        previous = self.last_codes[row.asset_id]
        current = next((code for code in row.event_codes if code not in BENIGN_CODES), row.event_codes[0] if row.event_codes else "OK")
        surprise = 0.0
        self.last_codes[row.asset_id] = current
        unknown = [code for code in row.event_codes if code not in self.vocabulary]
        fault_codes = [code for code in row.event_codes if code in FAULT_CODES]
        window = (recent + [current])[-8:]
        return {
            "previous": previous, "current": current, "transition_surprise": round(surprise, 3),
            "fault_code_rate": sum(code not in BENIGN_CODES for code in window) / len(window),
            "is_anomaly": bool(fault_codes),
            "known_fault_codes": fault_codes, "unknown_codes": unknown,
        }
