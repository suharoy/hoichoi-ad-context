"""Empirical calibration utilities for break-suitability features."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmpiricalCDF:
    """Frozen empirical CDF fitted from development-corpus values."""

    values: tuple[float, ...]

    @classmethod
    def fit(cls, values: list[float]) -> "EmpiricalCDF":
        cleaned = np.asarray(values, dtype=np.float64)

        if cleaned.size == 0:
            raise ValueError("Cannot fit an empirical CDF with no values")

        if not np.isfinite(cleaned).all():
            raise ValueError("Empirical CDF values must all be finite")

        cleaned.sort()

        return cls(tuple(float(value) for value in cleaned))

    def transform(self, value: float) -> float:
        """Return fraction of reference observations <= value."""
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("Cannot transform a non-finite value")

        rank = bisect_right(self.values, value)
        return rank / len(self.values)

    def to_dict(self) -> dict:
        return {
            "count": len(self.values),
            "minimum": self.values[0],
            "maximum": self.values[-1],
            "reference_values": list(self.values),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmpiricalCDF":
        return cls.fit(
            [float(value) for value in payload["reference_values"]]
        )


def harmonic_mean(left: float, right: float) -> float:
    """Harmonic mean used to represent two-sided text evidence length."""
    left = float(left)
    right = float(right)

    if left <= 0 or right <= 0:
        return 0.0

    return 2.0 * left * right / (left + right)


def percentile_summary(values: list[float]) -> dict[str, float]:
    """Return descriptive statistics for finite values."""
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Cannot summarize an empty collection")

    if not np.isfinite(array).all():
        raise ValueError("Summary values must be finite")

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }