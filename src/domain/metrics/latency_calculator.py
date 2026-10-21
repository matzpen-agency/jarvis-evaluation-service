"""
latency_calculator.py — Computes latency distribution statistics.

Uses numpy for accurate percentile computation.
Operates on a list of latency values in milliseconds.
"""

from __future__ import annotations

import numpy as np

from src.domain.entities.dataset_run import LatencyStats
from src.domain.metrics.base_metric_calculator import BaseMetricCalculator


class LatencyCalculator(BaseMetricCalculator[list[float], LatencyStats]):
    """
    Computes p50, p95, p99, average, min, and max from latency samples.

    Input: list of per-sample latencies in milliseconds.
    Output: LatencyStats dataclass.
    """

    def calculate(self, samples: list[float]) -> LatencyStats:
        if not samples:
            return LatencyStats()

        arr = np.array(samples, dtype=np.float64)
        return LatencyStats(
            p50=float(round(np.percentile(arr, 50), 2)),
            p95=float(round(np.percentile(arr, 95), 2)),
            p99=float(round(np.percentile(arr, 99), 2)),
            average=float(round(float(np.mean(arr)), 2)),
            minimum=float(round(float(np.min(arr)), 2)),
            maximum=float(round(float(np.max(arr)), 2)),
            total_samples=len(samples),
        )
