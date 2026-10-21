"""
base_metric_calculator.py — Generic interface for metric computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMetricCalculator[T, R](ABC):
    """
    Generic base class for computing a metric from a collection of samples.

    Type parameters:
      T - input type (e.g., list[float], list[EvaluationContext])
      R - output type (e.g., LatencyStats, FailureAnalysis)
    """

    @abstractmethod
    def calculate(self, samples: T) -> R:
        """Compute and return the metric from the provided samples."""
        ...

    def __call__(self, samples: T) -> R:
        return self.calculate(samples)
