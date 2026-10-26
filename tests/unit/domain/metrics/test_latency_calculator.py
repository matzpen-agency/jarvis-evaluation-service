"""
test_latency_calculator.py — Unit tests for LatencyCalculator.
"""

from __future__ import annotations

import pytest

from src.domain.metrics.latency_calculator import LatencyCalculator


@pytest.fixture
def calculator() -> LatencyCalculator:
    return LatencyCalculator()


@pytest.mark.unit
def test_empty_samples(calculator):
    stats = calculator.calculate([])
    assert stats.p50 == 0.0
    assert stats.total_samples == 0


@pytest.mark.unit
def test_single_sample(calculator):
    stats = calculator.calculate([500.0])
    assert stats.p50 == 500.0
    assert stats.p95 == 500.0
    assert stats.p99 == 500.0
    assert stats.minimum == 500.0
    assert stats.maximum == 500.0
    assert stats.average == 500.0
    assert stats.total_samples == 1


@pytest.mark.unit
def test_multiple_samples(calculator):
    samples = [100.0, 200.0, 300.0, 400.0, 500.0]
    stats = calculator.calculate(samples)
    assert stats.p50 == 300.0
    assert stats.minimum == 100.0
    assert stats.maximum == 500.0
    assert stats.average == 300.0
    assert stats.total_samples == 5


@pytest.mark.unit
def test_high_variance(calculator):
    samples = [10.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    stats = calculator.calculate(samples)
    assert stats.p95 >= 100.0  # the 1000ms outlier raises p95
    assert stats.p50 == 10.0
