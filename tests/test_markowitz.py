"""Tests del motor de Markowitz (puro, sin red)."""

from __future__ import annotations

import numpy as np

from markowitz_optimizer.engine import markowitz


def _inputs():
    tickers = ["A", "B", "C"]
    mu = np.array([0.05, 0.02, 0.40])      # C domina por retorno
    sigma = np.array([
        [0.04, 0.00, 0.00],
        [0.00, 0.02, 0.00],
        [0.00, 0.00, 0.09],
    ])
    return tickers, mu, sigma


def test_weights_sum_to_one_and_respect_cap():
    tickers, mu, sigma = _inputs()
    res = markowitz.optimize(tickers, mu, sigma, risk_free_rate=0.04,
                             frontier_points=10, monte_carlo_samples=50, max_weight=0.35)
    w = res.max_sharpe.weights
    assert abs(sum(w.values()) - 1.0) < 1e-3
    assert max(w.values()) <= 0.35 + 1e-3, f"cap violado: {w}"


def test_unconstrained_can_concentrate():
    # Sin tope, el óptimo puede concentrarse en el activo de mayor Sharpe.
    tickers, mu, sigma = _inputs()
    res = markowitz.optimize(tickers, mu, sigma, risk_free_rate=0.04,
                             frontier_points=10, monte_carlo_samples=50, max_weight=1.0)
    assert max(res.max_sharpe.weights.values()) > 0.5


def test_cap_diversifies_vs_unconstrained():
    tickers, mu, sigma = _inputs()
    capped = markowitz.optimize(tickers, mu, sigma, 0.04, 10, 50, max_weight=0.4).max_sharpe
    free = markowitz.optimize(tickers, mu, sigma, 0.04, 10, 50, max_weight=1.0).max_sharpe
    assert max(capped.weights.values()) < max(free.weights.values())
