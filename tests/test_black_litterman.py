"""Tests de Black-Litterman (puro, sin red)."""

from __future__ import annotations

import numpy as np

from markowitz_optimizer.engine.black_litterman import (
    black_litterman_returns,
    implied_equilibrium_returns,
)


def _sigma():
    # Covarianza PD simple de 3 activos.
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vol = np.array([0.20, 0.15, 0.10])
    return np.outer(vol, vol) * corr


def test_equilibrium_returns_sign_and_shape():
    sigma = _sigma()
    w = np.array([0.5, 0.3, 0.2])
    pi = implied_equilibrium_returns(sigma, w, delta=2.5)
    assert pi.shape == (3,)
    assert (pi > 0).all()  # con pesos y covarianzas positivas, equilibrio > 0


def test_posterior_between_prior_and_views():
    sigma = _sigma()
    mu_hist = np.array([0.30, 0.05, 0.02])  # view fuerte en el activo 0
    post = black_litterman_returns(sigma, mu_hist, rf=0.04, view_confidence=1.0)
    pi = implied_equilibrium_returns(sigma, np.full(3, 1 / 3), 2.5) + 0.04
    # El posterior del activo 0 queda entre el equilibrio y la view histórica.
    lo, hi = sorted([pi[0], mu_hist[0]])
    assert lo - 1e-6 <= post[0] <= hi + 1e-6


def test_confidence_moves_toward_history():
    sigma = _sigma()
    mu_hist = np.array([0.30, 0.05, 0.02])
    low = black_litterman_returns(sigma, mu_hist, rf=0.04, view_confidence=0.1)
    high = black_litterman_returns(sigma, mu_hist, rf=0.04, view_confidence=10.0)
    # Más confianza en las views => más cerca de la media histórica en el activo 0.
    assert abs(high[0] - mu_hist[0]) < abs(low[0] - mu_hist[0])
