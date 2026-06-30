"""Tests del estimador de covarianza (puro, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markowitz_optimizer.data.market_data import _estimate_cov


def _daily():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    return pd.DataFrame(rng.normal(0.0004, 0.012, size=(250, 4)),
                        index=idx, columns=["A", "B", "C", "D"])


def test_ledoit_wolf_returns_shrinkage_in_unit_interval():
    cov, lam = _estimate_cov(_daily(), "ledoit_wolf")
    assert cov.shape == (4, 4)
    assert 0.0 <= lam <= 1.0
    # simétrica y PSD (autovalores >= 0)
    assert np.allclose(cov.values, cov.values.T)
    assert (np.linalg.eigvalsh(cov.values) >= -1e-10).all()


def test_sample_method_has_no_shrinkage():
    cov, lam = _estimate_cov(_daily(), "sample")
    assert lam is None
    assert cov.shape == (4, 4)


def test_annualized_scale():
    # La covarianza anual debe ser ~252x la diaria.
    d = _daily()
    cov_a, _ = _estimate_cov(d, "sample")
    daily_var = d["A"].var(ddof=1)
    assert abs(cov_a.loc["A", "A"] / (daily_var * 252) - 1.0) < 1e-6
