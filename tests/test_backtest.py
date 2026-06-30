"""Tests del backtest a peso constante (puro, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markowitz_optimizer.engine.backtest import backtest, walk_forward


def _returns():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    data = rng.normal(0.0005, 0.01, size=(300, 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def test_backtest_shape_and_metrics():
    R = _returns()
    out = backtest(R, {"current": {"A": 0.5, "B": 0.5}, "opt": {"C": 1.0}}, rf=0.04)
    assert len(out["dates"]) == len(R)
    assert len(out["series"]["current"]) == len(R)
    # equity arranca en ~100
    assert abs(out["series"]["current"][0] - 100 * (1 + R.iloc[0] @ np.array([.5, .5, 0]))) < 1
    for name in ("current", "opt"):
        m = out["metrics"][name]
        assert set(m) == {"total_return", "cagr", "volatility", "sharpe", "max_drawdown"}
        assert m["max_drawdown"] <= 0.0
        assert m["volatility"] >= 0.0


def test_walk_forward_is_out_of_sample():
    R = _returns()  # 300 días > 126 + 21
    out = walk_forward(R, {"A": 0.34, "B": 0.33, "C": 0.33},
                       rf=0.04, lookback=126, rebalance=21)
    assert out["available"] is True
    # incluye benchmark equal-weight y la cartera actual
    assert set(out["series"]) == {"current", "max_sharpe", "min_variance", "equal_weight"}
    # la serie OOS arranca después del lookback (no usa toda la historia)
    assert len(out["dates"]) <= len(R) - 126
    assert out["params"]["rebalances"] >= 1
    for m in out["metrics"].values():
        assert m["max_drawdown"] <= 0.0


def test_walk_forward_short_window_unavailable():
    R = _returns().iloc[:100]  # < 126 + 21
    out = walk_forward(R, {"A": 1.0}, lookback=126, rebalance=21)
    assert out["available"] is False
    assert "reason" in out


def test_weights_normalized():
    R = _returns()
    # pesos que no suman 1 se normalizan -> mismo resultado que ya normalizados
    a = backtest(R, {"x": {"A": 2.0, "B": 2.0}}, rf=0.0)["series"]["x"]
    b = backtest(R, {"x": {"A": 0.5, "B": 0.5}}, rf=0.0)["series"]["x"]
    assert a == b
