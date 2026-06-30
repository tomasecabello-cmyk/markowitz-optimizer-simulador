"""
backtest — desempeño histórico de carteras a peso constante sobre la ventana.

Toma los retornos diarios ya alineados (y normalizados a USD si corresponde) y
simula cada conjunto de pesos como una cartera rebalanceada a diario (peso
constante). Devuelve curvas de equity (base 100) + métricas comparables:
retorno total, CAGR, volatilidad anual, Sharpe y máximo drawdown.

Es una función pura: no descarga datos ni optimiza, solo evalúa pesos sobre la
serie que recibe. Sirve para contrastar la cartera ACTUAL vs la ÓPTIMA sugerida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _metrics(equity: np.ndarray, daily: np.ndarray, rf: float) -> dict:
    total_return = float(equity[-1] / equity[0] - 1.0)
    years = max(len(daily) / TRADING_DAYS, 1e-9)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    ann_ret = float(daily.mean() * TRADING_DAYS)
    sharpe = float((ann_ret - rf) / vol) if vol > 0 else 0.0
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1.0).min())
    return {
        "total_return": round(total_return, 5),
        "cagr": round(cagr, 5),
        "volatility": round(vol, 5),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 5),
    }


def walk_forward(
    daily_returns: pd.DataFrame,
    current_weights: dict[str, float],
    rf: float = 0.0,
    max_weight: float = 1.0,
    cov_method: str = "ledoit_wolf",
    return_method: str = "black_litterman",
    bl_view_confidence: float = 1.0,
    lookback: int = 126,
    rebalance: int = 21,
) -> dict:
    """
    Backtest OUT-OF-SAMPLE (walk-forward): en cada fecha de rebalanceo se estiman
    μ y Σ SOLO con datos pasados (ventana expansiva), se optimiza la cartera y se
    aplica a los días siguientes (que el optimizador no vio). Así se evita el
    look-ahead del backtest in-sample.

    Compara: cartera actual (pesos fijos), máx Sharpe OOS, mín varianza OOS y un
    benchmark equal-weight (1/N), todos sobre el MISMO período fuera de muestra.

    Devuelve {available, dates, series, metrics, params}. Si la ventana es muy
    corta para entrenar + evaluar, available=False.
    """
    from . import markowitz
    from ..data.market_data import _estimate_cov
    from .black_litterman import black_litterman_returns

    cols = list(daily_returns.columns)
    n = len(cols)
    T = len(daily_returns)
    if T <= lookback + rebalance:
        return {"available": False,
                "reason": f"ventana insuficiente para walk-forward (días={T}, "
                          f"requiere > {lookback + rebalance})"}

    ew = np.full(n, 1.0 / n)
    cw = np.array([float(current_weights.get(c, 0.0)) for c in cols])
    cw = cw / cw.sum() if cw.sum() > 0 else ew.copy()

    rets: dict[str, list[float]] = {"current": [], "max_sharpe": [],
                                    "min_variance": [], "equal_weight": []}
    dates: list[str] = []
    n_rebal = 0
    t = lookback
    while t < T:
        train = daily_returns.iloc[:t]
        mu_hist = train.mean().to_numpy() * TRADING_DAYS
        sigma = _estimate_cov(train, cov_method)[0].to_numpy()
        if return_method == "black_litterman":
            mu = black_litterman_returns(sigma, mu_hist, rf=rf, view_confidence=bl_view_confidence)
        else:
            mu = mu_hist
        try:
            w_ms = markowitz.max_sharpe_portfolio(cols, mu, sigma, rf, max_weight).weights
            w_mv = markowitz.min_variance_portfolio(cols, mu, sigma, rf, max_weight).weights
            wms = np.array([w_ms.get(c, 0.0) for c in cols])
            wmv = np.array([w_mv.get(c, 0.0) for c in cols])
        except Exception:  # noqa: BLE001 - óptimo infactible: caer a equal-weight
            wms = wmv = ew.copy()
        n_rebal += 1

        seg = daily_returns.iloc[t:t + rebalance].to_numpy()
        seg_dates = daily_returns.index[t:t + rebalance]
        for i, dt in enumerate(seg_dates):
            r = seg[i]
            rets["current"].append(float(r @ cw))
            rets["max_sharpe"].append(float(r @ wms))
            rets["min_variance"].append(float(r @ wmv))
            rets["equal_weight"].append(float(r @ ew))
            dates.append(str(dt.date()))
        t += rebalance

    series, metrics = {}, {}
    for name, arr in rets.items():
        a = np.array(arr)
        eq = 100.0 * np.cumprod(1.0 + a)
        series[name] = [round(float(x), 4) for x in eq]
        metrics[name] = _metrics(eq, a, rf)

    return {"available": True, "dates": dates, "series": series, "metrics": metrics,
            "params": {"lookback": lookback, "rebalance": rebalance, "rebalances": n_rebal}}


def backtest(
    daily_returns: pd.DataFrame,
    weight_sets: dict[str, dict[str, float]],
    rf: float = 0.0,
) -> dict:
    """
    Simula carteras a peso constante.

    Args:
        daily_returns: retornos diarios (fechas x tickers), ya alineados.
        weight_sets: {nombre -> {ticker: peso}}. Pesos faltantes => 0.
        rf: tasa libre de riesgo anual para el Sharpe.

    Returns:
        {dates, series:{nombre:[equity base 100]}, metrics:{nombre:{...}}}
    """
    cols = list(daily_returns.columns)
    R = daily_returns.to_numpy()                      # (T, N)
    dates = [str(d.date()) for d in daily_returns.index]

    series: dict[str, list[float]] = {}
    metrics: dict[str, dict] = {}
    for name, weights in weight_sets.items():
        w = np.array([float(weights.get(t, 0.0)) for t in cols], dtype=float)
        s = w.sum()
        if s > 0:
            w = w / s                                  # normalizar por si no suma 1
        port_daily = R @ w                             # retorno diario de la cartera
        equity = 100.0 * np.cumprod(1.0 + port_daily)
        series[name] = [round(float(x), 4) for x in equity]
        metrics[name] = _metrics(equity, port_daily, rf)

    return {"dates": dates, "series": series, "metrics": metrics}
