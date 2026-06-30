"""
Smoke test sin red: valida el pipeline matemático (engine + risk + ai mock)
con datos sintéticos. Salida 0 = PASS.

No usa yfinance ni la API de Claude. El e2e con datos reales se prueba con la
app corriendo (ver README).

    python scripts/smoke.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from markowitz_optimizer.ai import analyst
from markowitz_optimizer.engine import markowitz
from markowitz_optimizer.risk.framework import assess_portfolio


def main() -> int:
    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    n = len(tickers)

    # Retornos diarios sintéticos correlacionados.
    days = 600
    base = rng.normal(0, 0.01, size=(days, 1))
    noise = rng.normal(0, 0.012, size=(days, n))
    daily = pd.DataFrame(0.5 * base + noise, columns=tickers)

    mu = daily.mean().to_numpy() * 252
    sigma = (daily.cov().to_numpy()) * 252
    corr = daily.corr()

    # 1. Frontera eficiente.
    mk = markowitz.optimize(tickers, mu, sigma, risk_free_rate=0.04,
                            frontier_points=20, monte_carlo_samples=500)
    d = mk.to_dict()
    assert abs(sum(d["max_sharpe"]["weights"].values()) - 1.0) < 1e-3, "pesos max_sharpe no suman 1"
    assert abs(sum(d["min_variance"]["weights"].values()) - 1.0) < 1e-3, "pesos min_var no suman 1"
    assert d["max_sharpe"]["sharpe"] >= d["min_variance"]["sharpe"] - 1e-6, "sharpe inconsistente"
    assert len(d["frontier"]) > 1, "frontera vacía"
    assert len(d["monte_carlo"]) == 500, "monte carlo incompleto"
    print(f"[OK] Markowitz: max Sharpe={d['max_sharpe']['sharpe']:.3f}, "
          f"min var vol={d['min_variance']['volatility']:.3f}")

    # 2. Framework de riesgo.
    weights = {t: 1.0 / n for t in tickers}
    info = {t: {"sector": "Tech", "country": "US", "currency": "USD",
                "beta": 1.1, "avg_volume": 2_000_000, "market_cap": 2e10} for t in tickers}
    info["AAA"]["sector"] = "Energy"  # algo de variación sectorial
    ra = assess_portfolio(weights, daily, corr, info).to_dict()
    assert 0 <= ra["risk_score"] <= 100, "risk_score fuera de rango"
    assert len(ra["dimensions"]) == 10, f"esperaba 10 dimensiones, hay {len(ra['dimensions'])}"
    assert ra["top_risks"], "sin top_risks"
    print(f"[OK] Riesgo: score={ra['risk_score']} ({ra['risk_band']}), "
          f"dims={len(ra['dimensions'])}, top={len(ra['top_risks'])}")

    # 3. Analista IA (mock determinístico, sin API key).
    ai = analyst.analyze_portfolio(
        portfolio={"weights": weights, "sharpe": d["max_sharpe"]["sharpe"]},
        market_summary={"mean_returns": {t: float(mu[i]) for i, t in enumerate(tickers)}},
        markowitz=d, risk=ra, api_key="",  # fuerza mock
    ).to_dict()
    assert ai["is_mock"] is True, "debería ser mock sin api key"
    assert ai["executive_summary"] and ai["hedging_strategies"], "análisis IA vacío"
    print(f"[OK] IA (mock): {len(ai['hedging_strategies'])} coberturas, "
          f"{len(ai['key_findings'])} hallazgos")

    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
