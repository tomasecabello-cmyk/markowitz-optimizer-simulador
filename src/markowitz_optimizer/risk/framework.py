"""
framework — análisis de riesgo de cartera en 10 dimensiones (determinístico).

Espejo de los 10 puntos del framework del proyecto. Es una función pura sobre
los pesos de la cartera + estadísticas de mercado + metadatos de los activos:
no llama a la IA ni a la red. La IA (capa ai/) interpreta y prioriza ESTE
output; acá se computan los hechos cuantitativos y auditables.

Dimensiones (cada una produce un `RiskDimension` con severity 0-100 y detalle):
  1. Correlación entre holdings        6. Liquidez por holding
  2. Concentración sectorial           7. Riesgo de acción individual
  3. Exposición geográfica/moneda      8. Tail risk (VaR/CVaR)
  4. Sensibilidad a tasas              9. Hedging (inputs para la IA)
  5. Stress test de recesión          10. Rebalanceo (delta vs óptimo)

`risk_score` global (0-100, mayor = más riesgo) es el promedio ponderado de las
severidades de las dimensiones cuantificables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Severidad -> etiqueta cualitativa.
SEVERITY_BANDS: list[tuple[float, str]] = [
    (20, "bajo"), (40, "moderado"), (60, "elevado"), (80, "alto"), (101, "crítico")
]

# Pesos de cada dimensión en el risk_score global (suman 1.0). Las dimensiones
# 9 y 10 son recomendaciones, no contribuyen al score.
DIMENSION_WEIGHTS: dict[str, float] = {
    "correlation": 0.15, "sector_concentration": 0.15, "geographic": 0.10,
    "rate_sensitivity": 0.10, "recession_stress": 0.20, "liquidity": 0.10,
    "single_stock": 0.10, "tail_risk": 0.10,
}

# Activos sensibles a tasas (proxy por categoría/símbolo).
_RATE_SENSITIVE_HINTS = ("TLT", "IEF", "BND", "AGG", "LQD", "TIP", "GOVT", "SHY", "ZB", "ZN")
# Shocks de mercado de referencia para el stress test (caída del benchmark).
_SCENARIO_SHOCKS = {"2008": -0.55, "covid_2020": -0.34}

# --- Calibración de severidad (0-100) --------------------------------------
# Cada dimensión mapea una métrica cruda a una severidad 0-100 de forma lineal y
# saturante (clip). Los umbrales salen de convenciones de gestión de riesgo y se
# nombran acá para que sean auditables y defendibles, no "números mágicos".

# Correlación media ponderada → severidad. ~0.91 satura en 100 (sin
# diversificación); corr ≤ 0 da 0 (diversificación plena).
CORR_SEVERITY_SLOPE = 110.0

# Concentración sectorial: se penaliza el exceso por encima de un sector
# "neutral" (20%) y satura al alcanzar 60% (= BASE + RANGE) en un solo sector.
SECTOR_BASE, SECTOR_RANGE = 0.20, 0.40

# Geografía/moneda: 100% en un país aporta 60; 100% fuera de la moneda base
# (USD) aporta 80. La suma se clipea a 100.
GEO_COUNTRY_SLOPE, GEO_FX_SLOPE = 60.0, 80.0

# Sensibilidad a tasas: el peso en instrumentos de duración satura ~en 83%.
RATE_SLOPE = 120.0

# Stress de recesión: la peor caída estimada (drawdown, valor abs) → severidad;
# ~71% de drawdown satura en 100.
STRESS_DD_SLOPE = 140.0

# Liquidez: ~67% de la cartera en baja liquidez satura en 100.
LIQUIDITY_SLOPE = 150.0

# Acción individual: se penaliza el exceso sobre 15% (regla de concentración
# tipo UCITS "5/10/40" simplificada) y satura al llegar a 50% (= BASE + RANGE).
SINGLE_BASE, SINGLE_RANGE = 0.15, 0.35

# Tail risk: VaR diario (valor abs) → severidad. Pendiente ≈ √252×120 ≈ 1900,
# manteniendo la calibración histórica del proyecto pero SOBRE el VaR diario
# (sin anualizar con √t, que supondría retornos iid-normales).
TAIL_VAR_SLOPE = 1900.0

# z del 5% de la normal estándar (norm.ppf(0.05)); fijo para no depender de scipy.
_Z_95 = -1.6448536269514722


def _modified_var(series: np.ndarray, alpha: float = 0.05) -> float:
    """VaR modificado de Cornish-Fisher (Favre & Galeano, 2002).

    Corrige el cuantil normal z por la asimetría S y la curtosis en exceso K
    muestrales, así el VaR refleja las colas gordas reales de los retornos:

        z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36
        mVaR = μ + z_cf·σ        (retorno diario; negativo = pérdida)

    Para series muy cortas (<4) cae al percentil empírico.
    """
    s = np.asarray(series, dtype=float)
    n = s.size
    if n < 4:
        return float(np.percentile(s, alpha * 100)) if n else 0.0
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd <= 0:
        return mu
    z = _Z_95  # alpha = 0.05 (95%)
    skew = float(((s - mu) ** 3).mean() / sd**3)
    exkurt = float(((s - mu) ** 4).mean() / sd**4) - 3.0
    z_cf = (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * exkurt / 24.0
        - (2 * z**3 - 5 * z) * skew**2 / 36.0
    )
    return mu + z_cf * sd


def _band(severity: float) -> str:
    for limit, label in SEVERITY_BANDS:
        if severity < limit:
            return label
    return "crítico"


@dataclass
class RiskDimension:
    key: str
    title: str
    severity: float          # 0-100
    band: str
    summary: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "severity": round(float(self.severity), 1),
            "band": self.band,
            "summary": self.summary,
            "metrics": self.metrics,
        }


@dataclass
class RiskAssessment:
    risk_score: float
    risk_band: str
    dimensions: list[RiskDimension]
    top_risks: list[str]

    def to_dict(self) -> dict:
        return {
            "risk_score": round(float(self.risk_score), 1),
            "risk_band": self.risk_band,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "top_risks": list(self.top_risks),
        }


def _dim(key, title, severity, summary, metrics) -> RiskDimension:
    severity = float(np.clip(severity, 0, 100))
    return RiskDimension(key, title, severity, _band(severity), summary, metrics)


def _weighted_avg_corr(weights: dict[str, float], corr: pd.DataFrame) -> float:
    tickers = [t for t in weights if t in corr.columns]
    if len(tickers) < 2:
        return 0.0
    num = den = 0.0
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            wpair = weights[a] * weights[b]
            num += wpair * float(corr.loc[a, b])
            den += wpair
    return num / den if den > 0 else 0.0


def assess_portfolio(
    weights: dict[str, float],
    daily_returns: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    asset_info: dict[str, dict],
) -> RiskAssessment:
    """Computa el assessment de 10 dimensiones. `weights` debe sumar ~1."""
    tickers = [t for t in weights if weights[t] > 0]
    dims: list[RiskDimension] = []

    # --- 1. Correlación entre holdings ---
    avg_corr = _weighted_avg_corr(weights, corr_matrix)
    sev = np.clip(avg_corr * CORR_SEVERITY_SLOPE, 0, 100)
    dims.append(_dim(
        "correlation", "Correlación entre Holdings", sev,
        f"Correlación media ponderada de {avg_corr:.2f}. "
        + ("Diversificación pobre: los activos se mueven juntos y amplifican pérdidas."
           if avg_corr > 0.6 else
           "Diversificación razonable entre posiciones." if avg_corr < 0.4 else
           "Correlación intermedia; hay margen para diversificar."),
        {"avg_weighted_correlation": round(avg_corr, 4)},
    ))

    # --- 2. Concentración sectorial ---
    sector_w: dict[str, float] = {}
    for t in tickers:
        s = asset_info.get(t, {}).get("sector", "Unknown")
        sector_w[s] = sector_w.get(s, 0.0) + weights[t]
    max_sector, max_sw = (max(sector_w.items(), key=lambda kv: kv[1])
                          if sector_w else ("Unknown", 0.0))
    sev = np.clip((max_sw - SECTOR_BASE) / SECTOR_RANGE * 100, 0, 100)
    dims.append(_dim(
        "sector_concentration", "Concentración Sectorial", sev,
        f"Mayor exposición: {max_sector} ({max_sw*100:.0f}%). "
        + (">40% en un sector es concentración significativa." if max_sw > 0.40
           else "Distribución sectorial dentro de límites razonables."),
        {"by_sector": {k: round(v, 4) for k, v in sorted(
            sector_w.items(), key=lambda kv: -kv[1])}, "max_sector_weight": round(max_sw, 4)},
    ))

    # --- 3. Exposición geográfica / moneda ---
    country_w: dict[str, float] = {}
    currency_w: dict[str, float] = {}
    for t in tickers:
        meta = asset_info.get(t, {})
        country_w[meta.get("country", "Unknown")] = (
            country_w.get(meta.get("country", "Unknown"), 0.0) + weights[t])
        currency_w[meta.get("currency", "USD")] = (
            currency_w.get(meta.get("currency", "USD"), 0.0) + weights[t])
    max_country_w = max(country_w.values()) if country_w else 0.0
    non_base_fx = 1.0 - currency_w.get("USD", 0.0)
    sev = np.clip(max_country_w * GEO_COUNTRY_SLOPE + non_base_fx * GEO_FX_SLOPE, 0, 100)
    dims.append(_dim(
        "geographic", "Exposición Geográfica y Moneda", sev,
        f"Concentración geográfica máxima {max_country_w*100:.0f}%; "
        f"exposición a moneda no-USD {non_base_fx*100:.0f}%.",
        {"by_country": {k: round(v, 4) for k, v in country_w.items()},
         "by_currency": {k: round(v, 4) for k, v in currency_w.items()}},
    ))

    # --- 4. Sensibilidad a tasas de interés ---
    rate_w = sum(weights[t] for t in tickers
                 if any(h in t for h in _RATE_SENSITIVE_HINTS)
                 or "bond" in str(asset_info.get(t, {}).get("sector", "")).lower())
    sev = np.clip(rate_w * RATE_SLOPE, 0, 100)
    dims.append(_dim(
        "rate_sensitivity", "Sensibilidad a Tasas de Interés", sev,
        f"{rate_w*100:.0f}% en instrumentos sensibles a duración (renta fija/bonos). "
        + ("Movimientos de tasas de la Fed/BCE impactarían fuerte." if rate_w > 0.3
           else "Exposición a duración acotada."),
        {"rate_sensitive_weight": round(rate_w, 4)},
    ))

    # --- 5. Stress test de recesión ---
    # β por activo: el estimado por regresión (CAPM vs S&P 500, poblado en
    # market_data para US *y* ARG). 1.0 sólo como último recurso si no hay β.
    betas = [_asset_beta(asset_info, t) for t in tickers]
    n_estimated = sum(1 for t in tickers if asset_info.get(t, {}).get("beta") is not None)
    pesos = [weights[t] for t in tickers]
    port_beta = float(np.dot(betas, pesos)) if tickers else 1.0
    scenarios = {sc: round(float(np.clip(port_beta * shock, -0.95, 0)), 4)
                 for sc, shock in _SCENARIO_SHOCKS.items()}
    hist_dd = _max_drawdown(_portfolio_series(weights, daily_returns))
    worst = min(min(scenarios.values()), hist_dd)
    sev = np.clip(abs(worst) * STRESS_DD_SLOPE, 0, 100)
    beta_note = (f" β estimada por regresión para {n_estimated}/{len(tickers)} activos."
                 if n_estimated else " β no disponible (se asumió 1.0).")
    dims.append(_dim(
        "recession_stress", "Stress Test de Recesión", sev,
        f"Beta de cartera {port_beta:.2f}. Drawdown estimado: "
        f"2008 {scenarios['2008']*100:.0f}%, COVID {scenarios['covid_2020']*100:.0f}%; "
        f"peor caída histórica en el período {hist_dd*100:.0f}%.{beta_note}",
        {"portfolio_beta": round(port_beta, 3), "scenario_drawdowns": scenarios,
         "historical_max_drawdown": round(hist_dd, 4),
         "betas_estimated": n_estimated, "n_holdings": len(tickers)},
    ))

    # --- 6. Liquidez por holding ---
    liq_ratings = {}
    illiquid_w = 0.0
    for t in tickers:
        meta = asset_info.get(t, {})
        rating = _liquidity_rating(meta.get("avg_volume"), meta.get("market_cap"))
        liq_ratings[t] = rating
        if rating == "LOW":
            illiquid_w += weights[t]
    sev = np.clip(illiquid_w * LIQUIDITY_SLOPE, 0, 100)
    dims.append(_dim(
        "liquidity", "Rating de Liquidez por Holding", sev,
        f"{illiquid_w*100:.0f}% de la cartera en posiciones de baja liquidez."
        if illiquid_w > 0 else "Todas las posiciones con liquidez media/alta.",
        {"ratings": liq_ratings, "low_liquidity_weight": round(illiquid_w, 4)},
    ))

    # --- 7. Riesgo de acción individual ---
    max_t, max_w = (max(weights.items(), key=lambda kv: kv[1])
                    if weights else ("-", 0.0))
    over = {t: round(w, 4) for t, w in weights.items() if w > SINGLE_BASE}
    sev = np.clip((max_w - SINGLE_BASE) / SINGLE_RANGE * 100, 0, 100)
    dims.append(_dim(
        "single_stock", "Riesgo de Acción Individual", sev,
        f"Posición mayor: {max_t} ({max_w*100:.0f}%). "
        + (">15-20% en un activo es sobredimensionamiento." if max_w > 0.15
           else "Ninguna posición sobredimensionada."),
        {"largest_position": {max_t: round(max_w, 4)}, "oversized_positions": over},
    ))

    # --- 8. Tail risk (VaR / CVaR / VaR modificado) ---
    # VaR y CVaR EMPÍRICOS (histórico, sin supuesto de distribución) + VaR
    # MODIFICADO de Cornish-Fisher (Favre & Galeano), que corrige el cuantil
    # gaussiano por la asimetría y curtosis reales. NO se anualiza con √252:
    # escalar un cuantil por √t supone retornos iid-normales, justo lo que las
    # colas gordas violan (se subestimaría la pérdida en la cola).
    series = _portfolio_series(weights, daily_returns)
    if len(series):
        var95 = float(np.percentile(series, 5))
        tail = series[series <= var95]
        cvar95 = float(tail.mean()) if tail.size else var95
        mvar95 = _modified_var(series)
    else:
        var95 = cvar95 = mvar95 = 0.0
    worst_var = min(var95, mvar95)  # el más conservador (más negativo)
    sev = np.clip(abs(worst_var) * TAIL_VAR_SLOPE, 0, 100)
    dims.append(_dim(
        "tail_risk", "Escenarios de Tail Risk", sev,
        f"VaR diario 95% {var95*100:.2f}% (histórico); VaR modificado "
        f"(Cornish-Fisher, ajusta por asimetría y curtosis) {mvar95*100:.2f}%; "
        f"CVaR {cvar95*100:.2f}%. Medidas diarias: no se anualizan por √t porque "
        f"los retornos no son normales.",
        {"var_95_daily": round(var95, 5), "cvar_95_daily": round(cvar95, 5),
         "modified_var_95_daily": round(mvar95, 5)},
    ))

    # --- 9. Hedging (inputs para la IA; severity = 0, no puntúa) ---
    dims.append(_dim(
        "hedging", "Estrategias de Hedging", 0,
        "Los 3 mayores riesgos identificados se pasan a la IA para proponer "
        "coberturas concretas (opciones, ETF inversos, oro).",
        {},
    ))

    # --- 10. Rebalanceo (delta vs óptimo; lo completa la API con el óptimo) ---
    dims.append(_dim(
        "rebalancing", "Recomendaciones de Rebalanceo", 0,
        "El delta entre la cartera actual y la de máximo Sharpe se calcula en la "
        "capa de optimización; la IA justifica los cambios.",
        {},
    ))

    # --- Score global ---
    score = sum(d.severity * DIMENSION_WEIGHTS.get(d.key, 0.0) for d in dims)
    score = score / sum(DIMENSION_WEIGHTS.values()) if DIMENSION_WEIGHTS else score
    scored = [d for d in dims if d.key in DIMENSION_WEIGHTS]
    top = sorted(scored, key=lambda d: -d.severity)[:3]

    return RiskAssessment(
        risk_score=score,
        risk_band=_band(score),
        dimensions=dims,
        top_risks=[f"{d.title} ({d.band}, {d.severity:.0f}/100)" for d in top],
    )


def _asset_beta(asset_info: dict[str, dict], ticker: str) -> float:
    """β del activo desde asset_info; 1.0 sólo si no hay ninguno estimado.

    Usa un chequeo explícito contra None (no `or`) para no descartar betas
    legítimamente chicos o negativos (ej. bonos, oro), que `x or 1.0` pisaría.
    """
    b = asset_info.get(ticker, {}).get("beta")
    return 1.0 if b is None else float(b)


def _portfolio_series(weights: dict[str, float], daily_returns: pd.DataFrame) -> np.ndarray:
    cols = [t for t in weights if t in daily_returns.columns]
    if not cols:
        return np.array([])
    w = np.array([weights[t] for t in cols])
    w = w / w.sum() if w.sum() > 0 else w
    return (daily_returns[cols].to_numpy() @ w)


def _max_drawdown(series: np.ndarray) -> float:
    if len(series) == 0:
        return 0.0
    cum = np.cumprod(1 + series)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def _liquidity_rating(avg_volume, market_cap) -> str:
    vol = avg_volume or 0
    cap = market_cap or 0
    if vol >= 1_000_000 or cap >= 10_000_000_000:
        return "HIGH"
    if vol >= 100_000 or cap >= 1_000_000_000:
        return "MED"
    return "LOW"
