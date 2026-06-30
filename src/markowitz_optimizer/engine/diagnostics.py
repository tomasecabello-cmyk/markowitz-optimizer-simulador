"""
diagnostics — tests de supuestos sobre los retornos, ANTES de optimizar.

El optimizador y las métricas (anualización ×252/√252, Sharpe, VaR gaussiano)
asumen que los retornos diarios son aproximadamente i.i.d. y normales. Este
módulo testea esos supuestos en vez de darlos por sentado:

  - Jarque-Bera        → ¿normales? (valida Sharpe y el VaR gaussiano)
  - Ljung-Box          → ¿independientes? (autocorrelación rompe la anualización)
  - Variance-Ratio     → ¿random walk? (Lo-MacKinlay, robusto a heterocedasticidad)
  - ARCH-LM (Engle)    → ¿volatility clustering? (Σ estática mal especificada → GARCH)

Implementación pura numpy/scipy (sin statsmodels). Funciones puras, sin red.
Cada test reporta su estadístico, p-valor y un veredicto al 5% de significancia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05            # significancia (5%)
MIN_OBS = 30            # menos que esto: la ventana es muy corta para testear


@dataclass
class DiagnosticTest:
    key: str
    title: str
    statistic: float
    p_value: float
    reject: bool              # ¿se rechaza H0 al 5%?
    h0: str                   # hipótesis nula
    verdict: str              # interpretación en criollo

    def to_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title,
            "statistic": round(float(self.statistic), 4),
            "p_value": round(float(self.p_value), 4),
            "reject": bool(self.reject), "h0": self.h0, "verdict": self.verdict,
        }


@dataclass
class DiagnosticsReport:
    available: bool
    n_obs: int
    tests: list[DiagnosticTest] = field(default_factory=list)
    normality_by_asset: dict = field(default_factory=dict)
    interpretation: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "n_obs": self.n_obs,
            "tests": [t.to_dict() for t in self.tests],
            "normality_by_asset": self.normality_by_asset,
            "interpretation": self.interpretation,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Tests individuales (numpy / scipy)
# ---------------------------------------------------------------------------

def jarque_bera(x: np.ndarray) -> tuple[float, float]:
    """JB = n/6 (S² + K²/4) ~ χ²(2). H0: normalidad."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = x.mean()
    sd = x.std(ddof=0)
    if sd <= 0:
        return 0.0, 1.0
    skew = np.mean((x - m) ** 3) / sd**3
    exkurt = np.mean((x - m) ** 4) / sd**4 - 3.0
    jb = n / 6.0 * (skew**2 + exkurt**2 / 4.0)
    return float(jb), float(stats.chi2.sf(jb, 2))


def _acf(x: np.ndarray, k: int) -> float:
    n = x.size
    m = x.mean()
    c0 = np.sum((x - m) ** 2) / n
    if c0 <= 0:
        return 0.0
    ck = np.sum((x[k:] - m) * (x[:-k] - m)) / n
    return ck / c0


def ljung_box(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """Q = n(n+2) Σ ρ_k²/(n-k) ~ χ²(h). H0: sin autocorrelación hasta el lag h."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = lags or min(10, max(1, n // 5))
    q = 0.0
    for k in range(1, h + 1):
        rho = _acf(x, k)
        q += rho**2 / (n - k)
    q *= n * (n + 2)
    return float(q), float(stats.chi2.sf(q, h)), h


def arch_lm(x: np.ndarray, lags: int = 5) -> tuple[float, float, int]:
    """LM = (n-q)·R² ~ χ²(q) de regresar ε²_t sobre q lags de ε². H0: sin efectos ARCH."""
    x = np.asarray(x, dtype=float)
    e = (x - x.mean()) ** 2
    n = e.size
    q = min(lags, max(1, (n - 2) // 2))
    if n <= q + 2:
        return 0.0, 1.0, q
    y = e[q:]
    cols = [np.ones(n - q)] + [e[q - k - 1 : n - k - 1] for k in range(q)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    lm = (n - q) * r2
    return float(lm), float(stats.chi2.sf(lm, q)), q


def variance_ratio(x: np.ndarray, q: int = 2) -> tuple[float, float, float]:
    """
    Lo-MacKinlay variance ratio con estadístico M2 robusto a heterocedasticidad.
    VR(q) = Var(retorno de q-períodos) / (q · Var(1-período)). H0 (random walk): VR=1.
    Devuelve (VR, M2, p_valor). M2 ~ N(0,1) bajo H0.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    if var1 <= 0 or n <= q:
        return 1.0, 0.0, 1.0
    # varianza de q-períodos (solapada, estimador insesgado)
    csum = np.cumsum(x)
    rq = csum[q - 1:] - np.concatenate(([0.0], csum[:-q]))   # sumas de q retornos
    # m absorbe el factor q: σ²_c(q) estima la varianza de 1 período bajo RW,
    # así que VR = σ²_c(q) / σ²_a(1) ≈ 1 (no se divide por q de nuevo).
    m = q * (n - q + 1) * (1.0 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    # error estándar robusto a heterocedasticidad (Lo-MacKinlay 1988)
    eps2 = (x - mu) ** 2
    s2 = np.sum(eps2)
    theta = 0.0
    for k in range(1, q):
        delta_k = np.sum(eps2[k:] * eps2[:-k]) / (s2**2)
        theta += (2.0 * (q - k) / q) ** 2 * delta_k
    if theta <= 0:
        return float(vr), 0.0, 1.0
    m2 = (vr - 1.0) / np.sqrt(theta)
    p = 2.0 * stats.norm.sf(abs(m2))
    return float(vr), float(m2), float(p)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def _portfolio_series(weights: dict[str, float], daily: pd.DataFrame) -> np.ndarray:
    cols = [t for t in weights if t in daily.columns and weights[t] > 0]
    if not cols:
        return np.array([])
    w = np.array([weights[t] for t in cols], dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    return daily[cols].to_numpy() @ w


def run_diagnostics(
    weights: dict[str, float], daily_returns: pd.DataFrame, vr_horizon: int = 2
) -> DiagnosticsReport:
    """Corre los 4 tests sobre la serie de la cartera + normalidad por activo."""
    series = _portfolio_series(weights, daily_returns)
    n = series.size
    if n < MIN_OBS:
        return DiagnosticsReport(
            available=False, n_obs=int(n),
            reason=f"ventana muy corta para testear supuestos (n={n}, requiere ≥ {MIN_OBS}).",
        )

    tests: list[DiagnosticTest] = []

    jb, jb_p = jarque_bera(series)
    jb_rej = jb_p < ALPHA
    tests.append(DiagnosticTest(
        "jarque_bera", "Normalidad (Jarque-Bera)", jb, jb_p, jb_rej,
        "Los retornos son normales.",
        ("Se RECHAZA la normalidad: hay colas gordas/asimetría. El Sharpe y el VaR "
         "gaussiano subestiman el riesgo de cola (por eso el VaR usa Cornish-Fisher)."
         if jb_rej else
         "No se rechaza la normalidad: el Sharpe y el VaR gaussiano son razonables."),
    ))

    lb, lb_p, lb_h = ljung_box(series)
    lb_rej = lb_p < ALPHA
    tests.append(DiagnosticTest(
        "ljung_box", f"Autocorrelación (Ljung-Box, {lb_h} lags)", lb, lb_p, lb_rej,
        "Los retornos son independientes (sin autocorrelación).",
        ("Se RECHAZA la independencia: hay autocorrelación. La anualización ×252/√252, "
         "que asume retornos i.i.d., es una aproximación."
         if lb_rej else
         "No se rechaza la independencia: la anualización ×252/√252 está justificada."),
    ))

    vr, m2, vr_p = variance_ratio(series, vr_horizon)
    vr_rej = vr_p < ALPHA
    tests.append(DiagnosticTest(
        "variance_ratio", f"Random walk (Variance-Ratio q={vr_horizon})", m2, vr_p, vr_rej,
        "La serie sigue un random walk (mercado eficiente débil).",
        (f"Se RECHAZA el random walk (VR={vr:.2f}): hay estructura temporal "
         f"({'momentum' if vr > 1 else 'reversión a la media'})."
         if vr_rej else
         f"No se rechaza el random walk (VR={vr:.2f}): consistente con eficiencia débil."),
    ))

    lm, lm_p, lm_q = arch_lm(series)
    lm_rej = lm_p < ALPHA
    tests.append(DiagnosticTest(
        "arch_lm", f"Volatility clustering (ARCH-LM, {lm_q} lags)", lm, lm_p, lm_rej,
        "La volatilidad es constante (sin clustering).",
        ("Se RECHAZAN efectos ARCH: hay volatility clustering. La Σ estática sobre la "
         "ventana subestima la dinámica del riesgo (motivaría un modelo GARCH)."
         if lm_rej else
         "No se detecta volatility clustering: la Σ estática es una aproximación razonable."),
    ))

    # Normalidad por activo (relevante para la covarianza)
    norm_by_asset: dict[str, dict] = {}
    for t in [c for c in weights if c in daily_returns.columns and weights[c] > 0]:
        xi = daily_returns[t].dropna().to_numpy()
        if xi.size >= MIN_OBS:
            ji, jpi = jarque_bera(xi)
            norm_by_asset[t] = {"jb": round(ji, 2), "p_value": round(jpi, 4),
                                "reject": bool(jpi < ALPHA)}
    n_non_normal = sum(1 for v in norm_by_asset.values() if v["reject"])

    # Interpretación global
    flags = [t.title.split(" (")[0] for t in tests if t.reject]
    if flags:
        interpretation = (
            "Varios supuestos no se cumplen "
            f"({', '.join(flags).lower()}). El optimizador igual funciona, pero las "
            "métricas que asumen normalidad/independencia (Sharpe, VaR gaussiano, "
            "anualización) deben leerse con cuidado. El motor ya mitiga esto con "
            "Ledoit-Wolf (Σ robusta) y VaR de Cornish-Fisher. "
            f"{n_non_normal}/{len(norm_by_asset)} activos rechazan normalidad individual."
        )
    else:
        interpretation = (
            "Los supuestos clave (normalidad, independencia, random walk, "
            "homocedasticidad) no se rechazan al 5%: las métricas estándar son válidas."
        )

    return DiagnosticsReport(
        available=True, n_obs=int(n), tests=tests,
        normality_by_asset=norm_by_asset, interpretation=interpretation,
    )
