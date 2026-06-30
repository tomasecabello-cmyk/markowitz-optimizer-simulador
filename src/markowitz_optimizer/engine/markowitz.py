"""
markowitz — frontera eficiente, optimización y simulación Monte Carlo.

Modelo: cartera long-only, pesos en [0, 1] que suman 1 (sin apalancamiento ni
cortos), sobre el supuesto clásico de un único período.

  E(Rp) = w · μ
  σ²p   = w · Σ · w
  Sharpe = (E(Rp) − Rf) / σp

Componentes:
  - min_variance_portfolio   : punto más a la izquierda de la frontera.
  - max_sharpe_portfolio     : cartera tangente a la Capital Market Line.
  - efficient_frontier       : conjunto de carteras óptimas por nivel de riesgo.
  - monte_carlo              : nube de carteras aleatorias (para visualizar).

Usa scipy.optimize (SLSQP) para los puntos óptimos exactos; el Monte Carlo es
solo ilustrativo. Las matrices/vectores entran como numpy alineados por orden
de tickers (ver build_inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

WEIGHT_CLEANUP_THRESHOLD: float = 1e-5
DEFAULT_RISK_FREE_RATE: float = 0.04  # 4% anual, tasa libre de riesgo por defecto


@dataclass
class PortfolioPoint:
    """Una cartera con sus pesos y métricas anualizadas."""

    label: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
            "expected_return": round(float(self.expected_return), 6),
            "volatility": round(float(self.volatility), 6),
            "sharpe": round(float(self.sharpe), 6),
        }


@dataclass
class MarkowitzResult:
    """Resultado completo de la optimización para el frontend."""

    tickers: list[str]
    risk_free_rate: float
    min_variance: PortfolioPoint
    max_sharpe: PortfolioPoint
    frontier: list[PortfolioPoint]
    monte_carlo: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "risk_free_rate": self.risk_free_rate,
            "min_variance": self.min_variance.to_dict(),
            "max_sharpe": self.max_sharpe.to_dict(),
            "frontier": [p.to_dict() for p in self.frontier],
            "monte_carlo": self.monte_carlo,
        }


# ---------------------------------------------------------------------------
# Métricas base
# ---------------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ sigma @ weights, 0.0)))


def portfolio_sharpe(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    vol = portfolio_volatility(weights, sigma)
    if vol <= 0:
        return 0.0
    return (portfolio_return(weights, mu) - rf) / vol


def _clean_weights(w: np.ndarray, tickers: list[str]) -> dict[str, float]:
    w = np.array(w, dtype=float)
    w[w < WEIGHT_CLEANUP_THRESHOLD] = 0.0
    total = w.sum()
    if total <= 0:
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = w / total
    return {t: float(w[i]) for i, t in enumerate(tickers)}


# ---------------------------------------------------------------------------
# Optimizadores
# ---------------------------------------------------------------------------

def _solve(objective, n: int, constraints: list[dict], max_weight: float = 1.0) -> np.ndarray:
    bounds = [(0.0, max_weight)] * n
    w0 = np.full(n, 1.0 / n)
    base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=base + constraints,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    return np.array(res.x, dtype=float)


def min_variance_portfolio(
    tickers: list[str], mu: np.ndarray, sigma: np.ndarray, rf: float, max_weight: float = 1.0
) -> PortfolioPoint:
    w = _solve(lambda w: float(w @ sigma @ w), len(tickers), [], max_weight)
    weights = _clean_weights(w, tickers)
    wv = np.array(list(weights.values()))
    return PortfolioPoint(
        label="min_variance",
        weights=weights,
        expected_return=portfolio_return(wv, mu),
        volatility=portfolio_volatility(wv, sigma),
        sharpe=portfolio_sharpe(wv, mu, sigma, rf),
    )


def max_sharpe_portfolio(
    tickers: list[str], mu: np.ndarray, sigma: np.ndarray, rf: float, max_weight: float = 1.0
) -> PortfolioPoint:
    # Maximizar Sharpe == minimizar -Sharpe.
    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ sigma @ w, 1e-12))
        return -((w @ mu) - rf) / vol

    w = _solve(neg_sharpe, len(tickers), [], max_weight)
    weights = _clean_weights(w, tickers)
    wv = np.array(list(weights.values()))
    return PortfolioPoint(
        label="max_sharpe",
        weights=weights,
        expected_return=portfolio_return(wv, mu),
        volatility=portfolio_volatility(wv, sigma),
        sharpe=portfolio_sharpe(wv, mu, sigma, rf),
    )


def _min_vol_for_target(
    tickers: list[str], mu: np.ndarray, sigma: np.ndarray, target: float, max_weight: float = 1.0
) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: float(w @ mu) - target}]
    return _solve(lambda w: float(w @ sigma @ w), len(tickers), constraints, max_weight)


def efficient_frontier(
    tickers: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float,
    points: int = 40,
    max_weight: float = 1.0,
) -> list[PortfolioPoint]:
    """Carteras de mínima varianza para una grilla de retornos objetivo."""
    mv = min_variance_portfolio(tickers, mu, sigma, rf, max_weight)
    r_min = mv.expected_return
    r_max = float(np.max(mu))
    if r_max <= r_min:
        return [mv]

    frontier: list[PortfolioPoint] = []
    for target in np.linspace(r_min, r_max, points):
        try:
            w = _min_vol_for_target(tickers, mu, sigma, float(target), max_weight)
        except Exception:  # noqa: BLE001
            continue
        weights = _clean_weights(w, tickers)
        wv = np.array(list(weights.values()))
        vol = portfolio_volatility(wv, sigma)
        ret = portfolio_return(wv, mu)
        frontier.append(
            PortfolioPoint(
                label="frontier",
                weights=weights,
                expected_return=ret,
                volatility=vol,
                sharpe=portfolio_sharpe(wv, mu, sigma, rf),
            )
        )
    return frontier


def monte_carlo(
    tickers: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float,
    samples: int = 4000,
    seed: int = 42,
) -> list[dict]:
    """Nube de carteras aleatorias (Dirichlet) para visualizar el espacio riesgo-retorno."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    weights = rng.dirichlet(np.ones(n), size=samples)
    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    sharpes = np.where(vols > 0, (rets - rf) / vols, 0.0)
    return [
        {
            "expected_return": round(float(rets[i]), 5),
            "volatility": round(float(vols[i]), 5),
            "sharpe": round(float(sharpes[i]), 5),
        }
        for i in range(samples)
    ]


def optimize(
    tickers: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    frontier_points: int = 40,
    monte_carlo_samples: int = 4000,
    max_weight: float = 1.0,
) -> MarkowitzResult:
    """Orquesta el cálculo completo de la frontera eficiente."""
    return MarkowitzResult(
        tickers=tickers,
        risk_free_rate=risk_free_rate,
        min_variance=min_variance_portfolio(tickers, mu, sigma, risk_free_rate, max_weight),
        max_sharpe=max_sharpe_portfolio(tickers, mu, sigma, risk_free_rate, max_weight),
        frontier=efficient_frontier(tickers, mu, sigma, risk_free_rate, frontier_points, max_weight),
        monte_carlo=monte_carlo(tickers, mu, sigma, risk_free_rate, monte_carlo_samples),
    )
