"""
market_data — arma las estadísticas que necesita Markowitz a partir de
instrumentos de múltiples fuentes (US vía yfinance, ARG vía data912).

Cada instrumento entra como (symbol, source). Se descarga la serie de cierres de
cada uno, se alinean por fechas comunes (inner join), y se derivan:

  - mean_returns : retorno esperado anual por instrumento
  - cov_matrix   : covarianza anual
  - corr_matrix  : correlación
  - volatility   : desvío anual

Trabaja en espacio de RETORNOS (cambios %): los activos ARG (ARS) y US (USD) se
combinan sin normalizar FX (simplificación documentada; ver providers.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .providers import (
    FX_KINDS,
    SOURCES,
    PriceSeries,
    ProviderError,
    benchmark_close,
    byma_on_info,
    data912_live,
    fetch_series,
    usd_ars_history,
)

TRADING_DAYS: int = 252


class MarketDataError(RuntimeError):
    """Error al obtener o procesar datos de mercado."""


@dataclass
class MarketData:
    tickers: list[str]
    sources: dict[str, str]              # symbol -> source
    currencies: dict[str, str]           # symbol -> ARS/USD
    price_history: pd.DataFrame
    daily_returns: pd.DataFrame
    mean_returns: pd.Series
    volatility: pd.Series
    cov_matrix: pd.DataFrame
    corr_matrix: pd.DataFrame
    start: str
    end: str
    dropped: list[dict]                  # [{symbol, source, reason}]
    normalized: bool = False             # True si se convirtió todo a USD
    fx: dict | None = None               # info del tipo de cambio aplicado
    window: dict | None = None           # ventana efectiva + instrumentos que la limitan
    cov: dict | None = None              # método de covarianza + shrinkage (Ledoit-Wolf)

    @property
    def mixed_currency(self) -> bool:
        if self.normalized:
            return False
        return len(set(self.currencies.values())) > 1

    def to_dict(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "start": self.start, "end": self.end,
            "mean_returns": {t: float(self.mean_returns[t]) for t in self.tickers},
            "volatility": {t: float(self.volatility[t]) for t in self.tickers},
            "corr_matrix": {
                t: {u: float(self.corr_matrix.loc[t, u]) for u in self.tickers}
                for t in self.tickers
            },
            "cov_matrix": {
                t: {u: float(self.cov_matrix.loc[t, u]) for u in self.tickers}
                for t in self.tickers
            },
        }


def _estimate_cov(daily: "pd.DataFrame", method: str) -> tuple["pd.DataFrame", float | None]:
    """
    Covarianza ANUAL de los retornos diarios.

    - "ledoit_wolf" (default): shrinkage de Ledoit-Wolf. La covarianza muestral es
      ruidosa y mal condicionada con pocas observaciones (típico en ARG: ventanas
      cortas, ONs jóvenes), lo que produce carteras inestables. El shrinkage la
      regulariza hacia un target estructurado y mejora la estabilidad del óptimo.
      Devuelve también la intensidad de shrinkage λ ∈ [0,1].
    - "sample": covarianza muestral clásica (para comparar / docencia).
    """
    cols = list(daily.columns)
    if method == "sample":
        return daily.cov() * TRADING_DAYS, None
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(daily.to_numpy())
    cov_annual = lw.covariance_ * TRADING_DAYS
    return (pd.DataFrame(cov_annual, index=cols, columns=cols), float(lw.shrinkage_))


def _dedup(instruments: list[dict]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for it in instruments:
        sym = str(it.get("symbol", "")).strip().upper()
        src = str(it.get("source", "us")).strip().lower()
        if not sym or src not in SOURCES:
            continue
        key = (sym, src)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def fetch_market_data(
    instruments: list[dict],
    period: str = "3y",
    normalize_currency: bool = True,
    fx_kind: str = "ccl",
    cov_method: str = "ledoit_wolf",
) -> MarketData:
    """
    Descarga y alinea series de cierres para instrumentos multi-fuente.

    Args:
        instruments: lista de {"symbol": str, "source": str}. source ∈ SOURCES.
        period: ventana ("1y", "3y", "5y", "max").
        normalize_currency: si True, convierte los activos ARS a USD con el FX
            histórico (CCL por defecto), para que retornos/covarianzas sean
            consistentes y no estén sesgados por la devaluación del peso.
        fx_kind: tipo de cambio para la conversión (ccl | mep | blue | ...).

    Raises:
        MarketDataError: si quedan < 2 instrumentos con histórico utilizable.
    """
    pairs = _dedup(instruments)
    if len(pairs) < 2:
        raise MarketDataError("Se requieren al menos 2 instrumentos distintos.")

    series: list[PriceSeries] = []
    dropped: list[dict] = []
    for sym, src in pairs:
        try:
            series.append(fetch_series(sym, src, period))
        except ProviderError as exc:
            dropped.append({"symbol": sym, "source": src, "reason": str(exc)})

    if len(series) < 2:
        detail = "; ".join(f"{d['symbol']}: {d['reason']}" for d in dropped)
        raise MarketDataError(
            f"No hay suficientes instrumentos con histórico. Descartados: {detail}"
        )

    orig_currencies = {ps.symbol: ps.currency for ps in series}

    # Normalización a USD: convertir las series ARS dividiendo por el FX histórico.
    normalized = False
    fx_info: dict | None = None
    has_ars = any(ps.currency == "ARS" for ps in series)
    if normalize_currency and has_ars:
        try:
            fx = usd_ars_history(fx_kind, period)
            for ps in series:
                if ps.currency == "ARS":
                    aligned = fx.reindex(ps.close.index.union(fx.index)).ffill()
                    aligned = aligned.reindex(ps.close.index)
                    ps.close = (ps.close / aligned).dropna()
                    ps.currency = "USD"
            normalized = True
            fx_info = {
                "applied": True, "kind": fx_kind,
                "label": FX_KINDS.get(fx_kind, FX_KINDS["ccl"])[1],
                "latest": round(float(fx.iloc[-1]), 2),
                "as_of": str(fx.index[-1].date()),
            }
        except ProviderError as exc:
            fx_info = {"applied": False, "kind": fx_kind, "reason": str(exc)}

    # Alinear por fechas comunes.
    frame = pd.concat({ps.symbol: ps.close for ps in series}, axis=1)
    frame = frame.sort_index().ffill().dropna()
    if len(frame) < 30:
        # Demasiada poca intersección: quedarse con la mayor cobertura común.
        raise MarketDataError(
            "Las series no tienen suficiente solapamiento de fechas. "
            "Probá un período más largo o instrumentos con historiales similares."
        )

    cols = list(frame.columns)
    by_symbol = {ps.symbol: ps for ps in series}

    # Ventana efectiva: el join por fechas comunes arranca donde empieza el
    # instrumento más joven. Si eso recorta bastante, lo reportamos.
    starts = {ps.symbol: ps.close.index.min() for ps in series}
    latest_start = max(starts.values())
    earliest_start = min(starts.values())
    gap_days = (latest_start - earliest_start).days
    limited_by = [s for s, dt in starts.items() if (latest_start - dt).days <= 5]
    window = {
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
        "effective_days": (frame.index[-1] - frame.index[0]).days,
        "requested": period,
        "limited_by": limited_by if gap_days > 45 else [],
        "limiting_start": str(latest_start.date()) if gap_days > 45 else None,
    }

    daily = frame.pct_change().dropna(how="all").dropna()
    if daily.empty:
        raise MarketDataError("No se pudieron calcular retornos diarios.")

    mean_returns = daily.mean() * TRADING_DAYS
    cov_matrix, shrinkage = _estimate_cov(daily, cov_method)
    # Correlación derivada de la MISMA covarianza (consistente con el optimizador).
    d = np.sqrt(np.diag(cov_matrix.values))
    corr_vals = cov_matrix.values / np.outer(d, d)
    corr_matrix = pd.DataFrame(corr_vals, index=cols, columns=cols)
    volatility = pd.Series({t: float(np.sqrt(cov_matrix.loc[t, t])) for t in cols},
                           name="volatility")
    cov_info = {"method": cov_method, "shrinkage": shrinkage}

    return MarketData(
        tickers=cols,
        sources={t: by_symbol[t].source for t in cols},
        currencies={t: orig_currencies[t] for t in cols},
        price_history=frame,
        daily_returns=daily,
        mean_returns=mean_returns,
        volatility=volatility,
        cov_matrix=cov_matrix,
        corr_matrix=corr_matrix,
        start=str(frame.index[0].date()),
        end=str(frame.index[-1].date()),
        dropped=dropped,
        normalized=normalized,
        fx=fx_info,
        window=window,
        cov=cov_info,
    )


# ---------------------------------------------------------------------------
# Metadatos de activos (para el framework de riesgo)
# ---------------------------------------------------------------------------

def _market_betas(daily: pd.DataFrame, bench_close: pd.Series) -> dict[str, float]:
    """
    β tipo CAPM por activo = Cov(activo, mercado) / Var(mercado), estimada sobre
    los retornos diarios alineados al benchmark (S&P 500).

    Funciona igual para US y ARG: para activos argentinos normalizados a USD, la
    β mide su co-movimiento con el mercado global, que es lo que necesita el
    stress test (shock del benchmark × β). Devuelve {} si no hay solapamiento
    suficiente (< 30 días) o el mercado tiene varianza nula.
    """
    bench_ret = bench_close.pct_change().dropna()
    common = daily.index.intersection(bench_ret.index)
    if len(common) < 30:
        return {}
    b = bench_ret.loc[common].to_numpy()
    var_b = float(b.var(ddof=1))
    if var_b <= 0:
        return {}
    b_centered = b - b.mean()
    betas: dict[str, float] = {}
    for t in daily.columns:
        a = daily[t].loc[common].to_numpy()
        cov_ab = float((a - a.mean()) @ b_centered / (len(common) - 1))
        betas[t] = cov_ab / var_b
    return betas


def fetch_asset_info(md: MarketData) -> dict[str, dict]:
    """
    Metadatos por instrumento. US: yfinance.info; ARG: derivado del source +
    liquidez del snapshot live de data912.

    La β se estima por regresión contra el S&P 500 (uniforme US/ARG) y sobrescribe
    la de yfinance (que falta en ARG), para que el stress test tenga una β real
    por activo en vez de asumir 1.0.
    """
    info: dict[str, dict] = {}
    live_cache: dict[str, dict[str, dict]] = {}

    for sym in md.tickers:
        src = md.sources[sym]
        meta = SOURCES[src]
        if src == "us":
            info[sym] = _us_info(sym)
            continue

        # ONs: liquidez/vencimiento desde BYMA. Resto ARG: live de data912.
        if src == "arg_corp":
            on = byma_on_info(sym) or {}
            info[sym] = {
                "name": sym, "sector": "Renta Fija ARG (ON)", "country": "Argentina",
                "currency": on.get("currency", "ARS"), "quote_type": "BOND",
                "beta": None, "market_cap": None,
                "avg_volume": on.get("trade_volume"),
                "maturity_date": on.get("maturity_date"),
                "days_to_maturity": on.get("days_to_maturity"),
            }
            continue

        cat = meta.get("d912_live")
        if cat and cat not in live_cache:
            live_cache[cat] = {row.get("symbol", ""): row for row in data912_live(cat)}
        live = live_cache.get(cat, {}).get(sym, {})
        info[sym] = {
            "name": sym,
            "sector": "Renta Fija ARG" if meta["kind"] == "bond" else f"Renta Variable ARG ({meta['label']})",
            "country": "Argentina",
            "currency": meta["currency"],
            "quote_type": "BOND" if meta["kind"] == "bond" else "EQUITY",
            "beta": None,
            "market_cap": None,
            "avg_volume": float(live.get("v") or 0) or None,
        }

    # β tipo CAPM por regresión vs S&P 500 (uniforme US/ARG): sobrescribe la β de
    # yfinance con una estimada sobre la MISMA ventana del análisis. Si no hay red
    # ni benchmark, se conserva la β previa (yfinance en US, None en ARG → 1.0).
    try:
        period = (md.window or {}).get("requested") or "3y"
        betas = _market_betas(md.daily_returns, benchmark_close(period))
        for sym, beta in betas.items():
            if sym in info:
                info[sym]["beta"] = round(float(beta), 3)
                info[sym]["beta_source"] = "regresión vs S&P 500 (CAPM)"
    except Exception:  # noqa: BLE001 - sin red/benchmark: degradar sin romper el análisis
        pass

    return info


def corp_live_info(symbols: list[str]) -> dict[str, dict]:
    """Info en vivo de ONs (BYMA) para mostrar incluso si quedan fuera del optimizador."""
    out: dict[str, dict] = {}
    for s in symbols:
        on = byma_on_info(s)
        if on:
            out[s] = on
    return out


def _us_info(symbol: str) -> dict:
    import yfinance as yf  # local: solo si hay activos US

    meta = {
        "name": symbol, "sector": "Unknown", "country": "United States",
        "currency": "USD", "quote_type": "EQUITY", "beta": None,
        "market_cap": None, "avg_volume": None,
    }
    try:
        raw = yf.Ticker(symbol).info or {}
        qt = (raw.get("quoteType") or "EQUITY").upper()
        meta.update({
            "name": raw.get("longName") or raw.get("shortName") or symbol,
            "sector": raw.get("sector") or ("ETF" if qt == "ETF" else "Unknown"),
            "country": raw.get("country") or "United States",
            "currency": raw.get("currency") or "USD",
            "quote_type": qt,
            "beta": raw.get("beta"),
            "market_cap": raw.get("marketCap"),
            "avg_volume": raw.get("averageVolume") or raw.get("averageVolume10days"),
        })
    except Exception:  # noqa: BLE001
        pass
    return meta
