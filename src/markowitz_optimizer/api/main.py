"""
FastAPI app — orquesta el pipeline completo:

  holdings → datos de mercado (yfinance) → frontera eficiente de Markowitz
           → framework de riesgo de 10 dimensiones → análisis con IA (Claude)

Sirve también el frontend estático en `/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from markowitz_optimizer.ai import analyst
from markowitz_optimizer.data.market_data import (
    MarketDataError,
    corp_live_info,
    fetch_asset_info,
    fetch_market_data,
)
from markowitz_optimizer.data.providers import SOURCES, fetch_risk_free_rate, universe
from markowitz_optimizer.engine import markowitz
from markowitz_optimizer.engine.backtest import backtest as run_backtest
from markowitz_optimizer.engine.diagnostics import run_diagnostics
from markowitz_optimizer.engine.backtest import walk_forward as run_walk_forward
from markowitz_optimizer.engine.black_litterman import black_litterman_returns as bl_returns
from markowitz_optimizer.risk.framework import assess_portfolio

from .schemas import AnalyzeRequest, AnalyzeResponse

load_dotenv()

app = FastAPI(
    title="Optimizador de Carteras Markowitz + IA",
    description="Frontera eficiente, framework de riesgo de 10 dimensiones y análisis con Claude.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/risk-free-rate")
def risk_free_rate(tenor: str = "3m") -> dict:
    """Tasa libre de riesgo en vivo (T-bill/Treasury EE.UU. vía yfinance)."""
    return fetch_risk_free_rate(tenor)


@app.get("/categories")
def categories() -> dict:
    """Categorías de instrumentos disponibles (para el selector del frontend)."""
    return {
        "categories": [
            {"value": k, "label": v["label"], "currency": v["currency"], "kind": v["kind"]}
            for k, v in SOURCES.items()
        ]
    }


@app.get("/universe")
def universe_endpoint(category: str) -> dict:
    """Lista de instrumentos disponibles para una categoría."""
    if category not in SOURCES:
        raise HTTPException(status_code=422, detail=f"Categoría inválida: {category}")
    return {"category": category, "instruments": universe(category)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    warnings: list[str] = []

    # 0. Tasa libre de riesgo: la enviada, o la tasa en vivo (T-bill EE.UU.).
    if req.risk_free_rate is not None:
        rf = req.risk_free_rate
        rf_info = {"rate": rf, "percent": round(rf * 100, 3), "source": "definida por el usuario"}
    else:
        rf_info = fetch_risk_free_rate()
        rf = rf_info["rate"]

    # 1. Datos de mercado reales (multi-fuente: yfinance US/.BA + data912/PPI ARG).
    instruments = [{"symbol": h.ticker, "source": h.source} for h in req.holdings]
    try:
        md = fetch_market_data(
            instruments, period=req.period,
            normalize_currency=req.normalize_currency, fx_kind=req.fx_kind,
            cov_method=req.cov_method,
        )
    except MarketDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    valid = md.tickers
    if len(valid) < 2:
        raise HTTPException(status_code=422, detail="Se necesitan al menos 2 instrumentos con histórico.")

    # 2. Pesos actuales. Si se normalizó, los montos en ARS se pasan a USD con el
    #    FX actual para que los pesos sean consistentes (un monto ARS y uno USD no
    #    son comparables en crudo). Solo se optimiza lo válido.
    fx_latest = md.fx.get("latest") if (md.fx and md.fx.get("applied")) else None

    def to_base(symbol: str, val: float) -> float:
        if fx_latest and md.currencies.get(symbol) == "ARS":
            return val / fx_latest
        return val

    value_by = {h.ticker: to_base(h.ticker, h.value) for h in req.holdings}
    invested = sum(value_by.get(t, 0.0) for t in valid)
    if invested <= 0:
        raise HTTPException(status_code=422, detail="El valor total invertido debe ser > 0.")
    total = invested
    weights = {t: value_by.get(t, 0.0) / invested for t in valid}

    # 3. Descartados + info de ONs (BYMA) aunque no entren al optimizador.
    dropped_corp = [d["symbol"] for d in md.dropped if d["source"] == "arg_corp"]
    on_info = corp_live_info(dropped_corp) if dropped_corp else {}
    for d in md.dropped:
        warnings.append(f"{d['symbol']} ({d['source']}) fuera del optimizador: {d['reason']}")
    if md.cov and md.cov.get("method") == "ledoit_wolf" and md.cov.get("shrinkage") is not None:
        warnings.append(
            f"Covarianza estimada con shrinkage de Ledoit-Wolf (λ={md.cov['shrinkage']:.2f}): "
            "regulariza la matriz muestral para un óptimo más estable."
        )
    if md.window and md.window.get("limited_by"):
        lb = ", ".join(md.window["limited_by"])
        warnings.append(
            f"Ventana recortada: el análisis arranca el {md.window['limiting_start']} "
            f"({md.window['effective_days']} días) por el historial más corto de {lb}. "
            "Quitá ese activo o usá un período acorde para una ventana más larga."
        )
    if md.normalized and md.fx:
        warnings.append(
            f"Activos ARG convertidos a USD con {md.fx['label']} "
            f"(${md.fx['latest']} al {md.fx['as_of']}). Todo se valúa en USD."
        )
    elif md.mixed_currency:
        warnings.append(
            "Cartera con monedas mixtas (ARS y USD) SIN normalizar. Los retornos están "
            "en moneda nativa: los activos en ARS aparecen inflados por la devaluación. "
            "Activá la normalización a USD para una comparación correcta."
        )
    elif md.fx and not md.fx.get("applied"):
        warnings.append(
            f"No se pudo normalizar a USD ({md.fx.get('reason', 'FX no disponible')}). "
            "Se usan retornos en moneda nativa."
        )

    # 3b. Vectores alineados para el optimizador.
    mu_hist = np.array([md.mean_returns[t] for t in valid])
    sigma = np.array([[md.cov_matrix.loc[a, b] for b in valid] for a in valid])

    # 3c. Retorno esperado μ. Black-Litterman (default) regulariza la media
    #     histórica hacia un prior de equilibrio; "historical" la usa tal cual.
    if req.return_method == "black_litterman":
        mu = bl_returns(sigma, mu_hist, rf=rf, view_confidence=req.bl_view_confidence)
        returns_info = {"method": "black_litterman", "view_confidence": req.bl_view_confidence}
    else:
        mu = mu_hist
        returns_info = {"method": "historical"}

    # 4. Frontera eficiente. El tope por activo no puede ser menor que 1/N
    #    (si no, los pesos no pueden sumar 1); lo subimos y avisamos.
    max_weight = req.max_weight
    floor = 1.0 / len(valid)
    if max_weight < floor:
        warnings.append(
            f"El tope por activo ({max_weight*100:.0f}%) es menor que 1/N "
            f"({floor*100:.0f}%) y haría imposible invertir el 100%. Se subió a {floor*100:.0f}%."
        )
        max_weight = floor
    mk = markowitz.optimize(valid, mu, sigma, risk_free_rate=rf, max_weight=max_weight)
    mk_dict = mk.to_dict()

    # 5. Métricas de la cartera actual.
    wv = np.array([weights[t] for t in valid])
    cur_ret = markowitz.portfolio_return(wv, mu)
    cur_vol = markowitz.portfolio_volatility(wv, sigma)
    cur_sharpe = markowitz.portfolio_sharpe(wv, mu, sigma, rf)
    current_portfolio = {
        "weights": {t: round(weights[t], 6) for t in valid},
        "total_value": round(total, 2),
        "total_currency": "USD" if md.normalized else "nativa",
        "expected_return": round(cur_ret, 6),
        "volatility": round(cur_vol, 6),
        "sharpe": round(cur_sharpe, 6),
    }

    # 6. Framework de riesgo (sobre la cartera actual).
    asset_info = fetch_asset_info(md)
    risk_assess = assess_portfolio(weights, md.daily_returns, md.corr_matrix, asset_info)
    risk_dict = risk_assess.to_dict()

    # 6b. Diagnóstico de supuestos: testea normalidad / iid / random walk / ARCH
    #     sobre los retornos ANTES de creerle a la anualización, el Sharpe y el VaR.
    diagnostics = run_diagnostics(weights, md.daily_returns).to_dict()

    # 7. Rebalanceo hacia máximo Sharpe: deltas de peso + montos concretos.
    #    Los montos están en la base del cómputo (USD si se normalizó).
    target = mk.max_sharpe.weights
    base_ccy = "USD" if md.normalized else "nativa"
    actions = {}
    for t in valid:
        cur_val = value_by.get(t, 0.0)
        tgt_val = target.get(t, 0.0) * invested
        delta_val = tgt_val - cur_val
        act = {
            "current_value": round(cur_val, 2),
            "target_value": round(tgt_val, 2),
            "delta_value": round(delta_val, 2),
            "action": "comprar" if delta_val > 0.01 else ("vender" if delta_val < -0.01 else "mantener"),
            "native_currency": md.currencies.get(t, "USD"),
        }
        # Montos en moneda nativa: los activos ARG se valúan en ARS con el FX actual,
        # así no tenés que reconvertir a mano (ej. "comprá US$X ≈ $Y ARS de YPFD").
        if md.currencies.get(t) == "ARS" and fx_latest:
            act["current_native"] = round(cur_val * fx_latest, 0)
            act["target_native"] = round(tgt_val * fx_latest, 0)
            act["delta_native"] = round(delta_val * fx_latest, 0)
        actions[t] = act
    rebalancing = {
        "target": "max_sharpe",
        "currency": base_ccy,
        "fx_rate": fx_latest,
        "invested": round(invested, 2),
        "deltas": {
            t: round(target.get(t, 0.0) - weights.get(t, 0.0), 6) for t in valid
        },
        "actions": actions,
        "improvement": {
            "expected_return": round(mk.max_sharpe.expected_return - cur_ret, 6),
            "volatility": round(mk.max_sharpe.volatility - cur_vol, 6),
            "sharpe": round(mk.max_sharpe.sharpe - cur_sharpe, 6),
        },
    }

    # 7b. Backtest a peso constante: actual vs óptimo sobre la ventana.
    backtest = run_backtest(
        md.daily_returns,
        {
            "current": weights,
            "max_sharpe": mk.max_sharpe.weights,
            "min_variance": mk.min_variance.weights,
        },
        rf=rf,
    )

    # 7c. Backtest out-of-sample (walk-forward): el contraste honesto.
    walk_forward = run_walk_forward(
        md.daily_returns, weights, rf=rf, max_weight=max_weight, cov_method=req.cov_method,
        return_method=req.return_method, bl_view_confidence=req.bl_view_confidence,
    )

    # 8. Análisis con IA (o mock determinístico sin API key).
    market_summary = {
        "asset_info": asset_info,
        "mean_returns": {t: round(float(md.mean_returns[t]), 6) for t in valid},
        "volatility": {t: round(float(md.volatility[t]), 6) for t in valid},
    }
    if req.run_ai:
        ai = analyst.analyze_portfolio(
            portfolio=current_portfolio,
            market_summary=market_summary,
            markowitz=mk_dict,
            risk=risk_dict,
        ).to_dict()
    else:
        ai = {"skipped": True}

    return AnalyzeResponse(
        tickers=valid,
        period={
            "start": md.start, "end": md.end, "window": req.period,
            "dropped": md.dropped,
            "currencies": md.currencies,
            "effective": md.window,
        },
        market={
            "mean_returns": market_summary["mean_returns"],
            "volatility": market_summary["volatility"],
            "correlation": md.to_dict()["corr_matrix"],
            "asset_info": asset_info,
        },
        current_portfolio=current_portfolio,
        markowitz=mk_dict,
        risk=risk_dict,
        diagnostics=diagnostics,
        rebalancing=rebalancing,
        backtest=backtest,
        walk_forward=walk_forward,
        ai_analysis=ai,
        risk_free=rf_info,
        fx=md.fx or {},
        covariance=md.cov or {},
        returns=returns_info,
        warnings=warnings,
        on_info=on_info,
    )


# Frontend estático (montado al final para no pisar /analyze, /health).
if _FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")
