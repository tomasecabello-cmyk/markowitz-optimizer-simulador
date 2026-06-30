"""Schemas Pydantic para la API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


VALID_SOURCES = {"us", "arg_stock", "arg_cedear", "arg_bond", "arg_corp"}


class Holding(BaseModel):
    ticker: str = Field(..., description="Símbolo del activo, ej. AAPL o GGAL o AL30")
    value: float = Field(..., gt=0, description="Monto invertido")
    source: str = Field(
        "us",
        description="Fuente/tipo: us | arg_stock | arg_cedear | arg_bond | arg_corp",
    )

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker no puede estar vacío")
        return v

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        v = (v or "us").strip().lower()
        if v not in VALID_SOURCES:
            raise ValueError(f"source inválido: {v}. Use uno de {sorted(VALID_SOURCES)}")
        return v


class AnalyzeRequest(BaseModel):
    holdings: list[Holding] = Field(..., min_length=2, description="Al menos 2 activos")
    period: str = Field("3y", description="Ventana histórica yfinance: 1y, 3y, 5y, max")
    risk_free_rate: float | None = Field(
        None, ge=0, le=0.2,
        description="Tasa libre de riesgo anual (decimal). Si se omite, se usa la tasa en vivo (T-bill EE.UU.).",
    )
    normalize_currency: bool = Field(
        True, description="Convertir activos ARS a USD (FX histórico) para consistencia.")
    fx_kind: str = Field("ccl", description="Tipo de cambio: ccl | mep | blue | oficial | mayorista")
    max_weight: float = Field(
        1.0, gt=0, le=1.0,
        description="Tope de peso por activo en el optimizador (1.0 = sin tope). Ej. 0.35 = 35%.")
    cov_method: str = Field(
        "ledoit_wolf",
        description="Estimador de covarianza: ledoit_wolf (shrinkage, robusto) | sample.")
    return_method: str = Field(
        "black_litterman",
        description="Estimador de retorno esperado μ: black_litterman (robusto) | historical.")
    bl_view_confidence: float = Field(
        1.0, gt=0, le=100,
        description="Confianza en la media histórica como view de Black-Litterman (>1 acerca μ a la media; <1 al equilibrio).")
    run_ai: bool = Field(True, description="Ejecutar el análisis con IA")


class AnalyzeResponse(BaseModel):
    tickers: list[str]
    period: dict
    market: dict
    current_portfolio: dict
    markowitz: dict
    risk: dict
    rebalancing: dict
    diagnostics: dict = {}    # tests de supuestos (normalidad, iid, random walk, ARCH)
    backtest: dict = {}       # curvas de equity in-sample (actual vs óptimo)
    walk_forward: dict = {}   # backtest out-of-sample (walk-forward) + equal-weight
    ai_analysis: dict
    risk_free: dict = {}      # tasa libre de riesgo usada + fuente
    fx: dict = {}             # tipo de cambio aplicado (normalización a USD)
    covariance: dict = {}     # estimador de covarianza + shrinkage
    returns: dict = {}        # estimador de retorno esperado μ (BL / histórico)
    warnings: list[str] = []
    on_info: dict = {}        # info en vivo de ONs (BYMA), incl. las no optimizadas
