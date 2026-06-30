"""
providers — fuentes de datos de mercado (híbrido ARG + US).

Dos proveedores detrás de una interfaz común que devuelve una serie de cierres
diarios (pd.Series indexada por fecha):

  - yfinance  : activos US (AAPL) y BYMA vía sufijo .BA (GGAL.BA, AAPL.BA).
  - data912   : API argentina gratuita (sin key). Histórico de acciones ARG,
                CEDEARs y bonos soberanos; snapshot en vivo de todo (incl.
                corporativos/ONs y MEP).

`SOURCES` enumera los orígenes soportados por instrumento. Los bonos corporativos
no tienen histórico gratuito (solo live) → no entran al optimizador.

Nota de moneda: los activos ARG cotizan en ARS y los US en USD. Acá las series
son en su moneda nativa; el optimizador trabaja en espacio de RETORNOS (cambios
porcentuales), una simplificación estándar que ignora el co-movimiento del FX.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import time
from dataclasses import dataclass

import pandas as pd
import requests
import urllib3
import yfinance as yf

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA912_BASE = "https://data912.com"
BYMA_BASE = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"
PPI_BASE = "https://clientapi.portfoliopersonal.com"
ARGENTINADATOS_BASE = "https://api.argentinadatos.com/v1/cotizaciones/dolares"
RAVA_BASE = "https://www.rava.com"
_HTTP_TIMEOUT = 25
_HEADERS = {"User-Agent": "markowitz-optimizer/0.1"}

# source -> (proveedor, categoría/handler)
# Ojo: data912 usa nombres distintos para histórico (`d912`) y live (`d912_live`):
#   histórico → /historical/stocks/SYM   live → /live/arg_stocks
SOURCES: dict[str, dict] = {
    "us":        {"label": "US (yfinance)", "currency": "USD", "kind": "equity"},
    "arg_stock": {"label": "Acción ARG", "currency": "ARS", "kind": "equity",
                  "d912": "stocks", "d912_live": "arg_stocks", "yf_suffix": ".BA"},
    "arg_cedear": {"label": "CEDEAR ARG", "currency": "ARS", "kind": "equity",
                   "d912": "cedears", "d912_live": "arg_cedears", "yf_suffix": ".BA"},
    "arg_bond":  {"label": "Bono soberano ARG", "currency": "ARS", "kind": "bond",
                  "d912": "bonds", "d912_live": "arg_bonds"},
    "arg_corp":  {"label": "Bono corporativo ARG (ON)", "currency": "ARS", "kind": "bond",
                  "d912": "corp", "d912_live": "arg_corp"},  # sin histórico gratuito → solo live
}

_PERIOD_DAYS = {"1y": 365, "3y": 1095, "5y": 1825, "max": None}

# Universo US curado (yfinance no lista todo; ofrecemos lo relevante). Long-only.
_US_UNIVERSE: list[dict] = [
    # Bonos / renta fija US (vía ETF)
    {"symbol": "BIL", "name": "T-Bills 1-3 meses (ETF)", "group": "Renta Fija"},
    {"symbol": "SHY", "name": "Treasury 1-3 años (ETF)", "group": "Renta Fija"},
    {"symbol": "IEF", "name": "Treasury 7-10 años (ETF)", "group": "Renta Fija"},
    {"symbol": "TLT", "name": "Treasury 20+ años (ETF)", "group": "Renta Fija"},
    {"symbol": "GOVT", "name": "US Treasury Bond (ETF)", "group": "Renta Fija"},
    {"symbol": "TIP", "name": "Treasury indexado a inflación (ETF)", "group": "Renta Fija"},
    {"symbol": "AGG", "name": "US Aggregate Bond (ETF)", "group": "Renta Fija"},
    {"symbol": "BND", "name": "Total Bond Market (ETF)", "group": "Renta Fija"},
    {"symbol": "LQD", "name": "Corporativos Investment Grade (ETF)", "group": "Renta Fija"},
    {"symbol": "HYG", "name": "Corporativos High Yield (ETF)", "group": "Renta Fija"},
    {"symbol": "EMB", "name": "Bonos emergentes en USD (ETF)", "group": "Renta Fija"},
    # Índices / ETF amplios US
    {"symbol": "SPY", "name": "S&P 500", "group": "Índices"},
    {"symbol": "VOO", "name": "S&P 500 (Vanguard)", "group": "Índices"},
    {"symbol": "RSP", "name": "S&P 500 Equal Weight", "group": "Índices"},
    {"symbol": "QQQ", "name": "Nasdaq 100", "group": "Índices"},
    {"symbol": "VTI", "name": "Total US Market", "group": "Índices"},
    {"symbol": "DIA", "name": "Dow Jones 30", "group": "Índices"},
    {"symbol": "IWM", "name": "Russell 2000 (small caps)", "group": "Índices"},
    {"symbol": "MDY", "name": "S&P MidCap 400", "group": "Índices"},
    {"symbol": "VUG", "name": "US Growth", "group": "Índices"},
    {"symbol": "VTV", "name": "US Value", "group": "Índices"},
    {"symbol": "SCHD", "name": "US Dividendos", "group": "Índices"},
    # Internacional
    {"symbol": "VT", "name": "Acciones mundo total", "group": "Internacional"},
    {"symbol": "ACWI", "name": "MSCI All-Country World", "group": "Internacional"},
    {"symbol": "VEA", "name": "Desarrollados ex-US", "group": "Internacional"},
    {"symbol": "EFA", "name": "MSCI EAFE (desarrollados)", "group": "Internacional"},
    {"symbol": "VWO", "name": "Emergentes (Vanguard)", "group": "Internacional"},
    {"symbol": "EEM", "name": "MSCI Emergentes", "group": "Internacional"},
    # Sectores US (SPDR Select + temáticos)
    {"symbol": "XLK", "name": "Tecnología (ETF)", "group": "Sectores"},
    {"symbol": "XLE", "name": "Energía (ETF)", "group": "Sectores"},
    {"symbol": "XLF", "name": "Financiero (ETF)", "group": "Sectores"},
    {"symbol": "XLV", "name": "Salud (ETF)", "group": "Sectores"},
    {"symbol": "XLI", "name": "Industrial (ETF)", "group": "Sectores"},
    {"symbol": "XLP", "name": "Consumo defensivo (ETF)", "group": "Sectores"},
    {"symbol": "XLY", "name": "Consumo discrecional (ETF)", "group": "Sectores"},
    {"symbol": "XLU", "name": "Utilities (ETF)", "group": "Sectores"},
    {"symbol": "XLB", "name": "Materiales (ETF)", "group": "Sectores"},
    {"symbol": "XLRE", "name": "Real Estate (ETF)", "group": "Sectores"},
    {"symbol": "XLC", "name": "Comunicaciones (ETF)", "group": "Sectores"},
    {"symbol": "SMH", "name": "Semiconductores (ETF)", "group": "Sectores"},
    {"symbol": "VNQ", "name": "REITs US (ETF)", "group": "Sectores"},
    # Commodities
    {"symbol": "GLD", "name": "Oro (ETF)", "group": "Commodities"},
    {"symbol": "SLV", "name": "Plata (ETF)", "group": "Commodities"},
    {"symbol": "DBC", "name": "Commodities diversificado (ETF)", "group": "Commodities"},
    # Acciones líderes US
    {"symbol": "AAPL", "name": "Apple", "group": "Acciones"},
    {"symbol": "MSFT", "name": "Microsoft", "group": "Acciones"},
    {"symbol": "NVDA", "name": "NVIDIA", "group": "Acciones"},
    {"symbol": "AMZN", "name": "Amazon", "group": "Acciones"},
    {"symbol": "GOOGL", "name": "Alphabet", "group": "Acciones"},
    {"symbol": "META", "name": "Meta", "group": "Acciones"},
    {"symbol": "TSLA", "name": "Tesla", "group": "Acciones"},
    {"symbol": "AVGO", "name": "Broadcom", "group": "Acciones"},
    {"symbol": "AMD", "name": "AMD", "group": "Acciones"},
    {"symbol": "NFLX", "name": "Netflix", "group": "Acciones"},
    {"symbol": "ORCL", "name": "Oracle", "group": "Acciones"},
    {"symbol": "CRM", "name": "Salesforce", "group": "Acciones"},
    {"symbol": "JPM", "name": "JPMorgan", "group": "Acciones"},
    {"symbol": "BAC", "name": "Bank of America", "group": "Acciones"},
    {"symbol": "V", "name": "Visa", "group": "Acciones"},
    {"symbol": "MA", "name": "Mastercard", "group": "Acciones"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway B", "group": "Acciones"},
    {"symbol": "LLY", "name": "Eli Lilly", "group": "Acciones"},
    {"symbol": "UNH", "name": "UnitedHealth", "group": "Acciones"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "group": "Acciones"},
    {"symbol": "ABBV", "name": "AbbVie", "group": "Acciones"},
    {"symbol": "MRK", "name": "Merck", "group": "Acciones"},
    {"symbol": "WMT", "name": "Walmart", "group": "Acciones"},
    {"symbol": "COST", "name": "Costco", "group": "Acciones"},
    {"symbol": "PG", "name": "Procter & Gamble", "group": "Acciones"},
    {"symbol": "KO", "name": "Coca-Cola", "group": "Acciones"},
    {"symbol": "PEP", "name": "PepsiCo", "group": "Acciones"},
    {"symbol": "MCD", "name": "McDonald's", "group": "Acciones"},
    {"symbol": "HD", "name": "Home Depot", "group": "Acciones"},
    {"symbol": "XOM", "name": "ExxonMobil", "group": "Acciones"},
    {"symbol": "CVX", "name": "Chevron", "group": "Acciones"},
    {"symbol": "DIS", "name": "Disney", "group": "Acciones"},
]


@dataclass
class PriceSeries:
    symbol: str
    source: str
    currency: str
    kind: str            # equity | bond
    close: pd.Series     # cierres diarios indexados por fecha (DatetimeIndex)


class ProviderError(RuntimeError):
    pass


_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^]{1,15}$")


def _safe_symbol(symbol: str) -> str:
    """
    Normaliza y valida un símbolo antes de meterlo en una URL/parámetro.

    Acepta solo [A-Z0-9.-^] (tickers reales). Rechaza cualquier otra cosa para
    evitar que entradas raras del usuario alteren las requests (path traversal,
    inyección de parámetros, etc.). Devuelve el símbolo en mayúsculas.
    """
    s = str(symbol or "").strip().upper()
    if not _SYMBOL_RE.match(s):
        raise ProviderError(f"Símbolo inválido: {symbol!r}")
    return s


def _cutoff(period: str) -> _dt.date | None:
    days = _PERIOD_DAYS.get(period, 1095)
    if days is None:
        return None
    return _dt.date.today() - _dt.timedelta(days=days)


# ---------------------------------------------------------------------------
# data912 (Argentina)
# ---------------------------------------------------------------------------

def data912_history(symbol: str, category: str, period: str) -> pd.Series:
    """Cierres diarios de data912: GET /historical/{category}/{symbol}."""
    symbol = _safe_symbol(symbol)
    url = f"{DATA912_BASE}/historical/{category}/{symbol}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise ProviderError(f"data912 no respondió para {symbol}: {exc}") from exc
    if resp.status_code == 404:
        raise ProviderError(f"data912 sin histórico para {category}/{symbol}.")
    if not resp.ok:
        raise ProviderError(f"data912 error {resp.status_code} para {symbol}.")
    try:
        rows = resp.json()
    except ValueError as exc:
        raise ProviderError(f"data912 respondió formato no-JSON para {symbol}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ProviderError(f"data912 sin histórico utilizable para {symbol}.")

    # Símbolos inválidos pueden devolver 200 con una forma inesperada (no dicts):
    # parseamos defensivamente y, si no hay puntos, lo tratamos como sin datos.
    pts = {
        pd.Timestamp(r["date"]): float(r["c"])
        for r in rows
        if isinstance(r, dict) and r.get("date") and r.get("c")
    }
    if not pts:
        raise ProviderError(f"data912 sin datos para {symbol}.")

    s = pd.Series(pts, name=symbol).sort_index()
    cut = _cutoff(period)
    if cut is not None:
        s = s[s.index >= pd.Timestamp(cut)]
    return s[s > 0]


_live_cache: dict[str, tuple[float, list]] = {}
_LIVE_TTL = 120


def data912_live(category: str) -> list[dict]:
    """Snapshot en vivo de una categoría (para metadatos/liquidez), cacheado 2 min."""
    cached = _live_cache.get(category)
    if cached and (time.time() - cached[0]) < _LIVE_TTL:
        return cached[1]
    url = f"{DATA912_BASE}/live/{category}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
    except requests.RequestException:
        data = []
    _live_cache[category] = (time.time(), data)
    return data


def universe(category: str) -> list[dict]:
    """
    Lista de instrumentos disponibles por categoría, para que el usuario elija
    desde el universo correcto en vez de tipear tickers a ciegas.

    US: lista curada. ARG: símbolos en vivo (data912 / BYMA), ordenados por
    volumen, con precio cuando está disponible.
    """
    if category == "us":
        return list(_US_UNIVERSE)

    if category == "arg_corp":
        rows = sorted(_byma_corp_snapshot().values(),
                      key=lambda r: -(r.get("tradeVolume") or 0))
        return [
            {"symbol": r["symbol"], "name": r.get("symbol"),
             "price": r.get("closingPrice") or r.get("previousClosingPrice"),
             "currency": r.get("denominationCcy", "ARS"),
             "maturity": r.get("maturityDate")}
            for r in rows if r.get("symbol")
        ]

    meta = SOURCES.get(category)
    cat = meta.get("d912_live") if meta else None
    if not cat:
        return []
    rows = sorted(data912_live(cat), key=lambda r: -(r.get("v") or 0))
    return [
        {"symbol": r["symbol"], "name": r.get("symbol"), "price": r.get("c")}
        for r in rows if r.get("symbol")
    ]


# ---------------------------------------------------------------------------
# Rava Bursátil — histórico de ONs (y otros) gratis vía /api/chart-history
# ---------------------------------------------------------------------------
_rava_session: requests.Session | None = None
_RAVA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _rava_get_session() -> requests.Session:
    global _rava_session
    if _rava_session is None:
        s = requests.Session()
        s.headers.update(_RAVA_HEADERS)
        try:
            s.get(RAVA_BASE + "/", timeout=_HTTP_TIMEOUT)  # cookies (SSL verificado)
        except requests.RequestException:
            pass
        _rava_session = s
    return _rava_session


def rava_history(symbol: str, period: str) -> pd.Series:
    """
    Histórico de cierres diarios (ARS) de Rava para un símbolo (ej. una ON).

    Usa POST /api/chart-history (especie=SYM). La profundidad depende de la
    antigüedad del instrumento. Lanza ProviderError si falla o no hay datos.
    """
    sym = _safe_symbol(symbol)
    s = _rava_get_session()
    try:
        resp = s.post(
            RAVA_BASE + "/api/chart-history",
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Referer": f"{RAVA_BASE}/perfil/{sym}"},
            data={"especie": sym}, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise ProviderError(f"Rava no respondió para {sym}: {exc}") from exc

    rows = payload.get("body") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ProviderError(f"Rava sin histórico para {sym}.")

    pts = {}
    for r in rows:
        fecha, cierre = r.get("fecha"), r.get("cierre")
        if fecha and cierre:
            pts[pd.Timestamp(fecha)] = float(cierre)
    if not pts:
        raise ProviderError(f"Rava: formato inesperado para {sym}.")

    s_close = pd.Series(pts, name=sym).sort_index()
    cut = _cutoff(period)
    if cut is not None:
        s_close = s_close[s_close.index >= pd.Timestamp(cut)]
    return s_close[s_close > 0]


# ---------------------------------------------------------------------------
# yfinance (US + .BA)
# ---------------------------------------------------------------------------

_RF_TICKERS = {
    "3m": ("^IRX", "T-bill EE.UU. 13 semanas"),
    "5y": ("^FVX", "Treasury EE.UU. 5 años"),
    "10y": ("^TNX", "Treasury EE.UU. 10 años"),
}
_rf_cache: dict[str, tuple[float, dict]] = {}


def fetch_risk_free_rate(tenor: str = "3m") -> dict:
    """
    Tasa libre de riesgo en vivo desde yfinance (yields de Treasuries EE.UU.).

    Por defecto el T-bill a 13 semanas (^IRX), el proxy estándar para el ratio de
    Sharpe. Devuelve {rate(decimal), percent, source, as_of, tenor}. Cachea 1h.
    Si falla, cae a 0.04 con source='fallback'.
    """
    cached = _rf_cache.get(tenor)
    if cached and (time.time() - cached[0]) < 3600:
        return cached[1]

    ticker, label = _RF_TICKERS.get(tenor, _RF_TICKERS["3m"])
    result = {"rate": 0.04, "percent": 4.0, "source": "fallback (yfinance no disponible)",
              "as_of": None, "tenor": tenor, "ticker": ticker}
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                         progress=False, auto_adjust=False, threads=False)
        if df is not None and not df.empty:
            close = df["Close"].dropna()
            pct = float(close.iloc[-1].item())
            result = {
                "rate": round(pct / 100.0, 5), "percent": round(pct, 3),
                "source": f"{label} ({ticker}, yfinance)",
                "as_of": str(close.index[-1].date()), "tenor": tenor, "ticker": ticker,
            }
    except Exception:  # noqa: BLE001
        pass
    _rf_cache[tenor] = (time.time(), result)
    return result


# Tipos de cambio USD/ARS (argentinadatos): CCL es el estándar para valuar
# activos invertibles en USD; MEP (bolsa) es la alternativa.
FX_KINDS = {
    "ccl": ("contadoconliqui", "Dólar CCL (contado con liquidación)"),
    "mep": ("bolsa", "Dólar MEP (bolsa)"),
    "blue": ("blue", "Dólar blue"),
    "oficial": ("oficial", "Dólar oficial"),
    "mayorista": ("mayorista", "Dólar mayorista"),
}
_fx_cache: dict[str, tuple[float, pd.Series]] = {}


def usd_ars_history(kind: str = "ccl", period: str = "max") -> pd.Series:
    """
    Serie histórica USD/ARS (ARS por dólar), precio medio compra/venta.

    Por defecto CCL (contado con liquidación), el tipo de cambio para convertir
    activos argentinos invertibles a USD. Cacheada 1h. Lanza ProviderError si falla.
    """
    casa = FX_KINDS.get(kind, FX_KINDS["ccl"])[0]
    cached = _fx_cache.get(casa)
    if cached and (time.time() - cached[0]) < 3600:
        s = cached[1]
    else:
        url = f"{ARGENTINADATOS_BASE}/{casa}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            rows = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"No se pudo obtener USD/ARS ({casa}): {exc}") from exc
        pts = {}
        for r in rows:
            v, b = r.get("venta"), r.get("compra")
            if r.get("fecha") and (v or b):
                mid = (float(v or b) + float(b or v)) / 2.0
                if mid > 0:
                    pts[pd.Timestamp(r["fecha"])] = mid
        if not pts:
            raise ProviderError(f"USD/ARS ({casa}) sin datos utilizables.")
        s = pd.Series(pts, name=f"USDARS_{kind}").sort_index()
        _fx_cache[casa] = (time.time(), s)

    cut = _cutoff(period)
    if cut is not None:
        s = s[s.index >= pd.Timestamp(cut)]
    return s


_BENCHMARK_TICKER = "SPY"  # proxy de mercado (S&P 500, USD) para β tipo CAPM
_benchmark_cache: dict[tuple[str, str], tuple[float, "pd.Series"]] = {}


def benchmark_close(period: str = "3y", ticker: str = _BENCHMARK_TICKER) -> pd.Series:
    """
    Cierres diarios del benchmark de mercado (S&P 500 vía SPY, en USD).

    Se usa para estimar β tipo CAPM (Sharpe, 1964) por regresión de los retornos
    de cada activo contra el mercado, de forma uniforme para activos US y ARG
    (estos últimos ya normalizados a USD). Cacheado 1h. Lanza ProviderError si
    no hay datos. Lo decide el caller cómo manejar el fallo.
    """
    key = (ticker, period)
    cached = _benchmark_cache.get(key)
    if cached and (time.time() - cached[0]) < 3600:
        return cached[1].copy()
    close = yfinance_history(ticker, period)
    _benchmark_cache[key] = (time.time(), close)
    return close.copy()


def yfinance_history(symbol: str, period: str) -> pd.Series:
    """Cierres diarios ajustados de yfinance para un único símbolo."""
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(f"yfinance falló para {symbol}: {exc}") from exc
    if raw is None or raw.empty:
        raise ProviderError(f"yfinance sin datos para {symbol}.")
    col = "Close" if "Close" in raw.columns else "Adj Close"
    s = raw[col]
    if isinstance(s, pd.DataFrame):  # MultiIndex con un solo ticker
        s = s.iloc[:, 0]
    s = s.dropna()
    s.name = symbol
    return s[s > 0]


# ---------------------------------------------------------------------------
# Resolver unificado
# ---------------------------------------------------------------------------

_series_cache: dict[tuple[str, str, str], tuple[float, PriceSeries]] = {}
_SERIES_TTL = 600  # 10 min


def fetch_series(symbol: str, source: str, period: str) -> PriceSeries:
    """
    Wrapper con caché (TTL 10 min) sobre `_fetch_series_uncached`.

    Cachea la serie CRUDA por (símbolo, fuente, período) y devuelve un PriceSeries
    nuevo con una copia de la serie en cada llamada. Es importante copiar: la
    normalización a USD en market_data reescribe `ps.close`, y no queremos tocar
    el objeto cacheado. Acelera reanálisis (cambiar normalización, tope, etc.).
    """
    sym = _safe_symbol(symbol)
    key = (sym, source, period)
    hit = _series_cache.get(key)
    if hit and (time.time() - hit[0]) < _SERIES_TTL:
        ps = hit[1]
        return PriceSeries(ps.symbol, ps.source, ps.currency, ps.kind, ps.close.copy())
    ps = _fetch_series_uncached(sym, source, period)
    _series_cache[key] = (time.time(), ps)
    return PriceSeries(ps.symbol, ps.source, ps.currency, ps.kind, ps.close.copy())


def _fetch_series_uncached(symbol: str, source: str, period: str) -> PriceSeries:
    """
    Devuelve la serie de cierres para un instrumento según su `source`.

    ARG: intenta data912 (histórico) y, para acciones/CEDEARs, cae a yfinance .BA.
    US: yfinance directo. Lanza ProviderError si no hay histórico utilizable.
    """
    symbol = _safe_symbol(symbol)
    meta = SOURCES.get(source)
    if meta is None:
        raise ProviderError(f"source desconocido: {source!r}")

    if source == "us":
        close = yfinance_history(symbol, period)
        return PriceSeries(symbol, source, "USD", "equity", close)

    # Bonos corporativos (ONs): data912 no tiene histórico. Fuente primaria Rava
    # (gratis, histórico real); PPI como fallback si hay credenciales.
    if source == "arg_corp":
        err: Exception | None = None
        try:
            close = rava_history(symbol, period)
            if len(close) >= 20:
                return PriceSeries(symbol, source, meta["currency"], meta["kind"], close)
            err = ProviderError(f"{symbol}: Rava devolvió muy pocos datos.")
        except ProviderError as e:
            err = e
        if ppi_credentials_present():
            close = ppi_history(symbol, period)
            return PriceSeries(symbol, source, meta["currency"], meta["kind"], close)
        raise ProviderError(
            f"{symbol}: sin histórico de ON utilizable ({err}). "
            "Su precio/datos en vivo se muestran igual (BYMA)."
        )

    # Resto de Argentina: data912 primero.
    err: Exception | None = None
    cat = meta.get("d912")
    if cat:
        try:
            close = data912_history(symbol, cat, period)
            if len(close) >= 30:
                return PriceSeries(symbol, source, meta["currency"], meta["kind"], close)
            err = ProviderError(f"data912 con muy pocos datos para {symbol}.")
        except ProviderError as e:
            err = e

    # Fallback a yfinance .BA para acciones/CEDEARs (no aplica a bonos).
    suffix = meta.get("yf_suffix")
    if suffix:
        try:
            close = yfinance_history(symbol + suffix, period)
            return PriceSeries(symbol, source, meta["currency"], meta["kind"], close)
        except ProviderError as e:
            err = e

    raise ProviderError(str(err) if err else f"sin datos para {symbol} ({source}).")


# ---------------------------------------------------------------------------
# BYMA open data — snapshot de Obligaciones Negociables (ONs)
# ---------------------------------------------------------------------------

_byma_cache: dict[str, tuple[float, dict]] = {}
_BYMA_TTL = 300  # 5 min


def _byma_corp_snapshot() -> dict[str, dict]:
    """Snapshot de ONs de BYMA, cacheado: {symbol -> record}."""
    cached = _byma_cache.get("corp")
    if cached and (time.time() - cached[0]) < _BYMA_TTL:
        return cached[1]
    try:
        # verify=False: el server de BYMA presenta una cadena de certificados
        # incompleta y falla la verificación estándar. Es un endpoint PÚBLICO de
        # SOLO LECTURA (snapshot de precios de ONs) y NO se envían credenciales,
        # así que el riesgo de MITM se limita a datos de mercado informativos.
        resp = requests.post(
            f"{BYMA_BASE}/negociable-obligations",
            headers={**_HEADERS, "Content-Type": "application/json", "Accept": "application/json"},
            json={"excludeZeroPxAndQty": False, "T2": True, "T1": True, "T0": True},
            timeout=_HTTP_TIMEOUT, verify=False,  # noqa: S501 (ver nota arriba)
        )
        resp.raise_for_status()
        data = {r["symbol"]: r for r in resp.json() if r.get("symbol")}
    except (requests.RequestException, ValueError, KeyError):
        data = {}
    _byma_cache["corp"] = (time.time(), data)
    return data


def byma_on_info(symbol: str) -> dict | None:
    """Info en vivo de una ON desde BYMA (precio, vencimiento, volumen, moneda)."""
    rec = _byma_corp_snapshot().get(symbol.strip().upper())
    if not rec:
        return None
    last = rec.get("closingPrice") or rec.get("trade") or rec.get("previousClosingPrice")
    return {
        "symbol": rec.get("symbol"),
        "last_price": last,
        "previous_close": rec.get("previousClosingPrice"),
        "currency": rec.get("denominationCcy", "ARS"),
        "maturity_date": rec.get("maturityDate"),
        "days_to_maturity": rec.get("daysToMaturity"),
        "trade_volume": rec.get("tradeVolume"),
        "volume_amount": rec.get("volumeAmount"),
        "high": rec.get("tradingHighPrice"),
        "low": rec.get("tradingLowPrice"),
        "market": rec.get("market", "BYMA"),
    }


# ---------------------------------------------------------------------------
# PPI (Portfolio Personal Inversiones) — histórico de ONs (requiere credenciales)
# ---------------------------------------------------------------------------
# Credenciales por entorno: PPI_API_KEY, PPI_API_SECRET (de tu cuenta PPI).
# Headers de cliente PPI estándar (AuthorizedClient/ClientKey son fijos del SDK).
# Si tu integración usa otros valores, ajustá _PPI_CLIENT.

_PPI_CLIENT = {"AuthorizedClient": "API_CLI", "ClientKey": "pp_client"}
_ppi_token: dict[str, object] = {"token": None, "exp": 0.0}


def ppi_credentials_present() -> bool:
    return bool(os.environ.get("PPI_API_KEY") and os.environ.get("PPI_API_SECRET"))


def _ppi_login() -> str:
    now = time.time()
    if _ppi_token["token"] and now < float(_ppi_token["exp"]):
        return str(_ppi_token["token"])
    headers = {
        **_PPI_CLIENT,
        "Content-Type": "application/json",
        "ApiKey": os.environ["PPI_API_KEY"],
        "ApiSecret": os.environ["PPI_API_SECRET"],
    }
    resp = requests.post(
        f"{PPI_BASE}/api/1.0/Account/LoginApi", headers=headers, json={}, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("accessToken") or data.get("access_token")
    if not token:
        raise ProviderError("PPI: login sin accessToken.")
    _ppi_token["token"] = token
    _ppi_token["exp"] = now + 600  # refrescar conservadoramente cada 10 min
    return token


def ppi_history(symbol: str, period: str, instrument_type: str = "BONOS",
                settlement: str = "A-24") -> pd.Series:
    """
    Histórico de cierres de una ON vía PPI MarketData/Historical.

    instrument_type/settlement pueden variar según el instrumento; por defecto
    BONOS / A-24. Si PPI rechaza, probá ajustarlos. Requiere credenciales.
    """
    try:
        token = _ppi_login()
    except requests.RequestException as exc:
        raise ProviderError(f"PPI login falló: {exc}") from exc

    cut = _cutoff(period) or (_dt.date.today() - _dt.timedelta(days=3650))
    headers = {**_PPI_CLIENT, "Authorization": f"Bearer {token}"}
    params = {
        "ticker": symbol.strip().upper(),
        "type": instrument_type,
        "settlement": settlement,
        "dateFrom": cut.strftime("%m/%d/%Y"),
        "dateTo": _dt.date.today().strftime("%m/%d/%Y"),
    }
    try:
        resp = requests.get(
            f"{PPI_BASE}/api/1.0/MarketData/Historical",
            headers=headers, params=params, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise ProviderError(f"PPI histórico falló para {symbol}: {exc}") from exc

    if not rows:
        raise ProviderError(f"PPI sin datos para {symbol} ({instrument_type}/{settlement}).")

    points = {}
    for r in rows:
        date = r.get("date") or r.get("Date")
        price = r.get("price") or r.get("Price") or r.get("close") or r.get("settlementPrice")
        if date and price:
            points[pd.Timestamp(date)] = float(price)
    if not points:
        raise ProviderError(f"PPI: formato inesperado para {symbol}.")
    s = pd.Series(points, name=symbol).sort_index()
    return s[s > 0]
