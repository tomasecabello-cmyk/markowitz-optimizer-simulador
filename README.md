# Optimizador de Carteras · Markowitz + IA

Optimizador de carteras **ARG + US** basado en la **frontera eficiente de Markowitz**,
con un **framework de riesgo de 10 dimensiones** y un **análisis con IA (Claude Opus 4.8)**.
Inspirado en el proyecto _"Optimizador de Portfolios con Markowitz e IA"_.

**Autores:** Tomás Emanuel Cabello · Lola Belén Lombardi

**Repositorio:** https://github.com/tomasecabello-cmyk/markowitz-optimizer (privado)

![CI](https://github.com/tomasecabello-cmyk/markowitz-optimizer/actions/workflows/ci.yml/badge.svg)

Entrada: tus posiciones (ticker + tipo de mercado + monto). Salida: frontera
eficiente, cartera de máximo Sharpe, matriz de correlación, score de riesgo en 10
dimensiones, propuesta de rebalanceo y un informe con IA.

### Fuentes de datos (híbrido ARG + US)

| Tipo (`source`)   | Activo                         | Fuente histórica            |
|-------------------|--------------------------------|-----------------------------|
| `us`              | Acciones/ETF US (AAPL, SPY)    | yfinance                    |
| `arg_stock`       | Acciones ARG (GGAL, YPFD)      | data912 (fallback .BA)      |
| `arg_cedear`      | CEDEARs (AAPL, etc.)           | data912 (fallback .BA)      |
| `arg_bond`        | Bonos soberanos (AL30, GD30)   | data912                     |
| `arg_corp`        | Obligaciones Negociables (ONs) | Rava (PPI fallback)         |

- **ONs**: el histórico sale de **Rava Bursátil** (gratis, vía `/api/chart-history`),
  así que **entran al optimizador** como cualquier otro activo (la profundidad
  depende de la antigüedad del bono). La info en vivo (precio/vencimiento/volumen)
  sigue saliendo de **BYMA**. Si Rava falla y hay `PPI_API_KEY`/`PPI_API_SECRET`,
  se usa PPI como respaldo.
- **Moneda (normalización a USD)**: los activos ARG cotizan en ARS y los US en USD.
  Por defecto se **convierte todo a USD** usando el dólar **CCL** histórico
  (`api.argentinadatos.com`, opciones: CCL/MEP/blue/oficial), tanto las series de
  precios (para μ/Σ) como los montos ingresados (para los pesos). Así los retornos
  ARG no quedan inflados por la devaluación del peso. Se puede desactivar; en ese
  caso se avisa que la comparación queda en moneda nativa.

> Demo educativo. **No es asesoramiento financiero.**

## Arquitectura

```
src/markowitz_optimizer/
  data/providers.py      # fuentes: yfinance (US/.BA), data912 (ARG), Rava (ON hist.), BYMA (ON info), argentinadatos (FX), PPI (fallback)
  data/market_data.py    # alinea instrumentos multi-fuente → retornos, μ, Σ, correlación, metadatos
  engine/markowitz.py    # frontera eficiente, Monte Carlo, máx Sharpe, mín varianza (scipy)
  risk/framework.py      # 10 dimensiones de riesgo (determinístico, función pura)
  ai/analyst.py          # análisis estilo Bridgewater con Claude Opus 4.8 (+ mock sin key)
  api/main.py            # FastAPI: POST /analyze orquesta todo + sirve el frontend
frontend/                # SPA con Plotly (frontera, heatmap, allocations, gauges, informe)
scripts/smoke.py         # test del pipeline sin red (datos sintéticos)
```

Flujo: `holdings → datos de mercado → Markowitz → framework de riesgo → IA`.
Cada capa es independiente; la IA **interpreta** los números, no los recalcula.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\maria\markowitz-optimizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env      # opcional: pegá tu ANTHROPIC_API_KEY para IA real
```

Sin `ANTHROPIC_API_KEY`, el análisis con IA usa un **mock determinístico** (mismo
formato) para que el demo corra igual.

## Correr

```powershell
# Opción 1 (recomendada para demo): doble-click a run.cmd, o desde la terminal:
.\run.cmd
# (run.cmd evita el bloqueo de scripts de PowerShell; abre el navegador solo)

# Si preferís el .ps1 directo y PowerShell lo bloquea, usá:
powershell -ExecutionPolicy Bypass -File .\run.ps1

# Opción 2 (manual): backend + frontend en http://127.0.0.1:8000 (Swagger en /docs)
python -m uvicorn markowitz_optimizer.api.main:app --reload
```

Abrí http://127.0.0.1:8000 en el navegador.

## Deploy

La app es **un solo servicio**: el backend FastAPI sirve la API y el frontend en
el mismo puerto (mismo origen, sin CORS). El frontend usa rutas relativas, así que
anda igual local y en producción.

```bash
# Docker local
docker build -t markowitz-optimizer .
docker run -p 8000:8000 markowitz-optimizer        # http://localhost:8000
```

En un PaaS (Render / Railway / Fly.io, todos con free tier):
1. Conectá el repo de GitHub y elegí build por **Docker** (hay `Dockerfile`).
2. El host inyecta `$PORT` (el contenedor ya lo respeta).
3. Opcional: seteá `ANTHROPIC_API_KEY` para el análisis con IA real.

Requiere salida a internet (yfinance, data912, Rava, BYMA, argentinadatos), que
está permitida en estos hosts. Ojo: al exponerlo, cualquiera con la URL puede
usarlo y disparar las llamadas externas (y la IA, si hay key).

## Tests

```powershell
python scripts/smoke.py        # pipeline matemático sin red (exit 0 = PASS)
python -m pytest -q            # si agregás tests bajo tests/
```

## API

`POST /analyze`

```json
{
  "holdings": [
    {"ticker": "GGAL", "source": "arg_stock", "value": 200000},
    {"ticker": "AL30", "source": "arg_bond",  "value": 200000},
    {"ticker": "GD30", "source": "arg_bond",  "value": 150000},
    {"ticker": "AAPL", "source": "us",        "value": 1000}
  ],
  "period": "3y",
  "run_ai": true
}
```

`risk_free_rate` es **opcional**: si se omite, se usa la **tasa libre de riesgo en
vivo** (T-bill EE.UU. 13 semanas, `^IRX` vía yfinance) — ver `GET /risk-free-rate`.
Para exposición soberana US, agregá ETFs de Treasuries (`BIL`, `SHY`, `IEF`, `TLT`,
`GOVT`) con `source: "us"`.

Devuelve `market`, `current_portfolio`, `markowitz` (frontera + máx Sharpe + Monte
Carlo), `risk` (10 dimensiones + score), `rebalancing` (delta vs óptimo),
`ai_analysis`, `warnings` (FX/instrumentos descartados) y `on_info` (datos en vivo
de ONs vía BYMA).

## Notas

- Datos de mercado reales vía **yfinance** (requiere internet). Tickers sin datos
  se ignoran y se reportan en `period.dropped`.
- El framework de riesgo cubre: correlación, concentración sectorial, exposición
  geográfica/moneda, sensibilidad a tasas, stress test de recesión (beta + drawdown
  histórico), liquidez, riesgo de acción individual, tail risk (VaR/CVaR), hedging
  y rebalanceo.
- Modelo de IA por defecto: `claude-opus-4-8` (configurable con `ANTHROPIC_MODEL`).
