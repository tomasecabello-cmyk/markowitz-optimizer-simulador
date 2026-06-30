# Optimizador de Carteras · Markowitz + IA

Optimizador de carteras **ARG + US** basado en la **frontera eficiente de Markowitz**,
con un **framework de riesgo de 10 dimensiones** y un **análisis con IA (Claude Opus 4.8)**.
Inspirado en el proyecto _"Optimizador de Portfolios con Markowitz e IA"_.

**Autores:** Tomás Emanuel Cabello · Lola Belén Lombardi

**Repositorio:** https://github.com/tomasecabello-cmyk/markowitz-optimizer-simulador (privado)

Entrada: tus posiciones (ticker + tipo de mercado + monto). Salida: frontera
eficiente, cartera de máximo Sharpe, matriz de correlación, score de riesgo en 10
dimensiones, propuesta de rebalanceo y un informe con IA.

## Inicio rápido (correrlo en el simulador / tu máquina)

Necesitás **Python 3.11+** y **salida a internet** (la app baja datos de mercado en vivo).
Probado de punta a punta: 36 tests en verde, smoke OK y la app levanta en `http://127.0.0.1:8000`.

```bash
# 1. Clonar
git clone https://github.com/tomasecabello-cmyk/markowitz-optimizer-simulador.git
cd markowitz-optimizer-simulador

# 2. Entorno virtual + dependencias
python -m venv .venv
# Windows:        .\.venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -e ".[dev]"

# 3. (opcional) verificar que todo anda, sin necesidad de internet
python scripts/smoke.py      # debe imprimir "SMOKE PASS"
pytest -q                    # 36 tests en verde

# 4. Levantar la app
python -m uvicorn markowitz_optimizer.api.main:app --port 8000
```

Abrí **http://127.0.0.1:8000** en el navegador. La doc interactiva de la API
(Swagger) queda en **http://127.0.0.1:8000/docs**.

- **No hace falta ninguna API key para correrlo.** Sin `ANTHROPIC_API_KEY`, el
  informe con IA usa un **mock determinístico** con el mismo formato; todo lo demás
  (Markowitz, riesgo, rebalanceo) funciona igual con datos reales.
- En **Windows** podés además hacer doble-click a `run.cmd` (levanta el server y abre
  el navegador solo).
- Si vas a correrlo en un servidor/simulador detrás de proxy, asegurate de permitir
  salida HTTPS a: `yfinance`, `data912`, `rava`, `byma`, `argentinadatos`.

> Demo educativo. **No es asesoramiento financiero.**

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

## Activar la IA real (opcional)

El optimizador corre completo sin ninguna key (la IA cae a un mock determinístico).
Para que el informe lo escriba Claude de verdad:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# editá .env y pegá tu ANTHROPIC_API_KEY=...
```

Modelo por defecto: `claude-opus-4-8` (configurable con `ANTHROPIC_MODEL`).

## Otras formas de correrlo

```powershell
# Windows, doble-click o desde la terminal (levanta server + abre navegador):
.\run.cmd
# Si PowerShell bloquea el .ps1:
powershell -ExecutionPolicy Bypass -File .\run.ps1

# Modo desarrollo con auto-reload:
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
python -m pytest -q            # suite completa (36 tests)
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
