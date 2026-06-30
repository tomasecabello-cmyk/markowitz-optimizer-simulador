"""
analyst — análisis de riesgo de cartera estilo "Bridgewater" con Claude.

Toma el output cuantitativo determinístico (frontera eficiente + framework de
riesgo de 10 dimensiones) y le pide a Claude que lo INTERPRETE: veredicto,
hallazgos, coberturas concretas para los 3 mayores riesgos y la justificación
del rebalanceo hacia la cartera de máximo Sharpe.

La IA interpreta; no recalcula. Toda la matemática viene de las capas engine/ y
risk/. Si no hay ANTHROPIC_API_KEY, se devuelve un análisis mock determinístico
(mismo schema) para que el demo corra sin clave.

Modelo por defecto: Claude Opus 4.8 (claude-opus-4-8), configurable con
ANTHROPIC_MODEL (p.ej. claude-haiku-4-5). Usa structured outputs
(output_config.format / json_schema), soportado en todos esos modelos. No usa
adaptive thinking ni effort: ambos dan 400 en Haiku 4.5 (ver skill claude-api).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = """\
Sos un analista cuantitativo de riesgo de carteras, con el rigor de una mesa \
institucional (estilo Bridgewater). Recibís el resultado de un motor determinístico: \
la frontera eficiente de Markowitz (cartera de mínima varianza y de máximo Sharpe) y \
un framework de riesgo de 10 dimensiones ya computado sobre la cartera actual del \
usuario.

Tu trabajo es INTERPRETAR esos números, no recalcularlos. Sé concreto y accionable. \
Para las coberturas, nombrá instrumentos reales (ETF inversos, oro/GLD, bonos largos/TLT, \
opciones put, etc.) atados a los 3 riesgos más altos del framework. Para el rebalanceo, \
justificá el movimiento desde la cartera actual hacia la de máximo Sharpe en términos de \
retorno/volatilidad/Sharpe.

Escribí en español, tono profesional y directo. No es asesoramiento financiero \
personalizado: es un análisis cuantitativo educativo. No inventes datos que no estén en \
el input."""


@dataclass
class HedgingIdea:
    risk: str
    instrument: str
    rationale: str


@dataclass
class AIAnalysis:
    executive_summary: str
    risk_verdict: str
    key_findings: list[str]
    hedging_strategies: list[HedgingIdea]
    rebalancing_rationale: str
    disclaimer: str
    model: str
    is_mock: bool = False

    def to_dict(self) -> dict:
        return {
            "executive_summary": self.executive_summary,
            "risk_verdict": self.risk_verdict,
            "key_findings": list(self.key_findings),
            "hedging_strategies": [asdict(h) for h in self.hedging_strategies],
            "rebalancing_rationale": self.rebalancing_rationale,
            "disclaimer": self.disclaimer,
            "model": self.model,
            "is_mock": self.is_mock,
        }


# Esquema JSON para structured outputs (sin constraints no soportadas).
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "risk_verdict": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "hedging_strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "instrument": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["risk", "instrument", "rationale"],
                "additionalProperties": False,
            },
        },
        "rebalancing_rationale": {"type": "string"},
    },
    "required": [
        "executive_summary", "risk_verdict", "key_findings",
        "hedging_strategies", "rebalancing_rationale",
    ],
    "additionalProperties": False,
}

_DISCLAIMER = (
    "Análisis cuantitativo educativo generado con datos de mercado históricos. "
    "No constituye asesoramiento financiero personalizado. Los retornos pasados no "
    "garantizan resultados futuros."
)


def _build_user_payload(
    portfolio: dict,
    market_summary: dict,
    markowitz: dict,
    risk: dict,
) -> str:
    """Empaqueta el contexto cuantitativo como JSON para el prompt."""
    context = {
        "cartera_actual": portfolio,
        "estadisticas_mercado": market_summary,
        "frontera_eficiente": {
            "min_variance": markowitz.get("min_variance"),
            "max_sharpe": markowitz.get("max_sharpe"),
            "risk_free_rate": markowitz.get("risk_free_rate"),
        },
        "framework_riesgo": {
            "risk_score": risk.get("risk_score"),
            "risk_band": risk.get("risk_band"),
            "top_risks": risk.get("top_risks"),
            "dimensiones": risk.get("dimensions"),
        },
    }
    return (
        "Analizá esta cartera. Acá está el output del motor determinístico "
        "(JSON):\n\n" + json.dumps(context, ensure_ascii=False, indent=2)
    )


def analyze_portfolio(
    portfolio: dict,
    market_summary: dict,
    markowitz: dict,
    risk: dict,
    model: str | None = None,
    api_key: str | None = None,
) -> AIAnalysis:
    """
    Genera el análisis con IA. Usa Claude si hay API key; si no, mock determinístico.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    if not key:
        return _mock_analysis(markowitz, risk, model)

    try:
        import anthropic
    except ImportError:
        return _mock_analysis(markowitz, risk, model)

    client = anthropic.Anthropic(api_key=key)
    user_payload = _build_user_payload(portfolio, market_summary, markowitz, risk)

    try:
        # Solo output_config.format (json_schema) para JSON estructurado: soportado
        # en Haiku 4.5, Opus 4.8, Sonnet 4.6 y Fable 5. NO usamos `thinking: adaptive`
        # ni `effort`: ambos dan 400 en Haiku 4.5 (extended thinking / effort no van
        # en Haiku). En Opus omitir `effort` ya equivale a "high".
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            output_config={"format": {
                "type": "json_schema", "schema": _OUTPUT_SCHEMA,
            }},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de API → mock con nota
        mock = _mock_analysis(markowitz, risk, model)
        mock.key_findings.insert(0, f"[Aviso] La IA no respondió ({exc}); se muestra análisis base.")
        return mock

    if response.stop_reason == "refusal":
        return _mock_analysis(markowitz, risk, model)

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _mock_analysis(markowitz, risk, model)

    return AIAnalysis(
        executive_summary=data.get("executive_summary", ""),
        risk_verdict=data.get("risk_verdict", ""),
        key_findings=list(data.get("key_findings", [])),
        hedging_strategies=[
            HedgingIdea(h.get("risk", ""), h.get("instrument", ""), h.get("rationale", ""))
            for h in data.get("hedging_strategies", [])
        ],
        rebalancing_rationale=data.get("rebalancing_rationale", ""),
        disclaimer=_DISCLAIMER,
        model=model,
        is_mock=False,
    )


def _mock_analysis(markowitz: dict, risk: dict, model: str) -> AIAnalysis:
    """Análisis determinístico (sin LLM) con el mismo schema que la IA real."""
    band = risk.get("risk_band", "moderado")
    score = risk.get("risk_score", 50)
    top = risk.get("top_risks", [])
    ms = markowitz.get("max_sharpe", {})
    cur_sharpe = ms.get("sharpe", 0)

    findings = [f"Score de riesgo de la cartera: {score}/100 ({band})."]
    findings += [f"Riesgo destacado: {t}" for t in top]

    hedges = []
    for t in top[:3]:
        name = t.split(" (")[0]
        if "Concentración" in name or "Acción" in name:
            instr, why = "ETF amplio (SPY/VTI)", "diluye la concentración en pocas posiciones"
        elif "Tasas" in name:
            instr, why = "Bonos cortos (SHY) / efectivo", "reduce la duración ante subas de tasas"
        elif "Recesión" in name or "Tail" in name:
            instr, why = "Oro (GLD) y puts sobre índice", "cobertura ante caídas severas de mercado"
        elif "Correlación" in name:
            instr, why = "Activos descorrelacionados (TLT, GLD)", "baja la correlación media de la cartera"
        else:
            instr, why = "Diversificación adicional", "mejora el perfil riesgo-retorno"
        hedges.append(HedgingIdea(risk=name, instrument=instr, rationale=why))

    return AIAnalysis(
        executive_summary=(
            f"La cartera presenta un nivel de riesgo {band} ({score}/100). "
            "El motor de Markowitz sugiere reasignar hacia la cartera de máximo Sharpe "
            f"(Sharpe objetivo {cur_sharpe:.2f}) para mejorar el retorno por unidad de riesgo."
        ),
        risk_verdict=f"Riesgo {band}. Revisar las 3 dimensiones más altas antes de invertir.",
        key_findings=findings,
        hedging_strategies=hedges,
        rebalancing_rationale=(
            "Mover los pesos hacia la cartera de máximo Sharpe de la frontera eficiente "
            "reduce la volatilidad para un retorno comparable, maximizando el ratio de Sharpe."
        ),
        disclaimer=_DISCLAIMER,
        model=model,
        is_mock=True,
    )
