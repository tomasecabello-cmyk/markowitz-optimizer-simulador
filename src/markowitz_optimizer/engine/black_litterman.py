"""
black_litterman — estimación robusta del retorno esperado μ (Black & Litterman, 1992).

PROBLEMA QUE RESUELVE
---------------------
Markowitz necesita μ (retorno esperado por activo). El estimador ingenuo, la media
histórica, es ruidoso: pequeños cambios en μ mueven mucho los pesos óptimos, así que
la cartera de máximo Sharpe queda inestable y concentrada. Black-Litterman ataca eso.

LA IDEA (en dos piezas)
-----------------------
1) PRIOR DE EQUILIBRIO (reverse optimization). En vez de partir de la media histórica,
   se parte de los retornos que *justificarían* una cartera de referencia w_ref bajo
   equilibrio de mercado. Si todos optimizan media-varianza y el mercado está en
   equilibrio, entonces el óptimo es w_ref y, dando vuelta la condición de primer
   orden de "maximizar  wᵀμ − (δ/2) wᵀΣw", se despeja:

        Π = δ Σ w_ref            (retornos de equilibrio implícitos, en exceso)

   δ = aversión al riesgo del inversor representativo. w_ref = cartera de referencia
   (acá equal-weight 1/N por defecto: no afirmamos conocer el "portafolio de mercado"
   de una canasta mixta ARG/US, así que anclamos en algo neutral).

2) VIEWS (visiones del inversor). El inversor puede tener opiniones sobre retornos.
   Acá usamos la MEDIA HISTÓRICA como una view por activo (P = I, Q = μ_hist − r_f),
   con una incertidumbre Ω. Es decir: "los datos sugieren esto, pero no del todo".

   El posterior bayesiano combina prior (Π) y views (Q) ponderando por su confianza:

        E[R] = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ [ (τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q ] + r_f

   τ escala la incertidumbre del prior; Ω la de las views. Con P = I y
   Ω = diag(P τΣ Pᵀ) (convención He & Litterman, 2002), el resultado es una mezcla
   por-activo entre el equilibrio y la media histórica, ponderada por varianzas.
   Más confianza en las views (view_confidence ↑) ⇒ μ se acerca a la media histórica;
   menos confianza ⇒ μ se acerca al equilibrio (más estable y diversificado).

Es, en esencia, un SHRINKAGE de la media histórica hacia un ancla con estructura
económica. Mismo espíritu que Ledoit-Wolf hace con Σ, pero para μ.

Referencias: Black & Litterman (1992); He & Litterman (2002); Idzorek (2007).
"""

from __future__ import annotations

import numpy as np

DEFAULT_DELTA = 2.5    # aversión al riesgo del inversor representativo (valor estándar de la literatura)
DEFAULT_TAU = 0.05     # incertidumbre del prior de equilibrio (escala pequeña, usual 0.025–0.05)


def implied_equilibrium_returns(
    sigma: np.ndarray, w_ref: np.ndarray, delta: float = DEFAULT_DELTA
) -> np.ndarray:
    """Π = δ Σ w_ref — retornos de equilibrio implícitos (en exceso sobre r_f)."""
    return delta * (sigma @ w_ref)


def black_litterman_returns(
    sigma: np.ndarray,
    mu_hist: np.ndarray,
    rf: float = 0.0,
    w_ref: np.ndarray | None = None,
    delta: float = DEFAULT_DELTA,
    tau: float = DEFAULT_TAU,
    view_confidence: float = 1.0,
) -> np.ndarray:
    """
    Retorno esperado posterior de Black-Litterman (en niveles, incluye r_f).

    Args:
        sigma: covarianza ANUAL (n x n), simétrica PD.
        mu_hist: media histórica ANUAL por activo (n,) — se usa como views.
        rf: tasa libre de riesgo anual (para trabajar en exceso de retorno).
        w_ref: cartera de referencia (n,). Default equal-weight 1/N.
        delta: aversión al riesgo (Π = δ Σ w_ref).
        tau: incertidumbre del prior.
        view_confidence: confianza en las views (media histórica). >1 acerca μ a la
            media histórica; <1 lo acerca al equilibrio.

    Returns:
        μ posterior (n,) en niveles (excess + rf).
    """
    sigma = np.asarray(sigma, dtype=float)
    mu_hist = np.asarray(mu_hist, dtype=float)
    n = sigma.shape[0]
    if w_ref is None:
        w_ref = np.full(n, 1.0 / n)
    w_ref = np.asarray(w_ref, dtype=float)

    pi = implied_equilibrium_returns(sigma, w_ref, delta)   # prior (exceso)
    tau_sigma = tau * sigma
    P = np.eye(n)
    Q = mu_hist - rf                                         # views como exceso de retorno

    # Ω = diag(P τΣ Pᵀ) / confianza  (He & Litterman). Epsilon evita singularidad.
    omega = np.diag(np.diag(P @ tau_sigma @ P.T)) / max(view_confidence, 1e-9)
    omega += np.eye(n) * 1e-12

    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)
    posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    posterior_excess = posterior_cov @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
    return posterior_excess + rf
