"""Tests del módulo de diagnósticos de supuestos."""
import numpy as np
import pandas as pd

from markowitz_optimizer.engine.diagnostics import (
    arch_lm,
    jarque_bera,
    ljung_box,
    run_diagnostics,
    variance_ratio,
)

ALPHA = 0.05


def _rng():
    return np.random.default_rng(7)


# --- Jarque-Bera --------------------------------------------------------------

def test_jarque_bera_no_rechaza_normal():
    x = _rng().standard_normal(1000) * 0.01
    _, p = jarque_bera(x)
    assert p > ALPHA


def test_jarque_bera_rechaza_colas_gordas():
    x = _rng().standard_t(4, 1000) * 0.01      # t de Student: colas gordas
    _, p = jarque_bera(x)
    assert p < ALPHA


# --- Ljung-Box ----------------------------------------------------------------

def test_ljung_box_no_rechaza_ruido_blanco():
    x = _rng().standard_normal(800) * 0.01
    _, p, _ = ljung_box(x)
    assert p > ALPHA


def test_ljung_box_rechaza_autocorrelacion():
    rng = _rng()
    x = np.zeros(800)
    for i in range(1, 800):
        x[i] = 0.4 * x[i - 1] + rng.standard_normal() * 0.01   # AR(1)
    _, p, _ = ljung_box(x)
    assert p < ALPHA


# --- Variance ratio (Lo-MacKinlay) -------------------------------------------

def test_variance_ratio_no_rechaza_random_walk():
    x = _rng().standard_normal(800) * 0.01
    vr, _, p = variance_ratio(x, 2)
    assert p > ALPHA
    assert abs(vr - 1.0) < 0.2


def test_variance_ratio_rechaza_momentum():
    rng = _rng()
    x = np.zeros(800)
    for i in range(1, 800):
        x[i] = 0.4 * x[i - 1] + rng.standard_normal() * 0.01
    vr, _, p = variance_ratio(x, 2)
    assert p < ALPHA
    assert vr > 1.0      # autocorrelación positiva → VR > 1


def test_variance_ratio_detecta_reversion():
    rng = _rng()
    x = np.zeros(800)
    for i in range(1, 800):
        x[i] = -0.3 * x[i - 1] + rng.standard_normal() * 0.01
    vr, _, p = variance_ratio(x, 2)
    assert p < ALPHA
    assert vr < 1.0      # autocorrelación negativa → VR < 1


# --- ARCH-LM ------------------------------------------------------------------

def test_arch_lm_no_rechaza_homocedastico():
    x = _rng().standard_normal(800) * 0.01
    _, p, _ = arch_lm(x)
    assert p > ALPHA


def test_arch_lm_rechaza_clustering():
    rng = _rng()
    g = np.zeros(800)
    s = 0.01
    for i in range(800):
        prev = g[i - 1] ** 2 if i > 0 else 0.0
        s = np.sqrt(1e-5 + 0.1 * prev + 0.85 * s**2)   # GARCH(1,1)
        g[i] = rng.standard_normal() * s
    _, p, _ = arch_lm(g)
    assert p < ALPHA


# --- run_diagnostics (orquestador) -------------------------------------------

def test_run_diagnostics_estructura():
    rng = _rng()
    dr = pd.DataFrame({
        "A": rng.standard_normal(400) * 0.02,
        "B": rng.standard_t(5, 400) * 0.02,
    })
    rep = run_diagnostics({"A": 0.5, "B": 0.5}, dr).to_dict()
    assert rep["available"] is True
    assert rep["n_obs"] == 400
    keys = {t["key"] for t in rep["tests"]}
    assert keys == {"jarque_bera", "ljung_box", "variance_ratio", "arch_lm"}
    assert "A" in rep["normality_by_asset"]
    assert rep["interpretation"]


def test_run_diagnostics_ventana_corta():
    dr = pd.DataFrame({"A": [0.01, -0.01, 0.0], "B": [0.0, 0.01, -0.01]})
    rep = run_diagnostics({"A": 0.5, "B": 0.5}, dr).to_dict()
    assert rep["available"] is False
    assert "corta" in rep["reason"]
