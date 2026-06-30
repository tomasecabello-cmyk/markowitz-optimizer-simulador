"""Tests de la capa de proveedores (sin red, con requests mockeado)."""

from __future__ import annotations

import pytest

from markowitz_optimizer.data import providers
from markowitz_optimizer.data.providers import ProviderError, _safe_symbol, data912_history


@pytest.mark.parametrize("bad", ["../etc", "AAPL;rm", "a b", "<script>", "", "x" * 16])
def test_safe_symbol_rejects_bad_input(bad):
    with pytest.raises(ProviderError):
        _safe_symbol(bad)


@pytest.mark.parametrize("ok,expected", [("aapl", "AAPL"), ("aapl.ba", "AAPL.BA"), ("al30", "AL30")])
def test_safe_symbol_normalizes_valid(ok, expected):
    assert _safe_symbol(ok) == expected


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def test_data912_history_non_dict_rows_raises_provider_error(monkeypatch):
    # Regression: ISSUE-002 — 500 on invalid ticker (data912 non-dict rows)
    # Found by /qa on 2026-06-14
    # Report: .gstack/qa-reports/qa-report-127-0-0-1-2026-06-14.md
    # Un símbolo inválido devolvía 200 con una lista de strings; r.get("c")
    # lanzaba AttributeError (no capturado) -> 500. Debe ser ProviderError.
    monkeypatch.setattr(providers.requests, "get",
                        lambda *a, **k: _FakeResp(["x", "y", "z"]))
    with pytest.raises(ProviderError):
        data912_history("ZZZZZ", "stocks", "1y")


def test_data912_history_empty_list_raises(monkeypatch):
    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: _FakeResp([]))
    with pytest.raises(ProviderError):
        data912_history("ZZZZZ", "stocks", "1y")


def test_data912_history_parses_valid_rows(monkeypatch):
    rows = [
        {"date": "2025-01-02", "c": 100.0},
        {"date": "2025-01-03", "c": 101.5},
        {"date": "2025-01-06", "c": 0},      # se ignora (precio 0)
        {"date": "2025-01-07"},               # se ignora (sin cierre)
    ]
    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: _FakeResp(rows))
    s = data912_history("GGAL", "stocks", "max")
    assert list(s.values) == [100.0, 101.5]
    assert len(s) == 2
