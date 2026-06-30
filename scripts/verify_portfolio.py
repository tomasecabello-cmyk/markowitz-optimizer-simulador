"""Reproduce el pipeline de /analyze para el portfolio del usuario y compara
current vs optimo (max Sharpe / min var) vs 1/N. Verifica por que el current
puede 'ganarle' al optimo en retorno realizado."""
from __future__ import annotations

import numpy as np

from markowitz_optimizer.data.market_data import fetch_asset_info, fetch_market_data
from markowitz_optimizer.engine import markowitz
from markowitz_optimizer.engine.backtest import backtest, walk_forward
from markowitz_optimizer.engine.black_litterman import black_litterman_returns

HOLDINGS = [
    ("GGAL", "arg_stock", 200000), ("YPFD", "arg_stock", 150000),
    ("AL30", "arg_bond", 200000), ("GD30", "arg_bond", 150000),
    ("AAPL", "us", 1000), ("TSLA", "us", 1000), ("NVDA", "us", 10000),
    ("KO", "us", 1000), ("JPM", "us", 10000),
]
RF = 0.03655
VC = 1.0

instruments = [{"symbol": s, "source": src} for s, src, _ in HOLDINGS]
md = fetch_market_data(instruments, period="3y", normalize_currency=True,
                       fx_kind="ccl", cov_method="ledoit_wolf")
valid = md.tickers
fx = md.fx.get("latest") if (md.fx and md.fx.get("applied")) else None
print(f"Ventana: {md.start} -> {md.end} | normalizado USD: {md.normalized} | CCL: {fx}")
print(f"Validos ({len(valid)}): {valid}")
print(f"Descartados: {[d['symbol'] for d in md.dropped]}")


def to_base(sym, val):
    if fx and md.currencies.get(sym) == "ARS":
        return val / fx
    return val


value_by = {s: to_base(s, v) for s, _, v in HOLDINGS if s in valid}
invested = sum(value_by.values())
weights = {t: value_by.get(t, 0.0) / invested for t in valid}
print("\nPesos current (en USD):")
for t in sorted(weights, key=lambda x: -weights[x]):
    print(f"  {t:6s} {weights[t]*100:6.2f}%  (${value_by[t]:.0f})")

mu_hist = np.array([md.mean_returns[t] for t in valid])
sigma = np.array([[md.cov_matrix.loc[a, b] for b in valid] for a in valid])
mu_bl = black_litterman_returns(sigma, mu_hist, rf=RF, view_confidence=VC)

mk = markowitz.optimize(valid, mu_bl, sigma, risk_free_rate=RF, max_weight=1.0)
print("\nPesos max_sharpe (BL):")
for t in sorted(mk.max_sharpe.weights, key=lambda x: -mk.max_sharpe.weights[x]):
    w = mk.max_sharpe.weights[t]
    if w > 0.001:
        print(f"  {t:6s} {w*100:6.2f}%")

# mu historico vs BL por activo (para ver el shrinkage)
print("\nmu historico vs BL (anual):")
for i, t in enumerate(valid):
    print(f"  {t:6s} hist {mu_hist[i]*100:7.1f}%   BL {mu_bl[i]*100:7.1f}%")

print("\n=== Backtest IN-SAMPLE (peso constante sobre la ventana) ===")
bt = backtest(md.daily_returns, {
    "current": weights, "max_sharpe": mk.max_sharpe.weights,
    "min_variance": mk.min_variance.weights,
}, rf=RF)
for name, m in bt["metrics"].items():
    print(f"  {name:13s} ret {m['total_return']*100:7.1f}%  CAGR {m['cagr']*100:6.1f}%  "
          f"vol {m['volatility']*100:5.1f}%  Sharpe {m['sharpe']:5.2f}  maxDD {m['max_drawdown']*100:6.1f}%")

print("\n=== Walk-forward OUT-OF-SAMPLE ===")
wf = walk_forward(md.daily_returns, weights, rf=RF, max_weight=1.0,
                  return_method="black_litterman", bl_view_confidence=VC)
if wf.get("available"):
    for name, m in wf["metrics"].items():
        print(f"  {name:13s} ret {m['total_return']*100:7.1f}%  CAGR {m['cagr']*100:6.1f}%  "
              f"vol {m['volatility']*100:5.1f}%  Sharpe {m['sharpe']:5.2f}  maxDD {m['max_drawdown']*100:6.1f}%")
else:
    print("  no disponible:", wf.get("reason"))
