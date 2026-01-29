import numpy as np
import pandas as pd

from data_provider import get_prices, is_europe_ticker
from risk_engine import (
    returns, annual_return, annual_vol, sharpe,
    max_drawdown, var_cvar, beta_alpha
)

def score_stock(ar, vol, sh, dd, target, max_vol):
    score = (
        0.35 * np.tanh(ar * 2)
        + 0.25 * (1 - np.tanh(vol * 2))
        + 0.25 * np.tanh(sh / 2 if sh == sh else 0)
        + 0.15 * (1 - np.tanh(abs(dd) * 2))
    )
    score = (score + 1) * 50

    if ar < target:
        score -= min(25, (target - ar) * 100)
    if vol > max_vol:
        score -= min(25, (vol - max_vol) * 100)

    return float(np.clip(score, 0, 100))

def analyze(
    tickers, period, interval, rf, target, max_vol,
    bench_us="SPY", bench_eu="^FCHI", var_level=0.95
):
    price_cache = {}
    rows = []

    bench_prices = {
        bench_us: get_prices(bench_us, period, interval),
        bench_eu: get_prices(bench_eu, period, interval)
    }

    for t in tickers:
        base = {
            "Ticker": t,
            "Last Price": np.nan,
            "Score": np.nan
        }

        try:
            df = get_prices(t, period, interval)
            price_cache[t] = df

            close = df["Close"]
            r = returns(close)

            ar = annual_return(r)
            vol = annual_vol(r)
            sh = sharpe(r, rf)
            dd = max_drawdown(close)
            var, cvar = var_cvar(r, var_level)

            b = bench_eu if is_europe_ticker(t) else bench_us
            br = returns(bench_prices[b]["Close"])
            beta, alpha = beta_alpha(r, br, rf)

            score = score_stock(ar, vol, sh, dd, target, max_vol)

            rows.append({
                "Ticker": t,
                "Last Price": close.iloc[-1],
                "Annual Return": ar,
                "Annual Volatility": vol,
                "Sharpe": sh,
                "Max Drawdown": dd,
                "VaR": var,
                "CVaR": cvar,
                "Beta": beta,
                "Alpha": alpha,
                "Benchmark": b,
                "Score": score
            })

        except Exception as e:
            base["Error"] = str(e)
            rows.append(base)

    return pd.DataFrame(rows), price_cache
