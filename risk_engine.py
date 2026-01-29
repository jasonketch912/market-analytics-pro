import numpy as np
import pandas as pd

TRADING_DAYS = 252

def returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()

def annual_return(r: pd.Series) -> float:
    return float((1 + r.mean()) ** TRADING_DAYS - 1)

def annual_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))

def sharpe(r: pd.Series, rf=0.0) -> float:
    vol = annual_vol(r)
    return float((annual_return(r) - rf) / vol) if vol != 0 else np.nan

def max_drawdown(close: pd.Series) -> float:
    dd = close / close.cummax() - 1
    return float(dd.min())

def var_cvar(r: pd.Series, level=0.95):
    q = np.quantile(r, 1 - level)
    var = float(q)
    cvar = float(r[r <= q].mean()) if (r <= q).any() else var
    return var, cvar

def beta_alpha(asset_r: pd.Series, bench_r: pd.Series, rf=0.0):
    df = pd.concat([asset_r, bench_r], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan

    ri, rm = df.iloc[:, 0], df.iloc[:, 1]
    beta = np.cov(ri, rm)[0, 1] / np.var(rm)
    alpha = (ri.mean() - beta * rm.mean()) * TRADING_DAYS
    return float(beta), float(alpha)
