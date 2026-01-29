import numpy as np
import pandas as pd
from optimizer import max_sharpe

def monthly_rebalance(prices: pd.DataFrame, rf, min_w, max_w):
    prices = prices.dropna(how="all").ffill().dropna()
    rets = prices.pct_change().dropna()

    rebalance_dates = rets.resample("M").last().index
    n = rets.shape[1]
    bounds = [(min_w, max_w)] * n

    w = np.ones(n) / n
    value = 1.0
    history = []

    for date in rets.index:
        if date in rebalance_dates and len(rets.loc[:date]) > 60:
            mu = rets.loc[:date].mean().values * 252
            cov = rets.loc[:date].cov().values * 252
            w = max_sharpe(mu, cov, rf, bounds)

        value *= (1 + w @ rets.loc[date].values)
        history.append((date, value))

    df = pd.DataFrame(history, columns=["Date", "Portfolio"]).set_index("Date")
    df["Drawdown"] = df["Portfolio"] / df["Portfolio"].cummax() - 1
    return df
