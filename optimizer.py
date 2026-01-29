import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252

def portfolio_return(w, mu):
    return float(w @ mu)

def portfolio_vol(w, cov):
    return float((w.T @ cov @ w) ** 0.5)

def max_sharpe(mu, cov, rf, bounds):
    n = len(mu)
    x0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def neg_sharpe(w):
        vol = portfolio_vol(w, cov)
        return -((portfolio_return(w, mu) - rf) / vol) if vol != 0 else 1e9

    res = minimize(
        neg_sharpe,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"
    )

    return res.x if res.success else x0
