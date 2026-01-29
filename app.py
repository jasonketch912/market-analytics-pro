import streamlit as st
import numpy as np
import pandas as pd

from analysis import analyze
from optimizer import max_sharpe
from backtest_engine import monthly_rebalance

st.set_page_config(page_title="Portfolio Research", layout="wide")
st.title("📊 Portfolio Research")

# -----------------------------
# Sidebar params
# -----------------------------
with st.sidebar:
    st.header("Paramètres")

    tickers = st.text_area(
        "Tickers (virgules)",
        "AMD, META, AAPL, MSFT, MC.PA, OR.PA"
    ).split(",")

    period = st.selectbox("Période", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Intervalle", ["1d", "1h"], index=0)

    budget = st.number_input("Budget (€)", 100.0, 1e7, 1000.0, step=100.0)

    st.divider()
    rf = st.number_input("Risk-free rate (annuel, ex 0.03)", 0.0, 0.2, 0.0, step=0.01)
    target = st.number_input("Objectif rendement annuel (%)", 0.0, 100.0, 10.0) / 100
    risk = st.slider("Tolérance au risque (1 faible → 10 élevé)", 1, 10, 5)
    max_vol = {1:0.15,2:0.18,3:0.20,4:0.23,5:0.26,6:0.30,7:0.35,8:0.40,9:0.45,10:0.55}[risk]
    st.caption(f"Vol max (scoring) ~ {int(max_vol*100)}%")

    st.divider()
    st.subheader("Benchmarks")
    bench_us = st.selectbox("Benchmark principal (US)", ["SPY", "QQQ", "DIA", "IWM"], index=0)
    bench_alt = st.selectbox("Benchmark secondaire", ["QQQ", "SPY", "DIA", "IWM"], index=0)
    bench_eu = st.text_input("Benchmark Europe (indice)", "^FCHI")  # CAC40
    var_level = st.selectbox("VaR/CVaR", [0.90, 0.95, 0.99], index=1)

    st.divider()
    st.subheader("Markowitz (contraintes)")
    min_w = st.number_input("Poids min", 0.0, 1.0, 0.0, step=0.01)
    max_w = st.number_input("Poids max", 0.01, 1.0, 0.25, step=0.01)

    run = st.button("🚀 Lancer l'analyse")

# -----------------------------
# Helpers
# -----------------------------
def safe_close_df(prices: dict) -> pd.DataFrame:
    series = {}
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        if "Close" in df.columns:
            series[t] = df["Close"].rename(t)
    if not series:
        return pd.DataFrame()
    out = pd.concat(series.values(), axis=1).dropna(how="all").ffill().dropna()
    return out

def cum_performance(close: pd.Series) -> pd.Series:
    r = close.pct_change().dropna()
    return (1 + r).cumprod()

def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1

def rolling_beta_alpha(asset_close: pd.Series, bench_close: pd.Series, window=60, rf=0.0):
    """
    Rolling beta/alpha (approx).
    alpha annualisée, beta standard.
    """
    ar = asset_close.pct_change()
    br = bench_close.pct_change()
    df = pd.concat([ar, br], axis=1).dropna()
    df.columns = ["ri", "rm"]

    betas, alphas, idx = [], [], []
    for i in range(window, len(df)):
        s = df.iloc[i-window:i]
        cov = np.cov(s["ri"], s["rm"])[0, 1]
        var = np.var(s["rm"])
        beta = cov / var if var != 0 else np.nan

        # alpha simple annualisée
        alpha = (s["ri"].mean() - beta * s["rm"].mean()) * 252
        betas.append(beta)
        alphas.append(alpha)
        idx.append(df.index[i])

    out = pd.DataFrame({"Beta (rolling)": betas, "Alpha (rolling, ann.)": alphas}, index=idx)
    return out

# -----------------------------
# Main
# -----------------------------
if run:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        st.error("Ajoute au moins un ticker.")
        st.stop()

    with st.spinner("Calcul en cours (prix, score, beta/alpha, VaR/CVaR)..."):
        results, prices = analyze(
            tickers=tickers,
            period=period,
            interval=interval,
            rf=rf,
            target=target,
            max_vol=max_vol,
            bench_us=bench_us,
            bench_eu=bench_eu,
            var_level=var_level
        )

    # Ensure columns exist
    for col in ["Last Price", "Annual Return", "Annual Volatility", "Sharpe", "Max Drawdown", "VaR", "CVaR", "Beta", "Alpha", "Score"]:
        if col not in results.columns:
            results[col] = np.nan

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Compare Markets", "Risk", "Portfolio / Backtest"])

    # -----------------------------
    # TAB 1: Overview
    # -----------------------------
    with tab1:
        st.subheader("Tableau complet (métriques + score + beta/alpha)")
        st.dataframe(results.sort_values("Score", ascending=False, na_position="last"), use_container_width=True)

        valid = results.dropna(subset=["Score"])
        if valid.empty:
            st.warning("Aucune action exploitable (vérifie tickers/API).")
        else:
            chosen = st.selectbox("Sélection action", valid["Ticker"].tolist(), key="ov_chosen")
            row = results[results["Ticker"] == chosen].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{row['Score']:.1f}")
            c2.metric("Return (ann.)", f"{row['Annual Return']:.2%}" if pd.notna(row["Annual Return"]) else "NA")
            c3.metric("Vol (ann.)", f"{row['Annual Volatility']:.2%}" if pd.notna(row["Annual Volatility"]) else "NA")
            c4.metric("Drawdown max", f"{row['Max Drawdown']:.2%}" if pd.notna(row["Max Drawdown"]) else "NA")

            c5, c6 = st.columns(2)
            c5.metric("Beta", f"{row['Beta']:.2f}" if pd.notna(row["Beta"]) else "NA")
            c6.metric("Alpha (ann.)", f"{row['Alpha']:.2%}" if pd.notna(row["Alpha"]) else "NA")

            if chosen in prices:
                close = prices[chosen]["Close"].dropna()
                st.write("Prix (Close)")
                st.line_chart(close)

                st.write("Performance cumulée")
                st.line_chart(cum_performance(close))

                st.write("Drawdown")
                st.line_chart(drawdown(close))

    # -----------------------------
    # TAB 2: Compare Markets
    # -----------------------------
    with tab2:
        st.subheader("Comparaison au marché (SPY / QQQ / etc.)")

        valid = results.dropna(subset=["Score"])
        if valid.empty:
            st.warning("Pas de données action.")
        else:
            chosen = st.selectbox("Action à comparer", valid["Ticker"].tolist(), key="cmp_chosen")

            # get action close
            if chosen not in prices:
                st.warning("Données action non dispo.")
            else:
                asset_close = prices[chosen]["Close"].dropna()

                # Fetch benchmarks via analyze's prices? (Not in cache) -> easiest: reuse data_provider
                from data_provider import get_prices

                bench1 = get_prices(bench_us, period, interval)["Close"].dropna()
                bench2 = get_prices(bench_alt, period, interval)["Close"].dropna()

                # Align by dates
                df = pd.concat({
                    chosen: asset_close,
                    bench_us: bench1,
                    bench_alt: bench2
                }, axis=1).dropna()

                st.write("Performance cumulée (normalisée)")
                perf = df.apply(cum_performance)
                st.line_chart(perf)

                st.write("Corrélations (sur la période)")
                rets = df.pct_change().dropna()
                st.dataframe(rets.corr(), use_container_width=True)

                st.write("Beta & Alpha rolling (60 jours) vs benchmark principal")
                roll = rolling_beta_alpha(df[chosen], df[bench_us], window=60, rf=rf)
                st.line_chart(roll)

    # -----------------------------
    # TAB 3: Risk
    # -----------------------------
    with tab3:
        st.subheader("Risk (VaR/CVaR + distribution + stress test)")

        valid = results.dropna(subset=["Score"])
        if valid.empty:
            st.warning("Pas de données exploitables.")
        else:
            chosen = st.selectbox("Action (Risk)", valid["Ticker"].tolist(), key="risk_chosen")
            row = results[results["Ticker"] == chosen].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volatilité (ann.)", f"{row['Annual Volatility']:.2%}" if pd.notna(row["Annual Volatility"]) else "NA")
            c2.metric(f"VaR {int(var_level*100)}% (daily)", f"{row['VaR']:.2%}" if pd.notna(row["VaR"]) else "NA")
            c3.metric(f"CVaR {int(var_level*100)}% (daily)", f"{row['CVaR']:.2%}" if pd.notna(row["CVaR"]) else "NA")
            c4.metric("Max Drawdown", f"{row['Max Drawdown']:.2%}" if pd.notna(row["Max Drawdown"]) else "NA")

            if chosen in prices:
                close = prices[chosen]["Close"].dropna()
                r = close.pct_change().dropna()

                st.write("Histogramme des rendements (daily)")
                hist = np.histogram(r.values, bins=40)
                hist_df = pd.DataFrame({"bin": range(len(hist[0])), "count": hist[0]})
                st.bar_chart(hist_df.set_index("bin"))

                st.write("Volatilité rolling (30 jours)")
                roll_vol = r.rolling(30).std() * np.sqrt(252)
                st.line_chart(roll_vol.dropna())

                # Stress test in €
                st.write("Stress test (perte en € selon allocation)")
                alloc_share = st.slider("Part du budget investie sur cette action (%)", 0, 100, 20) / 100
                invested = budget * alloc_share
                shocks = [-0.10, -0.20, -0.30, -0.40]
                stress = pd.DataFrame({
                    "Shock": [f"{int(s*100)}%" for s in shocks],
                    "P&L (€)": [invested * s for s in shocks]
                })
                st.dataframe(stress, use_container_width=True)

    # -----------------------------
    # TAB 4: Portfolio & Backtest
    # -----------------------------
    with tab4:
        st.subheader("Portefeuille (Markowitz) + Backtest rebalancing mensuel")

        close_df = safe_close_df(prices)
        if close_df.empty or close_df.shape[1] < 2:
            st.warning("Il faut au moins 2 actions valides pour optimiser/backtester.")
        else:
            returns_df = close_df.pct_change().dropna()
            mu = returns_df.mean().values * 252
            cov = returns_df.cov().values * 252

            bounds = [(min_w, max_w)] * len(mu)
            w = max_sharpe(mu, cov, rf=rf, bounds=bounds)

            alloc = pd.DataFrame({
                "Ticker": close_df.columns,
                "Weight": w,
                "Allocation (€)": w * budget,
                "Shares (fractional)": (w * budget) / close_df.iloc[-1].values
            }).sort_values("Weight", ascending=False)

            st.write("Poids optimaux (Max Sharpe) + allocation")
            st.dataframe(alloc, use_container_width=True)

            st.write("Backtest (rebalancing mensuel)")
            bt = monthly_rebalance(close_df, rf=rf, min_w=min_w, max_w=max_w)
            st.line_chart(bt["Portfolio"])
            st.line_chart(bt["Drawdown"])
