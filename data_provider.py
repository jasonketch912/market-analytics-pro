import os
import datetime as dt
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

try:
    from polygon import RESTClient
except Exception:
    RESTClient = None

EU_SUFFIXES = (
    ".PA", ".DE", ".AS", ".MI", ".MC", ".BR", ".LS",
    ".SW", ".ST", ".CO", ".HE", "^"
)

def is_europe_ticker(ticker: str) -> bool:
    t = ticker.upper().strip()
    return t.endswith(EU_SUFFIXES) or t.startswith("^")

def yf_history(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        auto_adjust=True
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"yfinance: aucune donnée pour {ticker}")
    return df

def polygon_history(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY manquant")

    if RESTClient is None:
        raise ValueError("polygon-api-client non installé")

    end = dt.date.today()
    days = {"6mo": 185, "1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
    start = end - dt.timedelta(days=days)

    multiplier, timespan = (1, "hour") if interval == "1h" else (1, "day")

    client = RESTClient(api_key)
    bars = client.get_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start.isoformat(),
        to=end.isoformat(),
        adjusted=True,
        sort="asc",
        limit=50000
    )

    if not bars:
        raise ValueError(f"Polygon: aucune donnée pour {ticker}")

    df = pd.DataFrame([{
        "Date": pd.to_datetime(b.timestamp, unit="ms"),
        "Open": b.open,
        "High": b.high,
        "Low": b.low,
        "Close": b.close,
        "Volume": b.volume
    } for b in bars]).set_index("Date")

    return df

def get_prices(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    """
    Europe / indices → yfinance
    US → Polygon (fallback yfinance si erreur)
    """
    t = ticker.upper().strip()

    if is_europe_ticker(t):
        return yf_history(t, period, interval)

    try:
        return polygon_history(t, period, interval)
    except Exception:
        return yf_history(t, period, interval)
