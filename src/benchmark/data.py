"""Data acquisition and causal feature engineering.

Daily prices are downloaded from Yahoo Finance. Because yfinance versions on hosted notebooks change often
(e.g. "Impersonating chromeXXX is not supported"), the downloader tries yfinance first and falls back to the
Yahoo v8 chart endpoint. Prices are adjusted for dividends and splits (equivalent to ``auto_adjust=True``).
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

START, END = "2019-07-01", "2025-07-01"
TARGET = "Close"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
TA = ["SMA10", "SMA20", "EMA10", "EMA20", "RSI14", "MACD", "MACD_sig", "ret1", "vol10"]
EXT = ["JKSE", "USDIDR"]
FEATURES = OHLCV + TA + EXT                     # 16 features used in the paper
FEATURE_SETS = {"Close only": ["Close"], "OHLCV": OHLCV, "OHLCV + TA": OHLCV + TA, "OHLCV + TA + EXT": FEATURES}
STOCKS = ["SIDO.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]
EXOGENOUS = {"JKSE": "^JKSE", "USDIDR": "USDIDR=X"}


def _chart_api(ticker: str, start: str, end: str) -> pd.DataFrame:
    import requests
    p1, p2 = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    last = None
    for host in ("query1", "query2"):
        r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}",
                         params=dict(period1=p1, period2=p2, interval="1d", events="div,splits"),
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"},
                         timeout=30)
        last = r.status_code
        if r.status_code != 200:
            continue
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        d = pd.DataFrame({"Date": pd.to_datetime(res["timestamp"], unit="s"), "Open": q["open"], "High": q["high"],
                          "Low": q["low"], "Close": q["close"], "Volume": q["volume"], "Adj": adj}).dropna()
        factor = d["Adj"] / d["Close"]
        for c in ("Open", "High", "Low", "Close"):
            d[c] = d[c] * factor
        d["Date"] = (d["Date"] + pd.Timedelta(hours=7)).dt.normalize()
        return d.drop(columns="Adj")
    raise RuntimeError(f"chart API status {last}")


def download(ticker: str, start: str = START, end: str = END, cols=tuple(OHLCV), tries: int = 6) -> pd.DataFrame:
    """Download adjusted daily prices; alternates yfinance.download, yfinance.Ticker and the chart endpoint."""
    last = None
    for i in range(tries):
        try:
            m = i % 3
            if m == 2:
                d = _chart_api(ticker, start, end)
            else:
                import yfinance as yf
                d = (yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False) if m == 0
                     else yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True))
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if "Date" not in d.columns:
                d = d.reset_index()
                if "Date" not in d.columns:
                    d = d.rename(columns={d.columns[0]: "Date"})
            d["Date"] = pd.to_datetime(d["Date"])
            if getattr(d["Date"].dt, "tz", None) is not None:
                d["Date"] = d["Date"].dt.tz_localize(None)
            d["Date"] = d["Date"].dt.normalize()
            if len(d) > 200:
                return d[["Date", *cols]].dropna().reset_index(drop=True)
            last = f"{len(d)} rows"
        except Exception as ex:  # noqa: BLE001
            last = repr(ex)[:160]
        time.sleep(10 * (i + 1) + random.random() * 10)
    raise RuntimeError(f"all download methods failed for {ticker}: {last}")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Nine causal technical indicators computed from current and past values only."""
    d = df.copy()
    c = d["Close"]
    d["SMA10"], d["SMA20"] = c.rolling(10).mean(), c.rolling(20).mean()
    d["EMA10"], d["EMA20"] = _ema(c, 10), _ema(c, 20)
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI14"] = 100 - 100 / (1 + rs)
    macd = _ema(c, 12) - _ema(c, 26)
    d["MACD"], d["MACD_sig"] = macd, _ema(macd, 9)
    d["ret1"] = c.pct_change()
    d["vol10"] = d["ret1"].rolling(10).std()
    return d


def load_stock(ticker: str = "SIDO.JK", start: str = START, end: str = END, exogenous: bool = True) -> pd.DataFrame:
    """OHLCV + technical indicators + JCI and USD/IDR (forward/back filled), warm-up rows removed."""
    d = add_technical(download(ticker, start, end))
    if exogenous:
        for name, tk in EXOGENOUS.items():
            e = download(tk, start, end, cols=("Close",)).rename(columns={"Close": name})
            d = d.merge(e, on="Date", how="left")
            d[name] = d[name].ffill().bfill()
    return d.dropna().reset_index(drop=True)
