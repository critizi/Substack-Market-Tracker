"""
market_data.py — yfinance OHLCV pulls + volume snapshots

Public API
----------
fetch_ohlcv(ticker, dates, window=5) -> pd.DataFrame
pull_market_data(conn, window=5)     -> int  (total snapshots stored)
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range_for_windows(dates: list[str], window: int) -> tuple[str, str]:
    """
    Given a list of ISO date strings and a ±window (calendar days), return
    the overall (start, end) pair that covers every per-date window.

    Returns (start_str, end_str) where end_str is already +1 day so it can be
    passed directly to yfinance (which uses an exclusive upper bound).
    """
    parsed = [datetime.fromisoformat(d) for d in dates]
    earliest = min(parsed) - timedelta(days=window)
    latest   = max(parsed) + timedelta(days=window) + timedelta(days=1)  # exclusive end
    return earliest.strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d")


def _dates_in_any_window(all_dates: pd.Index, mention_dates: list[str], window: int) -> pd.Index:
    """
    Filter a DatetimeIndex to only rows that fall within ±window calendar days
    of at least one mention date.
    """
    parsed_mentions = [datetime.fromisoformat(d) for d in mention_dates]
    mask = pd.Series(False, index=all_dates)
    for mention_dt in parsed_mentions:
        lower = pd.Timestamp(mention_dt - timedelta(days=window))
        upper = pd.Timestamp(mention_dt + timedelta(days=window))
        mask |= (all_dates >= lower) & (all_dates <= upper)
    return all_dates[mask.values]


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, dates: list[str], window: int = 5) -> pd.DataFrame:
    """
    Download OHLCV data for *ticker* covering a ±window calendar-day window
    around each date in *dates*.

    Parameters
    ----------
    ticker : str
        Equity / ETF ticker symbol (e.g. "AAPL").
    dates  : list[str]
        ISO-formatted date strings (YYYY-MM-DD).
    window : int
        Number of calendar days to expand around each mention date.

    Returns
    -------
    pd.DataFrame
        Columns: Date (str YYYY-MM-DD), Open, High, Low, Close, Volume.
        Returns an empty DataFrame if yfinance returns nothing or raises.
    """
    _EMPTY = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    if not dates:
        return _EMPTY

    start, end = _date_range_for_windows(dates, window)

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            # Suppress the multi-level column header that newer yfinance emits
            # when only one ticker is requested.
        )
    except Exception as exc:
        print(f"WARNING [{ticker}]: yfinance raised {type(exc).__name__}: {exc}")
        return _EMPTY

    if raw is None or raw.empty:
        return _EMPTY

    # yfinance ≥ 0.2.x may return a MultiIndex column when a single ticker is
    # passed; flatten it if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Ensure the standard columns exist.
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(raw.columns)):
        return _EMPTY

    # The index is a DatetimeIndex; keep only rows inside the per-date windows.
    keep_idx = _dates_in_any_window(raw.index, dates, window)
    df = raw.loc[keep_idx, list(required)].copy()

    if df.empty:
        return _EMPTY

    # Reset index and stringify the Date column.
    df = df.reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)

    # The index column might be named "Date" already (common in recent yfinance).
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    if date_col != "Date":
        df.rename(columns={date_col: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Guarantee column order.
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# pull_market_data
# ---------------------------------------------------------------------------

def pull_market_data(conn, window: int = 5) -> int:
    """
    Iterate over every ticker in the database, fetch OHLCV windows around each
    mention date, and persist them as volume snapshots.

    Parameters
    ----------
    conn   : sqlite3.Connection (or compatible)
        Open database connection passed through to database.*  helpers.
    window : int
        Calendar-day expansion passed to fetch_ohlcv.

    Returns
    -------
    int
        Total number of snapshot rows stored across all tickers.
    """
    tickers = database.get_all_tickers(conn)
    total_stored = 0

    for ticker in tickers:
        try:
            mentions = database.get_ticker_mentions(conn, ticker)
            mention_dates = [m["published_date"] for m in mentions if m.get("published_date")]

            if not mention_dates:
                print(f"{ticker}: no mention dates (skipped)")
                continue

            df = fetch_ohlcv(ticker, mention_dates, window)

            if df.empty:
                print(f"{ticker}: no data (skipped)")
                time.sleep(0.5)
                continue

            rows_stored = 0
            for _, row in df.iterrows():
                database.insert_volume_snapshot(
                    conn,
                    ticker,
                    row["Date"],
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    int(row["Volume"]),
                )
                rows_stored += 1

            total_stored += rows_stored
            print(f"{ticker}: {rows_stored} snapshots stored")

        except Exception as exc:
            print(f"WARNING [{ticker}]: unexpected error — {type(exc).__name__}: {exc}")

        time.sleep(0.5)

    return total_stored
