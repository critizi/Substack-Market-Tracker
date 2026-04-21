"""
correlation.py — Correlation analysis for the Substack Finance Scraper.

For each ticker found in the database, computes three metrics that describe
how Substack mention activity relates to market behaviour:

    1. volume_spike_ratio   – average ratio of post-day volume to the
                              preceding 30-day average volume.
    2. avg_price_return_pct – average intraday return (close-open)/open on
                              the day of each mention, expressed as a
                              percentage.
    3. pearson_r / pearson_p – Pearson correlation between weekly mention
                               counts and weekly average volume.

Public API
----------
    compute_correlations(conn) -> pd.DataFrame
"""

from __future__ import annotations

import datetime
import math
from typing import Optional

import pandas as pd

from database import get_all_tickers, get_ticker_mentions, get_volume_snapshots

# ---------------------------------------------------------------------------
# Optional scipy import – degrade gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    from scipy.stats import pearsonr as _scipy_pearsonr

    def _pearsonr(x: list[float], y: list[float]) -> tuple[float, float]:
        r, p = _scipy_pearsonr(x, y)
        return float(r), float(p)

except Exception:  # ImportError or anything else at import time
    def _pearsonr(x: list[float], y: list[float]) -> tuple[float, float]:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iso_week(date_str: str) -> Optional[int]:
    """Return the ISO week number for a YYYY-MM-DD string, or None on error."""
    try:
        return datetime.date.fromisoformat(date_str[:10]).isocalendar()[1]
    except (ValueError, TypeError):
        return None


def _iso_year_week(date_str: str) -> Optional[tuple[int, int]]:
    """Return (ISO year, ISO week) for a YYYY-MM-DD string, or None on error.

    Using (year, week) rather than just week avoids collisions across year
    boundaries.
    """
    try:
        cal = datetime.date.fromisoformat(date_str[:10]).isocalendar()
        return (cal[0], cal[1])  # (isoyear, isoweek)
    except (ValueError, TypeError):
        return None


def _parse_date(date_str: str) -> Optional[datetime.date]:
    """Parse a YYYY-MM-DD (or longer ISO) string to a date, or None on error."""
    try:
        return datetime.date.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Per-ticker metric computations
# ---------------------------------------------------------------------------

def _compute_volume_spike_ratio(
    mentions: list[dict],
    snapshots: list[dict],
) -> float:
    """
    For each mention, find the closest volume snapshot to the mention date.
    Compute post_day_volume / 30d_avg_volume, where the 30-day average uses
    all snapshots strictly within the 30 days *before* that mention date.
    Return the mean spike ratio across all mentions; NaN if none could be
    computed.
    """
    if not snapshots:
        return float("nan")

    # Build a mapping date -> volume for quick lookup
    date_to_volume: dict[datetime.date, Optional[float]] = {}
    # Also keep a sorted list of (date, volume) for 30-day window lookups
    dated_volumes: list[tuple[datetime.date, float]] = []

    for snap in snapshots:
        d = _parse_date(snap["snapshot_date"])
        if d is None:
            continue
        vol = snap.get("volume")
        if vol is not None:
            try:
                vol = float(vol)
            except (TypeError, ValueError):
                vol = None
        date_to_volume[d] = vol
        if vol is not None:
            dated_volumes.append((d, vol))

    dated_volumes.sort(key=lambda x: x[0])

    if not dated_volumes:
        return float("nan")

    spike_ratios: list[float] = []

    for mention in mentions:
        mention_date = _parse_date(mention.get("published_date", ""))
        if mention_date is None:
            continue

        # Find the closest snapshot date to the mention date
        if not date_to_volume:
            continue

        closest_date = min(date_to_volume.keys(), key=lambda d: abs((d - mention_date).days))
        post_vol = date_to_volume.get(closest_date)
        if post_vol is None or post_vol == 0:
            continue

        # 30-day average: snapshots strictly before the mention date, within 30 days
        window_start = mention_date - datetime.timedelta(days=30)
        prior_vols = [
            v for (d, v) in dated_volumes
            if window_start <= d < mention_date
        ]
        if not prior_vols:
            # No prior snapshots – skip this mention
            continue

        avg_30d = sum(prior_vols) / len(prior_vols)
        if avg_30d == 0:
            continue

        spike_ratios.append(post_vol / avg_30d)

    if not spike_ratios:
        return float("nan")

    return sum(spike_ratios) / len(spike_ratios)


def _compute_avg_price_return(
    mentions: list[dict],
    snapshots: list[dict],
) -> float:
    """
    For each mention date, find the snapshot for exactly that date and compute
    (close - open) / open as a percentage.  Return the mean across all
    mentions; NaN if none could be computed.
    """
    # Build date -> (open, close) mapping
    date_to_ohlc: dict[datetime.date, tuple[float, float]] = {}
    for snap in snapshots:
        d = _parse_date(snap["snapshot_date"])
        if d is None:
            continue
        try:
            o = float(snap["open"])
            c = float(snap["close"])
        except (TypeError, ValueError):
            continue
        date_to_ohlc[d] = (o, c)

    returns: list[float] = []
    for mention in mentions:
        mention_date = _parse_date(mention.get("published_date", ""))
        if mention_date is None:
            continue
        ohlc = date_to_ohlc.get(mention_date)
        if ohlc is None:
            continue
        o, c = ohlc
        if o == 0:
            continue
        returns.append((c - o) / o * 100.0)

    if not returns:
        return float("nan")

    return sum(returns) / len(returns)


def _compute_pearson(
    mentions: list[dict],
    snapshots: list[dict],
) -> tuple[float, float]:
    """
    Compute Pearson r between weekly mention counts and weekly average volume.

    Both series are keyed by (ISO year, ISO week).  Only weeks present in
    *both* series are used.  Returns (NaN, NaN) if fewer than 3 overlapping
    weeks exist.
    """
    # --- weekly mention counts ---
    mention_weeks: dict[tuple[int, int], int] = {}
    for mention in mentions:
        yw = _iso_year_week(mention.get("published_date", ""))
        if yw is None:
            continue
        mention_weeks[yw] = mention_weeks.get(yw, 0) + 1

    # --- weekly average volumes ---
    volume_weeks: dict[tuple[int, int], list[float]] = {}
    for snap in snapshots:
        yw = _iso_year_week(snap.get("snapshot_date", ""))
        if yw is None:
            continue
        vol = snap.get("volume")
        if vol is None:
            continue
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            continue
        volume_weeks.setdefault(yw, []).append(vol)

    avg_volume_weeks: dict[tuple[int, int], float] = {
        yw: sum(vols) / len(vols) for yw, vols in volume_weeks.items()
    }

    # --- align on common weeks ---
    common_weeks = sorted(set(mention_weeks) & set(avg_volume_weeks))
    if len(common_weeks) < 3:
        return float("nan"), float("nan")

    mention_series = [float(mention_weeks[yw]) for yw in common_weeks]
    volume_series  = [avg_volume_weeks[yw]     for yw in common_weeks]

    # Guard against zero-variance series (pearsonr would raise/return NaN anyway)
    if len(set(mention_series)) < 2 or len(set(volume_series)) < 2:
        return float("nan"), float("nan")

    try:
        return _pearsonr(mention_series, volume_series)
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_correlations(conn) -> pd.DataFrame:
    """
    Compute per-ticker correlation metrics and return a DataFrame.

    Columns
    -------
    ticker               : str   – ticker symbol
    mention_count        : int   – total number of mention rows
    volume_spike_ratio   : float – mean (post-day volume / 30d-avg volume)
    avg_price_return_pct : float – mean intraday % return on mention days
    pearson_r            : float – Pearson r (weekly mentions vs weekly avg vol)
    pearson_p            : float – two-tailed p-value for pearson_r
    snapshot_count       : int   – number of volume snapshot rows

    Rows are sorted by pearson_r descending (NaN sorts to the bottom).
    Tickers with fewer than 3 mentions or fewer than 5 volume snapshots are
    skipped entirely.
    """
    tickers = get_all_tickers(conn)

    rows: list[dict] = []

    for ticker in tickers:
        mentions   = get_ticker_mentions(conn, ticker)
        snapshots  = get_volume_snapshots(conn, ticker)

        mention_count  = len(mentions)
        snapshot_count = len(snapshots)

        # Minimum-data guard
        if mention_count < 3 or snapshot_count < 5:
            continue

        volume_spike_ratio   = _compute_volume_spike_ratio(mentions, snapshots)
        avg_price_return_pct = _compute_avg_price_return(mentions, snapshots)
        pearson_r, pearson_p = _compute_pearson(mentions, snapshots)

        rows.append(
            {
                "ticker":               ticker,
                "mention_count":        mention_count,
                "volume_spike_ratio":   volume_spike_ratio,
                "avg_price_return_pct": avg_price_return_pct,
                "pearson_r":            pearson_r,
                "pearson_p":            pearson_p,
                "snapshot_count":       snapshot_count,
            }
        )

    if not rows:
        # Return an empty DataFrame with the correct schema
        return pd.DataFrame(
            columns=[
                "ticker",
                "mention_count",
                "volume_spike_ratio",
                "avg_price_return_pct",
                "pearson_r",
                "pearson_p",
                "snapshot_count",
            ]
        )

    df = pd.DataFrame(rows)

    # Sort by pearson_r descending; NaN values go to the bottom
    df.sort_values("pearson_r", ascending=False, na_position="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
