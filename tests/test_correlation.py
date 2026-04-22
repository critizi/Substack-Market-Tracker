"""
Tests for correlation.py.

Pure-function tests use crafted dicts directly.
Integration tests use an in-memory SQLite DB seeded via the database module.
"""

import math
import pytest

from database import init_db, insert_post, insert_ticker_mention, insert_volume_snapshot
from correlation import (
    _compute_volume_spike_ratio,
    _compute_avg_price_return,
    _compute_pearson,
    compute_correlations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db():
    """Fresh in-memory SQLite connection for each test."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


def _make_snapshots(dates_and_vols, open_=100.0, close=105.0):
    """Build snapshot dicts from a list of (date_str, volume) tuples."""
    return [
        {
            "snapshot_date": d,
            "volume": v,
            "open": open_,
            "close": close,
        }
        for d, v in dates_and_vols
    ]


def _make_mentions(dates):
    return [{"published_date": d} for d in dates]


# ---------------------------------------------------------------------------
# _compute_avg_price_return
# ---------------------------------------------------------------------------

class TestAvgPriceReturn:
    def test_basic(self):
        mentions  = _make_mentions(["2024-01-10"])
        snapshots = _make_snapshots([("2024-01-10", 1000)], open_=100.0, close=105.0)
        assert abs(_compute_avg_price_return(mentions, snapshots) - 5.0) < 1e-9

    def test_negative_return(self):
        mentions  = _make_mentions(["2024-01-10"])
        snapshots = _make_snapshots([("2024-01-10", 1000)], open_=100.0, close=90.0)
        assert abs(_compute_avg_price_return(mentions, snapshots) - (-10.0)) < 1e-9

    def test_multiple_mentions_averaged(self):
        mentions  = _make_mentions(["2024-01-10", "2024-01-11"])
        snapshots = _make_snapshots(
            [("2024-01-10", 1000), ("2024-01-11", 2000)],
            open_=100.0,
            close=110.0,  # 10% each day
        )
        assert abs(_compute_avg_price_return(mentions, snapshots) - 10.0) < 1e-9

    def test_no_matching_snapshot_returns_nan(self):
        mentions  = _make_mentions(["2024-01-10"])
        snapshots = _make_snapshots([("2024-01-15", 1000)])
        assert math.isnan(_compute_avg_price_return(mentions, snapshots))

    def test_zero_open_skipped(self):
        mentions  = _make_mentions(["2024-01-10"])
        snapshots = [{"snapshot_date": "2024-01-10", "volume": 500, "open": 0, "close": 5}]
        assert math.isnan(_compute_avg_price_return(mentions, snapshots))

    def test_empty_mentions(self):
        snapshots = _make_snapshots([("2024-01-10", 1000)])
        assert math.isnan(_compute_avg_price_return([], snapshots))

    def test_bad_date_in_mention_skipped(self):
        mentions  = [{"published_date": "not-a-date"}, {"published_date": "2024-01-10"}]
        snapshots = _make_snapshots([("2024-01-10", 1000)], open_=100.0, close=110.0)
        assert abs(_compute_avg_price_return(mentions, snapshots) - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# _compute_volume_spike_ratio
# ---------------------------------------------------------------------------

class TestVolumeSpikeRatio:
    def test_basic(self):
        # 30-day prior avg = 100, post-day vol = 200  → ratio 2.0
        mentions = _make_mentions(["2024-02-01"])
        snapshots = _make_snapshots(
            [("2024-01-02", 100), ("2024-01-15", 100), ("2024-02-01", 200)]
        )
        result = _compute_volume_spike_ratio(mentions, snapshots)
        assert abs(result - 2.0) < 1e-9

    def test_no_prior_snapshots_returns_nan(self):
        # mention is before any snapshot
        mentions  = _make_mentions(["2024-01-01"])
        snapshots = _make_snapshots([("2024-01-05", 500), ("2024-01-10", 500)])
        assert math.isnan(_compute_volume_spike_ratio(mentions, snapshots))

    def test_empty_snapshots_returns_nan(self):
        mentions = _make_mentions(["2024-01-10"])
        assert math.isnan(_compute_volume_spike_ratio(mentions, []))

    def test_zero_avg_skipped(self):
        mentions  = _make_mentions(["2024-02-01"])
        snapshots = _make_snapshots([("2024-01-15", 0), ("2024-02-01", 200)])
        # avg_30d == 0, so mention is skipped → NaN
        assert math.isnan(_compute_volume_spike_ratio(mentions, snapshots))

    def test_multiple_mentions_averaged(self):
        snapshots = _make_snapshots(
            [
                ("2024-01-02", 100),
                ("2024-01-15", 100),
                ("2024-02-01", 300),  # ratio 3.0
                ("2024-02-15", 100),
                ("2024-02-28", 100),
                ("2024-03-15", 200),  # prior avg ~133, ratio ~1.5
            ]
        )
        mentions = _make_mentions(["2024-02-01", "2024-03-15"])
        result = _compute_volume_spike_ratio(mentions, snapshots)
        assert not math.isnan(result)
        assert result > 1.0


# ---------------------------------------------------------------------------
# _compute_pearson
# ---------------------------------------------------------------------------

class TestComputePearson:
    def _spread_mentions(self, year_weeks):
        """Turn [(year, week, count)] into a flat mention list."""
        mentions = []
        for (y, w, count) in year_weeks:
            # First day of that ISO week
            import datetime
            d = datetime.date.fromisocalendar(y, w, 1).isoformat()
            mentions.extend([{"published_date": d}] * count)
        return mentions

    def _spread_snapshots(self, year_weeks_vols):
        """Turn [(year, week, volume)] into snapshot list (one per week)."""
        import datetime
        snaps = []
        for (y, w, v) in year_weeks_vols:
            d = datetime.date.fromisocalendar(y, w, 1).isoformat()
            snaps.append({"snapshot_date": d, "volume": v, "open": 100, "close": 100})
        return snaps

    def test_perfect_positive_correlation(self):
        # mentions and volume move in lockstep across 5 weeks
        data = [(2024, w, w) for w in range(1, 6)]
        mentions  = self._spread_mentions(data)
        snapshots = self._spread_snapshots(data)
        r, p = _compute_pearson(mentions, snapshots)
        assert abs(r - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        mentions_data  = [(2024, w, w)     for w in range(1, 6)]
        snapshots_data = [(2024, w, 6 - w) for w in range(1, 6)]
        r, p = _compute_pearson(
            self._spread_mentions(mentions_data),
            self._spread_snapshots(snapshots_data),
        )
        assert abs(r - (-1.0)) < 1e-6

    def test_fewer_than_3_weeks_returns_nan(self):
        data = [(2024, 1, 2), (2024, 2, 3)]
        mentions  = self._spread_mentions(data)
        snapshots = self._spread_snapshots(data)
        r, p = _compute_pearson(mentions, snapshots)
        assert math.isnan(r)

    def test_no_overlap_returns_nan(self):
        mentions  = self._spread_mentions([(2024, w, 1) for w in range(1, 6)])
        snapshots = self._spread_snapshots([(2024, w, 100) for w in range(10, 15)])
        r, p = _compute_pearson(mentions, snapshots)
        assert math.isnan(r)

    def test_constant_mention_series_returns_nan(self):
        # zero variance in mentions → Pearson undefined
        data = [(2024, w, 1) for w in range(1, 6)]  # all counts == 1
        r, p = _compute_pearson(
            self._spread_mentions(data),
            self._spread_snapshots(data),
        )
        assert math.isnan(r)


# ---------------------------------------------------------------------------
# compute_correlations (integration)
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def _seed(self, conn, ticker, mention_dates, snapshots):
        """Insert the minimum rows needed for compute_correlations."""
        post_id = insert_post(conn, "test-pub", "Test Post", f"https://example.com/{ticker}", None, None)
        for date in mention_dates:
            conn.execute(
                "UPDATE posts SET published_date = ? WHERE id = ?", (date, post_id)
            )
            insert_ticker_mention(conn, post_id, ticker, None, "dollar_sign")
        for s in snapshots:
            insert_volume_snapshot(
                conn, ticker, s["date"], s["open"], None, None, s["close"], s["volume"]
            )
        conn.commit()

    def test_returns_dataframe_with_correct_columns(self, mem_db):
        import datetime
        dates = [datetime.date.fromisocalendar(2024, w, 1).isoformat() for w in range(1, 8)]
        snaps = [
            {"date": d, "open": 100.0, "close": 105.0, "volume": 1000 + i * 100}
            for i, d in enumerate(dates)
        ]
        self._seed(mem_db, "AAPL", dates[:5], snaps)

        df = compute_correlations(mem_db)
        expected_cols = {
            "ticker", "mention_count", "volume_spike_ratio",
            "avg_price_return_pct", "pearson_r", "pearson_p", "snapshot_count",
        }
        assert expected_cols == set(df.columns)

    def test_ticker_with_too_few_mentions_excluded(self, mem_db):
        snaps = [{"date": f"2024-01-{i:02d}", "open": 100.0, "close": 105.0, "volume": 1000}
                 for i in range(1, 10)]
        # Only 2 mentions — below the 3-mention floor
        self._seed(mem_db, "XYZ", ["2024-01-05", "2024-01-06"], snaps)
        df = compute_correlations(mem_db)
        assert "XYZ" not in df["ticker"].values

    def test_ticker_with_too_few_snapshots_excluded(self, mem_db):
        # 5 mentions but only 3 snapshots — below the 5-snapshot floor
        snaps = [{"date": f"2024-01-{i:02d}", "open": 100.0, "close": 105.0, "volume": 1000}
                 for i in range(1, 4)]
        self._seed(mem_db, "LOW", ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], snaps)
        df = compute_correlations(mem_db)
        assert "LOW" not in df["ticker"].values

    def test_empty_db_returns_empty_dataframe(self, mem_db):
        df = compute_correlations(mem_db)
        assert df.empty
        assert list(df.columns) == [
            "ticker", "mention_count", "volume_spike_ratio",
            "avg_price_return_pct", "pearson_r", "pearson_p", "snapshot_count",
        ]

    def test_sorted_by_pearson_r_descending(self, mem_db):
        import datetime

        def seed_ticker(ticker, vol_per_week):
            """One post per week so each mention gets its own published_date."""
            for w, vol in enumerate(vol_per_week, start=1):
                d = datetime.date.fromisocalendar(2024, w, 1).isoformat()
                post_id = insert_post(
                    mem_db, "pub", f"{ticker}-w{w}",
                    f"https://example.com/{ticker}/w{w}", d, None,
                )
                # mention count grows week-over-week for both tickers
                for _ in range(w):
                    insert_ticker_mention(mem_db, post_id, ticker, None, "dollar_sign")
                insert_volume_snapshot(mem_db, ticker, d, 100.0, None, None, 105.0, vol)
            mem_db.commit()

        # GOOD: volume increases with week number → r ≈ +1
        seed_ticker("GOOD", [1100, 1200, 1300, 1400, 1500, 1600, 1700])
        # BAD: volume decreases as week number rises → r ≈ -1
        seed_ticker("BAD",  [1700, 1600, 1500, 1400, 1300, 1200, 1100])

        df = compute_correlations(mem_db)
        tickers = df["ticker"].tolist()
        assert tickers.index("GOOD") < tickers.index("BAD")
