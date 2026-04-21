"""
database.py — SQLite database layer for the Substack Finance Scraper.

Provides schema initialization and all query helpers needed by the scraper.
"""

import sqlite3
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    substack_slug  TEXT    NOT NULL,
    title          TEXT,
    url            TEXT    UNIQUE NOT NULL,
    published_date TEXT,
    body_text      TEXT,
    scraped_at     TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_mentions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id         INTEGER REFERENCES posts(id),
    ticker          TEXT    NOT NULL,
    mention_context TEXT,
    match_type      TEXT    -- 'dollar_sign' or 'context_word'
);

CREATE TABLE IF NOT EXISTS volume_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    snapshot_date TEXT    NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        INTEGER,
    UNIQUE(ticker, snapshot_date)
);
"""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db(db_path: str = "./substack.db") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at *db_path*, apply the schema, and
    return the connection.

    The connection is configured with:
    - row_factory = sqlite3.Row  (so callers can access columns by name)
    - WAL journal mode            (better concurrent read performance)
    - Foreign-key enforcement enabled
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(_DDL)
    conn.commit()

    return conn


# ---------------------------------------------------------------------------
# posts helpers
# ---------------------------------------------------------------------------

def insert_post(
    conn: sqlite3.Connection,
    slug: str,
    title: Optional[str],
    url: str,
    published_date: Optional[str],
    body_text: Optional[str],
) -> int:
    """
    Insert a post row and return its *id*.

    Uses INSERT OR IGNORE so duplicate URLs are silently skipped.  In either
    case the existing row's id is returned, making the function idempotent.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO posts (substack_slug, title, url, published_date, body_text)
        VALUES (?, ?, ?, ?, ?)
        """,
        (slug, title, url, published_date, body_text),
    )
    conn.commit()

    row = conn.execute("SELECT id FROM posts WHERE url = ?", (url,)).fetchone()
    return row["id"]


# ---------------------------------------------------------------------------
# ticker_mentions helpers
# ---------------------------------------------------------------------------

def insert_ticker_mention(
    conn: sqlite3.Connection,
    post_id: int,
    ticker: str,
    context: Optional[str],
    match_type: Optional[str],
) -> None:
    """
    Insert a ticker mention linked to *post_id*.

    Duplicate mentions (same post + ticker + match_type) are allowed by design
    because the same ticker can be mentioned multiple times in one post.
    """
    conn.execute(
        """
        INSERT INTO ticker_mentions (post_id, ticker, mention_context, match_type)
        VALUES (?, ?, ?, ?)
        """,
        (post_id, ticker, context, match_type),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# volume_snapshots helpers
# ---------------------------------------------------------------------------

def insert_volume_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    open: Optional[float],
    high: Optional[float],
    low: Optional[float],
    close: Optional[float],
    volume: Optional[int],
) -> None:
    """
    Insert an OHLCV snapshot for *ticker* on *date*.

    The UNIQUE(ticker, snapshot_date) constraint means re-inserting an
    existing (ticker, date) pair is silently ignored.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO volume_snapshots
            (ticker, snapshot_date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, open, high, low, close, volume),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_all_tickers(conn: sqlite3.Connection) -> list[str]:
    """Return a sorted list of unique ticker symbols found in ticker_mentions."""
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM ticker_mentions ORDER BY ticker"
    ).fetchall()
    return [row["ticker"] for row in rows]


def get_ticker_mentions(conn: sqlite3.Connection, ticker: str) -> list[dict]:
    """
    Return every mention of *ticker* enriched with post metadata.

    Each dict contains:
        mention_id, ticker, mention_context, match_type,
        post_id, substack_slug, title, url, published_date, scraped_at
    """
    rows = conn.execute(
        """
        SELECT
            tm.id            AS mention_id,
            tm.ticker,
            tm.mention_context,
            tm.match_type,
            p.id             AS post_id,
            p.substack_slug,
            p.title,
            p.url,
            p.published_date,
            p.scraped_at
        FROM ticker_mentions tm
        JOIN posts p ON p.id = tm.post_id
        WHERE tm.ticker = ?
        ORDER BY p.published_date DESC, tm.id
        """,
        (ticker,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_volume_snapshots(conn: sqlite3.Connection, ticker: str) -> list[dict]:
    """
    Return all OHLCV snapshots for *ticker*, ordered by date ascending.

    Each dict contains: id, ticker, snapshot_date, open, high, low, close, volume
    """
    rows = conn.execute(
        """
        SELECT id, ticker, snapshot_date, open, high, low, close, volume
        FROM volume_snapshots
        WHERE ticker = ?
        ORDER BY snapshot_date
        """,
        (ticker,),
    ).fetchall()
    return [dict(row) for row in rows]
