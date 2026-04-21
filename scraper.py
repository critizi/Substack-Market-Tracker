import re
import time
from datetime import datetime, timezone, timedelta

import feedparser
import requests
from bs4 import BeautifulSoup

import database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKLIST = {
    "A", "I", "AT", "BE", "IT", "OR", "IN", "IS", "BY", "ON", "AN", "AM",
    "IF", "AS", "ARE", "THE", "AND", "BUT", "FOR", "NOT", "ALL", "NEW",
    "NOW", "CEO", "CFO", "COO", "GDP", "IPO", "ETF", "USD", "EUR", "GBP",
    "FED", "SEC", "IMF", "WHO", "ESG", "REITs",
}

# Matches $AAPL, $TSLA, etc.
DOLLAR_SIGN_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Matches "buying NVDA", "bullish on MSFT", etc.
CONTEXT_WORD_RE = re.compile(
    r"(?:buying|bullish on|bearish on|long|short|shares of|"
    r"position in|invested in|holds?|owns?)\s+([A-Z]{2,5})\b"
)

CONTEXT_WINDOW = 50  # characters on each side for surrounding context


# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Return plain text extracted from an HTML string."""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")


def _get_entry_body(entry) -> str:
    """Extract the full body text from a feedparser entry."""
    try:
        raw = entry.content[0].value
    except (AttributeError, IndexError, KeyError):
        raw = getattr(entry, "summary", "") or ""
    return _strip_html(raw)


def _parse_published(entry) -> str:
    """Return the published date as an ISO 8601 string (UTC), or today if missing."""
    try:
        struct = entry.published_parsed  # time.struct_time in UTC
        dt = datetime(*struct[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    except (AttributeError, TypeError):
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------

def fetch_feed(slug: str) -> list[dict]:
    """
    Fetch and parse the RSS feed for a Substack slug.

    Returns a list of post dicts with keys:
        title, published (ISO string), link, body
    Raises requests.RequestException on network failure.
    """
    url = f"https://{slug}.substack.com/feed"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    feed = feedparser.parse(response.content)

    posts = []
    for entry in feed.entries:
        posts.append({
            "title": getattr(entry, "title", ""),
            "published": _parse_published(entry),
            "link": getattr(entry, "link", ""),
            "body": _get_entry_body(entry),
        })
    return posts


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------

def _surrounding_context(text: str, match: re.Match) -> str:
    """Return up to CONTEXT_WINDOW characters on each side of a regex match."""
    start = max(0, match.start() - CONTEXT_WINDOW)
    end = min(len(text), match.end() + CONTEXT_WINDOW)
    return text[start:end].strip()


def extract_tickers(text: str) -> list[dict]:
    """
    Extract ticker mentions from plain text.

    Returns a list of dicts with keys:
        ticker, context, match_type
    Duplicates (same ticker + match_type) within a single post are kept so
    that every occurrence is stored independently.
    """
    mentions = []

    # Dollar-sign style: $AAPL
    for match in DOLLAR_SIGN_RE.finditer(text):
        ticker = match.group(1)
        if ticker in BLOCKLIST:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "dollar_sign",
        })

    # Context-word style: "buying NVDA"
    for match in CONTEXT_WORD_RE.finditer(text):
        ticker = match.group(1)
        if ticker in BLOCKLIST:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "context_word",
        })

    return mentions


# ---------------------------------------------------------------------------
# Main scraping entry point
# ---------------------------------------------------------------------------

def scrape_substacks(slugs: list[str], lookback_days: int, conn) -> int:
    """
    Scrape a list of Substack slugs, store results in the database.

    Args:
        slugs:         List of Substack publication slugs (e.g. ["doomberg", "moontower"]).
        lookback_days: Only posts published within this many days are stored.
        conn:          An open database connection passed to database helpers.

    Returns:
        Total number of posts scraped across all slugs.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    total_posts = 0

    for i, slug in enumerate(slugs):
        # Rate-limit: pause between requests (skip before the very first slug)
        if i > 0:
            time.sleep(1)

        try:
            posts = fetch_feed(slug)
        except Exception as exc:
            print(f"WARNING: could not fetch '{slug}': {exc}")
            continue

        slug_posts = 0
        slug_tickers = 0

        for post in posts:
            # Filter by lookback window
            try:
                published_dt = datetime.fromisoformat(post["published"])
                # Ensure timezone-aware for comparison
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                # Unparseable date — include it to be safe
                published_dt = datetime.now(timezone.utc)

            if published_dt < cutoff:
                continue

            # Persist the post and get back its row id
            post_id = database.insert_post(
                conn,
                slug=slug,
                title=post["title"],
                url=post["link"],
                published_date=post["published"],
                body_text=post["body"],
            )

            # Extract and persist ticker mentions
            mentions = extract_tickers(post["body"])
            for mention in mentions:
                database.insert_ticker_mention(
                    conn,
                    post_id=post_id,
                    ticker=mention["ticker"],
                    context=mention["context"],
                    match_type=mention["match_type"],
                )

            slug_posts += 1
            slug_tickers += len(mentions)

        print(f"{slug}: {slug_posts} posts, {slug_tickers} tickers found")
        total_posts += slug_posts

    return total_posts
