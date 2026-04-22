"""Tests for scraper.py."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from database import init_db
from scraper import (
    _detect_paywall,
    _get_entry_body,
    _parse_published,
    _strip_html,
    _surrounding_context,
    extract_tickers,
    fetch_feed,
    scrape_substacks,
    BLOCKLIST,
    COMPANY_NAME_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_db():
    conn = init_db(":memory:")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# _detect_paywall
# ---------------------------------------------------------------------------

class TestDetectPaywall:
    def test_subscribe_to_read(self):
        assert _detect_paywall("Subscribe to read the full article.") is True

    def test_subscribe_to_continue(self):
        assert _detect_paywall("Subscribe to continue reading.") is True

    def test_paid_subscribers(self):
        assert _detect_paywall("This post is for paid subscribers only.") is True

    def test_subscribers_without_paid(self):
        assert _detect_paywall("This post is for subscribers only.") is True

    def test_become_subscriber(self):
        assert _detect_paywall("Become a paid subscriber to unlock this content.") is True

    def test_upgrade_subscription(self):
        assert _detect_paywall("Upgrade your subscription to access this post.") is True

    def test_free_article_not_flagged(self):
        assert _detect_paywall("AAPL reported earnings today, beating estimates.") is False

    def test_empty_string(self):
        assert _detect_paywall("") is False

    def test_case_insensitive(self):
        assert _detect_paywall("SUBSCRIBE TO READ MORE") is True

    def test_partial_match_not_flagged(self):
        assert _detect_paywall("I subscribe to this newsletter voluntarily.") is False


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_tags(self):
        result = _strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result and "world" in result

    def test_plain_text_unchanged(self):
        assert _strip_html("no html here") == "no html here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_nested_tags(self):
        result = _strip_html("<div><ul><li>item</li></ul></div>")
        assert "item" in result
        assert "<" not in result


# ---------------------------------------------------------------------------
# _surrounding_context
# ---------------------------------------------------------------------------

class TestSurroundingContext:
    def _match(self, text, pattern):
        import re
        return re.search(pattern, text)

    def test_returns_text_around_match(self):
        text = "a" * 60 + "MATCH" + "b" * 60
        m = self._match(text, "MATCH")
        ctx = _surrounding_context(text, m)
        assert "MATCH" in ctx
        assert ctx.startswith("a")
        assert ctx.endswith("b")

    def test_clips_at_start_of_string(self):
        text = "MATCH" + "x" * 100
        m = self._match(text, "MATCH")
        ctx = _surrounding_context(text, m)
        assert ctx.startswith("MATCH")

    def test_clips_at_end_of_string(self):
        text = "x" * 100 + "MATCH"
        m = self._match(text, "MATCH")
        ctx = _surrounding_context(text, m)
        assert ctx.endswith("MATCH")


# ---------------------------------------------------------------------------
# _parse_published
# ---------------------------------------------------------------------------

class TestParsePublished:
    def _entry(self, struct):
        e = MagicMock()
        e.published_parsed = struct
        return e

    def test_valid_struct(self):
        struct = time.strptime("2024-03-15 12:00:00", "%Y-%m-%d %H:%M:%S")
        result = _parse_published(self._entry(struct))
        assert "2024-03-15" in result

    def test_missing_attribute_returns_today(self):
        e = MagicMock(spec=[])  # no attributes
        result = _parse_published(e)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in result

    def test_none_attribute_returns_today(self):
        e = MagicMock()
        e.published_parsed = None
        result = _parse_published(e)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in result


# ---------------------------------------------------------------------------
# _get_entry_body
# ---------------------------------------------------------------------------

class TestGetEntryBody:
    def test_uses_content_field(self):
        entry = MagicMock()
        entry.content = [MagicMock(value="<p>Hello</p>")]
        assert _get_entry_body(entry) == "Hello"

    def test_falls_back_to_summary(self):
        entry = MagicMock(spec=["summary"])
        entry.summary = "<b>Summary text</b>"
        assert _get_entry_body(entry) == "Summary text"

    def test_empty_when_nothing_available(self):
        entry = MagicMock(spec=[])
        assert _get_entry_body(entry) == ""


# ---------------------------------------------------------------------------
# extract_tickers
# ---------------------------------------------------------------------------

class TestExtractTickers:
    def test_dollar_sign_match(self):
        results = extract_tickers("I love $AAPL and $TSLA right now.")
        tickers = {r["ticker"] for r in results}
        assert tickers == {"AAPL", "TSLA"}

    def test_context_word_match(self):
        results = extract_tickers("I am buying NVDA this week.")
        assert any(r["ticker"] == "NVDA" and r["match_type"] == "context_word" for r in results)

    def test_blocklist_dollar_sign_excluded(self):
        results = extract_tickers("$THE $AND $FED are all blocklisted.")
        assert all(r["ticker"] not in {"THE", "AND", "FED"} for r in results)

    def test_blocklist_context_word_excluded(self):
        results = extract_tickers("I am buying CEO this week.")
        assert not any(r["ticker"] == "CEO" for r in results)

    def test_match_type_recorded(self):
        results = extract_tickers("$MSFT and bullish on GOOG")
        types = {r["ticker"]: r["match_type"] for r in results}
        assert types.get("MSFT") == "dollar_sign"
        assert types.get("GOOG") == "context_word"

    def test_duplicate_mentions_kept(self):
        results = extract_tickers("$AAPL is up. I also like $AAPL a lot.")
        aapl = [r for r in results if r["ticker"] == "AAPL"]
        assert len(aapl) == 2

    def test_context_snippet_included(self):
        text = "some text before $AAPL and some after"
        results = extract_tickers(text)
        assert any("AAPL" in r["context"] for r in results)

    def test_no_tickers_returns_empty(self):
        assert extract_tickers("The quick brown fox.") == []

    def test_empty_string(self):
        assert extract_tickers("") == []

    def test_ticker_too_long_not_matched(self):
        # Dollar sign regex only matches 1-5 uppercase letters
        results = extract_tickers("$TOOLONG")
        assert not any(r["ticker"] == "TOOLONG" for r in results)

    def test_all_context_words(self):
        phrases = [
            "long AMZN", "short META", "shares of NFLX",
            "position in COIN", "invested in HOOD",
            "holds PLTR", "owns UBER",
        ]
        for phrase in phrases:
            results = extract_tickers(phrase)
            # A phrase may produce both a context_word and company_name match; at minimum
            # there should be exactly one context_word match.
            context_word_hits = [r for r in results if r["match_type"] == "context_word"]
            assert len(context_word_hits) == 1, f"Expected 1 context_word result for: {phrase}"

    def test_analyst_language_context_words(self):
        phrases = [
            ("overweight NVDA", "NVDA"),
            ("underweight INTC", "INTC"),
            ("Overweight MSFT", "MSFT"),   # capital first letter
            ("outperforming AAPL", "AAPL"),
            ("accumulating TSM", "TSM"),
            ("trimming AMZN", "AMZN"),
            ("exposed to SMCI", "SMCI"),
            ("stake in PLTR", "PLTR"),
        ]
        for phrase, expected_ticker in phrases:
            results = extract_tickers(phrase)
            tickers = [r["ticker"] for r in results]
            assert expected_ticker in tickers, f"Expected {expected_ticker} in: {phrase!r}"
            assert any(r["match_type"] == "context_word" for r in results if r["ticker"] == expected_ticker)

    def test_exchange_prefix(self):
        results = extract_tickers("The company NYSE: AAPL reported strong earnings.")
        assert any(r["ticker"] == "AAPL" and r["match_type"] == "exchange_prefix" for r in results)

    def test_exchange_prefix_nasdaq(self):
        results = extract_tickers("Nvidia (NASDAQ: NVDA) beat estimates.")
        assert any(r["ticker"] == "NVDA" for r in results)

    def test_parenthetical_ticker(self):
        results = extract_tickers("Apple (AAPL) reported record revenue.")
        assert any(r["ticker"] == "AAPL" and r["match_type"] == "parenthetical" for r in results)

    def test_parenthetical_blocklisted_skipped(self):
        results = extract_tickers("The company (CEO) resigned.")
        assert not any(r["ticker"] == "CEO" for r in results)

    def test_tech_jargon_blocklisted(self):
        jargon = ["CPU", "GPU", "DRAM", "HBM", "ASIC", "API", "SDK", "LLM", "MMA"]
        for term in jargon:
            assert term in BLOCKLIST, f"{term} should be in BLOCKLIST"

    def test_tech_jargon_not_extracted(self):
        text = "The GPU and CPU are key to AI inference. HBM enables fast DRAM access."
        results = extract_tickers(text)
        tickers = {r["ticker"] for r in results}
        assert not tickers.intersection({"GPU", "CPU", "HBM", "DRAM"})

    def test_optical_jargon_blocklisted(self):
        assert "ELS" in BLOCKLIST, "ELS (external light source) should be in BLOCKLIST"
        assert "PD" in BLOCKLIST, "PD (physical design / prefill-decode) should be in BLOCKLIST"
        assert "OE" in BLOCKLIST, "OE (optical engine) should be in BLOCKLIST"

    def test_optical_jargon_not_extracted(self):
        text = "optical engines (OEs) and external light source (ELS) modules"
        results = extract_tickers(text)
        tickers = {r["ticker"] for r in results}
        assert "ELS" not in tickers
        assert "OES" not in tickers

    def test_paren_minimum_three_chars(self):
        # 2-letter parenthetical should not match (e.g. "prefill-decode (PD)")
        results = extract_tickers("prefill-decode (PD) disaggregation")
        assert not any(r["ticker"] == "PD" for r in results)

    # Company name matching
    def test_company_name_nvidia(self):
        results = extract_tickers("Nvidia announced a new GPU.")
        assert any(r["ticker"] == "NVDA" and r["match_type"] == "company_name" for r in results)

    def test_company_name_case_insensitive(self):
        results = extract_tickers("NVIDIA reported earnings. nvidia stock surged.")
        nvda = [r for r in results if r["ticker"] == "NVDA" and r["match_type"] == "company_name"]
        assert len(nvda) == 2

    def test_company_name_tsmc_and_alias(self):
        results = extract_tickers("Taiwan Semiconductor and TSMC are both mentioned.")
        tickers = [r["ticker"] for r in results if r["match_type"] == "company_name"]
        assert tickers.count("TSM") == 2

    def test_company_name_multiword(self):
        results = extract_tickers("Applied Materials reported strong guidance.")
        assert any(r["ticker"] == "AMAT" and r["match_type"] == "company_name" for r in results)

    def test_company_name_private_skipped(self):
        # OpenAI and Anthropic map to None — should not produce a mention
        results = extract_tickers("OpenAI and Anthropic are both private companies.")
        tickers = {r["ticker"] for r in results}
        assert None not in tickers
        assert "OPENAI" not in tickers

    def test_company_name_no_blocklist_check(self):
        # PagerDuty maps to "PD" which is in BLOCKLIST, but company_name bypasses blocklist
        results = extract_tickers("PagerDuty reported strong earnings this quarter.")
        assert any(r["ticker"] == "PD" and r["match_type"] == "company_name" for r in results)

    def test_company_name_map_no_none_tickers_stored(self):
        # All non-None values in the map should be valid-looking ticker strings
        for name, ticker in COMPANY_NAME_MAP.items():
            if ticker is not None:
                assert isinstance(ticker, str) and len(ticker) >= 1, f"Bad ticker for {name!r}"

    def test_company_name_context_snippet(self):
        text = "Palantir continues to win government contracts across the board."
        results = extract_tickers(text)
        assert any(r["ticker"] == "PLTR" and "Palantir" in r["context"] for r in results)


# ---------------------------------------------------------------------------
# fetch_feed
# ---------------------------------------------------------------------------

class TestFetchFeed:
    def _fake_response(self, xml: str):
        resp = MagicMock()
        resp.content = xml.encode()
        resp.raise_for_status = MagicMock()
        return resp

    def _rss(self, entries_xml=""):
        return f"""<?xml version="1.0"?>
        <rss version="2.0"><channel>
            <title>Test Feed</title>
            {entries_xml}
        </channel></rss>"""

    def test_returns_list_of_posts(self):
        xml = self._rss("""
            <item>
                <title>Post 1</title>
                <link>https://example.substack.com/p/post1</link>
                <pubDate>Mon, 15 Jan 2024 12:00:00 +0000</pubDate>
                <description>Hello world</description>
            </item>
        """)
        with patch("scraper.requests.get", return_value=self._fake_response(xml)):
            posts = fetch_feed("example")
        assert len(posts) == 1
        assert posts[0]["title"] == "Post 1"
        assert posts[0]["link"] == "https://example.substack.com/p/post1"

    def test_empty_feed_returns_empty_list(self):
        with patch("scraper.requests.get", return_value=self._fake_response(self._rss())):
            posts = fetch_feed("empty-pub")
        assert posts == []

    def test_network_error_propagates(self):
        import requests
        with patch("scraper.requests.get", side_effect=requests.RequestException("timeout")):
            with pytest.raises(requests.RequestException):
                fetch_feed("bad-slug")

    def test_post_has_required_keys(self):
        xml = self._rss("""
            <item>
                <title>My Post</title>
                <link>https://x.substack.com/p/1</link>
                <pubDate>Mon, 15 Jan 2024 12:00:00 +0000</pubDate>
                <description>body text</description>
            </item>
        """)
        with patch("scraper.requests.get", return_value=self._fake_response(xml)):
            posts = fetch_feed("x")
        assert {"title", "published", "link", "body", "is_paywalled"} == set(posts[0].keys())


# ---------------------------------------------------------------------------
# scrape_substacks (integration)
# ---------------------------------------------------------------------------

class TestScrapeSubstacks:
    def _fake_posts(self, slug, count=2):
        return [
            {
                "title": f"{slug} post {i}",
                "published": "2099-01-01T00:00:00+00:00",  # far future → always within lookback
                "link": f"https://{slug}.substack.com/p/{i}",
                "body": f"I love $AAPL — post {i}",
            }
            for i in range(count)
        ]

    def test_returns_total_post_count(self, mem_db):
        with patch("scraper.fetch_feed", side_effect=lambda slug: self._fake_posts(slug)):
            count = scrape_substacks(["pub-a", "pub-b"], lookback_days=9999, conn=mem_db)
        assert count == 4

    def test_posts_written_to_db(self, mem_db):
        with patch("scraper.fetch_feed", return_value=self._fake_posts("mypub", count=3)):
            scrape_substacks(["mypub"], lookback_days=9999, conn=mem_db)
        rows = mem_db.execute("SELECT * FROM posts").fetchall()
        assert len(rows) == 3

    def test_ticker_mentions_written_to_db(self, mem_db):
        with patch("scraper.fetch_feed", return_value=self._fake_posts("mypub", count=2)):
            scrape_substacks(["mypub"], lookback_days=9999, conn=mem_db)
        rows = mem_db.execute("SELECT * FROM ticker_mentions").fetchall()
        assert len(rows) == 2  # one $AAPL mention per post

    def test_old_posts_excluded(self, mem_db):
        old_post = [{
            "title": "Old",
            "published": "2000-01-01T00:00:00+00:00",
            "link": "https://pub.substack.com/p/old",
            "body": "$TSLA is interesting",
        }]
        with patch("scraper.fetch_feed", return_value=old_post):
            count = scrape_substacks(["pub"], lookback_days=30, conn=mem_db)
        assert count == 0

    def test_failed_slug_skipped_others_continue(self, mem_db):
        import requests as req

        def side_effect(slug):
            if slug == "bad":
                raise req.RequestException("timeout")
            return self._fake_posts(slug, count=1)

        with patch("scraper.fetch_feed", side_effect=side_effect):
            count = scrape_substacks(["bad", "good"], lookback_days=9999, conn=mem_db)
        assert count == 1
