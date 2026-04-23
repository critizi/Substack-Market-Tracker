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
    # Common English words / articles
    "A", "I", "AT", "BE", "IT", "OR", "IN", "IS", "BY", "ON", "AN", "AM",
    "IF", "AS", "ARE", "THE", "AND", "BUT", "FOR", "NOT", "ALL", "NEW",
    "NOW",
    # Finance / regulatory acronyms
    "CEO", "CFO", "COO", "CTO", "GDP", "IPO", "ETF", "USD", "EUR", "GBP",
    "FED", "SEC", "IMF", "WHO", "ESG", "REIT", "REITs", "PE", "VC", "EPS",
    "CAPEX", "OPEX", "EBITDA",
    # SaaS / business metrics
    "ARR", "MRR", "CAC", "LTV", "GMV", "FCF", "NPS", "TAM", "SAM",
    "LBO", "DCF", "IRR", "TTM", "NDA", "EV",
    # Hardware / semiconductor jargon that appears in tech newsletters
    "CPU", "GPU", "NPU", "TPU", "ASIC", "FPGA",
    "DRAM", "HBM", "DDR", "NAND", "SSD", "HDD", "RAM", "ROM",
    "PCIe", "CXL", "NVMe",
    # Optical / photonics jargon (semianalysis, damnang newsletters)
    "ELS",  # external light source
    "PD",   # physical design / prefill-decode
    "OE",   # optical engine
    "CW",   # continuous wave (laser)
    "WDM",  # wavelength-division multiplexing
    "DSP",  # digital signal processor
    "TIA",  # transimpedance amplifier
    "FEC",  # forward error correction
    "DAC",  # digital-to-analog converter
    "ADC",  # analog-to-digital converter
    # Networking / software
    "API", "SDK", "USB", "OTA", "LTE", "NFC",
    # AI / ML terms
    "ML", "AI",   # AI is also NYSE: AI (C3.ai) but causes too many false positives
    "LLM", "NLP", "GAN", "RAG",
    # Misc abbreviations common in finance writing
    "MMA", "PIC", "AUM", "NAV", "ABS", "CLO", "CDO", "MBS",
    # General tech / engineering
    "IC",   # integrated circuit
    "PCB",  # printed circuit board
    "BOM",  # bill of materials
    "EOL",  # end of life
    "SKU",  # stock-keeping unit
    "OEM",  # original equipment manufacturer
    "ODM",  # original design manufacturer
    "IP",   # intellectual property (also a 2-letter ticker but too ambiguous)
}

# Matches $AAPL, $TSLA, etc.
DOLLAR_SIGN_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Matches "buying NVDA", "bullish on MSFT", etc.
# (?i:...) makes the keyword clause case-insensitive while [A-Z]{2,5} stays uppercase-only.
CONTEXT_WORD_RE = re.compile(
    r"(?i:"
    # Sentiment
    r"bull(?:ish)? on|bear(?:ish)? on|"
    # Trade actions
    r"buying|selling|accumulating|trimming|reducing|exiting|"
    # Position language
    r"long|short|"
    # Ownership
    r"holds?|owns?|we own|we hold|we like|stake in|"
    # Analyst ratings
    r"overweight|underweight|outperform(?:ing)?|underperform(?:ing)?|"
    r"initiat(?:ing|ed)(?: coverage)?(?: (?:on|of))?|coverage of|rating on|"
    # Exposure
    r"exposed to|"
    # Classic phrases
    r"shares of|position in|invested in|"
    # Watchlist
    r"watching|tracking"
    r")\s+([A-Z]{2,5})\b"
)

# Matches "NYSE: AAPL", "NASDAQ: NVDA", "Nasdaq: TSLA"
EXCHANGE_RE = re.compile(
    r"\b(?:NYSE|NASDAQ|Nasdaq|AMEX|OTC|TSX|LSE):\s*([A-Z]{1,5})\b"
)

# Matches "Apple (AAPL)", "Nvidia (NVDA)" — ticker in parentheses after a company name.
# Requires ≥3 uppercase letters; 2-letter captures are almost always jargon abbreviations.
PAREN_RE = re.compile(r"\(([A-Z]{3,5})\)")

# ---------------------------------------------------------------------------
# Company name → ticker mapping
# ---------------------------------------------------------------------------

# Maps lowercase company names to their primary US ticker symbol.
# Used to catch mentions like "Nvidia announced..." or "TSMC reported..."
# Note: None values are private companies — extracted from text but not stored.
COMPANY_NAME_MAP: dict[str, str | None] = {
    # Mega-cap tech
    "nvidia":                   "NVDA",
    "apple":                    "AAPL",
    "microsoft":                "MSFT",
    "alphabet":                 "GOOGL",
    "google":                   "GOOGL",
    "amazon":                   "AMZN",
    "meta":                     "META",
    "tesla":                    "TSLA",
    "netflix":                  "NFLX",
    "uber":                     "UBER",
    "lyft":                     "LYFT",
    "airbnb":                   "ABNB",
    "doordash":                 "DASH",
    "spotify":                  "SPOT",
    # Semiconductors / hardware
    "amd":                      "AMD",
    "intel":                    "INTC",
    "qualcomm":                 "QCOM",
    "broadcom":                 "AVGO",
    "taiwan semiconductor":     "TSM",
    "tsmc":                     "TSM",
    "samsung":                  "SSNLF",
    "micron":                   "MU",
    "applied materials":        "AMAT",
    "asml":                     "ASML",
    "lam research":             "LRCX",
    "kla":                      "KLAC",
    "arm holdings":             "ARM",
    "arm":                      "ARM",
    "marvell":                  "MRVL",
    "on semiconductor":         "ON",
    "onsemi":                   "ON",
    "skyworks":                 "SWKS",
    "wolfspeed":                "WOLF",
    "monolithic power":         "MPWR",
    "lattice semiconductor":    "LSCC",
    "axcelis":                  "ACLS",
    "entegris":                 "ENTG",
    "coherent":                 "COHR",
    "amkor":                    "AMKR",
    "silicon motion":           "SIMO",
    "maxlinear":                "MXL",
    "allegro":                  "ALGM",
    "impinj":                   "PI",
    "rambus":                   "RMBS",
    "kulicke":                  "KLIC",
    "kulicke and soffa":        "KLIC",
    "camtek":                   "CAMT",
    "indie semiconductor":      "INDI",
    "lumentum":                 "LITE",
    "synaptics":                "SYNA",
    "semtech":                  "SMTC",
    "silicon labs":             "SLAB",
    "silicon laboratories":     "SLAB",
    "ultra clean":              "UCTT",
    "onto innovation":          "ONTO",
    "western digital":          "WDC",
    "seagate":                  "STX",
    "corning":                  "GLW",
    "ii-vi":                    "COHR",  # merged into Coherent
    "macom":                    "MTSI",
    "pdf solutions":            "PDFS",
    "photronics":               "PLAB",
    # Networking / infrastructure
    "cisco":                    "CSCO",
    "arista":                   "ANET",
    "juniper":                  "JNPR",
    "ciena":                    "CIEN",
    "calix":                    "CALX",
    "viavi":                    "VIAV",
    # Cloud / SaaS
    "salesforce":               "CRM",
    "servicenow":               "NOW",
    "workday":                  "WDAY",
    "snowflake":                "SNOW",
    "datadog":                  "DDOG",
    "cloudflare":               "NET",
    "mongodb":                  "MDB",
    "palantir":                 "PLTR",
    "confluent":                "CFLT",
    "crowdstrike":              "CRWD",
    "okta":                     "OKTA",
    "twilio":                   "TWLO",
    "hubspot":                  "HUBS",
    "veeva":                    "VEEV",
    "elastic":                  "ESTC",
    "gitlab":                   "GTLB",
    "pagerduty":                "PD",
    "zoom":                     "ZM",
    "docusign":                 "DOCU",
    "asana":                    "ASAN",
    "braze":                    "BRZE",
    "dynatrace":                "DT",
    "appian":                   "APPN",
    "samsara":                  "IOT",
    "oracle":                   "ORCL",
    "sap":                      "SAP",
    "ibm":                      "IBM",
    "dell":                     "DELL",
    # AI / frontier labs (private, included for completeness)
    "openai":                   None,
    "anthropic":                None,
    # Fintech
    "coinbase":                 "COIN",
    "robinhood":                "HOOD",
    "paypal":                   "PYPL",
    "affirm":                   "AFRM",
    "upstart":                  "UPST",
    "sofi":                     "SOFI",
    "nubank":                   "NU",
    # Telecom
    "verizon":                  "VZ",
    "comcast":                  "CMCSA",
    "t-mobile":                 "TMUS",
    # Banks / finance
    "jpmorgan":                 "JPM",
    "jp morgan":                "JPM",
    "goldman sachs":            "GS",
    "morgan stanley":           "MS",
    "bank of america":          "BAC",
    "wells fargo":              "WFC",
    "blackrock":                "BLK",
    # Defense
    "lockheed":                 "LMT",
    "raytheon":                 "RTX",
    "northrop":                 "NOC",
    "boeing":                   "BA",
    # Energy
    "exxon":                    "XOM",
    "chevron":                  "CVX",
}

# Compile into a single regex (longest names first to prevent partial matches).
_name_alts = sorted(COMPANY_NAME_MAP, key=len, reverse=True)
COMPANY_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _name_alts) + r")\b",
    re.IGNORECASE,
)

CONTEXT_WINDOW = 50  # characters on each side for surrounding context

# Substack RSS feeds truncate paid posts and inject one of these phrases
_PAYWALL_RE = re.compile(
    r"subscribe to (read|continue|unlock)|"
    r"this post is for (paid )?subscribers|"
    r"become a (paid )?subscriber|"
    r"this is a (free )?preview|"
    r"upgrade (your )?subscription",
    re.IGNORECASE,
)


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


def _detect_paywall(body_text: str) -> bool:
    """Return True if *body_text* contains Substack paywall indicators."""
    return bool(_PAYWALL_RE.search(body_text))


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

def _feed_url(slug_or_url: str) -> str:
    """
    Convert a slug or a custom-domain URL into a feed URL.

    - "doomberg"                        → https://doomberg.substack.com/feed
    - "newsletter.semianalysis.com"     → https://newsletter.semianalysis.com/feed
    - "https://newsletter.semianalysis.com" → https://newsletter.semianalysis.com/feed
    - "https://newsletter.semianalysis.com/feed" → unchanged
    """
    s = slug_or_url.strip().rstrip("/")
    if s.startswith("http://") or s.startswith("https://"):
        return s if s.endswith("/feed") else s + "/feed"
    # Bare domain (contains a dot) — treat as custom domain, not a substack slug
    if "." in s:
        return f"https://{s}/feed"
    return f"https://{s}.substack.com/feed"


def _slug_label(slug_or_url: str) -> str:
    """Return a short human-readable identifier for storage / display."""
    s = slug_or_url.strip().rstrip("/")
    if s.startswith("http://") or s.startswith("https://"):
        # e.g. "https://newsletter.semianalysis.com" → "newsletter.semianalysis.com"
        from urllib.parse import urlparse
        return urlparse(s).netloc
    return s


def fetch_feed(slug_or_url: str) -> list[dict]:
    """
    Fetch and parse the RSS feed for a Substack slug or custom-domain URL.

    Accepts:
      - A bare slug:  "doomberg"
      - A full URL:   "https://newsletter.semianalysis.com"

    Returns a list of post dicts with keys:
        title, published (ISO string), link, body, is_paywalled
    Raises requests.RequestException on network failure.
    """
    url = _feed_url(slug_or_url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RSS-reader/1.0)"}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()

    feed = feedparser.parse(response.content)

    posts = []
    for entry in feed.entries:
        body = _get_entry_body(entry)
        posts.append({
            "title": getattr(entry, "title", ""),
            "published": _parse_published(entry),
            "link": getattr(entry, "link", ""),
            "body": body,
            "is_paywalled": _detect_paywall(body),
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

    # Context-word style: "overweight NVDA", "bullish on MSFT", "buying AAPL"
    for match in CONTEXT_WORD_RE.finditer(text):
        ticker = match.group(1)
        if ticker in BLOCKLIST:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "context_word",
        })

    # Exchange-prefix style: "NYSE: AAPL", "NASDAQ: NVDA"
    for match in EXCHANGE_RE.finditer(text):
        ticker = match.group(1)
        if ticker in BLOCKLIST:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "exchange_prefix",
        })

    # Parenthetical style: "Nvidia (NVDA)", "Apple (AAPL)"
    for match in PAREN_RE.finditer(text):
        ticker = match.group(1)
        if ticker in BLOCKLIST:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "parenthetical",
        })

    # Company name style: "Nvidia announced...", "TSMC reported...", "Palantir beats..."
    # BLOCKLIST is intentionally not applied here — the lookup is unambiguous.
    for match in COMPANY_NAME_RE.finditer(text):
        name_key = match.group(1).lower()
        ticker = COMPANY_NAME_MAP.get(name_key)
        if ticker is None:
            continue
        mentions.append({
            "ticker": ticker,
            "context": _surrounding_context(text, match),
            "match_type": "company_name",
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

    for i, slug_or_url in enumerate(slugs):
        # Rate-limit: pause between requests (skip before the very first slug)
        if i > 0:
            time.sleep(1)

        label = _slug_label(slug_or_url)

        try:
            posts = fetch_feed(slug_or_url)
        except Exception as exc:
            print(f"WARNING: could not fetch '{label}': {exc}")
            continue

        slug_posts = 0
        slug_tickers = 0
        slug_paywalled = 0

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

            is_paywalled = post.get("is_paywalled", False)

            # Persist the post and get back its row id
            post_id = database.insert_post(
                conn,
                slug=label,
                title=post["title"],
                url=post["link"],
                published_date=post["published"],
                body_text=post["body"],
                is_paywalled=is_paywalled,
            )

            # Extract and persist ticker mentions from whatever body text is available
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
            if is_paywalled:
                slug_paywalled += 1

        paywalled_note = f", {slug_paywalled} paywalled" if slug_paywalled else ""
        print(f"{label}: {slug_posts} posts{paywalled_note}, {slug_tickers} tickers found")
        total_posts += slug_posts

    return total_posts
