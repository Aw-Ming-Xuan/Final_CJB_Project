"""Collect market news from the Finnhub API into the RawNewsPost table.

Changes from the first version:
- The watchlist now comes from the tickers StockTwits just surfaced (read from
  our own stocktwits_posts table, then the StockTwits API, then a fixed fallback
  list), so the pipeline discovers tickers instead of echoing a hardcoded list.
- Articles are de-duplicated by id, URL and normalised title, so a story that
  your RSS collector already saved is not counted twice.
- Tickers Finnhub already knows (the symbol queried + the `related` field) are
  appended to the body as cashtags ($NVDA), so Phase 1 can trust them instead
  of regex-guessing from the text.
- US focus: the watchlist is limited to US common stocks (us_universe.py).
- Articles with no timestamp are skipped instead of being dated 1970.

Setup:
1. Free key at finnhub.io, put it in .env as FINNHUB_API_KEY=...
2. pip install requests python-dotenv
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from sqlalchemy import func

from data_base_creation.database import RawNewsPost, StockTwitsPost, SessionLocal, init_db
from data_base_creation.us_universe import load_us_symbols

load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"
HEADERS = {"X-Finnhub-Token": API_KEY or ""}

STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
# Same User-Agent as stocktwits_collector.py; a shorter one gets HTTP 403.
ST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Used ONLY if the StockTwits trending call fails.
FALLBACK_WATCHLIST = ["NVDA", "AAPL", "TSLA", "AMD", "MSFT", "AMZN", "META"]
MAX_SYMBOLS = 25        # 25 symbols * ~1.1 s pause stays well under 60 calls/min
DAYS_BACK = 7
PAUSE_SECONDS = 1.1     # free tier is 60 calls/minute


def get_json(path: str, params: dict) -> list:
    """Call a Finnhub endpoint. Returns [] and prints the status on failure."""
    try:
        res = requests.get(
            f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=15
        )
    except requests.RequestException as err:
        print(f"  Request error for {path}: {err}")
        return []

    time.sleep(PAUSE_SECONDS)

    if res.status_code != 200:
        # 401 = bad key, 403 = endpoint not in your plan, 429 = rate limited
        print(f"  HTTP {res.status_code} for {path} {params}")
        return []
    return res.json()


def symbols_from_stocktwits_db(session) -> list:
    """Tickers the StockTwits collector just saw (last 2 days). No extra API call."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
    rows = (
        session.query(StockTwitsPost.symbol, func.max(StockTwitsPost.created_at))
        .filter(StockTwitsPost.created_at >= cutoff)
        .group_by(StockTwitsPost.symbol)
        .order_by(func.max(StockTwitsPost.created_at).desc())
        .all()
    )
    symbols = [r[0] for r in rows if r[0] and r[0].isalpha() and r[0].isupper() and len(r[0]) <= 5]
    return symbols[:MAX_SYMBOLS]


def symbols_from_stocktwits_api() -> list:
    """Direct call to the StockTwits trending endpoint (needs a browser-like User-Agent)."""
    try:
        res = requests.get(STOCKTWITS_TRENDING_URL, headers=ST_HEADERS, timeout=15)
        if res.status_code == 200:
            raw = [s.get("symbol", "") for s in res.json().get("symbols", [])]
            symbols = [s.upper() for s in raw if s.isalpha() and len(s) <= 5]
            return list(dict.fromkeys(symbols))[:MAX_SYMBOLS]
        print(f"  StockTwits trending returned HTTP {res.status_code}")
    except (requests.RequestException, ValueError) as err:
        print(f"  Could not fetch StockTwits trending: {err}")
    return []


def get_watchlist(session) -> list:
    """Order of preference: tickers already in our StockTwits table, then the
    StockTwits API, then the fixed fallback list. Always limited to US stocks."""
    us_symbols = load_us_symbols(session)

    def us_only(symbols):
        return [s for s in symbols if not us_symbols or s in us_symbols][:MAX_SYMBOLS]

    symbols = us_only(symbols_from_stocktwits_db(session))
    if symbols:
        print("  Watchlist source: StockTwits posts already in the database.")
        return symbols

    symbols = us_only(symbols_from_stocktwits_api())
    if symbols:
        print("  Watchlist source: StockTwits trending API.")
        return symbols

    print("  WARNING: using the fixed fallback watchlist; results will be biased "
          "toward these tickers.")
    return FALLBACK_WATCHLIST


def normalise_title(title: str) -> str:
    """Lowercase letters/digits only, so syndicated copies of a headline match."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def parse_related(related: str) -> set:
    """Finnhub's `related` field is a comma-separated ticker string."""
    return {t.strip().upper() for t in (related or "").split(",")
            if t.strip().isalpha() and len(t.strip()) <= 5}


def add_cashtags(body: str, tickers: set) -> str:
    """Append any missing $TICKER tags to a body string."""
    missing = [t for t in sorted(tickers) if f"${t}" not in body]
    if not missing:
        return body
    return body + " " + " ".join(f"${t}" for t in missing)


def save_articles(session, articles, symbol, seen, staged) -> int:
    """Store new articles.

    seen   = dict with sets: ids, urls, titles (existing DB rows + this run)
    staged = {unique_id: RawNewsPost} for articles added during this run, so an
             article returned under a second symbol gets that symbol's cashtag
             instead of being silently dropped.
    """
    new_count = 0
    for article in articles:
        article_id = article.get("id")
        url = article.get("url")
        timestamp = article.get("datetime")
        if not article_id or not url or not timestamp:
            continue

        unique_id = f"fh_{article_id}"
        tickers = parse_related(article.get("related", ""))
        if symbol:
            tickers.add(symbol)

        # Same article already added in this run under another symbol:
        # keep one row, but record the extra ticker.
        if unique_id in staged:
            post = staged[unique_id]
            post.body = add_cashtags(post.body, tickers)
            continue

        headline = article.get("headline", "") or ""
        title_key = normalise_title(headline)

        # Already stored (by Finnhub earlier, or by the RSS collector)
        if (unique_id in seen["ids"] or url in seen["urls"]
                or (title_key and title_key in seen["titles"])):
            continue

        summary = article.get("summary", "") or ""
        body = add_cashtags(f"{headline}. {summary}", tickers)

        # Finnhub timestamps are unix seconds (UTC). Stored naive, like the RSS script.
        published = datetime.fromtimestamp(timestamp, timezone.utc).replace(tzinfo=None)

        post = RawNewsPost(
            id=unique_id,
            source=f"Finnhub_{symbol or 'general'}",
            publisher=(article.get("source") or "Finnhub")[:100],   # e.g. Yahoo, MarketWatch, SeekingAlpha
            title=headline,
            body=body,
            published_at=published,
            url=url,
            link=url,
        )
        session.add(post)
        staged[unique_id] = post
        seen["ids"].add(unique_id)
        seen["urls"].add(url)
        if title_key:
            seen["titles"].add(title_key)
        new_count += 1
    return new_count


def fetch_news_api():
    if not API_KEY:
        raise SystemExit("Set FINNHUB_API_KEY in your .env file first.")

    init_db()
    session = SessionLocal()
    total = 0

    try:
        # Pre-load existing ids, urls and titles (from ALL sources, incl. RSS)
        rows = session.query(
            RawNewsPost.id, RawNewsPost.url, RawNewsPost.title
        ).all()
        seen = {
            "ids": {r[0] for r in rows},
            "urls": {r[1] for r in rows if r[1]},
            "titles": {normalise_title(r[2]) for r in rows if r[2]},
        }
        staged = {}

        print("Fetching general market news...")
        articles = get_json("/news", {"category": "general"})
        total += save_articles(session, articles, None, seen, staged)

        watchlist = get_watchlist(session)
        print(f"Watchlist ({len(watchlist)}): {watchlist}")

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=DAYS_BACK)

        for symbol in watchlist:
            print(f"Fetching company news for ${symbol}...")
            articles = get_json(
                "/company-news",
                {"symbol": symbol, "from": start.isoformat(), "to": today.isoformat()},
            )
            total += save_articles(session, articles, symbol, seen, staged)

        session.commit()
        print(f"\nDone! Saved {total} new articles.")

    except Exception as e:
        session.rollback()
        print(f"Error running finnhub_news_collector: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    fetch_news_api()
