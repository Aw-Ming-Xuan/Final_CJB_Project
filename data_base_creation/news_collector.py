"""Collect market headlines from RSS feeds into the RawNewsPost table.

Two kinds of feeds:
  1. GENERAL market feeds (CNBC, MarketWatch, Yahoo Finance, Investing.com). We do not choose
     which stocks they mention, so they count as "unbiased" news in Phase 1.
  2. PER-TICKER feeds from Nasdaq.com for the stocks StockTwits is currently talking about.
     Each item carries the original publisher (e.g. The Motley Fool, Barchart, Zacks), so the
     frontend can show a wider range of news sources than Finnhub alone. These are saved as
     source "Nasdaq_<TICKER>" and treated like Finnhub's per-ticker news (trusted ticker, but
     not counted as unbiased chatter).

Other behaviour:
- De-duplicates by id, URL and normalised title across ALL sources (incl. Finnhub).
- Strips HTML tags/entities from summaries.
- Stores the publisher of every article in the `publisher` column.
- One broken feed never aborts the run; empty feeds are reported so you can drop them.

Setup: pip install feedparser
"""

import hashlib
import html
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
from sqlalchemy import func

from data_base_creation.database import RawNewsPost, SessionLocal, StockTwitsPost, init_db

# name -> (url, publisher shown to users)
RSS_FEEDS = {
    "CNBC_US_Markets": ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "CNBC"),
    "CNBC_Investing": ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069", "CNBC"),
    "CNBC_Earnings": ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135", "CNBC"),
    "CNBC_Technology": ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910", "CNBC"),
    "MarketWatch_TopStories": ("http://feeds.marketwatch.com/marketwatch/topstories", "MarketWatch"),
    "MarketWatch_Headlines": ("http://feeds.marketwatch.com/marketwatch/realtimeheadlines", "MarketWatch"),
    "Yahoo_Finance": ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance"),
    "Investing_StockNews": ("https://www.investing.com/rss/news_25.rss", "Investing.com"),
}
NASDAQ_TICKER_FEED = "https://www.nasdaq.com/feed/rssoutbound?symbol={symbol}"
MAX_TICKER_FEEDS = 25
TICKER_FEED_PAUSE = 1.0

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove HTML tags and entities, collapse whitespace."""
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalise_title(title: str) -> str:
    """Lowercase letters/digits only, so syndicated copies of a headline match."""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def trending_symbols(session) -> list:
    """Tickers the StockTwits collector saw in the last 2 days (no extra API call)."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
    rows = (session.query(StockTwitsPost.symbol, func.max(StockTwitsPost.created_at))
            .filter(StockTwitsPost.created_at >= cutoff)
            .group_by(StockTwitsPost.symbol)
            .order_by(func.max(StockTwitsPost.created_at).desc()).all())
    return [r[0] for r in rows if r[0] and r[0].isalpha() and r[0].isupper()][:MAX_TICKER_FEEDS]


class Store:
    """Adds articles while de-duplicating against everything already stored."""

    def __init__(self, session):
        self.session = session
        rows = session.query(RawNewsPost.id, RawNewsPost.url, RawNewsPost.title).all()
        self.ids = {r[0] for r in rows}
        self.urls = {r[1] for r in rows if r[1]}
        self.titles = {normalise_title(r[2]) for r in rows if r[2]}
        self.added = 0

    def add(self, entry, source: str, publisher: str, ticker: str = None) -> bool:
        link = entry.get("link", "")
        if not link:
            return False
        unique_id = hashlib.md5(link.encode("utf-8")).hexdigest()   # same id scheme as before
        title = clean_text(entry.get("title", ""))
        key = normalise_title(title)
        if unique_id in self.ids or link in self.urls or (key and key in self.titles):
            return False

        summary = clean_text(entry.get("summary", entry.get("description", "")))
        body = f"{title}. {summary}"
        if ticker and f"${ticker}" not in body:
            body += f" ${ticker}"            # same convention as the Finnhub collector

        if getattr(entry, "published_parsed", None):
            published = datetime(*entry.published_parsed[:6])
        else:
            published = datetime.now(timezone.utc).replace(tzinfo=None)

        self.session.add(RawNewsPost(id=unique_id, source=source, publisher=publisher[:100], title=title,
                                     body=body, published_at=published, url=link, link=link))
        self.ids.add(unique_id)
        self.urls.add(link)
        if key:
            self.titles.add(key)
        self.added += 1
        return True


def parse(url: str, label: str):
    try:
        parsed = feedparser.parse(url, agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")
    except Exception as err:
        print(f"  Could not parse {label}: {err}")
        return []
    if not parsed.entries:
        print(f"  No entries returned for {label}.")
    return parsed.entries


def fetch_news():
    init_db()   # also adds the new `publisher` column to an older database
    session = SessionLocal()
    try:
        store = Store(session)

        # 1. General market feeds
        for name, (url, publisher) in RSS_FEEDS.items():
            print(f"Fetching {name} headlines...")
            for entry in parse(url, name):
                store.add(entry, source=name, publisher=publisher)
        general = store.added

        # 2. Per-ticker Nasdaq feeds for what StockTwits is discussing
        symbols = trending_symbols(session)
        if symbols:
            print(f"Fetching Nasdaq.com news for {len(symbols)} trending tickers...")
        for symbol in symbols:
            for entry in parse(NASDAQ_TICKER_FEED.format(symbol=symbol), f"Nasdaq ${symbol}"):
                author = clean_text(entry.get("author", "")) or "Nasdaq"
                store.add(entry, source=f"Nasdaq_{symbol}", publisher=author, ticker=symbol)
            time.sleep(TICKER_FEED_PAUSE)

        session.commit()
        print(f"Done! Saved {store.added} new RSS articles "
              f"({general} from general feeds, {store.added - general} from per-ticker feeds).")
    except Exception as e:
        session.rollback()
        print(f"Error running news_collector: {e}")
    finally:
        session.close()


# Explicit entry point for terminal execution
if __name__ == "__main__":
    fetch_news()
