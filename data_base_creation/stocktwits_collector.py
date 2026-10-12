"""Collect StockTwits messages for trending US tickers into StockTwitsPost.

Changes from the first version:
- Stores the message's REAL timestamp (it used the collection time before, which
  breaks any recency / z-score logic).
- Covers up to MAX_SYMBOLS trending tickers instead of only 6, with a pause
  between calls and a clean stop on HTTP 429 (rate limited).
- Drops copy-paste spam: identical text from the same author is saved once.
- A message that appears in several symbol streams is saved once, but every
  ticker it carries is appended as a $CASHTAG, so no association is lost.
- Fallback list no longer contains ETFs (SPY/QQQ/IWM are not stock picks).
- US focus: trending tickers are filtered to US common stocks (us_universe.py),
  so ETFs, ADRs and OTC names are skipped before any stream is fetched.
- Reads the user's Bullish/Bearish tag from message["entities"]["sentiment"]
  (the old code looked at message["sentiment"], so every post was "Unspecified").
- Saves author join date, follower count and post count (new columns in
  database.py) so Phase 2 can drop bots and brand-new accounts.
"""

import html
import re
import time
from datetime import datetime, timezone

import requests

from data_base_creation.database import SessionLocal, StockTwitsPost, init_db
from data_base_creation.us_universe import load_us_symbols

STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
STOCKTWITS_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Used ONLY if the trending call fails or returns nothing usable.
FALLBACK_WATCHLIST = ["NVDA", "AAPL", "TSLA", "AMD", "MSFT", "AMZN", "META"]
MAX_SYMBOLS = 25        # unauthenticated limit is roughly 200 requests/hour
PAUSE_SECONDS = 1.0


def is_us_stock_symbol(symbol: str) -> bool:
    if symbol.endswith(".X") or "USD" in symbol or "." in symbol or "-" in symbol:
        return False
    return bool(re.match(r"^[A-Z]{1,5}$", symbol))


def get_trending_symbols(us_symbols=None) -> list:
    try:
        res = requests.get(STOCKTWITS_TRENDING_URL, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            raw = [item.get("symbol", "") for item in res.json().get("symbols", [])]
            symbols = [s for s in dict.fromkeys(raw) if is_us_stock_symbol(s)]
            if us_symbols:   # keep only US common stocks (drops ETFs, ADRs, OTC)
                dropped = [s for s in symbols if s not in us_symbols]
                symbols = [s for s in symbols if s in us_symbols]
                if dropped:
                    print(f"  Skipped non-US-stock tickers: {dropped}")
            if symbols:
                return symbols[:MAX_SYMBOLS]
        else:
            print(f"  Trending returned HTTP {res.status_code}")
    except Exception as err:
        print(f"  Could not fetch trending symbols: {err}")
    print("  Using fallback watchlist.")
    return FALLBACK_WATCHLIST


def parse_created_at(value: str) -> datetime:
    """StockTwits gives e.g. '2026-10-06T08:15:30Z'. Stored naive UTC."""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_join_date(value):
    """StockTwits user.join_date is usually 'YYYY-MM-DD' (sometimes a full ISO stamp)."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
    return None


def extract_user_sentiment(msg: dict) -> str:
    """The user's own Bullish/Bearish tag. StockTwits stores it in
    message["entities"]["sentiment"]["basic"]; the top-level "sentiment" key is
    checked too, in case the format changes."""
    for container in ((msg.get("entities") or {}).get("sentiment"), msg.get("sentiment")):
        if isinstance(container, dict) and container.get("basic") in ("Bullish", "Bearish"):
            return container["basic"]
    return "Unspecified"


def add_cashtags(body: str, tickers: set) -> str:
    missing = [t for t in sorted(tickers) if f"${t}" not in body.upper()]
    if not missing:
        return body
    return body + " " + " ".join(f"${t}" for t in missing)


def msg_tickers(msg: dict, stream_symbol: str, us_symbols=None) -> set:
    """Tickers StockTwits attached to the message, plus the stream we found it in."""
    tickers = {stream_symbol}
    for item in msg.get("symbols", []) or []:
        sym = (item.get("symbol") or "").upper()
        if is_us_stock_symbol(sym) and (not us_symbols or sym in us_symbols):
            tickers.add(sym)
    return tickers


def fetch_stocktwits():
    init_db()  # also upgrades an older database with the new author columns
    session = SessionLocal()
    new_count = 0

    try:
        # Existing post ids, to skip duplicates across runs and streams
        seen_ids = {res[0] for res in session.query(StockTwitsPost.id).all()}
        staged = {}          # msg_id -> StockTwitsPost added in this run
        seen_text = set()    # (author, normalised body) to drop copy-paste spam

        us_symbols = load_us_symbols(session)
        if not us_symbols:
            print("  WARNING: US universe not loaded; ETFs/ADRs will not be filtered.")
        symbols = get_trending_symbols(us_symbols)
        print(f"Discovered {len(symbols)} trending tickers: {symbols}")

        for symbol in symbols:
            print(f"Fetching US stock stream for ${symbol}...")
            url = STOCKTWITS_STREAM_URL.format(symbol=symbol)
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                time.sleep(PAUSE_SECONDS)

                if res.status_code == 429:
                    print("  Rate limited (HTTP 429). Stopping early; try again later.")
                    break
                if res.status_code != 200:
                    print(f"  HTTP {res.status_code} for ${symbol}")
                    continue

                for msg in res.json().get("messages", []):
                    msg_id = f"st_{msg['id']}"
                    tickers = msg_tickers(msg, symbol, us_symbols)

                    # Same message already staged under another symbol:
                    # keep one row, record the extra tickers.
                    if msg_id in staged:
                        post = staged[msg_id]
                        post.body = add_cashtags(post.body, tickers)
                        continue

                    # Already stored in an earlier run
                    if msg_id in seen_ids:
                        continue

                    user_obj = msg.get("user") or {}
                    user = user_obj.get("username", "anonymous")
                    raw_body = html.unescape(msg.get("body", "") or "")   # &#39; -> '  

                    text_key = (user, re.sub(r"\s+", " ", raw_body.lower()).strip())
                    if text_key in seen_text:
                        continue
                    seen_text.add(text_key)

                    user_sentiment = extract_user_sentiment(msg)

                    post = StockTwitsPost(
                        id=msg_id,
                        symbol=symbol,
                        author=user,
                        body=add_cashtags(raw_body, tickers),
                        sentiment=user_sentiment,
                        created_at=parse_created_at(msg.get("created_at")),
                        author_join_date=parse_join_date(user_obj.get("join_date")),
                        author_followers=user_obj.get("followers"),
                        author_post_count=user_obj.get("ideas"),
                    )
                    session.add(post)
                    staged[msg_id] = post
                    seen_ids.add(msg_id)
                    new_count += 1
            except Exception as err:
                print(f"  Error fetching ${symbol}: {err}")

        session.commit()
        print(f"Saved {new_count} new US market messages into database.")

    except Exception as e:
        session.rollback()
        print(f"Error running stocktwits_collector: {e}")
    finally:
        session.close()


# Entry point so running python stocktwits_collector.py directly works
if __name__ == "__main__":
    fetch_stocktwits()