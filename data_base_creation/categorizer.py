"""Turn raw news (RSS + Finnhub) and StockTwits posts into ProcessedNews rows.

Changes from the first version:
- No hardcoded watchlist. Tickers come from (in order of trust):
    1. StockTwits symbol / Finnhub source tag     (e.g. Finnhub_HPE)
    2. $CASHTAGS in the text (the collectors append these for known tickers)
    3. Exchange tags in headlines, e.g. "(NASDAQ: NVDA)"
    4. A small company-name map (extend COMPANY_NAMES as you like)
  Plain ALL-CAPS words are NOT treated as tickers, which is what let COO, CPA,
  FCF, USA, TV, IP etc. into the pipeline.
- US focus: tickers are kept only if they are US-listed common stocks
  (us_universe.py). ETFs like SPY/QQQ/XLF and ADRs like PBR/TSM become "MARKET".
- One ProcessedNews row per (post, ticker), so a post about NVDA and AMD counts
  for both. Posts with no ticker get "MARKET"; roundups with too many tickers
  are also "MARKET".
- Category = the one with the MOST keyword hits (not the first one that matches).
- Separate sentiment word lists for news vs social. "call" is gone from the
  news list because "earnings call" is not a bullish signal.
- process_unassigned_data(session, rebuild=True) wipes ProcessedNews and
  re-processes everything. Run it once now since the old rows were built with
  the old logic:   python categorizer.py --rebuild
"""

import re
import sys
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from data_base_creation.database import RawNewsPost, StockTwitsPost, ProcessedNews
from data_base_creation.us_universe import load_us_symbols

MAX_TICKERS_PER_POST = 5
# Sources fetched FOR a specific ticker ("<Feed>_<TICKER>"): the ticker in the name is trusted
QUERY_PREFIXES = ("Finnhub_", "Nasdaq_")

CATEGORY_RULES = {
    "Earnings & Financials": [r"\bearnings\b", r"\bquarterly\b", r"\brevenue\b", r"\beps\b", r"\bguidance\b", r"\bprofit\b"],
    "Macro & Fed Policy": [r"\bfed\b", r"\binflation\b", r"\brates\b", r"\bpowell\b", r"\bcpi\b", r"\brecession\b"],
    "M&A & Dealmaking": [r"\bacquire[sd]?\b", r"\bacquisition\b", r"\bmerger\b", r"\bbuyout\b", r"\btakeover\b"],
    "Legal & Regulation": [r"\bsec\b", r"\blawsuit\b", r"\bregulatory\b", r"\bprobe\b", r"\bftc\b", r"\bcourt\b"],
    "Product & Innovation": [r"\blaunch(es|ed)?\b", r"\bai\b", r"\bchips?\b", r"\bpatent\b", r"\bunveil(s|ed)?\b"],
}

# News: headline-style words. No "call"/"short"/"put" (earnings call, short seller...)
NEWS_BULLISH = {"surge", "surges", "jump", "jumps", "record", "growth", "beat", "beats",
                "upgrade", "upgraded", "outperform", "rally", "rallies", "gains", "soar", "soars"}
NEWS_BEARISH = {"plunge", "plunges", "drop", "drops", "miss", "misses", "downgrade", "downgraded",
                "decline", "slump", "lawsuit", "loss", "losses", "fall", "falls", "tumble", "tumbles"}

# Social: retail slang and options talk
SOCIAL_BULLISH = NEWS_BULLISH | {"calls", "moon", "bullish", "long", "buy", "breakout"}
SOCIAL_BEARISH = NEWS_BEARISH | {"puts", "bearish", "short", "sell", "dump", "crash"}

# Plain-text company names -> ticker (lowercase). Extend freely.
COMPANY_NAMES = {
    "nvidia": "NVDA", "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
    "amazon": "AMZN", "meta platforms": "META", "alphabet": "GOOGL", "google": "GOOGL",
    "netflix": "NFLX", "advanced micro devices": "AMD", "micron": "MU",
    "broadcom": "AVGO", "qualcomm": "QCOM", "intel": "INTC", "oracle": "ORCL",
    "salesforce": "CRM", "adobe": "ADBE", "palantir": "PLTR", "taiwan semiconductor": "TSM",
    "tsmc": "TSM", "alibaba": "BABA", "sony": "SONY", "asml": "ASML", "novo nordisk": "NVO",
    "toyota": "TM", "baidu": "BIDU", "pdd holdings": "PDD", "temu": "PDD", "nio": "NIO",
    "sap": "SAP", "astrazeneca": "AZN", "shopify": "SHOP",
    "petrobras": "PBR", "infosys": "INFY", "unilever": "UL", "delta air lines": "DAL", "hewlett packard enterprise": "HPE",
    "dell": "DELL", "ibm": "IBM", "nike": "NKE", "costco": "COST", "boeing": "BA",
    "coinbase": "COIN", "super micro": "SMCI", "arm holdings": "ARM", "uber": "UBER",
    "disney": "DIS", "walmart": "WMT", "jpmorgan": "JPM", "goldman sachs": "GS",
    "marvell": "MRVL", "sandisk": "SNDK", "synopsys": "SNPS", "cloudflare": "NET",
    "crowdstrike": "CRWD", "mongodb": "MDB", "rivian": "RIVN",
}

# Only applied to cashtags found in RSS text (untrusted); words that are not tickers.
NOT_TICKERS = {"COO", "CPA", "FCF", "USA", "TV", "IP", "ET", "SA", "UK", "EU", "FT", "MW",
               "CEO", "CFO", "ETF", "IPO", "SEC", "FED", "GDP", "CPI", "EPS", "PE", "AI", "US"}

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})(?![A-Za-z0-9.\-])")
EXCHANGE_TAG_RE = re.compile(
    r"\((?:NASDAQ|NYSE|NYSEARCA|NYSE American|AMEX|OTC)\s*:\s*([A-Z]{1,5})\)"
)
NAME_PATTERNS = [(re.compile(r"\b" + re.escape(name) + r"\b"), tick)
                 for name, tick in COMPANY_NAMES.items()]


def is_valid_symbol(sym: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}", sym or ""))


def classify_text(text: str) -> str:
    clean = text.lower()
    best, best_hits = "General Market / Social", 0
    for category, patterns in CATEGORY_RULES.items():
        hits = sum(len(re.findall(p, clean)) for p in patterns)
        if hits > best_hits:
            best, best_hits = category, hits
    return best


def calculate_sentiment(text: str, bullish: set, bearish: set) -> float:
    tokens = re.findall(r"\w+", text.lower())
    pos = sum(1 for w in tokens if w in bullish)
    neg = sum(1 for w in tokens if w in bearish)
    total = pos + neg
    return round((pos - neg) / total, 2) if total else 0.0


def get_sentiment_label(score: float) -> str:
    if score > 0.1:
        return "Bullish"
    if score < -0.1:
        return "Bearish"
    return "Neutral"


def extract_tickers(source: str, text: str, trusted: bool, us_symbols=None) -> list:
    """Return an ordered, de-duplicated list of tickers (empty if none found)."""
    found = []

    # 1. Per-ticker source tag, e.g. Finnhub_HPE or Nasdaq_HPE ("general" has no ticker)
    if source.startswith(QUERY_PREFIXES):
        tag = source.split("_", 1)[1]
        if is_valid_symbol(tag):
            found.append(tag)

    # 2. Cashtags ($NVDA). Trusted sources (Finnhub/StockTwits) keep everything;
    #    RSS cashtags are filtered against the not-a-ticker list.
    for tag in CASHTAG_RE.findall(text):
        if trusted or tag not in NOT_TICKERS:
            found.append(tag)

    # 3. Exchange tags: "(NASDAQ: NVDA)"
    found.extend(EXCHANGE_TAG_RE.findall(text))

    # 4. Known company names
    lower = text.lower()
    for pattern, tick in NAME_PATTERNS:
        if pattern.search(lower):
            found.append(tick)

    found = list(dict.fromkeys(found))
    if us_symbols:   # US focus: drop ETFs, ADRs, OTC and anything not a US common stock
        found = [t for t in found if t in us_symbols]
    return found


def finalize_tickers(tickers: list) -> list:
    if not tickers or len(tickers) > MAX_TICKERS_PER_POST:
        return ["MARKET"]
    return tickers


def process_unassigned_data(session: Session, rebuild: bool = False) -> int:
    processed_count = 0
    us_symbols = load_us_symbols(session)
    if not us_symbols:
        print("  WARNING: US universe not loaded; tickers are not being filtered to US stocks.")

    try:
        if rebuild:
            session.query(ProcessedNews).delete()
            session.commit()
            print("Cleared ProcessedNews; re-processing everything.")

        # 1. News (RSS + Finnhub)
        unprocessed_news = (
            session.query(RawNewsPost)
            .outerjoin(ProcessedNews, RawNewsPost.id == ProcessedNews.raw_post_id)
            .filter(ProcessedNews.id.is_(None))
            .all()
        )

        for post in unprocessed_news:
            full_text = f"{post.title} {post.body or ''}"
            category = classify_text(full_text)
            score = calculate_sentiment(full_text, NEWS_BULLISH, NEWS_BEARISH)
            trusted = post.source.startswith(QUERY_PREFIXES)
            tickers = finalize_tickers(extract_tickers(post.source, full_text, trusted, us_symbols))
            summary = (post.body[:250] + "...") if post.body and len(post.body) > 250 \
                else (post.body or post.title)

            for ticker in tickers:
                session.add(ProcessedNews(
                    raw_post_id=post.id,
                    source_type="NEWS",
                    source_name=post.source.split("_")[0],
                    source=post.source,
                    ticker=ticker,
                    category=category,
                    sentiment_score=score,
                    sentiment=get_sentiment_label(score),
                    title=post.title,
                    summary=summary,
                    created_at=post.published_at or datetime.now(timezone.utc).replace(tzinfo=None),
                ))
                processed_count += 1

        # 2. Social (StockTwits)
        unprocessed_social = (
            session.query(StockTwitsPost)
            .outerjoin(ProcessedNews, StockTwitsPost.id == ProcessedNews.raw_post_id)
            .filter(ProcessedNews.id.is_(None))
            .all()
        )

        for post in unprocessed_social:
            body = post.body or ""
            if post.sentiment == "Bullish":
                score = 1.0
            elif post.sentiment == "Bearish":
                score = -1.0
            else:
                score = calculate_sentiment(body, SOCIAL_BULLISH, SOCIAL_BEARISH)
            label = get_sentiment_label(score)

            tickers = extract_tickers("StockTwits", body, True, us_symbols)
            if post.symbol and post.symbol not in tickers and (not us_symbols or post.symbol in us_symbols):
                tickers.insert(0, post.symbol)
            tickers = finalize_tickers(tickers)
            category = classify_text(body)

            for ticker in tickers:
                session.add(ProcessedNews(
                    raw_post_id=post.id,
                    source_type="SOCIAL",
                    source_name="StockTwits",
                    source="StockTwits",
                    ticker=ticker,
                    category=category,
                    sentiment_score=score,
                    sentiment=label,
                    title=f"StockTwits post on ${ticker}",
                    summary=body[:250],
                    created_at=post.created_at or datetime.now(timezone.utc).replace(tzinfo=None),
                ))
                processed_count += 1

        session.commit()

    except Exception as e:
        session.rollback()
        print(f"Error processing unassigned data: {e}")
        raise e

    return processed_count


if __name__ == "__main__":
    from data_base_creation.database import SessionLocal
    db = SessionLocal()
    count = process_unassigned_data(db, rebuild="--rebuild" in sys.argv)
    db.close()
    print(f"Successfully processed and categorized {count} new records!")
