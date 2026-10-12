"""Phase 1: bot elimination, then ticker extraction.

Sources: StockTwits (social) + RSS and Finnhub (news). Reddit is no longer used.

Order, as in the plan:
  1. BOT ELIMINATION (StockTwits only; news has no accounts)
       - account younger than MIN_ACCOUNT_AGE_DAYS when it posted
       - "karma" equivalent: fewer than MIN_FOLLOWERS followers
       - hyperactive account: lifetime posts / account age above MAX_POSTS_PER_DAY
       - rate cap: more than MAX_POSTS_PER_MINUTE posts inside any 60 seconds
         (the whole account is discarded)
       - copy-paste spam: same text posted by COPYPASTE_MIN_AUTHORS+ different accounts
  2. TICKER EXTRACTION, most trusted first, always limited to US common stocks:
       - StockTwits symbol / Finnhub source tag / $CASHTAGS      (categorizer.extract_tickers)
       - exchange tags "(NASDAQ: NVDA)" and company names          (categorizer.extract_tickers)
       - "(NVDA)" in parentheses, if it is a real US ticker and not a common word
       - spaCy NER: ORG entities in headlines, matched to company names from the
         US universe (e.g. "Nvidia" -> NVDA)
     Plain ALL-CAPS words are never treated as tickers on their own.

Output: one row per (post, ticker) mention, plus a per-ticker summary.
`chatter_score` = StockTwits posts + `unbiased_news` (articles from RSS or Finnhub's
general feed that name the company). It is the number to rank by, because we did not
choose those tickers when fetching. Per-company Finnhub articles are NOT counted:
`news_text_mentions` includes them, and `news_query_only` counts articles tagged with a
ticker only because our own query asked for it. Both mostly measure what we downloaded.
Every run also stores the summary in phase1_history, the baseline for Phase 2's z-score.

Run:  python filtering_of_data/phase1_filter.py
Requires: python data_base_creation/trigger.py has been run (US universe + data).
Optional: python -m spacy download en_core_web_sm   (without it the NER step is skipped)
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from data_base_creation.database import (
    SessionLocal, RawNewsPost, StockTwitsPost, USUniverse, Phase1History, init_db,
)
from data_base_creation.us_universe import load_us_symbols, ALLOWED_TYPES, ALLOWED_MICS
from data_base_creation.categorizer import (
    extract_tickers, COMPANY_NAMES, NOT_TICKERS, MAX_TICKERS_PER_POST, QUERY_PREFIXES,
)

# ----------------------------------------------------------------------
# Settings (starting points: tune them using the drop counts printed below)
# ----------------------------------------------------------------------
LOOKBACK_DAYS = 7
MIN_ACCOUNT_AGE_DAYS = 30
MIN_FOLLOWERS = 3
MAX_POSTS_PER_DAY = 150        # lifetime posts / account age in days
MAX_POSTS_PER_MINUTE = 5
COPYPASTE_MIN_AUTHORS = 3
MIN_COPYPASTE_LENGTH = 15      # ignore very short texts such as "buy" or "lol"

# Share classes of one company count as one ticker (the first is kept)
SHARE_CLASS_ALIASES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "UA": "UAA", "ZG": "Z"}

PAREN_RE = re.compile(r"\(([A-Z]{2,5})\)")
# The Finnhub collector appends "$SYMBOL" tags to every article it fetches for SYMBOL
TRAILING_TAGS_RE = re.compile(r"(?:\s+\$[A-Z]{1,5})+\s*$")
SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|"
    r"group|class [a-c]|cl [a-c]|common stock|ordinary shares|trust|sa|nv|ag|lp|llc|"
    r"sponsored adr|spons adr|spon adr|sp adr|adr|ads|american depositary shares?|se|asa|ab|spa|kk|"
    r"a s|ny reg shs|reg shs|ny shs)\b"
)
# Company-name fragments that are ordinary words; never match these as companies
NAME_STOPLIST = {"target", "general", "first", "american", "united", "national", "energy",
                 "global", "capital", "technologies", "systems", "international",
                 "financial", "bancorp", "partners", "resources", "industries"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_social(session, since) -> pd.DataFrame:
    rows = session.query(StockTwitsPost).filter(StockTwitsPost.created_at >= since).all()
    df = pd.DataFrame([{
        "post_id": r.id, "symbol": r.symbol, "author": r.author, "body": r.body or "",
        "st_sentiment": r.sentiment, "created_at": r.created_at,
        "author_join_date": r.author_join_date,
        "author_followers": r.author_followers,
        "author_post_count": r.author_post_count,
    } for r in rows])
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["author_join_date"] = pd.to_datetime(df["author_join_date"], errors="coerce")
    df["author_followers"] = pd.to_numeric(df["author_followers"], errors="coerce")
    df["author_post_count"] = pd.to_numeric(df["author_post_count"], errors="coerce")
    return df


def load_news(session, since) -> pd.DataFrame:
    rows = session.query(RawNewsPost).filter(RawNewsPost.published_at >= since).all()
    return pd.DataFrame([{
        "post_id": r.id, "source": r.source or "", "title": r.title or "",
        "body": r.body or "", "created_at": r.published_at,
    } for r in rows])


# ----------------------------------------------------------------------
# Step 1: bot elimination (StockTwits)
# ----------------------------------------------------------------------
def normalise_text(text: str) -> str:
    text = re.sub(r"\$[a-z]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text)).strip()


def rate_flagged_authors(social: pd.DataFrame) -> set:
    """Authors with more than MAX_POSTS_PER_MINUTE posts inside any 60 seconds."""
    d = social.sort_values(["author", "created_at"])
    later = d.groupby("author")["created_at"].shift(-MAX_POSTS_PER_MINUTE)
    span = later - d["created_at"]
    return set(d.loc[span <= pd.Timedelta(seconds=60), "author"])


def copypaste_mask(social: pd.DataFrame) -> pd.Series:
    """Rows whose text was posted by COPYPASTE_MIN_AUTHORS or more different accounts."""
    key = social["body"].map(normalise_text)
    long_enough = key.str.len() >= MIN_COPYPASTE_LENGTH
    tmp = pd.DataFrame({"key": key, "author": social["author"]})[long_enough]
    n_authors = tmp.groupby("key")["author"].nunique()
    bad_keys = set(n_authors[n_authors >= COPYPASTE_MIN_AUTHORS].index)
    return long_enough & key.isin(bad_keys)


def eliminate_bots(social: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if social.empty:
        return social

    age_days = (social["created_at"] - social["author_join_date"]).dt.days
    has_age = social["author_join_date"].notna()

    too_new = has_age & (age_days < MIN_ACCOUNT_AGE_DAYS)
    low_followers = social["author_followers"].notna() & (social["author_followers"] < MIN_FOLLOWERS)
    posts_per_day = social["author_post_count"] / age_days.clip(lower=1)
    hyperactive = has_age & (posts_per_day > MAX_POSTS_PER_DAY)
    too_fast = social["author"].isin(rate_flagged_authors(social))
    copy_spam = copypaste_mask(social)

    bad = too_new | low_followers | hyperactive | too_fast | copy_spam

    if verbose:
        print(f"  StockTwits posts in window: {len(social)}")
        print(f"    account < {MIN_ACCOUNT_AGE_DAYS} days old : {int(too_new.sum())}")
        print(f"    followers < {MIN_FOLLOWERS}            : {int(low_followers.sum())}")
        print(f"    > {MAX_POSTS_PER_DAY} posts/day lifetime  : {int(hyperactive.sum())}")
        print(f"    > {MAX_POSTS_PER_MINUTE} posts/minute         : {int(too_fast.sum())}")
        print(f"    copy-paste across accounts : {int(copy_spam.sum())}")
        print(f"    (rules overlap) removed {int(bad.sum())}, kept {int((~bad).sum())}")
        missing = int((~has_age).sum())
        if missing:
            print(f"    NOTE: {missing} posts have no join date, so the age rule could not be applied.")
    return social[~bad].copy()


# ----------------------------------------------------------------------
# Step 2: ticker extraction
# ----------------------------------------------------------------------
def normalise_name(description: str) -> str:
    s = re.sub(r"[^a-z0-9& ]", " ", (description or "").lower())
    s = SUFFIX_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_name_map(session) -> dict:
    """normalised company name -> ticker, from the US universe descriptions."""
    rows = (
        session.query(USUniverse.symbol, USUniverse.description)
        .filter(USUniverse.type.in_(ALLOWED_TYPES), USUniverse.mic.in_(ALLOWED_MICS))
        .all()
    )
    by_name = {}
    for symbol, desc in rows:
        name = normalise_name(desc)
        if len(name) < 4 or name in NAME_STOPLIST:
            continue
        by_name.setdefault(name, []).append(symbol)
    # Several share classes (GOOG/GOOGL): pick the shortest ticker, then override by hand
    name_map = {n: sorted(s, key=lambda x: (len(x), x))[0] for n, s in by_name.items()}
    name_map.update(COMPANY_NAMES)
    return name_map


def load_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as err:
        print(f"  NOTE: spaCy NER unavailable ({err}); skipping the NER step. "
              "Install with: python -m spacy download en_core_web_sm")
        return None


def ner_tickers(nlp, titles, name_map) -> list:
    """For each headline: tickers for ORG entities that match a known company name."""
    if nlp is None:
        return [[] for _ in titles]
    out = []
    for doc in nlp.pipe(titles, batch_size=64):
        found = []
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            ticker = name_map.get(normalise_name(ent.text)) or COMPANY_NAMES.get(ent.text.lower())
            if ticker:
                found.append(ticker)
        out.append(list(dict.fromkeys(found)))
    return out


def extract_social_mentions(social: pd.DataFrame, us_symbols: set) -> list:
    rows = []
    for r in social.itertuples(index=False):
        tickers = extract_tickers("StockTwits", r.body, True, us_symbols)
        if r.symbol in us_symbols and r.symbol not in tickers:
            tickers.insert(0, r.symbol)
        if not tickers or len(tickers) > MAX_TICKERS_PER_POST:
            continue
        for t in tickers:
            rows.append({
                "post_id": r.post_id, "ticker": t, "source_type": "SOCIAL",
                "source_name": "StockTwits", "author": r.author,
                "created_at": r.created_at, "text": r.body[:500],
                "st_sentiment": r.st_sentiment, "origin": "SOCIAL", "unbiased": True,
            })
    return rows


def extract_news_mentions(news: pd.DataFrame, us_symbols: set, name_map: dict, nlp) -> list:
    """origin = TEXT  : the article itself names the company/ticker (real chatter)
       origin = QUERY : the ticker is only there because OUR Finnhub query asked for it
                        (it says how much we fetched, not how much people discuss it)"""
    if news.empty:
        return []
    ner = ner_tickers(nlp, news["title"].tolist(), name_map)
    rows = []
    for r, ner_found in zip(news.itertuples(index=False), ner):
        text = f"{r.title} {r.body}"
        is_finnhub = r.source.startswith(QUERY_PREFIXES)   # fetched for a specific ticker

        # Tickers the article text supports on its own (ignoring our appended tags)
        clean = TRAILING_TAGS_RE.sub("", text) if is_finnhub else text
        organic = set(extract_tickers("RSS", clean, False, us_symbols))
        organic.update(t for t in PAREN_RE.findall(clean) if t in us_symbols and t not in NOT_TICKERS)
        organic.update(t for t in ner_found if t in us_symbols)

        tickers = extract_tickers(r.source, text, is_finnhub, us_symbols)
        tickers += [t for t in PAREN_RE.findall(text) if t in us_symbols and t not in NOT_TICKERS]
        tickers += [t for t in ner_found if t in us_symbols]
        tickers = list(dict.fromkeys(tickers))
        if not tickers or len(tickers) > MAX_TICKERS_PER_POST:
            continue

        source_name = r.source.split("_")[0]
        for t in tickers:
            rows.append({
                "post_id": r.post_id, "ticker": t, "source_type": "NEWS",
                "source_name": source_name, "author": source_name,
                "created_at": r.created_at, "text": r.title[:500],
                "st_sentiment": None, "origin": "TEXT" if t in organic else "QUERY",
                # Unbiased = we did not choose the ticker when fetching: RSS, or Finnhub's
                # general feed. Per-company Finnhub articles only measure what we downloaded.
                "unbiased": (t in organic) and (not is_finnhub or r.source == "Finnhub_general"),
            })
    return rows


# ----------------------------------------------------------------------
# Summary and entry point
# ----------------------------------------------------------------------
def summarize(mentions: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "chatter_score", "social_mentions", "unbiased_news",
            "news_text_mentions", "news_query_only", "organic_mentions",
            "unique_authors", "unique_author_ratio", "total_mentions"]
    if mentions.empty:
        return pd.DataFrame(columns=cols)
    social = mentions[mentions["source_type"] == "SOCIAL"]
    news = mentions[mentions["source_type"] == "NEWS"]
    news_text = news[news["origin"] == "TEXT"]
    news_query = news[news["origin"] == "QUERY"]
    news_unbiased = news[news["unbiased"]]
    organic = mentions[mentions["origin"] != "QUERY"]

    out = pd.DataFrame({"total_mentions": mentions.groupby("ticker")["post_id"].nunique()})
    out["organic_mentions"] = organic.groupby("ticker")["post_id"].nunique()
    out["social_mentions"] = social.groupby("ticker")["post_id"].nunique()
    out["unbiased_news"] = news_unbiased.groupby("ticker")["post_id"].nunique()
    out["news_text_mentions"] = news_text.groupby("ticker")["post_id"].nunique()
    out["news_query_only"] = news_query.groupby("ticker")["post_id"].nunique()
    out["unique_authors"] = social.groupby("ticker")["author"].nunique()
    out = out.fillna(0).astype(int)
    # chatter_score = what people actually said: StockTwits posts + news we did not select by ticker
    out["chatter_score"] = out["social_mentions"] + out["unbiased_news"]
    # Author ratio is only meaningful for social posts (news "authors" are publishers)
    out["unique_author_ratio"] = (
        out["unique_authors"] / out["social_mentions"].where(out["social_mentions"] > 0)
    ).round(2)
    out = out.sort_values(["chatter_score", "organic_mentions"], ascending=False)
    return out.reset_index()[cols]


def save_snapshot(summary: pd.DataFrame, snapshot_date=None) -> int:
    """Store today's summary in phase1_history (replaces rows for the same date)."""
    if summary.empty:
        return 0
    init_db()
    snapshot_date = snapshot_date or _utcnow().date()
    session = SessionLocal()
    try:
        session.query(Phase1History).filter(Phase1History.snapshot_date == snapshot_date).delete()
        session.bulk_insert_mappings(Phase1History, [{
            "snapshot_date": snapshot_date,
            "ticker": r.ticker,
            "chatter_score": int(r.chatter_score),
            "social_mentions": int(r.social_mentions),
            "unbiased_news": int(r.unbiased_news),
            "unique_authors": int(r.unique_authors),
            "unique_author_ratio": None if pd.isna(r.unique_author_ratio) else float(r.unique_author_ratio),
        } for r in summary.itertuples(index=False)])
        session.commit()
        return len(summary)
    except Exception as err:
        session.rollback()
        print(f"  Could not save phase1 snapshot: {err}")
        return 0
    finally:
        session.close()


LAST_STATS = {}     # counts from the latest run, read by run_pipeline.py for the provenance report


def run_phase1(lookback_days: int = LOOKBACK_DAYS, verbose: bool = True, save_history: bool = True):
    """Returns (mentions, summary). `mentions` has one row per (post, ticker).
    With save_history=True the summary is also stored in phase1_history (one snapshot per day)."""
    session = SessionLocal()
    try:
        us_symbols = load_us_symbols(session)
        if not us_symbols:
            raise RuntimeError("US universe is empty. Run data_base_creation/trigger.py first.")

        since = _utcnow() - timedelta(days=lookback_days)
        if verbose:
            print("=== Phase 1: Bot elimination + ticker extraction ===")
            print(f"Window: last {lookback_days} days | US stocks in universe: {len(us_symbols)}")

        social = load_social(session, since)
        news = load_news(session, since)
        if verbose:
            print(f"\n[1] Bot elimination")
            print(f"  News articles in window: {len(news)} (no accounts, so no bot rules; "
                  "duplicates were removed at collection)")
        n_social_raw = len(social)
        social = eliminate_bots(social, verbose)
        LAST_STATS.update(news_in_window=len(news), social_in_window=n_social_raw,
                          social_after_bots=len(social), lookback_days=lookback_days)

        if verbose:
            print("\n[2] Ticker extraction")
        name_map = build_name_map(session)
        nlp = load_nlp()
        rows = extract_social_mentions(social, us_symbols) if not social.empty else []
        rows += extract_news_mentions(news, us_symbols, name_map, nlp)
    finally:
        session.close()

    mentions = pd.DataFrame(rows)
    if not mentions.empty:
        mentions["ticker"] = mentions["ticker"].replace(SHARE_CLASS_ALIASES)
        mentions = mentions.drop_duplicates(["post_id", "ticker"]).reset_index(drop=True)
    summary = summarize(mentions)
    if save_history and not summary.empty:
        saved = save_snapshot(summary)
        if verbose:
            print(f"  Saved {saved} rows to phase1_history (today's snapshot).")
    if verbose:
        print(f"  Mentions isolated: {len(mentions)} across {summary.shape[0]} tickers")
        print("\nTop 15 tickers after Phase 1:")
        print(summary.head(15).to_string(index=False))
    return mentions, summary


if __name__ == "__main__":
    run_phase1()
